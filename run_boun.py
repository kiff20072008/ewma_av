"""
Эксперимент на наборе BOUN DDoS (кампусная сеть университета Богазичи).

    python run_boun.py --dir boun --out results_boun          # из исходных CSV
    python run_boun.py --counters results_boun --out results_boun   # из уже
                                                  # собранных counters_*.csv

Протокол (записи короткие — около 8 мин):
  * первые 60 с — разогрев: по ним оцениваются начальные значения,
    в показатели качества они не входят;
  * 3 с после каждого окна атаки исключены (полоса завершения атаки,
    см. ewma_av/boun.py) — и из калибровки, и из показателей;
  * порог каждого метода — квантиль статистики по нормальным отсчётам
    записи, дающий долю превышений 0,01; калибровка внутривыборочная
    (записи слишком коротки для разделения) и одинакова для всех методов;
    площади под кривыми от порога не зависят;
  * атака считается обнаруженной, если хотя бы одна тревога выдана
    внутри её 20-секундного окна (без допуска);
  * Хольт — Уинтерс не участвует: суточного периода в записи нет.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ewma_av.boun import prepare_boun, schedule_labels, verify_labels, MONITORED
from ewma_av.detectors import (EwmaAV, SigmaRule, EwmaFixed, AEWMA,
                               Cusum, HampelMAD)
from ewma_av import metrics as M

TARGET_FPR = 0.01
WARMUP = 60
WINDOWS = [(80, 100), (180, 200), (280, 300), (380, 400)]


def detectors():
    return [EwmaAV(warmup=WARMUP), AEWMA(0.2, 3.0, warmup=WARMUP),
            Cusum(warmup=WARMUP), HampelMAD(WARMUP), SigmaRule(WARMUP),
            EwmaFixed(0.05, warmup=WARMUP), EwmaFixed(0.2, warmup=WARMUP)]


def evaluate(det, x, truth, ignore):
    s = det.score(x)
    keep = np.ones(len(x), bool)
    keep[:WARMUP] = False
    keep &= ~ignore
    thr = M.calibrate_threshold(s[keep], TARGET_FPR, clean_mask=~truth[keep])
    p_full = (s > thr) & keep
    ss, t, p = s[keep], truth[keep], p_full[keep]
    r = {"method": det.name}
    r.update(M.point_metrics(p, t))
    r.update(M.auc_scores(ss, t))
    per_attack = [int(p_full[a:b].sum()) for a, b in WINDOWS]
    r["seconds_per_attack"] = " ".join(map(str, per_attack))
    r["attacks_detected"] = int(sum(v > 0 for v in per_attack))
    return r


def load_counters(path: Path):
    cnt = pd.read_csv(path)
    name = path.stem.lower()
    metric = MONITORED["udp"] if "udp" in name else MONITORED["tcp"]
    truth, ignore = schedule_labels(len(cnt))
    print(f"файл {path.name}: контролируемая величина {metric}")
    verify_labels(cnt, truth, ignore)
    return cnt[metric].to_numpy(float), truth, ignore, cnt, metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="папка с исходными CSV набора")
    ap.add_argument("--counters", help="папка с counters_*.csv")
    ap.add_argument("--out", default="results_boun")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    items = []
    if a.counters:
        for f in sorted(Path(a.counters).glob("counters_*.csv")):
            x, truth, ignore, cnt, metric = load_counters(f)
            items.append((f.stem.replace("counters_", ""), x, truth, ignore, metric))
    elif a.dir:
        files = [f for f in sorted(Path(a.dir).rglob("*.csv"))
                 if not f.name.startswith(("counters_", "comparison_", "table_"))]
        for f in files:
            x, truth, ignore, cnt, metric = prepare_boun(f)
            cnt.to_csv(out / f"counters_{f.stem}.csv", index=False)
            items.append((f.stem, x, truth, ignore, metric))
    else:
        raise SystemExit("укажите --dir или --counters")

    rows = []
    for name, x, truth, ignore, metric in items:
        bg = x[(~truth) & (~ignore) & (np.arange(len(x)) >= WARMUP)]
        med, sd = float(np.median(bg)), float(bg.std())
        snr = [round(float((x[a:b].mean() - med) / sd), 1) for a, b in WINDOWS]
        print(f"  {name}: прирост при атаках, в СКО фона: {snr}")
        for d in detectors():
            r = evaluate(d, x, truth, ignore)
            r.update({"file": name, "metric": metric, "attack_snr": " ".join(map(str, snr))})
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out / "comparison_boun_full.csv", index=False)
    cols = ["file", "metric", "method", "P", "R", "F1", "FPR", "AUC_ROC", "AUC_PR",
            "attacks_detected", "seconds_per_attack"]
    df[cols].to_csv(out / "table_comparison_boun.csv", index=False)
    pd.set_option("display.width", 220)
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nрезультаты записаны в {out}/")


if __name__ == "__main__":
    main()
