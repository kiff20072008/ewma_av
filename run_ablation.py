"""
Вклад компонентов схемы: отключение приостановки обновления и пересчёта (5);
доля отсчётов в приостановке и наибольшая её длительность на участке оценки UGR'16.

    python run_ablation.py --repo ugr16 --boun results_boun --out results_ablation

Протокол тот же, что в run_ugr16.py и run_boun.py (калибровка, участки
оценки, разогрев). Сравниваются три варианта предлагаемой схемы:
  * base       — схема целиком (параметры по умолчанию);
  * no_freeze  — без приостановки обновления (порог приостановки L = ∞);
  * no_correct — без пересчёта оценки к масштабу процесса (формула (5) статьи).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ewma_av.detectors import EwmaAV
from ewma_av.boun import schedule_labels, MONITORED
import run_ugr16 as U
import run_boun as B

VARIANTS = {"base": {}, "no_freeze": {"L": float("inf")},
            "no_correct": {"bias_correction": False}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ugr16", help="клон UGR16_FeatureData")
    ap.add_argument("--boun", default="results_boun", help="папка с counters_*.csv")
    ap.add_argument("--out", default="results_ablation")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    X = pd.read_csv(Path(a.repo) / "csv" / "UGR16v1.Xtest.csv", usecols=[U.MAIN_METRIC])
    Y = pd.read_csv(Path(a.repo) / "csv" / "UGR16v1.Ytest.csv")
    x, truth = X[U.MAIN_METRIC].to_numpy(float), Y["labeldos"].to_numpy() > 0
    for v, kw in VARIANTS.items():
        det = EwmaAV(**kw)
        r, *_ = U.evaluate(det.score(x), truth)
        fr = det.frozen_[U.WARM:U.EVAL_END]
        longest, cur = 0, 0
        for f in fr:
            cur = cur + 1 if f else 0
            longest = max(longest, cur)
        rows.append({"data": "UGR16 " + U.MAIN_METRIC, "variant": v, **{
            k: r[k] for k in ("P", "R", "F1", "FPR", "AUC_ROC", "AUC_PR", "events_detected")},
            "freeze_share_eval": float(fr.mean()), "freeze_longest_min": int(longest)})

    for f in sorted(Path(a.boun).glob("counters_*.csv")):
        cnt = pd.read_csv(f)
        metric = MONITORED["udp"] if "udp" in f.stem.lower() else MONITORED["tcp"]
        t, ig = schedule_labels(len(cnt))
        xb = cnt[metric].to_numpy(float)
        for v, kw in VARIANTS.items():
            r = B.evaluate(EwmaAV(warmup=B.WARMUP, **kw), xb, t, ig)
            rows.append({"data": f"BOUN {f.stem.replace('counters_', '')} {metric}", "variant": v,
                         **{k: r[k] for k in ("P", "R", "F1", "FPR", "AUC_ROC", "AUC_PR")},
                         "events_detected": r["attacks_detected"],
                         "seconds_per_attack": r["seconds_per_attack"]})

    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nрезультаты записаны в {out}/")


if __name__ == "__main__":
    main()
