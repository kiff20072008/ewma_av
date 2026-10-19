"""
Граница применимости: медленно нарастающая (трендовая) аномалия на
синтетическом ряде с известной разметкой.

    python run_synthetic.py --runs 30 --out results_synthetic

Модель ряда (ewma_av/generators.py, сценарий "Slow DDoS"):
  * 20 160 одноминутных отсчётов (14 сут), базовый уровень A = 1000;
  * суточная синусоида с относительной амплитудой 0,4;
  * мультипликативный логнормальный шум с коэффициентом вариации 0,1;
  * аномалия: линейное нарастание от 0 до 0,8 A (т. е. до 8 условных СКО
    шума 0,1 A) за 3000 отсчётов, начиная с середины ряда.
Протокол: порог каждого метода — по первой трети ряда (в ней аномалии нет)
на долю превышений 0,002; показатели — по оставшимся двум третям.
Хольту — Уинтерсу для фазы I дан отдельный чистый ряд той же модели длиной
10 сут (другое начальное значение генератора).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ewma_av.generators import SCENARIOS, TrafficConfig, generate
from ewma_av.detectors import (EwmaAV, SigmaRule, EwmaFixed, AEWMA,
                               Cusum, HoltWinters, HampelMAD)
from ewma_av import metrics as M

TARGET_FPR = 0.002


def run_once(seed: int):
    base = SCENARIOS["Slow DDoS"]
    cfg = TrafficConfig(**{**base.to_dict(), "seed": seed})
    x, truth = generate(cfg)
    tr_cfg = TrafficConfig(**{**base.to_dict(), "seed": seed + 100000,
                              "anomaly_kind": "none", "n_anomalies": 0,
                              "n": 10 * cfg.period_day})
    train, _ = generate(tr_cfg)
    n = len(x)
    cal = slice(0, n // 3)
    ev = slice(n // 3, n)
    assert not truth[cal].any()
    dets = [EwmaAV(), AEWMA(0.2, 3.0), Cusum(), HampelMAD(200), SigmaRule(200),
            HoltWinters(train, cfg.period_day), EwmaFixed(0.05), EwmaFixed(0.2)]
    rows = []
    for d in dets:
        s = d.score(x)
        thr = float(np.quantile(s[cal], 1 - TARGET_FPR))
        p, t = s[ev] > thr, truth[ev]
        r = {"method": d.name, "seed": seed}
        r.update(M.point_metrics(p, t))
        r.update(M.auc_scores(s[ev], t))
        r["delay"] = M.detection_delay(p, t)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default="results_synthetic")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(a.runs):
        rows += run_once(1000 + k)
        if (k + 1) % 5 == 0:
            print(f"  прогонов: {k + 1}/{a.runs}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "slow_ramp_runs.csv", index=False)
    g = df.groupby("method")
    summ = pd.DataFrame({
        "F1_mean": g["F1"].mean(),
        "F1_ci95": 1.96 * g["F1"].std(ddof=1) / np.sqrt(g.size()),
        "R_mean": g["R"].mean(),
        "AUC_PR_mean": g["AUC_PR"].mean(),
        "delay_mean": g["delay"].mean(),
        "runs": g.size(),
    }).sort_values("F1_mean", ascending=False)
    summ.to_csv(out / "table_slow_ramp.csv")
    print(summ.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
