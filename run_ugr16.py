"""
Эксперимент на реальных данных UGR'16 (одноминутная агрегация).

    git clone --depth 1 https://github.com/josecamachop/UGR16_FeatureData.git ugr16
    python run_ugr16.py --repo ugr16 --out results_ugr16

Протокол
--------
Нестационарность (обучающая часть, 19.03–25.06.2016):
  * суточный и недельный профили — по всем 134 262 отсчётам;
  * периодограмма — на непрерывном участке длиной ровно 11 недель
    (10.04–25.06.2016). В обучающей части есть пропуск 5,7 сут (01–06.04),
    склейка ряда через него сдвигает суточную фазу; длина, кратная неделе,
    совмещает суточную и недельную частоты с узлами частотной сетки.

Сравнение детекторов (тестовая часть, 28.07–26.08.2016):
  * все 450 минут с атаками DoS приходятся на 28.07–09.08 (первые 13 сут);
  * порог каждого метода — квантиль его статистики на участке без атак
    10.08–26.08 (17 сут), дающий долю превышений 0,002 (калибровка);
  * показатели качества — на участке 28.07–09.08, без первых 240 мин
    (разогрев: наибольшее окно инициализации среди методов — 200 мин);
  * участки калибровки и оценки не пересекаются;
  * Хольт — Уинтерс оценивается по самому длинному непрерывному участку
    обучающей части (06.04–25.06, 80,3 сут) и затем идёт по тестовому ряду
    онлайн; остальные методы обучающей выборки не используют;
  * 95 % доверительные интервалы — бутстрэп по суткам участка оценки
    (2000 повторений), в том числе для разности с AEWMA.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

from ewma_av.ugr16_minute import load_ugr16_minute
from ewma_av.datasets import profile_by_hour, profile_by_weekday
from ewma_av.detectors import (EwmaAV, SigmaRule, EwmaFixed, AEWMA,
                               Cusum, HoltWinters, HampelMAD)
from ewma_av import metrics as M

TARGET_FPR = 0.002
PERIOD = 1440
WARM = 240                    # минут разогрева, не входят в оценку
EVAL_END = 13 * PERIOD        # 28.07 00:00 — 09.08 23:59
N_BOOT = 2000
MAIN_METRIC = "dporthttp"
OTHER_METRICS = ["tcpflagsSYN", "protocoltcp"]


def contiguous_segments(ts: pd.DatetimeIndex):
    d = np.diff(ts.values).astype("timedelta64[m]").astype(int)
    cuts = np.where(d > 1)[0]
    b = [0] + list(cuts + 1) + [len(ts)]
    return list(zip(b[:-1], b[1:]))


def periodogram_11w(x, ts):
    a, b = max(contiguous_segments(ts), key=lambda ab: ab[1] - ab[0])
    n = 11 * 7 * PERIOD
    v = np.array(x[b - n:b], dtype=float)
    v -= v.mean()
    f = np.fft.rfftfreq(n, d=60.0)[1:]
    p = (np.abs(np.fft.rfft(v)) ** 2)[1:]
    df = pd.DataFrame({"period_hours": 1 / f / 3600, "power_pct": 100 * p / p.sum()})
    return df, ts[b - n], ts[b - 1]


def load_train(repo, metric):
    X = pd.read_csv(Path(repo) / "csv" / "UGR16v1.Xtrain.csv", usecols=["Row", metric])
    ts = pd.to_datetime(X["Row"].astype(str), format="%Y%m%d%H%M")
    o = np.argsort(ts.values)
    X, ts = X.iloc[o].reset_index(drop=True), pd.DatetimeIndex(ts.values[o])
    a, b = max(contiguous_segments(ts), key=lambda ab: ab[1] - ab[0])
    return X[metric].to_numpy(float)[a:b], ts[a], ts[b - 1]


def detectors(train):
    return [EwmaAV(), AEWMA(0.2, 3.0), Cusum(), HampelMAD(200), SigmaRule(200),
            HoltWinters(train, PERIOD), EwmaFixed(0.05), EwmaFixed(0.2)]


def evaluate(score, truth):
    """Порог по участку калибровки, показатели — по участку оценки."""
    calib = np.arange(EVAL_END, len(score))
    assert not truth[calib].any(), "на участке калибровки есть атаки"
    thr = float(np.quantile(score[calib], 1 - TARGET_FPR))
    ev = slice(WARM, EVAL_END)
    s, t = score[ev], truth[ev]
    p = s > thr
    r = M.point_metrics(p, t)
    r.update(M.auc_scores(s, t))
    e = M.event_metrics(p, t, tolerance=0)
    r.update({"events_detected": e["n_detected"], "events_total": e["n_events"]})
    r["threshold"] = thr
    return r, s, t, p


def f1(p, t):
    tp = np.sum(p & t)
    fp = np.sum(p & ~t)
    fn = np.sum(~p & t)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def bootstrap(per_method, t, ref="EWMA-AV", other="AEWMA (Capizzi-Masarotto)"):
    """Бутстрэп по суткам участка оценки: ДИ показателей и разностей."""
    rng = np.random.default_rng(20261019)
    idx = np.arange(WARM, EVAL_END) // PERIOD          # номер суток
    days = np.unique(idx)
    pos = {d: np.flatnonzero(idx == d) for d in days}
    names = list(per_method)
    stats = {n: {"F1": [], "AUC_ROC": [], "AUC_PR": []} for n in names}
    diffs = {n: {"F1": [], "AUC_ROC": [], "AUC_PR": []} for n in names if n != ref}
    for _ in range(N_BOOT):
        pick = np.concatenate([pos[d] for d in rng.choice(days, size=days.size)])
        tt = t[pick]
        if tt.min() == tt.max():
            continue
        cur = {}
        for n in names:
            s, p = per_method[n]
            cur[n] = {"F1": f1(p[pick], tt),
                      "AUC_ROC": roc_auc_score(tt, s[pick]),
                      "AUC_PR": average_precision_score(tt, s[pick])}
            for k, v in cur[n].items():
                stats[n][k].append(v)
        for n in diffs:
            for k in ("F1", "AUC_ROC", "AUC_PR"):
                diffs[n][k].append(cur[ref][k] - cur[n][k])

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return ({n: {k: ci(v) for k, v in d.items()} for n, d in stats.items()},
            {n: {k: ci(v) + [float(np.mean(np.asarray(v) > 0))] for k, v in d.items()}
             for n, d in diffs.items()})


def sensitivity(x, truth):
    grid = {"beta": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
            "L": [2.5, 3.0, 3.5, 4.0],
            "lam_min": [0.01, 0.02, 0.05, 0.1],
            "lam_max": [0.2, 0.3, 0.4, 0.5],
            "k_release": [1, 2, 3, 5]}
    base = dict(beta=0.02, L=3.0, lam_min=0.05, lam_max=0.30, k_release=2)
    rows = []
    for prm, vals in grid.items():
        for v in vals:
            kw = dict(base)
            kw[prm] = v
            r, *_ = evaluate(EwmaAV(**kw).score(x), truth)
            rows.append({"param": prm, "value": v, "F1": r["F1"], "AUC_ROC": r["AUC_ROC"],
                         "AUC_PR": r["AUC_PR"], "FPR": r["FPR"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ugr16")
    ap.add_argument("--out", default="results_ugr16")
    ap.add_argument("--skip-bootstrap", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ---------- 1. нестационарность ----------
    x, _, ts = load_ugr16_minute(a.repo, split="train")
    ph, pw = profile_by_hour(x, ts), profile_by_weekday(x, ts)
    pg, t0, t1 = periodogram_11w(x, ts)
    ph.to_csv(out / "profile_hour.csv")
    pw.to_csv(out / "profile_weekday.csv")
    pg.to_csv(out / "periodogram_11w.csv", index=False)
    top = pg.sort_values("power_pct", ascending=False).head(6)
    summary["nonstationarity"] = {
        "n_train": int(len(x)), "train_from": str(ts.min()), "train_to": str(ts.max()),
        "night_to_peak": float(ph["ratio_to_peak"].min()),
        "hour_min": int(ph["ratio_to_peak"].idxmin()), "hour_max": int(ph["ratio_to_peak"].idxmax()),
        "cv_min": float(ph["cv"].min()), "cv_max": float(ph["cv"].max()),
        "weekend_to_weekday": float(pw.loc[[5, 6], "median"].mean() / pw.loc[[0, 1, 2, 3, 4], "median"].mean()),
        "minday_to_maxday": float(pw["median"].min() / pw["median"].max()),
        "spectrum_segment": [str(t0), str(t1)],
        "spectrum_top": top.round(4).to_dict("records"),
    }
    # тот же суточный профиль для основной контролируемой величины (порт 80)
    Xm = pd.read_csv(Path(a.repo) / "csv" / "UGR16v1.Xtrain.csv", usecols=["Row", MAIN_METRIC])
    tm = pd.to_datetime(Xm["Row"].astype(str), format="%Y%m%d%H%M")
    om = np.argsort(tm.values)
    phm = profile_by_hour(Xm[MAIN_METRIC].to_numpy(float)[om], pd.DatetimeIndex(tm.values[om]))
    phm.to_csv(out / f"profile_hour_{MAIN_METRIC}.csv")
    summary["nonstationarity_" + MAIN_METRIC] = {
        "min_to_max": float(phm["ratio_to_peak"].min()),
        "hour_min": int(phm["ratio_to_peak"].idxmin()), "hour_max": int(phm["ratio_to_peak"].idxmax()),
        "cv_min": float(phm["cv"].min()), "cv_max": float(phm["cv"].max()),
        "std_min": float(phm["std"].min()), "std_max": float(phm["std"].max()),
    }
    print(json.dumps(summary["nonstationarity"], ensure_ascii=False, indent=1))
    print(json.dumps(summary["nonstationarity_" + MAIN_METRIC], ensure_ascii=False, indent=1))

    # ---------- 2. сравнение детекторов ----------
    X = pd.read_csv(Path(a.repo) / "csv" / "UGR16v1.Xtest.csv")
    Y = pd.read_csv(Path(a.repo) / "csv" / "UGR16v1.Ytest.csv")
    truth = Y["labeldos"].to_numpy() > 0
    summary["protocol"] = {
        "n_test": int(len(X)), "dos_minutes": int(truth.sum()),
        "dos_minutes_eval": int(truth[WARM:EVAL_END].sum()),
        "events_eval": int(M.event_metrics(truth[WARM:EVAL_END], truth[WARM:EVAL_END])["n_events"]),
        "dos_in_calibration": int(truth[EVAL_END:].sum()),
    }
    print(summary["protocol"])

    rows, keep = [], {}
    for metric in [MAIN_METRIC] + OTHER_METRICS:
        xs = X[metric].to_numpy(float)
        train, tr0, tr1 = load_train(a.repo, metric)
        for d in detectors(train):
            s = d.score(xs)
            r, se, te, pe = evaluate(s, truth)
            r.update({"metric": metric, "method": d.name})
            if isinstance(d, HoltWinters):
                r.update({"hw_alpha": d.alpha, "hw_beta": d.beta, "hw_gamma": d.gamma,
                          "hw_converged": d.converged, "hw_train": f"{tr0} — {tr1}"})
            rows.append(r)
            if metric == MAIN_METRIC:
                keep[d.name] = (se, pe)
                t_eval = te
        print(f"  метрика {metric}: готово")
    df = pd.DataFrame(rows)
    df.to_csv(out / "comparison_ugr16_full.csv", index=False)
    cols = ["metric", "method", "P", "R", "F1", "FPR", "AUC_ROC", "AUC_PR",
            "events_detected", "events_total"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not a.skip_bootstrap:
        ci, diff = bootstrap(keep, t_eval)
        summary["bootstrap_ci"] = ci
        summary["bootstrap_diff_vs"] = diff
        print("\n95 % ДИ (бутстрэп по суткам):")
        for n, d in ci.items():
            print(f"  {n:<28} " + "  ".join(f"{k} [{v[0]:.3f}; {v[1]:.3f}]" for k, v in d.items()))
        print("\nРазность «предлагаемая − метод», 95 % ДИ и доля повторений с разностью > 0:")
        for n, d in diff.items():
            print(f"  {n:<28} " + "  ".join(f"{k} [{v[0]:+.3f}; {v[1]:+.3f}] p>0={v[2]:.3f}" for k, v in d.items()))

    sens = sensitivity(X[MAIN_METRIC].to_numpy(float), truth)
    sens.to_csv(out / "table_sensitivity_ugr16.csv", index=False)
    summary["sensitivity"] = sens.groupby("param").agg(
        F1_min=("F1", "min"), F1_max=("F1", "max"),
        AUC_PR_min=("AUC_PR", "min"), AUC_PR_max=("AUC_PR", "max"),
        AUC_ROC_min=("AUC_ROC", "min"), AUC_ROC_max=("AUC_ROC", "max")).round(4).to_dict("index")
    print("\nУстойчивость:\n", sens.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    (out / "summary_ugr16.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(f"\nрезультаты записаны в {out}/")


if __name__ == "__main__":
    main()
