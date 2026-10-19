"""
Метрики и протокол сравнения.

Ключевые принципы, отвечающие на замечания рецензента:

1. Единая калибровка.  Порог каждого метода подбирается так, чтобы на
   калибровочном участке достигалась одна и та же целевая доля ложных
   срабатываний.  Без этого сравнение F1 между методами лишено смысла.

2. Две схемы подсчёта.  Поточечная (point-based) и посегментная
   (event-based): атака считается обнаруженной, если помечен хотя бы один
   отсчёт внутри её окна.  Посегментная схема ближе к практике эксплуатации.

3. AUC-PR наряду с AUC-ROC.  При редких аномалиях ROC-кривая даёт
   оптимистичную оценку, поэтому приводятся обе.

4. Доверительные интервалы и проверка значимости — результаты по множеству
   независимых прогонов, а не по одному.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

__all__ = [
    "calibrate_threshold", "point_metrics", "event_metrics",
    "auc_scores", "detection_delay", "arl0",
    "aggregate", "wilcoxon_holm",
]


def calibrate_threshold(score: np.ndarray, target_fpr: float,
                        clean_mask: np.ndarray | None = None) -> float:
    """Порог, дающий заданную долю превышений на нормальных отсчётах.

    clean_mask — булев массив «отсчёт заведомо нормальный».  Если не задан,
    калибровка ведётся по всему ряду (допустимо лишь при малой доле аномалий).
    """
    s = score if clean_mask is None else score[clean_mask]
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.inf
    return float(np.quantile(s, 1.0 - target_fpr))


def point_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Поточечные Precision / Recall / F1 / FPR."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"P": p, "R": r, "F1": f1, "FPR": fpr,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Границы связных участков True в булевом массиве."""
    mask = mask.astype(bool)
    if not mask.any():
        return []
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [mask.size]
    return list(zip(starts, ends))


def event_metrics(pred: np.ndarray, truth: np.ndarray,
                  tolerance: int = 0) -> dict:
    """Посегментные метрики: событие найдено, если помечен хоть один отсчёт.

    tolerance — допустимое расширение окна события в обе стороны (отсчётов).
    """
    pred, truth = pred.astype(bool), truth.astype(bool)
    events = _segments(truth)
    n = truth.size
    detected = 0
    covered = np.zeros(n, dtype=bool)
    for a, b in events:
        lo, hi = max(0, a - tolerance), min(n, b + tolerance)
        covered[lo:hi] = True
        if pred[lo:hi].any():
            detected += 1
    recall = detected / len(events) if events else float("nan")
    false_alarms = _segments(pred & ~covered)
    n_fa = len(false_alarms)
    n_normal = int(np.sum(~covered))
    precision = (detected / (detected + n_fa)) if (detected + n_fa) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if events and precision + recall else 0.0)
    return {"event_R": recall, "event_P": precision, "event_F1": f1,
            "n_events": len(events), "n_detected": detected,
            "n_false_alarm_segments": n_fa,
            "false_alarm_rate_per_1000": 1000.0 * n_fa / max(n_normal, 1)}


def auc_scores(score: np.ndarray, truth: np.ndarray) -> dict:
    """AUC-ROC и AUC-PR (average precision)."""
    t = truth.astype(int)
    if t.min() == t.max():
        return {"AUC_ROC": float("nan"), "AUC_PR": float("nan")}
    s = np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=0.0)
    return {"AUC_ROC": float(roc_auc_score(t, s)),
            "AUC_PR": float(average_precision_score(t, s))}


def detection_delay(pred: np.ndarray, truth: np.ndarray) -> float:
    """Средняя задержка обнаружения в отсчётах (по обнаруженным событиям)."""
    delays = []
    for a, b in _segments(truth):
        idx = np.where(pred[a:b])[0]
        if idx.size:
            delays.append(int(idx[0]))
    return float(np.mean(delays)) if delays else float("nan")


def arl0(pred: np.ndarray, truth: np.ndarray) -> float:
    """Средняя длина серии до ложной тревоги на нормальных участках.

    Важно: из-за механизма заморозки ложные тревоги не независимы, поэтому
    доля ложных срабатываний по отсчётам не является биномиальной и ARL_0
    даёт более честную картину.
    """
    normal = ~truth.astype(bool)
    idx = np.where(pred & normal)[0]
    if idx.size < 2:
        return float(np.sum(normal))
    return float(np.mean(np.diff(idx)))


def aggregate(rows: list[dict]) -> dict:
    """Среднее и полуширина 95 % доверительного интервала по прогонам."""
    out = {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out[k], out[k + "_ci95"] = float("nan"), float("nan")
            continue
        m = float(v.mean())
        half = 1.96 * float(v.std(ddof=1)) / np.sqrt(v.size) if v.size > 1 else 0.0
        out[k], out[k + "_ci95"] = m, half
    out["n_runs"] = len(rows)
    return out


def wilcoxon_holm(results: dict[str, list[float]], reference: str) -> dict:
    """Попарное сравнение с эталонным методом, поправка Холма на множественность.

    results   — {имя метода: список значений метрики по прогонам}
    reference — имя метода, с которым сравниваются остальные
    """
    from scipy.stats import wilcoxon
    others = [k for k in results if k != reference]
    raw = {}
    for k in others:
        a = np.asarray(results[reference], float)
        b = np.asarray(results[k], float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5 or np.allclose(a[m], b[m]):
            raw[k] = float("nan")
            continue
        try:
            raw[k] = float(wilcoxon(a[m], b[m]).pvalue)
        except ValueError:
            raw[k] = float("nan")
    # поправка Холма
    valid = [(k, p) for k, p in raw.items() if np.isfinite(p)]
    valid.sort(key=lambda kv: kv[1])
    m = len(valid)
    adj, prev = {}, 0.0
    for i, (k, p) in enumerate(valid):
        val = min(1.0, max(prev, (m - i) * p))
        adj[k] = val
        prev = val
    for k, p in raw.items():
        if not np.isfinite(p):
            adj[k] = float("nan")
    return {"p_raw": raw, "p_holm": adj}
