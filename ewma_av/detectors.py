"""
Детекторы аномалий во временном ряду метрики трафика.

Все детекторы имеют единый интерфейс:

    score(x) -> np.ndarray
        Непрерывная статистика аномальности, одно значение на отсчёт.
        Больше = аномальнее. Порог применяется отдельно (см. metrics.calibrate),
        что обеспечивает единую калибровку всех методов по целевому FPR.

Такой интерфейс — принципиальное требование корректного сравнения: методы
сопоставляются при одинаковой доле ложных срабатываний, а не при произвольно
выбранных «родных» порогах каждого метода.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "EwmaAV", "SigmaRule", "EwmaFixed", "AEWMA",
    "Cusum", "HoltWinters", "HampelMAD", "ALL_DETECTORS",
]


def _robust_init(x: np.ndarray, warmup: int) -> tuple[float, float]:
    """Робастная инициализация: медиана и MAD-оценка СКО по окну разогрева.

    Робастность здесь существенна: окно разогрева может содержать аномалию,
    и обычные среднее/СКО были бы ею загрязнены.
    """
    w = x[:warmup]
    z0 = float(np.median(w))
    s0 = float(1.4826 * np.median(np.abs(w - z0)))
    if not np.isfinite(s0) or s0 <= 0:
        s0 = float(np.std(w)) or 1.0
    return z0, s0


class EwmaAV:
    """Адаптивное экспоненциальное сглаживание с онлайн-оценкой дисперсии.

    Схема (номера формул — по статье)
    ---------------------------------
    (1) Z_t = lam_t x_t + (1 - lam_t) Z_{t-1},  e_t = x_t - Z_{t-1},
        r_t = e_t / sigma~_{t-1};  выход score(x) = |r_t|, порог тревоги
        калибруется отдельно по целевой доле ложных срабатываний.
    (2) lam_t = l_min + (l_max - l_min) * (1 - exp(-r_t^2 / 2)) — ограниченная
        форма (лемма 1).  Исходная форма l_min + (l_max - l_min) r_t^2 не
        ограничена: при r ~ N(0,1) её среднее равно l_max, а примерно в 5 %
        шагов lam_t > 1, и рекурсия перестаёт быть выпуклой комбинацией.
    (3) sigma^_t^2 = beta e_t^2 + (1 - beta) sigma^_{t-1}^2 — оценка дисперсии
        НЕВЯЗКИ: при постоянном lam её предел равен sigma^2 * 2/(2 - lam)
        (лемма 2), где sigma^2 — дисперсия самого процесса.
    (5) sigma~_t^2 = (2 - lam_bar)/2 * sigma^_t^2 — пересчёт к масштабу
        процесса, в единицах которого заданы L и форма (2); lam_bar —
        экспоненциальное среднее lam_t (коэффициент 0,01).  Пересчёт точен
        при постоянном lam и приближён при адаптивном (см. check_theory.py).
    Приостановка: при |r_t| > L обновление Z, sigma^2 и lam_bar
    приостанавливается и возобновляется после k_release подряд идущих
    отсчётов с |r_t| <= L — это защищает оценки от «сползания» вслед за
    продолжительной аномалией.

    Параметры
    ---------
    lam_min, lam_max : границы коэффициента сглаживания, 0 < lam_min < lam_max < 1
    beta             : коэффициент забывания в оценке дисперсии
    L                : порог приостановки в единицах sigma (порог тревоги —
                       другой параметр, он калибруется отдельно)
    k_release        : число нормальных отсчётов для снятия приостановки
    lam_form         : "exp" (гладкая, рекомендуется) либо "clip" (усечение)
    bias_correction  : применять ли пересчёт (5) к масштабу процесса
    warmup           : длина окна робастной инициализации
    """

    name = "EWMA-AV"

    def __init__(self, lam_min=0.05, lam_max=0.30, beta=0.02, L=3.0,
                 k_release=2, lam_form="exp", bias_correction=True,
                 warmup=200):
        if not 0 < lam_min < lam_max < 1:
            raise ValueError("требуется 0 < lam_min < lam_max < 1")
        if not 0 < beta < 1:
            raise ValueError("требуется 0 < beta < 1")
        self.lam_min, self.lam_max = lam_min, lam_max
        self.beta, self.L = beta, L
        self.k_release = k_release
        self.lam_form = lam_form
        self.bias_correction = bias_correction
        self.warmup = warmup

    def _lambda(self, r2: float) -> float:
        """Коэффициент сглаживания как функция квадрата нормализованной невязки.

        Обе формы гарантируют lam in [lam_min, lam_max] при любом r2 >= 0.
        Для формы "exp" среднее значение в стационарном режиме равно
            lam_min + (lam_max - lam_min) * (1 - 1/sqrt(2)) ~ lam_min + 0.293*d.
        """
        d = self.lam_max - self.lam_min
        if self.lam_form == "exp":
            return self.lam_min + d * (1.0 - np.exp(-0.5 * r2))
        return self.lam_min + d * min(1.0, r2 / (self.L ** 2))

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.size
        w = min(self.warmup, max(10, n // 4))
        Z, s0 = _robust_init(x, w)
        s2 = s0 ** 2

        out = np.zeros(n)
        frozen_flags = np.zeros(n, dtype=bool)   # для отчёта о приостановке
        frozen = False
        ok_run = 0
        lam_bar = 0.5 * (self.lam_min + self.lam_max)  # стартовая оценка
        lam_ewm_beta = 0.01                            # сглаживание lam_bar

        for t in range(n):
            c = 2.0 / (2.0 - lam_bar) if self.bias_correction else 1.0
            sigma = np.sqrt(max(s2 / c, 1e-12))
            e = x[t] - Z
            r = e / sigma
            out[t] = abs(r)                 # непрерывная статистика

            if abs(r) > self.L:
                frozen, ok_run = True, 0
            else:
                ok_run += 1
                if frozen and ok_run >= self.k_release:
                    frozen = False

            frozen_flags[t] = frozen
            if not frozen:
                lam = self._lambda(r * r)
                lam_bar = (1 - lam_ewm_beta) * lam_bar + lam_ewm_beta * lam
                s2 = self.beta * e * e + (1.0 - self.beta) * s2
                Z = lam * x[t] + (1.0 - lam) * Z
        self.frozen_ = frozen_flags
        return out


class SigmaRule:
    """Правило k-сигма по скользящему окну (базовая линия)."""

    name = "3-sigma"

    def __init__(self, window=200):
        self.window = window

    def score(self, x):
        x = np.asarray(x, float)
        n, w = x.size, self.window
        out = np.zeros(n)
        for t in range(n):
            lo = max(0, t - w)
            seg = x[lo:t] if t > 0 else x[:1]
            if seg.size < 5:
                continue
            mu, sd = seg.mean(), seg.std()
            out[t] = abs(x[t] - mu) / max(sd, 1e-12)
        return out


class EwmaFixed:
    """Классическая EWMA (Roberts, 1959) с фиксированным коэффициентом.

    Дисперсия невязки оценивается по окну разогрева и далее не обновляется —
    именно это делает метод непригодным при дрейфе дисперсии.
    """

    def __init__(self, lam=0.2, warmup=200):
        self.lam, self.warmup = lam, warmup
        self.name = f"EWMA({lam})"

    def score(self, x):
        x = np.asarray(x, float)
        n = x.size
        w = min(self.warmup, max(10, n // 4))
        Z, s0 = _robust_init(x, w)
        # поправка на схему «невязка относительно сглаженного среднего»
        sigma_e = s0 * np.sqrt(2.0 / (2.0 - self.lam))
        out = np.zeros(n)
        for t in range(n):
            e = x[t] - Z
            out[t] = abs(e) / max(sigma_e, 1e-12)
            Z = self.lam * x[t] + (1.0 - self.lam) * Z
        return out


class AEWMA:
    """Адаптивная EWMA (Capizzi & Masarotto, 2003) в роли прогнозирующего
    сглаживателя.

    Сглаживание по Капицци — Мазаротто с функцией Хубера:
        Z_t = Z_{t-1} + phi(e_t),  e_t = x_t - Z_{t-1},
        phi(e) = e - (1 - lam) k s,   e >  k s
                 lam e,               |e| <= k s
                 e + (1 - lam) k s,   e < -k s,
    где s — масштаб невязки, оцениваемый один раз по окну разогрева (фаза I).
    При малых отклонениях схема ведёт себя как EWMA с коэффициентом lam, при
    больших — почти как карта Шухарта: оценка среднего быстро переходит
    к новому уровню.  Контролируется, как и в предлагаемой схеме, невязка
    одношагового прогноза |e_t| / s.
    """

    name = "AEWMA (Capizzi-Masarotto)"

    def __init__(self, lam=0.2, k=3.0, warmup=200):
        self.lam, self.k, self.warmup = lam, k, warmup

    def score(self, x):
        x = np.asarray(x, float)
        n = x.size
        w = min(self.warmup, max(10, n // 4))
        Z, s0 = _robust_init(x, w)          # фаза I: масштаб фиксируется здесь
        s = max(s0 * np.sqrt(2.0 / (2.0 - self.lam)), 1e-12)
        ks = self.k * s
        out = np.zeros(n)
        for t in range(n):
            e = x[t] - Z
            out[t] = abs(e) / s
            if e > ks:
                phi = e - (1.0 - self.lam) * ks
            elif e < -ks:
                phi = e + (1.0 - self.lam) * ks
            else:
                phi = self.lam * e
            Z = Z + phi
        return out


class Cusum:
    """Двусторонняя кумулятивная сумма (Page, 1954).

    Включён специально: ожидается, что именно CUSUM обнаруживает медленно
    нарастающие трендовые атаки, на которых схемы скользящего среднего
    принципиально неработоспособны.
    """

    name = "CUSUM"

    def __init__(self, k=0.5, h=5.0, warmup=200):
        self.k, self.h, self.warmup = k, h, warmup

    def score(self, x):
        x = np.asarray(x, float)
        n = x.size
        w = min(self.warmup, max(10, n // 4))
        mu, sd = _robust_init(x, w)
        sd = max(sd, 1e-12)
        gp = gm = 0.0
        out = np.zeros(n)
        for t in range(n):
            z = (x[t] - mu) / sd
            gp = max(0.0, gp + z - self.k)
            gm = max(0.0, gm - z - self.k)
            out[t] = max(gp, gm)
            # Сброс после превышения проектного порога h.  Без сброса
            # накопленная сумма остаётся высокой сколь угодно долго, что
            # даёт заведомо заниженные показатели и делает сравнение
            # с CUSUM некорректным.
            if out[t] > self.h:
                gp = gm = 0.0
        return out


class HoltWinters:
    """Аддитивная модель Хольта — Уинтерса в варианте Брутлага (2000).

    Параметры alpha, beta, gamma оцениваются (statsmodels, эвристическая
    инициализация) по обучающему ряду train — это и есть фаза I метода.
    Затем модель идёт по контролируемому ряду онлайн, с прогнозом на один
    шаг, и для каждого сезонного отсчёта ведёт сглаженное абсолютное
    отклонение d (Brutlag, 2000):
        e_t = x_t - (l_{t-1} + b_{t-1} + s_{t-m}),
        d_t = gamma |e_t| + (1 - gamma) d_{t-m},
    статистика аномальности — |e_t| / d_{t-m}.

    Требование: train заканчивается на границе сезона (последний отсчёт —
    последняя минута суток), а контролируемый ряд начинается с начала сезона.
    Сезонные компоненты и отклонения берутся из конца train, уровень заново
    оценивается по окну разогрева контролируемого ряда.
    """

    name = "Holt-Winters"

    def __init__(self, train, period=1440, warmup=200, dev_days=7):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        train = np.asarray(train, float)
        if train.size < 3 * period:
            raise ValueError("обучающий ряд короче трёх сезонов")
        self.period, self.warmup = period, warmup
        fit = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=period,
            initialization_method="heuristic",
        ).fit(optimized=True)
        prm = fit.params
        self.alpha = float(prm["smoothing_level"])
        self.beta = float(prm["smoothing_trend"])
        self.gamma = float(prm["smoothing_seasonal"])
        self.converged = bool(getattr(fit, "mle_retvals", {}).get("success", True))
        self.season0 = np.asarray(fit.season, float)[-period:].copy()
        resid = np.abs(np.asarray(train - fit.fittedvalues, float))
        k = min(dev_days, train.size // period)
        r = resid[-k * period:].reshape(k, period)
        self.dev0 = np.maximum(r.mean(axis=0), 1e-9)

    def score(self, x):
        x = np.asarray(x, float)
        n, m = x.size, self.period
        a, b, g = self.alpha, self.beta, self.gamma
        s = list(self.season0)
        d = list(self.dev0)
        w = min(self.warmup, max(10, n // 4))
        lvl = float(np.median(x[:w] - np.resize(self.season0, w)))
        trd = 0.0
        out = np.zeros(n)
        for t in range(n):
            j = t % m
            s_old, d_old = s[j], d[j]
            e = x[t] - (lvl + trd + s_old)
            out[t] = abs(e) / max(d_old, 1e-9)
            new_lvl = a * (x[t] - s_old) + (1 - a) * (lvl + trd)
            trd = b * (new_lvl - lvl) + (1 - b) * trd
            lvl = new_lvl
            s[j] = g * (x[t] - lvl) + (1 - g) * s_old
            d[j] = g * abs(e) + (1 - g) * d_old
        return out


class HampelMAD:
    """Скользящая медиана + MAD (фильтр Хампеля) — робастная базовая линия."""

    name = "Median+MAD"

    def __init__(self, window=200):
        self.window = window

    def score(self, x):
        x = np.asarray(x, float)
        n, w = x.size, self.window
        out = np.zeros(n)
        for t in range(n):
            lo = max(0, t - w)
            seg = x[lo:t] if t > 0 else x[:1]
            if seg.size < 5:
                continue
            med = np.median(seg)
            mad = 1.4826 * np.median(np.abs(seg - med))
            out[t] = abs(x[t] - med) / max(mad, 1e-12)
        return out


def ALL_DETECTORS():
    """Набор методов без обучающей выборки (Хольт — Уинтерс создаётся отдельно:
    ему нужен обучающий ряд)."""
    return [
        EwmaAV(),
        SigmaRule(),
        EwmaFixed(0.05), EwmaFixed(0.1), EwmaFixed(0.2), EwmaFixed(0.3),
        AEWMA(),
        Cusum(),
        HampelMAD(),
    ]
