"""
Генератор синтетических сценариев трафика.

Полностью документирован — это прямой ответ на замечание рецензента об
отсутствии описания способа генерации данных.  Каждый сценарий задаётся
явной моделью, все параметры вынесены в поля датакласса, seed фиксируется.

Модель фона
-----------
    x_t = A * s(t) * (1 + мультипликативный шум)          [логнормальный шум]
    s(t) = 1 + a_d*sin(2*pi*t/T_d) + a_w*sin(2*pi*t/T_w)

Логнормальный шум выбран потому, что интенсивность сетевого трафика
неотрицательна и имеет правостороннюю асимметрию; гауссовский шум в этой
роли систематически завышает качество любого порогового детектора и потому
даёт неоправданно оптимистичные результаты.

ВАЖНО: синтетика здесь — контролируемый стенд для анализа чувствительности
к параметрам, где известна вся истина.  Основные выводы статьи должны
опираться на реальные данные (см. ewma_av/datasets.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np

__all__ = ["TrafficConfig", "generate", "SCENARIOS"]


@dataclass
class TrafficConfig:
    """Параметры генерации одного сценария."""
    n: int = 20160                 # длина ряда: 14 суток при агрегации 1 мин
    period_day: int = 1440         # суточный период, отсчётов
    period_week: int = 10080       # недельный период, отсчётов
    base_level: float = 1000.0     # A, условных пакетов/с
    amp_day: float = 0.0           # a_d, относительная амплитуда суточной волны
    amp_week: float = 0.0          # a_w, относительная амплитуда недельной волны
    noise_cv: float = 0.10         # коэффициент вариации шума
    # параметры аномалий
    anomaly_kind: str = "none"     # none | burst | step | ramp
    n_anomalies: int = 12
    anomaly_len: int = 5           # длительность события, отсчётов
    anomaly_amp: float = 6.0       # амплитуда в единицах СКО фона
    ramp_len: int = 2000           # длительность нарастания для kind="ramp"
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def _seasonal(cfg: TrafficConfig) -> np.ndarray:
    t = np.arange(cfg.n)
    s = np.ones(cfg.n)
    if cfg.amp_day:
        s = s + cfg.amp_day * np.sin(2 * np.pi * t / cfg.period_day)
    if cfg.amp_week:
        s = s + cfg.amp_week * np.sin(2 * np.pi * t / cfg.period_week)
    return np.clip(s, 0.05, None)


def generate(cfg: TrafficConfig) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (ряд, булеву разметку аномалий)."""
    rng = np.random.default_rng(cfg.seed)
    s = _seasonal(cfg)
    mu_level = cfg.base_level * s

    # логнормальный мультипликативный шум с заданным коэффициентом вариации
    sig_ln = np.sqrt(np.log1p(cfg.noise_cv ** 2))
    noise = rng.lognormal(mean=-0.5 * sig_ln ** 2, sigma=sig_ln, size=cfg.n)
    x = mu_level * noise

    truth = np.zeros(cfg.n, dtype=bool)
    if cfg.anomaly_kind == "none" or cfg.n_anomalies == 0:
        return x, truth

    local_sd = cfg.base_level * cfg.noise_cv
    margin = max(cfg.period_day // 2, 1000)

    if cfg.anomaly_kind == "ramp":
        # одна медленно нарастающая трендовая атака — граница применимости
        start = cfg.n // 2
        end = min(cfg.n, start + cfg.ramp_len)
        ramp = np.linspace(0.0, cfg.anomaly_amp * local_sd, end - start)
        x[start:end] += ramp
        truth[start:end] = True
        return x, truth

    starts = rng.choice(
        np.arange(margin, cfg.n - margin - cfg.anomaly_len),
        size=cfg.n_anomalies, replace=False,
    )
    for st in np.sort(starts):
        en = st + cfg.anomaly_len
        if cfg.anomaly_kind == "burst":
            x[st:en] += cfg.anomaly_amp * local_sd
        elif cfg.anomaly_kind == "step":
            x[st:en] += cfg.anomaly_amp * local_sd * np.linspace(1, 1, en - st)
        truth[st:en] = True
    return x, truth


#: Пять сценариев.  Baseline и Diurnal проверяют режим ложных тревог и
#: устойчивость к дрейфу; Microburst и SYN Flood — резкие аномалии;
#: Slow DDoS — заявленная граница применимости.
SCENARIOS: dict[str, TrafficConfig] = {
    "Baseline": TrafficConfig(
        amp_day=0.0, anomaly_kind="none", n_anomalies=0),
    "Diurnal": TrafficConfig(
        amp_day=0.40, amp_week=0.15, anomaly_kind="burst",
        n_anomalies=12, anomaly_len=5, anomaly_amp=6.0),
    "Microburst": TrafficConfig(
        amp_day=0.40, anomaly_kind="burst",
        n_anomalies=20, anomaly_len=2, anomaly_amp=8.0),
    "SYN Flood": TrafficConfig(
        amp_day=0.40, anomaly_kind="step",
        n_anomalies=6, anomaly_len=30, anomaly_amp=10.0),
    "Slow DDoS": TrafficConfig(
        amp_day=0.40, anomaly_kind="ramp", ramp_len=3000, anomaly_amp=8.0),
}
