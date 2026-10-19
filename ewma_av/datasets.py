"""
Пайплайн подготовки реальных данных.

Поддерживаются три источника:

  UGR'16   — https://nesg.ugr.es/nesg-ugr16/
             Реальный NetFlow провайдера Tier-3, более 4 месяцев, размеченные
             DoS / сканирования / ботнет.  Датасет создан специально для
             оценки детекторов, учитывающих циклостационарность: суточная и
             недельная периодичность выражены явно.  Основной источник для
             характеризации нестационарности и для сравнения детекторов.

  BOUN DDoS — Erhan D., Anarim E. // Data in Brief. 2020. Vol. 32. 106187.
             Трафик реальной кампусной сети университета Богазичи, снятый с
             зеркального порта магистрального маршрутизатора, с реальными
             TCP SYN и UDP flood атаками.  Отвечает на замечание о том, что
             метод не проверялся на трафике образовательного учреждения.
             Ограничение: короткие записи, сезонной структуры в них нет.

  MAWILab  — http://www.fontugne.org/mawilab/
             Размеченные аномалии на ежедневных 15-минутных трейсах
             магистрали WIDE.  Полезен как третий независимый источник.

Ни один из наборов нельзя скачать автоматически без согласия с условиями
использования — ссылки и порядок получения описаны в README.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

__all__ = [
    "aggregate_ugr16_flows", "load_ugr16_counters",
    "pcap_to_counters", "load_boun",
    "profile_by_hour", "profile_by_weekday", "periodogram",
]

# --------------------------------------------------------------------------
# UGR'16
# --------------------------------------------------------------------------

#: Порядок колонок в csv-выгрузке потоков UGR'16.
UGR16_COLUMNS = [
    "te", "td", "sa", "da", "sp", "dp", "pr", "flg",
    "fwd", "stos", "pkt", "byt", "label",
]

#: Метки, считающиеся аномальными.  "background" — легитимный фон.
UGR16_ANOMALY_LABELS = {
    "dos", "scan11", "scan44", "nerisbotnet",
    "anomaly-spam", "blacklist", "dos11", "dos53s",
}


def aggregate_ugr16_flows(csv_path: str | Path, out_path: str | Path,
                          bin_seconds: int = 60,
                          chunksize: int = 2_000_000) -> Path:
    """Свернуть csv-выгрузку потоков UGR'16 в счётчики по временным интервалам.

    Читает потоково, поэтому работает на файлах в десятки гигабайт.
    Результат — parquet со столбцами:
        ts        начало интервала
        pkts      суммарное число пакетов
        bytes     суммарный объём
        flows     число потоков
        n_anom    число аномальных потоков в интервале
        is_anom   булева разметка интервала

    Получение исходных файлов:
        1. Скачать calibration / test с https://nesg.ugr.es/nesg-ugr16/
        2. Если файлы в формате nfcapd, преобразовать:
             nfdump -r nfcapd.XXXX -o "fmt:%te,%td,%sa,%da,%sp,%dp,%pr,\\
                    %flg,%fwd,%stos,%ipkt,%ibyt,%lbl" -q > flows.csv
    """
    csv_path, out_path = Path(csv_path), Path(out_path)
    acc: dict[int, np.ndarray] = {}

    reader = pd.read_csv(
        csv_path, names=UGR16_COLUMNS, header=None,
        usecols=["te", "pkt", "byt", "label"],
        chunksize=chunksize, on_bad_lines="skip",
    )
    for chunk in reader:
        ts = pd.to_datetime(chunk["te"], errors="coerce")
        chunk = chunk.loc[ts.notna()]
        ts = ts.dropna()
        bins = (ts.astype("int64") // 10 ** 9 // bin_seconds).to_numpy()
        pkt = pd.to_numeric(chunk["pkt"], errors="coerce").fillna(0).to_numpy()
        byt = pd.to_numeric(chunk["byt"], errors="coerce").fillna(0).to_numpy()
        lab = chunk["label"].astype(str).str.strip().str.lower().to_numpy()
        anom = np.isin(lab, list(UGR16_ANOMALY_LABELS)).astype(np.int64)

        for b, p, y, a in zip(bins, pkt, byt, anom):
            row = acc.get(b)
            if row is None:
                acc[b] = np.array([p, y, 1, a], dtype=np.float64)
            else:
                row += (p, y, 1, a)

    keys = np.sort(np.fromiter(acc.keys(), dtype=np.int64))
    arr = np.vstack([acc[k] for k in keys])
    df = pd.DataFrame({
        "ts": pd.to_datetime(keys * bin_seconds, unit="s"),
        "pkts": arr[:, 0], "bytes": arr[:, 1],
        "flows": arr[:, 2], "n_anom": arr[:, 3],
    })
    df["is_anom"] = df["n_anom"] > 0
    # заполнить пропущенные интервалы нулями, иначе ряд не эквидистантен
    full = pd.date_range(df["ts"].min(), df["ts"].max(),
                         freq=f"{bin_seconds}s")
    df = (df.set_index("ts").reindex(full, fill_value=0)
            .rename_axis("ts").reset_index())
    df["is_anom"] = df["is_anom"].astype(bool)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def load_ugr16_counters(parquet_path: str | Path,
                        metric: str = "pkts") -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Загрузить подготовленные счётчики: (ряд, разметка, метки времени)."""
    df = pd.read_parquet(parquet_path)
    return (df[metric].to_numpy(float),
            df["is_anom"].to_numpy(bool),
            pd.DatetimeIndex(df["ts"]))


# --------------------------------------------------------------------------
# BOUN DDoS (pcap)
# --------------------------------------------------------------------------

def pcap_to_counters(pcap_path: str | Path,
                     bin_seconds: float = 1.0) -> pd.DataFrame:
    """Свернуть pcap в счётчики пакетов и байтов по интервалам.

    Используется dpkt (быстрее scapy на больших файлах).  Альтернатива без
    Python — tshark:
        tshark -r file.pcap -q -z io,stat,1
    """
    import dpkt

    ts_list, sz_list = [], []
    with open(pcap_path, "rb") as fh:
        try:
            reader = dpkt.pcap.Reader(fh)
        except ValueError:
            fh.seek(0)
            reader = dpkt.pcapng.Reader(fh)
        for ts, buf in reader:
            ts_list.append(ts)
            sz_list.append(len(buf))

    ts = np.asarray(ts_list, float)
    sz = np.asarray(sz_list, float)
    if ts.size == 0:
        raise ValueError(f"пустой файл: {pcap_path}")

    t0 = ts.min()
    bins = ((ts - t0) // bin_seconds).astype(np.int64)
    n_bins = int(bins.max()) + 1
    pkts = np.bincount(bins, minlength=n_bins).astype(float)
    byts = np.bincount(bins, weights=sz, minlength=n_bins)
    return pd.DataFrame({
        "ts": pd.to_datetime(t0 + np.arange(n_bins) * bin_seconds, unit="s"),
        "pkts": pkts, "bytes": byts,
    })


def load_boun(pcap_path: str | Path, attack_intervals: list[tuple[float, float]],
              bin_seconds: float = 1.0) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Счётчики BOUN с разметкой по интервалам атак.

    attack_intervals — список (начало, конец) в секундах от начала записи;
    берутся из сопроводительной таблицы датасета, а не задаются на глаз.
    """
    df = pcap_to_counters(pcap_path, bin_seconds)
    n = len(df)
    truth = np.zeros(n, dtype=bool)
    for a, b in attack_intervals:
        lo = int(a / bin_seconds)
        hi = min(n, int(b / bin_seconds) + 1)
        truth[max(0, lo):hi] = True
    return df["pkts"].to_numpy(float), truth, pd.DatetimeIndex(df["ts"])


# --------------------------------------------------------------------------
# Характеризация нестационарности
# --------------------------------------------------------------------------

def profile_by_hour(x: np.ndarray, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Суточный профиль: медиана и квантили по часам.

    Даёт измеренные значения вместо утверждений «ночью 3-8 % от дневного
    максимума».  Столбец ratio_to_peak — искомое отношение.
    """
    df = pd.DataFrame({"x": x, "h": ts.hour})
    g = df.groupby("h")["x"]
    out = pd.DataFrame({
        "median": g.median(), "q10": g.quantile(0.10),
        "q90": g.quantile(0.90), "mean": g.mean(), "std": g.std(),
    })
    out["cv"] = out["std"] / out["mean"]
    out["ratio_to_peak"] = out["median"] / out["median"].max()
    return out


def profile_by_weekday(x: np.ndarray, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Недельный профиль: будни против выходных."""
    df = pd.DataFrame({"x": x, "d": ts.dayofweek})
    g = df.groupby("d")["x"]
    out = pd.DataFrame({"median": g.median(), "mean": g.mean(), "std": g.std()})
    out["cv"] = out["std"] / out["mean"]
    out["ratio_to_peak"] = out["median"] / out["median"].max()
    return out


def periodogram(x: np.ndarray, bin_seconds: int = 60, top: int = 6) -> pd.DataFrame:
    """Спектральный анализ: доказательство наличия суточного и недельного циклов.

    Возвращает наиболее выраженные периоды в часах — при корректных данных
    среди них должны быть 24 ч и 168 ч.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = x.size
    freqs = np.fft.rfftfreq(n, d=bin_seconds)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs, power = freqs[1:], power[1:]           # отбрасываем нулевую частоту
    idx = np.argsort(power)[::-1][:top]
    return pd.DataFrame({
        "period_hours": 1.0 / freqs[idx] / 3600.0,
        "power_rel": power[idx] / power.sum(),
    }).sort_values("power_rel", ascending=False).reset_index(drop=True)
