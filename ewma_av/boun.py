"""
Загрузка набора BOUN DDoS (кампусная сеть университета Богазичи).

Источник:
    Erhan D., Anarım E. Boğaziçi University distributed denial of service
    dataset // Data in Brief. 2020. Vol. 32. 106187.
    DOI: 10.1016/j.dib.2020.106187
    Данные: https://data.mendeley.com/datasets/mfnn9bh42m

Что в наборе
------------
Два файла CSV (экспорт из Wireshark), по ~8 минут каждый:
    BOUN_TCP_Anon.csv — 9,3 млн пакетов, атака TCP SYN flood
    BOUN_UDP_Anon.csv — 8,0 млн пакетов, атака UDP flood
Трафик снят с зеркального порта магистрального маршрутизатора кампуса при
более чем 4000 активных пользователей. Колонки (с подчёркиваниями):
    Time, Frame Number, Frame_length, Source_ip, Destination_IP,
    Source_Port, Destination_Port, SYN, ACK, RST, TTL, TCP_Protocol
Флаги записаны строками «Set» / «Not set».

Разметка
--------
Готового столбца с меткой атаки нет. По описанию набора каждая запись
состоит из циклов: 80 с фонового трафика, затем 20 с атаки; скорость атаки
от цикла к циклу растёт (1000, 1500, 2000, 2500 пакетов/с).

Схема проверена НЕЗАВИСИМО от оцениваемых детекторов — по адресу
получателя. Оба вида атаки направлены на один узел 10.50.199.86:
  * в TCP-файле он получает 125 557 пакетов SYN без ACK, следующий по числу
    адрес — 1 477;
  * в UDP-файле его UDP-трафик коррелирует с циклом 80/20 с коэффициентом
    0,82, у следующего кандидата — 0,17.
Атакующий трафик к этому узлу сосредоточен в окнах 80–100, 180–200,
280–300, 380–400 с. После каждого окна наблюдается «хвост» в 1–3 с
(завершение генератора и буферизация), поэтому 3 с после каждого окна
выделяются в защитную полосу и исключаются из калибровки и из подсчёта
метрик. Это стандартный приём при разметке с неточными границами.

Детекторы адреса получателя не используют: они видят только агрегат по
магистрали (см. MONITORED), поэтому подтверждение разметки по адресу не
создаёт порочного круга.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

__all__ = ["VICTIM", "MONITORED", "counters_from_csv", "schedule_labels",
           "verify_labels", "prepare_boun"]

#: Атакуемый узел, установленный по анализу адресов (см. описание модуля).
VICTIM = "10.50.199.86"

#: Контролируемая величина по типу файла — агрегат по всей магистрали.
#:   TCP: число пакетов SYN без ACK в секунду (классический признак SYN flood:
#:        отбрасывает SYN-ACK ответы нормальных рукопожатий);
#:   UDP: общее число UDP-пакетов в секунду.
MONITORED = {"tcp": "syn_noack", "udp": "udp"}

_FLAG_TRUE = {"1", "set", "true", "yes"}


def _norm(name: str) -> str:
    """'Frame_length' / 'Frame length' / ' frame  length ' -> 'frame length'."""
    return " ".join(str(name).replace("_", " ").strip().lower().split())


def _flag(series: pd.Series) -> np.ndarray:
    """Флаг TCP как булев массив, независимо от того, как pandas хранит строки.

    Новые версии pandas хранят текст в типе 'str', а не 'object'; проверка
    по dtype поэтому ненадёжна. Здесь значение всегда приводится к строке.
    """
    return series.astype(str).str.strip().str.lower().isin(_FLAG_TRUE).to_numpy()


def counters_from_csv(path: str | Path, bin_seconds: float = 1.0,
                      victim: str = VICTIM, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Свернуть CSV в посекундные счётчики. Читает частями, память не растёт.

    Столбцы результата:
        t_start     начало интервала, с
        pkts        все пакеты
        bytes       суммарный объём
        syn         пакеты с флагом SYN
        syn_noack   пакеты SYN без ACK
        udp         UDP-пакеты
        to_victim   пакеты к атакуемому узлу (ТОЛЬКО для проверки разметки,
                    детекторам не подаётся)
    """
    acc: dict[str, np.ndarray] = {}
    n_max = 0

    def add(key, b, w=None):
        nonlocal n_max
        v = np.bincount(b, weights=w, minlength=1).astype(float)
        cur = acc.get(key)
        if cur is None or cur.size < v.size:
            new = np.zeros(max(v.size, 0 if cur is None else cur.size))
            if cur is not None:
                new[:cur.size] = cur
            cur = new
        cur[:v.size] += v
        acc[key] = cur
        n_max = max(n_max, cur.size)

    reader = pd.read_csv(path, dtype=str, keep_default_na=False,
                         chunksize=chunksize)
    for ch in reader:
        ch.columns = [_norm(c) for c in ch.columns]
        if "time" not in ch.columns:
            raise ValueError(f"нет колонки времени; заголовки: {list(ch.columns)}")
        t = pd.to_numeric(ch["time"], errors="coerce").to_numpy()
        ok = np.isfinite(t)
        ch, t = ch.loc[ok], t[ok]
        b = (t // bin_seconds).astype(np.int64)

        add("pkts", b)
        if "frame length" in ch.columns:
            ln = pd.to_numeric(ch["frame length"], errors="coerce").fillna(0).to_numpy()
            add("bytes", b, ln)
        syn = _flag(ch["syn"]) if "syn" in ch.columns else np.zeros(len(ch), bool)
        ack = _flag(ch["ack"]) if "ack" in ch.columns else np.zeros(len(ch), bool)
        add("syn", b, syn.astype(float))
        add("syn_noack", b, (syn & ~ack).astype(float))
        proto = (ch["tcp protocol"].astype(str).str.strip().str.upper().to_numpy()
                 if "tcp protocol" in ch.columns else np.array([""] * len(ch)))
        add("udp", b, (proto == "UDP").astype(float))
        if "destination ip" in ch.columns:
            add("to_victim", b, (ch["destination ip"].to_numpy() == victim).astype(float))

    out = {"t_start": np.arange(n_max) * bin_seconds}
    for k, v in acc.items():
        full = np.zeros(n_max)
        full[:v.size] = v
        out[k] = full
    return pd.DataFrame(out)


def schedule_labels(n_bins: int, bin_seconds: float = 1.0, idle: float = 80.0,
                    attack: float = 20.0, guard: float = 3.0, n_cycles: int = 4):
    """Разметка по документированному циклу.

    Возвращает (truth, ignore): truth — интервалы атаки, ignore — защитная
    полоса длительностью guard секунд после каждого окна атаки.
    n_cycles — число атак в записи: по описанию набора их четыре (1000,
    1500, 2000 и 2500 пакетов/с). Без этого ограничения цикл «продолжается»
    за последнюю атаку и в конце записи появляется несуществующее окно.
    """
    t = np.arange(n_bins) * bin_seconds
    period = idle + attack
    phase = np.mod(t, period)
    cycle = (t // period).astype(int)
    truth = (phase >= idle) & (cycle < n_cycles)
    ignore = (phase < guard) & (cycle >= 1) & (cycle <= n_cycles)
    return truth, ignore


def verify_labels(cnt: pd.DataFrame, truth: np.ndarray, ignore: np.ndarray) -> dict:
    """Проверка разметки по трафику к атакуемому узлу.

    Детекторы этот столбец не видят, поэтому проверка независима.
    """
    v = cnt["to_victim"].to_numpy()
    total = v.sum()
    inside = v[truth].sum()
    in_guard = v[ignore].sum()
    outside = total - inside - in_guard
    share_in = inside / total if total else 0.0
    share_guard = in_guard / total if total else 0.0
    rate_in = v[truth].mean() if truth.any() else 0.0
    rate_out = v[~truth & ~ignore].mean()

    print(f"  длительность записи: {len(cnt)} с")
    print(f"  пакетов к атакуемому узлу {VICTIM}: {int(total)}")
    print(f"    в окнах атаки по схеме: {100 * share_in:.1f} %")
    print(f"    в защитных полосах 3 с: {100 * share_guard:.1f} %")
    print(f"    вне окон: {100 * outside / max(total, 1):.1f} %")
    print(f"  средняя интенсивность к узлу: в окнах {rate_in:.0f} пак/с, "
          f"вне окон {rate_out:.1f} пак/с")
    ok = share_in + share_guard > 0.9
    print("  РАЗМЕТКА ПОДТВЕРЖДЕНА" if ok else
          "  ВНИМАНИЕ: разметка НЕ подтверждается — пришлите этот вывод")
    return {"share_in": share_in, "share_guard": share_guard,
            "rate_in": rate_in, "rate_out": rate_out, "ok": ok}


def prepare_boun(csv_path: str | Path, bin_seconds: float = 1.0,
                 metric: str | None = None):
    """CSV -> (ряд, разметка, маска игнорирования, счётчики, имя метрики)."""
    name = Path(csv_path).name.lower()
    if metric is None:
        metric = MONITORED["udp"] if "udp" in name else MONITORED["tcp"]
    cnt = counters_from_csv(csv_path, bin_seconds)
    print(f"файл {Path(csv_path).name}: {int(cnt['pkts'].sum())} пакетов, "
          f"контролируемая величина: {metric}")
    truth, ignore = schedule_labels(len(cnt), bin_seconds)
    verify_labels(cnt, truth, ignore)
    return cnt[metric].to_numpy(float), truth, ignore, cnt, metric
