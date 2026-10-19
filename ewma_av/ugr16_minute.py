"""
Загрузка одноминутных признаковых данных UGR'16.

Источник:
    https://github.com/josecamachop/UGR16_FeatureData
    Camacho J., Rodríguez-Gómez R. A. Data quality tools to enhance a network
    anomaly detection benchmark // Data. 2025. Vol. 10, No. 3. Art. 33.
    Maciá-Fernández G. и др. UGR'16: a new dataset for the evaluation of
    cyclostationarity-based network IDSs // Computers & Security. 2018.
    Vol. 73. P. 411-424.

Это официальная агрегация исходного UGR'16 в интервалы по одной минуте,
выполненная авторами датасета инструментом FCParser.  Практическое отличие
от сырых nfcapd: 662 МБ вместо десятков гигабайт, и агрегация уже сделана
теми же людьми, что собирали данные, — то есть на неё можно ссылаться.

Обучающая часть: 134262 минуты (около 93 суток, март-июнь 2016).
Тестовая часть: 43200 минут (30 суток) с размеченными атаками.

Столбцы разметки Y: labeldos, labelscan11, labelscan44, labelnerisbotnet,
labelblacklist, labelanomalyidpscan, labelanomalysshscan, labelanomalyspam.
Значение — число потоков соответствующего типа в данную минуту.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

__all__ = ["load_ugr16_minute", "ATTACK_LABELS"]

#: Метки, трактуемые как атаки.  labelblacklist исключён по умолчанию:
#: он присутствует практически в каждой минуте фона и представляет собой
#: попадание в списки репутации, а не атаку на наблюдаемую сеть.
ATTACK_LABELS = [
    "labeldos", "labelscan11", "labelscan44", "labelnerisbotnet",
    "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyspam",
]

#: Признаки-счётчики по протоколам; их сумма равна общему числу потоков.
PROTOCOL_COLS = ["protocoltcp", "protocoludp", "protocolicmp",
                 "protocoligmp", "protocolother"]


def load_ugr16_minute(repo_dir: str | Path, variant: str = "UGR16v1",
                      split: str = "test", min_attack_flows: int = 1,
                      attack_labels: list[str] | None = None):
    """Вернуть (ряд интенсивности, разметка, метки времени).

    Параметры
    ---------
    repo_dir   : путь к склонированному UGR16_FeatureData
    variant    : UGR16v1 | UGR16v2 | UGR16v3 | UGR16v4 | UGR16v3v4
    split      : "train" (93 суток, для характеризации сезонности) либо
                 "test" (30 суток с размеченными атаками)
    min_attack_flows : минута считается аномальной, если суммарное число
                 атакующих потоков в ней не меньше этого значения
    """
    repo = Path(repo_dir) / "csv"
    suffix = "train" if split == "train" else "test"
    X = pd.read_csv(repo / f"{variant}.X{suffix}.csv")
    Y = pd.read_csv(repo / f"{variant}.Y{suffix}.csv")

    ts = pd.to_datetime(X["Row"].astype(str), format="%Y%m%d%H%M")
    flows = X[PROTOCOL_COLS].sum(axis=1).to_numpy(float)

    labels = attack_labels or ATTACK_LABELS
    present = [c for c in labels if c in Y.columns]
    truth = (Y[present].sum(axis=1).to_numpy() >= min_attack_flows)

    order = np.argsort(ts.to_numpy())
    return flows[order], truth[order], pd.DatetimeIndex(ts.to_numpy()[order])
