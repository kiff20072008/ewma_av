"""
Замер скорости детектора для раздела «Вычислительные затраты».

    python bench.py

Печатает модель процессора, версии Python и NumPy, число обрабатываемых
отсчётов в секунду (медиана по 7 повторам, один поток) и время на отсчёт.
"""
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

from ewma_av.detectors import EwmaAV
from ewma_av.generators import TrafficConfig, generate


def cpu_name() -> str:
    try:
        if sys.platform.startswith("win"):
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def main():
    x, _ = generate(TrafficConfig(n=200_000, amp_day=0.4, seed=7))
    det = EwmaAV()
    det.score(x[:5000])                       # прогрев
    rates = []
    for _ in range(7):
        t0 = time.perf_counter()
        det.score(x)
        rates.append(len(x) / (time.perf_counter() - t0))
    rate = statistics.median(rates)
    print("Процессор:        ", cpu_name())
    print("Python / NumPy:   ", platform.python_version(), "/", np.__version__)
    print("ОС:               ", platform.platform())
    print(f"Отсчётов в секунду (медиана из 7): {rate:,.0f}".replace(",", " "))
    print(f"Время на отсчёт:  {1e6 / rate:.2f} мкс")


if __name__ == "__main__":
    main()
