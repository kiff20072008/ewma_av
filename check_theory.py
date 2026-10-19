"""
Численная проверка всех теоретических утверждений статьи (Монте-Карло).

    python check_theory.py

Выводит числа, которые приводятся в тексте:
  1. Лемма 2: предел E[σ̂²] при постоянном λ равен σ²·2/(2 − λ).
  2. Лемма 3: относительное СКО оценки σ̂² равно sqrt(2β/(2 − β)).
  3. Исходная форма λ_t = λmin + (λmax − λmin) r²: среднее и доля λ_t > 1.
  4. Ограниченная форма (2): среднее λ_t при r ~ N(0, 1).
  5. Детектор целиком (адаптивный λ_t, пересчёт (5), приостановка при L = 3,
     k = 2) на стационарном гауссовом ряде: E[σ̃²]/σ², дисперсия r_t,
     среднее λ_t, доля превышений |r_t| > 3.
  6. То же без приостановки: насколько пересчёт (5) точен при переменном λ_t.
"""
import numpy as np
from scipy.stats import norm

RNG = np.random.default_rng(20261019)
LMIN, LMAX = 0.05, 0.30


def ewma_resid_var(x, lam, beta, s2_0=1.0):
    Z, s2 = x[0], s2_0
    out = np.empty(x.size)
    for t in range(x.size):
        e = x[t] - Z
        s2 = beta * e * e + (1 - beta) * s2
        out[t] = s2
        Z = lam * x[t] + (1 - lam) * Z
    return out


def detector(x, L=3.0, freeze=True, beta=0.02, k_release=2, beta_lam=0.01, burn=2000):
    """Та же рекурсия, что в ewma_av.detectors.EwmaAV (инициализация точная)."""
    Z, s2 = 0.0, 2.0 / (2.0 - 0.125)
    lam_bar = 0.5 * (LMIN + LMAX)
    frozen, ok = False, 0
    s2c, rr, lams = [], [], []
    for t in range(x.size):
        c = 2.0 / (2.0 - lam_bar)
        sig = np.sqrt(s2 / c)
        e = x[t] - Z
        r = e / sig
        if t >= burn:
            s2c.append(s2 / c)
            rr.append(r)
        if freeze and abs(r) > L:
            frozen, ok = True, 0
        else:
            ok += 1
            if frozen and ok >= k_release:
                frozen = False
        if not frozen:
            lam = LMIN + (LMAX - LMIN) * (1 - np.exp(-0.5 * r * r))
            if t >= burn:
                lams.append(lam)
            lam_bar = (1 - beta_lam) * lam_bar + beta_lam * lam
            s2 = beta * e * e + (1 - beta) * s2
            Z = lam * x[t] + (1 - lam) * Z
    rr = np.asarray(rr)
    return np.mean(s2c), np.var(rr), np.mean(lams), np.mean(np.abs(rr) > 3.0)


def main():
    print("1. Лемма 2 (σ² = 1, β = 0,02, 400 рядов по 4000 отсчётов)")
    for lam in (0.05, 0.1, 0.2, 0.3):
        v = [ewma_resid_var(RNG.normal(0, 1, 4000), lam, 0.02)[1500:].mean() for _ in range(400)]
        th = 2 / (2 - lam)
        print(f"   λ={lam:.2f}: эмпирически {np.mean(v):.4f}, теория {th:.4f}, "
              f"отклонение {100 * (np.mean(v) - th) / th:+.2f} %")

    print("2. Лемма 3 (λ = 0,1; 200 рядов по 20 000 отсчётов)")
    for beta in (0.2, 0.1, 0.05, 0.02, 0.01):
        v = np.concatenate([ewma_resid_var(RNG.normal(0, 1, 20000), 0.1, beta)[5000:]
                            for _ in range(200)])
        emp, th = v.std() / v.mean(), np.sqrt(2 * beta / (2 - beta))
        print(f"   β={beta:.2f}: отн. СКО σ̂² эмпирически {emp:.3f}, "
              f"теория {th:.3f}, отклонение {100 * (emp - th) / th:+.1f} %")

    z = RNG.normal(0, 1, 2_000_000)
    lam3 = LMIN + (LMAX - LMIN) * z ** 2
    print("3. Исходная форма λ_t (r ~ N(0,1), 2·10⁶ значений)")
    print(f"   среднее {lam3.mean():.4f}; доля λ_t > 1: {100 * np.mean(lam3 > 1):.2f} %; "
          f"порог |r| > {np.sqrt((1 - LMIN) / (LMAX - LMIN)):.2f}")
    lam4 = LMIN + (LMAX - LMIN) * (1 - np.exp(-z ** 2 / 2))
    print("4. Форма (2) при r ~ N(0,1)")
    print(f"   среднее {lam4.mean():.4f}; теория {LMIN + (LMAX - LMIN) * (1 - 2 ** -0.5):.4f}; "
          f"максимум {lam4.max():.4f}")

    for freeze in (True, False):
        res = np.array([detector(RNG.normal(0, 1, 30000), freeze=freeze) for _ in range(40)])
        s2c, vr, ml, fpr = res.mean(axis=0)
        tag = "5. Детектор с приостановкой (L = 3)" if freeze else "6. Детектор без приостановки"
        print(f"{tag}, 40 рядов по 30 000 отсчётов")
        print(f"   E[σ̃²]/σ² = {s2c:.4f}; дисперсия r_t = {vr:.4f}; среднее λ_t = {ml:.4f}; "
              f"доля |r_t| > 3: {fpr:.5f}")
        if freeze:
            print(f"   номинал 2[1−Φ(3)] = {2 * (1 - norm.cdf(3)):.5f}; "
                  f"с учётом дисперсии r_t: 2[1−Φ(3/√{vr:.3f})] = {2 * (1 - norm.cdf(3 / np.sqrt(vr))):.5f}")


if __name__ == "__main__":
    main()
