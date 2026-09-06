"""
5차 리뷰 C15 대응 스크립트.

C8이 "8/11 패턴에서 결손이 도움이 된다"를 확정한 순간, rate>0 전체 집계(n=1,320)는
서로 반대 방향으로 작용하는 두 메커니즘(약수 편향 vs 배수 tie 해소)을 뭉친 값이 되었다.
이 스크립트는 Appendix G의 원래 실험을 rate별로 재실행하며, 이번에는 rate마다 별도의
대응표본(paired) 벡터를 저장해 rate별 McNemar 검정(기하2단계 vs ACF(전역최대))까지
산출한다. 시드가 동일하므로 집계 성공률은 Appendix G/H와 완전히 일치해야 한다.
"""

import sys
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from find_n import IMAGE_DIR, OUT_DIR, _ensure_image_dir, generate_periodic_tact
from verification_v2 import wilson_ci
from verification_v3 import METHODS, RelPathTee, banner, run_all_methods, holm
from verification_tooth import all_physics_patterns, inject_random_tooth_glitches


def mcnemar_exact(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return n01, n10, float(min(1.0, 2 * tail))


def paired_bootstrap_diff(a, b, n_boot=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return float(b.mean() - a.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_stratified(patterns, rates, n_reps=30, L=15_000, N_max=20, seed0=6000, n_boot=200):
    """rate별로 별도의 paired 벡터를 유지하며 결손 실험을 재실행한다."""
    banner("[실험 J / C15] rate별 층화 재실행 (기존 Appendix G와 동일 시드)")
    print("Appendix G와 완전히 동일한 조건·시드로 재실행한다 — 집계 성공률이 일치하는지")
    print("확인하고, 이번에는 rate마다 대응표본(paired) 벡터를 별도로 저장한다.\n")

    per_rate_hits = {r: {m: [0, 0] for m in METHODS} for r in rates}
    per_rate_paired = {r: {m: [] for m in METHODS} for r in rates}

    for P, name, pat in patterns:
        for rate in rates:
            hits = {m: 0 for m in METHODS}
            for r in range(n_reps):
                seed = seed0 + r
                x0, y0 = generate_periodic_tact(L, pat, seed=seed)
                rng = np.random.default_rng(seed + 2_000_000)
                if rate > 0:
                    x, y, ng = inject_random_tooth_glitches(y0, rate, mode="delete", rng=rng)
                else:
                    x, y = x0, y0
                res = run_all_methods(x, y, N_max, seed, n_boot=n_boot)
                for m in METHODS:
                    ok = int(res[m] == P)
                    hits[m] += ok
                    per_rate_hits[rate][m][0] += ok
                    per_rate_hits[rate][m][1] += 1
                    per_rate_paired[rate][m].append(ok)
            print(f"  [P={P} {name} rate={rate*100:4.1f}%] " +
                  ", ".join(f"{m}={hits[m]}/{n_reps}" for m in METHODS))

    return per_rate_hits, per_rate_paired


def report_stratified_table(per_rate_hits, rates):
    banner("[C15-1] rate별 층화 성공률 표 (11패턴 x 30회 = 330회/rate)")
    header = f"{'rate':>6} | " + " | ".join(f"{m:>18}" for m in METHODS) + " | 순위(1위)"
    print(header)
    print("-" * len(header))
    ranks = {}
    for rate in rates:
        cells = []
        vals = {}
        for m in METHODS:
            s, n = per_rate_hits[rate][m]
            pct = 100 * s / n
            vals[m] = pct
            lo, hi = wilson_ci(s, n)
            cells.append(f"{pct:5.1f} [{100*lo:4.1f},{100*hi:5.1f}]".rjust(18))
        top = max(vals, key=vals.get)
        ranks[rate] = sorted(vals, key=lambda m: -vals[m])
        print(f"{rate*100:>5.1f}% | " + " | ".join(cells) + f" | {top}")

    print("\n[rate별 전체 순위 (내림차순)]")
    for rate in rates:
        print(f"  rate={rate*100:4.1f}%: " + " > ".join(ranks[rate]))

    print("\n[특기 사항]")
    geo = [per_rate_hits[r]["기하2단계"][0] / per_rate_hits[r]["기하2단계"][1] * 100 for r in rates]
    acfg = [per_rate_hits[r]["ACF(전역최대)"][0] / per_rate_hits[r]["ACF(전역최대)"][1] * 100 for r in rates]
    print(f"  기하2단계 rate별 추이: {[f'{v:.1f}' for v in geo]}")
    print(f"  ACF(전역최대) rate별 추이: {[f'{v:.1f}' for v in acfg]}")
    monotone = all(acfg[i] <= acfg[i+1] for i in range(len(acfg)-2))  # rate=0~2%까지 단조 확인(4%는 붕괴 가능)
    print(f"  ACF(전역최대)가 rate=0~2%에서 단조 증가하는가: {monotone}")
    r05 = rates[1] if len(rates) > 1 else None
    if r05 is not None:
        top_at_05 = max(per_rate_hits[r05], key=lambda m: per_rate_hits[r05][m][0])
        print(f"  rate={r05*100:.1f}%에서의 단독 1위: {top_at_05} "
              f"({100*per_rate_hits[r05][top_at_05][0]/per_rate_hits[r05][top_at_05][1]:.1f}%)")


def report_stratified_mcnemar(per_rate_paired, rates, a="기하2단계", b="ACF(전역최대)"):
    banner(f"[C15-5] rate별 McNemar: {a} vs {b}")
    print(f"각 rate 내에서 대응표본(n=330)으로 검정한다. 5개 rate에 걸친 다중비교이므로")
    print(f"Holm-Bonferroni를 적용한다.\n")

    raw = []
    rows = []
    rng = np.random.default_rng(4242)
    for rate in rates:
        va = np.array(per_rate_paired[rate][a])
        vb = np.array(per_rate_paired[rate][b])
        n01, n10, p = mcnemar_exact(va, vb)
        d, lo, hi = paired_bootstrap_diff(va, vb, rng=rng)
        raw.append(p)
        rows.append((rate, n01, n10, d, lo, hi, p))

    adj = holm(raw)
    print(f"{'rate':>6} | {'a만성공':>7} | {'b만성공':>7} | {'차이(b-a)':>10} | {'95% CI':>20} | {'Holm p':>10} | 판정")
    print("-" * 90)
    for (rate, n01, n10, d, lo, hi, p), pa in zip(rows, adj):
        sig = "유의" if pa < 0.05 else "n.s."
        direction = f"{a} 우세" if d < 0 and pa < 0.05 else (f"{b} 우세" if d > 0 and pa < 0.05 else "동률")
        print(f"{rate*100:>5.1f}% | {n01:>7} | {n10:>7} | {100*d:>+9.1f}%p | "
              f"[{100*lo:>+7.1f},{100*hi:>+7.1f}]%p | {pa:>10.2e} | {sig} ({direction})")


def plot_stratified(per_rate_hits, rates, out_name):
    _ensure_image_dir()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rates_pct = [r * 100 for r in rates]
    for m in METHODS:
        ys = [100 * per_rate_hits[r][m][0] / per_rate_hits[r][m][1] for r in rates]
        ax.plot(rates_pct, ys, marker="o", label=m)
    ax.set_xlabel("결손 비율 rate(%)")
    ax.set_ylabel("성공률(%) [strict, 11패턴 평균]")
    ax.set_title("rate별 층화 성공률 (동일 가중 집계와 달리 rate마다 순위가 다르다)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = IMAGE_DIR / out_name
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out}")


def main():
    tee = RelPathTee(OUT_DIR / "verification_tooth_v3_log.txt", OUT_DIR)
    sys.stdout = tee
    try:
        print("=" * 72)
        print("5차 리뷰 C15 대응 로그 (rate별 층화 분석)")
        print("=" * 72)

        patterns = all_physics_patterns()
        rates = (0.0, 0.005, 0.01, 0.02, 0.04)
        per_rate_hits, per_rate_paired = run_stratified(patterns, rates)
        report_stratified_table(per_rate_hits, rates)
        report_stratified_mcnemar(per_rate_paired, rates)
        plot_stratified(per_rate_hits, rates, "이미지_11_rate별_층화_성공률.png")

        print("\n" + "=" * 72)
        print("검증 로그 종료")
        print("=" * 72)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
    print("완료: verification_tooth_v3_log.txt")


if __name__ == "__main__":
    main()
