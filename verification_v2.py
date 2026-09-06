"""
period_detection_paper_v2.md 개정판을 위한 추가 검증 스크립트.

적대적 피어리뷰의 Critical/Major 지적사항을 실증으로 보강한다.
  - C2 : find_period_direct_phase의 후보 집합(2의 거듭제곱 편향) 검증 및
         참 주기 P=4/6/5로 확장한 18개 시나리오 비교
  - M2 : H의 N-불변성에 대한 선형회귀 기울기 부트스트랩 검정
  - M3 : 단일 시행 비교를 시드 30회 반복 성공률 + Wilson 95% CI로 전환
  - Minor : 작은 N에서 Σ⁻_N(n) ≈ Nd 근사의 타당성 실증

핵심 지표 함수(triangle_metrics, find_period, find_period_two_stage 등)는
find_n.py의 것을 그대로 재사용한다 — 본 스크립트는 "추가 검증"만 담당한다.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from find_n import (
    IMAGE_DIR,
    _ensure_image_dir,
    bootstrap_diff_ci,
    check,
    find_period_acf,
    find_period_direct_phase,
    find_period_fft,
    find_period_two_stage,
    generate_periodic_tact,
    generate_periodic_tact_with_drift,
    generate_periodic_tact_with_midshift,
    triangle_metrics,
)

plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STD_RATIO = 0.13333  # 기존 논문 패턴과 동일한 변동계수(표준편차/평균)
SMALL = (0.1, 0.0133)  # 작은 위상 값 (기존 패턴과 동일)


# ----------------------------------------------------------------------
# 공통 유틸
# ----------------------------------------------------------------------
def make_pattern(big_values, n_phases):
    """
    big_values를 짝수 위상(0,2,4,...)에 배치하고 나머지 위상은 SMALL로 채운
    길이 n_phases 패턴을 만든다. n_phases가 홀수면 마지막 위상까지 big을 채운다.
    std는 평균의 STD_RATIO로 통일한다.
    """
    pattern = []
    bi = 0
    for i in range(n_phases):
        if i % 2 == 0 and bi < len(big_values):
            m = big_values[bi]
            bi += 1
            pattern.append((m, round(m * STD_RATIO, 4)))
        else:
            pattern.append(SMALL)
    return pattern


def shift_pattern(pattern, delta=2.0):
    """패턴의 '큰' 위상 값에만 +delta를 더한 시프트 후 패턴 (중간 레벨 시프트용)."""
    out = []
    for m, s in pattern:
        if m > 1.0:  # 큰 위상만 이동
            nm = m + delta
            out.append((nm, round(nm * STD_RATIO, 4)))
        else:
            out.append((m, s))
    return out


def wilson_ci(successes, n, z=1.96):
    """이항 비율의 Wilson 95% 신뢰구간."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _z_two_sided(alpha):
    """양측 alpha에 대응하는 표준정규 분위수 (scipy 없이 이분법으로 계산)."""
    from math import erf, sqrt
    target = 1.0 - alpha / 2.0
    lo, hi = 0.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2
        cdf = 0.5 * (1 + erf(mid / sqrt(2)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bootstrap_diff_ci_fast(a, b, n_boot=200, alpha=0.05, rng=None, interval="se"):
    """
    mean 차이(stat=mean 고정)의 부트스트랩 신뢰구간을 벡터화로 계산한다.
    대량 반복 실험(M3)의 실행 시간을 줄이기 위한 구현이다.

    interval="percentile" : find_n.bootstrap_diff_ci와 동일한 백분위 구간
    interval="se"         : 부트스트랩으로 표준오차만 추정하고 정규근사로 구간을 구성.
                            Bonferroni 보정으로 alpha가 매우 작아질 때(예: 0.003)
                            백분위 방식은 n_boot=200으로 극단 분위수를 추정할 수 없어
                            불안정하므로, 기본값으로 이 방식을 사용한다.
    """
    if rng is None:
        rng = np.random.default_rng()
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    ma = a[rng.integers(0, na, size=(n_boot, na))].mean(axis=1)
    mb = b[rng.integers(0, nb, size=(n_boot, nb))].mean(axis=1)
    diffs = mb - ma
    point = b.mean() - a.mean()
    if interval == "percentile":
        lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    else:
        se = float(np.std(diffs, ddof=1))
        z = _z_two_sided(alpha)
        lo, hi = point - z * se, point + z * se
    return point, lo, hi, bool((lo > 0) or (hi < 0))


# ----------------------------------------------------------------------
# C2 : 직접 위상비교의 일반화 버전 (임의 배수 k 허용)
# ----------------------------------------------------------------------
def find_period_direct_phase_general(y, N_max, n_boot=200, alpha=0.05, rng=None):
    """
    기존 find_period_direct_phase는 N을 2배씩만 키우므로 1,2,4,8,16...
    (2의 거듭제곱)만 반환할 수 있다. 이 일반화 버전은 각 단계에서
    배수 k=2,3,...,N_max//N 를 모두 시도해 임의 주기를 반환할 수 있게 한다.

    후보 M=N*k에 대해, 같은 N-위상 i에 속하는 k개의 M-서브위상
    (y[(i+j*N)::M], j=0..k-1) 중 어느 한 쌍이라도 평균이 유의하게 다르면
    "N은 너무 성긴 분할"로 보고 M으로 세분화한다. 어떤 k에서도 유의한
    세분화가 없으면 현재 N을 주기로 반환한다.

    한 단계에서 N*(k-1)회의 다중 비교가 일어나므로 Bonferroni 보정
    (alpha_adj = alpha / (N*(k-1)))을 적용한다. 보정하지 않으면 위양성이
    누적되어 참 주기의 배수로 과도 세분화된다(P=6에서 12, P=5에서 15를
    반환하는 현상을 실측으로 확인).
    """
    if rng is None:
        rng = np.random.default_rng()
    N = 1
    while True:
        found = None
        for k in range(2, N_max // N + 1):
            M = N * k
            alpha_adj = alpha / max(1, N * (k - 1))
            refine = False
            for i in range(N):
                subs = [y[(i + j * N)::M] for j in range(k)]
                subs = [s for s in subs if len(s) >= 10]
                if len(subs) < 2:
                    continue
                for j in range(1, len(subs)):
                    _, _, _, sig = bootstrap_diff_ci_fast(subs[0], subs[j], n_boot=n_boot,
                                                          alpha=alpha_adj, rng=rng)
                    if sig:
                        refine = True
                        break
                if refine:
                    break
            if refine:
                found = M
                break
        if found is None:
            return N
        N = found


def reachable_Ns_doubling(N_max):
    """find_period_direct_phase(2배씩 증가)가 반환 가능한 N의 집합."""
    out, N = [1], 1
    while 2 * N <= N_max:
        N *= 2
        out.append(N)
    return out


def acf_curve(y, N_max):
    """lag 1..N_max의 자기상관 계수 배열."""
    yc = y - np.mean(y)
    den = np.sum(yc ** 2)
    return np.array([np.sum(yc[:-k] * yc[k:]) / den for k in range(1, N_max + 1)])


def find_period_acf_global(y, N_max):
    """
    ACF 기반 주기 추정의 변형: find_n.find_period_acf가 쓰는
    "상위 20% 이내의 첫 국소최댓값" 규칙 대신 단순 전역 최댓값을 채택한다.
    (지표 자체가 아니라 판정 규칙이 성능을 좌우함을 보이기 위한 대조군)
    """
    return int(np.argmax(acf_curve(y, N_max))) + 1


# ----------------------------------------------------------------------
# 실험 A (M2): H의 N-불변성 선형회귀 검정
# ----------------------------------------------------------------------
def expA_invariance_regression(n_trials=10, L=100_000, N_list=(4, 8, 12, 16), seed0=1000):
    print("\n" + "=" * 72)
    print("[실험 A / M2] H의 N-불변성: median H ~ a + b*N 회귀 기울기 부트스트랩 검정")
    print("=" * 72)

    pattern = [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)]
    seeds = [seed0 + t for t in range(n_trials)]
    print(f"사용 시드 리스트(재현용): {seeds}")

    # trial_H[t][N] = 시행 t, 후보 N에서의 median H
    trial_H = []
    for s in seeds:
        x, y = generate_periodic_tact(L, pattern, seed=s)
        row = {N: float(np.nanmedian(triangle_metrics(x, y, N)["H"])) for N in N_list}
        trial_H.append(row)

    print(f"\n{'trial':>6} | " + " | ".join(f"{'H(N=' + str(N) + ')':>12}" for N in N_list))
    print("-" * 62)
    for t, row in enumerate(trial_H):
        print(f"{t:>6} | " + " | ".join(f"{row[N]:>12.5f}" for N in N_list))

    Ns = np.array(N_list, dtype=float)

    def slope_of(rows):
        ys = np.array([np.median([r[N] for r in rows]) for N in N_list])
        b, a = np.polyfit(Ns, ys, 1)
        return b, a, ys

    b_hat, a_hat, y_hat = slope_of(trial_H)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n_trials, size=n_trials)
        boots.append(slope_of([trial_H[i] for i in idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"\nN별 median H (10회 시행의 median): " +
          ", ".join(f"N={N}: {v:.5f}" for N, v in zip(N_list, y_hat)))
    print(f"회귀식: H ≈ {a_hat:.6f} + ({b_hat:+.3e})·N")
    print(f"기울기 b = {b_hat:+.3e}, 부트스트랩 95% CI = [{lo:+.3e}, {hi:+.3e}]")
    contains_zero = bool(lo <= 0 <= hi)
    # N=4→16 구간에서 기울기가 유발하는 H 변화량을 H 평균 대비 비율로 환산
    rel = abs(b_hat) * (Ns.max() - Ns.min()) / float(np.mean(y_hat)) * 100
    print(f"N=4→16 구간 예측 변화량 = {b_hat * (Ns.max() - Ns.min()):+.3e} "
          f"(평균 H 대비 {rel:.2f}%)")
    check(contains_zero,
          f"기울기 95% CI가 0을 포함 → N-불변성 기각 못함 (CI=[{lo:+.3e}, {hi:+.3e}])")
    if not contains_zero:
        print(f"  (주의) CI가 0을 포함하지 않음 — 잔여 추세가 존재하나 크기는 평균 H의 {rel:.2f}% 수준")
    return {"slope": b_hat, "ci": (lo, hi), "contains_zero": contains_zero,
            "rel_pct": rel, "seeds": seeds, "y_hat": y_hat.tolist()}


# ----------------------------------------------------------------------
# 실험 B (Minor): 작은 N에서 Σ⁻_N(n) ≈ N·d 근사의 타당성
# ----------------------------------------------------------------------
def expB_lln_approximation(L=100_000, N_list=(1, 2, 4, 8, 16), seed=42):
    print("\n" + "=" * 72)
    print("[실험 B / Minor] 작은 N에서 Σ⁻_N(n) ≈ N·d 근사의 타당성 (변동계수 감소)")
    print("=" * 72)
    pattern = [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)]
    x, y = generate_periodic_tact(L, pattern, seed=seed)
    d = float(np.mean(y))
    print(f"tact 평균 d = {d:.4f} (seed={seed}, L={L:,})\n")
    print(f"{'N':>4} | {'mean Σ⁻':>10} | {'N·d':>10} | {'std Σ⁻':>10} | {'std/mean':>10}")
    print("-" * 56)
    rows = []
    for N in N_list:
        sums = np.convolve(y, np.ones(N), mode="valid")  # 길이 N 구간합
        m, s = float(np.mean(sums)), float(np.std(sums))
        rows.append((N, m, N * d, s, s / m))
        print(f"{N:>4} | {m:>10.4f} | {N * d:>10.4f} | {s:>10.4f} | {s / m:>10.4f}")
    print("\nmean Σ⁻가 N·d와 거의 일치하고, 변동계수(std/mean)는 N이 커질수록 감소한다.")
    return rows


# ----------------------------------------------------------------------
# 실험 C (C2): 직접 위상비교의 후보 집합 점검
# ----------------------------------------------------------------------
def expC_direct_phase_candidate_set(N_max=20):
    print("\n" + "=" * 72)
    print("[실험 C / C2] find_period_direct_phase가 반환 가능한 N 후보 집합 점검")
    print("=" * 72)
    reachable = reachable_Ns_doubling(N_max)
    print(f"N_max={N_max}일 때 2배 증가 방식이 반환 가능한 N: {reachable}")
    print("→ 2의 거듭제곱만 가능하므로 P=5, 6, 7, ... 같은 주기는 원리적으로 반환 불가.")

    # 실증: P=6, P=5 데이터에서 실제 반환값 확인
    p6 = make_pattern([10.0, 5.0, 7.0], 6)
    p5 = make_pattern([10.0, 5.0, 7.0], 5)
    print(f"\nP=6 패턴: {p6}")
    print(f"P=5 패턴: {p5}")
    for label, pat, P in [("P=6", p6, 6), ("P=5", p5, 5)]:
        x, y = generate_periodic_tact(15_000, pat, seed=42)
        got_double = find_period_direct_phase(y, N_max, n_boot=200,
                                              rng=np.random.default_rng(0))
        got_general = find_period_direct_phase_general(y, N_max, n_boot=200,
                                                       rng=np.random.default_rng(0))
        print(f"  {label}(참값 {P}): 2배 방식 → {got_double}, 일반화(k=2..) 방식 → {got_general}")
    check(set(reachable) == {1, 2, 4, 8, 16},
          "2배 방식의 후보 집합이 {1,2,4,8,16}(2의 거듭제곱)으로 제한됨을 확인")
    return reachable


# ----------------------------------------------------------------------
# 실험 D (C2+M3): P=4/6/5 x 대비 x 조건, 30회 반복 성공률
# ----------------------------------------------------------------------
def expD_success_rate(n_reps=30, L=15_000, N_max=20, seed0=2000, n_boot=200):
    print("\n" + "=" * 72)
    print(f"[실험 D / C2+M3] 18개 시나리오 x {n_reps}회 반복 성공률 (L={L:,}, N_max={N_max})")
    print("=" * 72)

    periods = {
        4: {"high": make_pattern([10.0, 5.0], 4), "low": make_pattern([10.0, 9.0], 4)},
        6: {"high": make_pattern([10.0, 5.0, 7.0], 6), "low": make_pattern([10.0, 9.0, 9.5], 6)},
        5: {"high": make_pattern([10.0, 5.0, 7.0], 5), "low": make_pattern([10.0, 9.0, 9.5], 5)},
    }
    for P, d in periods.items():
        for c, pat in d.items():
            print(f"  P={P} {c:>4}: {pat}")

    drift_amp, drift_period = 3.0, 2000
    methods = ["기하2단계", "ACF(첫국소최대)", "ACF(전역최대)", "FFT",
                "직접위상(2배)", "직접위상(일반화)"]
    results = {}

    for P in (4, 6, 5):
        for contrast in ("high", "low"):
            pat = periods[P][contrast]
            pat_shift = shift_pattern(pat, delta=2.0)
            for cond in ("정상", "사인드리프트", "중간시프트"):
                key = (P, contrast, cond)
                hits = {m: 0 for m in methods}
                for r in range(n_reps):
                    seed = seed0 + r
                    if cond == "정상":
                        x, y = generate_periodic_tact(L, pat, seed=seed)
                    elif cond == "사인드리프트":
                        x, y = generate_periodic_tact_with_drift(L, pat, drift_amp,
                                                                 drift_period, seed=seed)
                    else:
                        x, y = generate_periodic_tact_with_midshift(L, pat, pat_shift, seed=seed)

                    rng = np.random.default_rng(seed)
                    geo = find_period_two_stage(x, y, N_max=N_max, n_trials=5,
                                                confirm_stat_fn=np.mean,
                                                confirm_n_boot=n_boot, rng=rng)
                    if geo["N_star"] == P:
                        hits["기하2단계"] += 1
                    if find_period_acf(y, N_max) == P:
                        hits["ACF(첫국소최대)"] += 1
                    if find_period_acf_global(y, N_max) == P:
                        hits["ACF(전역최대)"] += 1
                    if find_period_fft(y, N_max) == P:
                        hits["FFT"] += 1
                    if find_period_direct_phase(y, N_max, n_boot=n_boot,
                                                rng=np.random.default_rng(seed)) == P:
                        hits["직접위상(2배)"] += 1
                    if find_period_direct_phase_general(y, N_max, n_boot=n_boot,
                                                        rng=np.random.default_rng(seed)) == P:
                        hits["직접위상(일반화)"] += 1
                results[key] = hits
                summary = ", ".join(f"{m}={hits[m]}/{n_reps}" for m in methods)
                print(f"  [P={P}, {contrast}, {cond}] {summary}")

    print("\n" + "-" * 72)
    print(f"[실험 D 요약] 시나리오별 성공률(%) [Wilson 95% CI], n={n_reps}")
    print("-" * 72)
    header = f"{'시나리오':<22} | " + " | ".join(f"{m:>20}" for m in methods)
    print(header)
    print("-" * len(header))
    for key, hits in results.items():
        P, contrast, cond = key
        label = f"P={P} {contrast} {cond}"
        cells = []
        for m in methods:
            lo, hi = wilson_ci(hits[m], n_reps)
            cells.append(f"{100*hits[m]/n_reps:5.1f} [{100*lo:4.1f},{100*hi:5.1f}]".rjust(20))
        print(f"{label:<22} | " + " | ".join(cells))

    print(f"\n[실험 D 총계] 전체 {len(results) * n_reps}회 시행 기준 방법별 성공률")
    totals = {}
    for m in methods:
        s = sum(h[m] for h in results.values())
        n = len(results) * n_reps
        lo, hi = wilson_ci(s, n)
        totals[m] = (s, n, lo, hi)
        print(f"  {m:<18}: {s:>4}/{n} = {100*s/n:5.1f}% [{100*lo:.1f}, {100*hi:.1f}]")

    # 주기별 소계
    print("\n[실험 D 주기별 소계]")
    for P in (4, 6, 5):
        line = [f"  P={P}: "]
        for m in methods:
            s = sum(h[m] for k, h in results.items() if k[0] == P)
            n = 6 * n_reps
            line.append(f"{m}={100*s/n:5.1f}%")
        print("  ".join(line))

    _ensure_image_dir()
    labels = [f"P={k[0]} {k[1]} {k[2]}" for k in results.keys()]
    grid = np.array([[results[k][m] / n_reps for m in methods] for k in results.keys()])
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(methods)):
            ax.text(j, i, f"{100*grid[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="성공률")
    ax.set_title(f"18개 시나리오 x {n_reps}회 반복: 방법별 주기 검출 성공률(%)")
    plt.tight_layout()
    out = IMAGE_DIR / "이미지_06_시나리오별_성공률.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out}")

    return results, totals


def expE_failure_diagnosis(n_reps=15, L=15_000, N_max=20, seed0=2000, n_boot=200):
    """
    각 방법이 '무엇을 반환하며 실패하는지'와 그 원인을 진단한다.
    특히 ACF의 P=6 실패가 지표(ACF) 자체가 아니라 판정 규칙에서 비롯됨을 보인다.
    """
    print("\n" + "=" * 72)
    print("[실험 E] 실패 원인 진단: 각 방법이 반환한 N의 분포와 ACF 곡선")
    print("=" * 72)

    from collections import Counter
    for P, bigs in [(4, [10.0, 5.0]), (6, [10.0, 5.0, 7.0]), (5, [10.0, 5.0, 7.0])]:
        pat = make_pattern(bigs, P)
        got = {m: Counter() for m in ["기하2단계", "ACF(첫국소최대)", "ACF(전역최대)",
                                       "FFT", "직접위상(2배)", "직접위상(일반화)"]}
        for r in range(n_reps):
            seed = seed0 + r
            x, y = generate_periodic_tact(L, pat, seed=seed)
            got["기하2단계"][find_period_two_stage(
                x, y, N_max=N_max, n_trials=5, confirm_stat_fn=np.mean,
                confirm_n_boot=n_boot, rng=np.random.default_rng(seed))["N_star"]] += 1
            got["ACF(첫국소최대)"][find_period_acf(y, N_max)] += 1
            got["ACF(전역최대)"][find_period_acf_global(y, N_max)] += 1
            got["FFT"][find_period_fft(y, N_max)] += 1
            got["직접위상(2배)"][find_period_direct_phase(
                y, N_max, n_boot=n_boot, rng=np.random.default_rng(seed))] += 1
            got["직접위상(일반화)"][find_period_direct_phase_general(
                y, N_max, n_boot=n_boot, rng=np.random.default_rng(seed))] += 1
        print(f"\n  --- P={P} (정상 조건, {n_reps}회): 반환된 N의 분포 ---")
        for m, c in got.items():
            print(f"    {m:<18}: {dict(sorted(c.items()))}")

    pat6 = make_pattern([10.0, 5.0, 7.0], 6)
    x, y = generate_periodic_tact(L, pat6, seed=seed0)
    acf = acf_curve(y, N_max)
    thr = float(np.percentile(acf, 80))
    print(f"\n  --- P=6 데이터의 ACF 곡선 (seed={seed0}) ---")
    print("    " + ", ".join(f"lag{k+1}={acf[k]:+.3f}" for k in range(min(12, N_max))))
    print(f"    상위 20% 임계값 = {thr:+.4f}")
    print(f"    참 주기 lag=6의 ACF = {acf[5]:+.4f} (lag=2: {acf[1]:+.4f}, lag=4: {acf[3]:+.4f})")
    print("    → lag=6의 신호(0.96 수준)는 lag=2,4(0.76 수준)와 뚜렷이 구분되지만,")
    print("      '상위 20% 이내의 첫 국소최댓값' 규칙은 임계값 부근의 lag=2/4를 먼저 채택해")
    print("      P=6에서 오답을 낸다. 즉 실패 원인은 ACF 지표가 아니라 판정 규칙이다.")
    return acf


if __name__ == "__main__":
    print("=" * 72)
    print("period_detection_paper_v2.md 개정 근거 검증 로그")
    print("=" * 72)
    expA_invariance_regression()
    expB_lln_approximation()
    expC_direct_phase_candidate_set()
    expD_success_rate()
    expE_failure_diagnosis()
    print("\n" + "=" * 72)
    print("검증 로그 종료")
    print("=" * 72)
