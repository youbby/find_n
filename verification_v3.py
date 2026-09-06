"""
period_detection_paper_v3.md 개정판을 위한 검증 스크립트 (2차 피어리뷰 대응).

v2 대비 추가/변경된 검증
  - C3 : 6개 방법이 동일 데이터셋에서 평가된 대응표본이므로, Wilson CI 겹침이
         아니라 McNemar 검정 + Holm 보정으로 방법 간 차이를 검정한다.
  - C4 : "성공"의 조작적 정의를 명문화하고 strict(N==P) / lenient(N이 P의 정배수)
         두 기준으로 성공률을 병기한다.
  - M4 : H~N 회귀 기울기 CI 끝값 기준 변화율을 실제로 계산해 출력한다.
  - M5 : 드리프트 진폭 스윕으로 "드리프트가 성능을 올리는" 역설을 진단한다.
  - M6 : 위상 대비를 정량 지표로 정의하고, 대비를 맞춘 패턴 세트로 재실험한다.
  - M8 : v1/v2에서 로그 근거가 없던 실험 1~4(4.3절, 6.2절, 7.2절)도 여기서
         재실행해 로그를 남긴다.

핵심 지표 함수는 find_n.py의 것을 그대로 재사용한다.
"""

import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from find_n import (
    IMAGE_DIR,
    OUT_DIR,
    _ensure_image_dir,
    bootstrap_diff_ci,
    experiment1_scale_dependency,
    experiment2_multiple_vs_nonmultiple,
    experiment9_phase_contrast_sensitivity,
    experiment10_mean_bootstrap_confirmation,
    find_period_acf,
    find_period_direct_phase,
    find_period_fft,
    find_period_two_stage,
    generate_periodic_tact,
    generate_periodic_tact_with_drift,
    generate_periodic_tact_with_midshift,
    triangle_metrics,
)
from verification_v2 import (
    STD_RATIO,
    acf_curve,
    find_period_acf_global,
    find_period_direct_phase_general,
    make_pattern,
    reachable_Ns_doubling,
    shift_pattern,
    wilson_ci,
)

plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METHODS = ["기하2단계", "ACF(첫국소최대)", "ACF(전역최대)", "FFT",
           "직접위상(2배)", "직접위상(일반화)"]


class RelPathTee:
    """콘솔+파일 동시 기록. 로컬 절대경로를 저장소 상대경로로 치환한다(M8)."""

    def __init__(self, path, root):
        self.file = open(path, "w", encoding="utf-8")
        self.roots = [str(root), str(root).replace("\\", "/")]
        # 콘솔이 cp949 등 한글 완성형이면 em-dash 같은 문자에서 죽는다.
        # 로그 파일은 항상 UTF-8이므로, 콘솔 쪽만 대체 문자로 넘어가게 한다.
        try:
            sys.__stdout__.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    def _rel(self, text):
        for r in self.roots:
            text = text.replace(r + "\\", "").replace(r + "/", "").replace(r, ".")
        return text

    def write(self, text):
        text = self._rel(text)
        self.file.write(text)          # 로그 파일이 정본. 콘솔 실패가 실행을 막지 않게 먼저 쓴다.
        try:
            sys.__stdout__.write(text)
        except UnicodeEncodeError:
            enc = getattr(sys.__stdout__, "encoding", "utf-8") or "utf-8"
            sys.__stdout__.write(text.encode(enc, errors="replace").decode(enc))

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ======================================================================
# 통계 도구 (C3)
# ======================================================================
def mcnemar_exact(a, b):
    """
    대응표본 이항 결과 a, b(길이 n의 0/1 배열)에 대한 McNemar 정확검정.

    불일치 쌍만 사용한다: n01 = a성공/b실패, n10 = a실패/b성공.
    귀무가설 하에서 불일치 쌍은 p=0.5 이항분포를 따른다.

    Returns
    -------
    (n01, n10, p_value)
    """
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
    """
    동일 시행에서 평가된 두 방법의 성공률 차이(b−a)에 대한 대응표본 부트스트랩 CI.
    시행 단위로 재표본하므로 두 방법의 상관을 그대로 보존한다.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return float(b.mean() - a.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def holm(pvals):
    """Holm-Bonferroni 보정된 p값 리스트를 원래 순서로 반환한다."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


# ======================================================================
# M6 : 위상 대비의 정량 지표
# ======================================================================
def contrast_index(pattern):
    """
    위상 대비 지표 C = min|μi − μj| / mean(μ), 단 '큰' 위상(μ > 1.0)들 사이에서만 계산.

    작은 위상(0.1)은 모든 패턴에서 동일하게 반복되므로 두 작은 위상 간 차이는
    항상 0이다. 실제로 방법을 헷갈리게 만드는 것은 서로 값이 가까운 '큰' 위상
    쌍이므로, 그 최소 상대 간격을 난이도 지표로 삼는다. C가 작을수록 어렵다.
    """
    mus = np.array([m for m, _ in pattern], dtype=float)
    bigs = mus[mus > 1.0]
    if len(bigs) < 2:
        return float("nan")
    gaps = [abs(bigs[i] - bigs[j]) for i in range(len(bigs)) for j in range(i + 1, len(bigs))]
    return float(min(gaps) / mus.mean())


def matched_pattern(P, mean_level=6.0, gap=1.2):
    """
    대비를 맞춘 패턴(M6): P개 위상을 mean_level 중심으로 gap 간격의 등차수열로 배치.

    모든 P에서 mean(μ)=mean_level, 최소 간격=gap 이므로 contrast_index가 동일해진다.
    부수적으로 '큰 값/작은 값 교대' 구조가 사라지므로 6.2절의 홀짝 혼동원도 함께
    제거된다 — 이는 의도된 통제이자 동시에 이 세트의 한계다(본문에 명시).
    """
    offs = (np.arange(P) - (P - 1) / 2.0) * gap
    return [(round(float(mean_level + o), 4), round(float(mean_level + o) * STD_RATIO, 4))
            for o in offs]


# ======================================================================
# 실험 1~4 재실행 (M8): 4.3절 / 6.2절 / 7.2절 표의 로그 근거
# ======================================================================
def exp1_relog_legacy_tables():
    banner("[실험 1 / M8] 4.3절·7.2절 표 재실행 (v1에서 로그 근거가 없던 표)")
    print("주의: 아래 experiment1은 시드를 고정하지 않는 '10회 반복' 설계이므로")
    print("      재실행하면 소수점 아래 값이 달라진다(4.3절 각주의 의도된 설계).")
    print("      experiment2는 seed=42 고정이라 결정적으로 재현된다.")
    experiment1_scale_dependency()
    experiment2_multiple_vs_nonmultiple()


def exp2_scale_decomposition(L=100_000, N_list=(4, 8, 12, 16), seed=42):
    """7.2절 스케일 분해표: S/N, |AB|/N, |AC|/N, BC/N, sinθ·N, κ·N², median H."""
    banner("[실험 2 / M8] 7.2절 스케일 분해표 (seed=42, median 기준)")
    pattern = [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)]
    x, y = generate_periodic_tact(L, pattern, seed=seed)
    print(f"패턴: {pattern}, L={L:,}, seed={seed}\n")
    print(f"{'N':>4} | {'S/N':>8} | {'|AB|/N':>8} | {'|AC|/N':>8} | {'BC/N':>8} | "
          f"{'sinθ·N':>8} | {'κ·N²':>10} | {'median H':>10}")
    print("-" * 84)
    for N in N_list:
        m = triangle_metrics(x, y, N)
        print(f"{N:>4} | {np.nanmedian(m['S'])/N:>8.4f} | {np.nanmedian(m['AB'])/N:>8.4f} | "
              f"{np.nanmedian(m['AC'])/N:>8.4f} | {np.nanmedian(m['BC'])/N:>8.4f} | "
              f"{np.nanmedian(m['sin_theta'])*N:>8.4f} | {np.nanmedian(m['kappa'])*N*N:>10.6f} | "
              f"{np.nanmedian(m['H']):>10.5f}")
    print("\nS/N, |AB|/N, |AC|/N, BC/N이 상수 → 모두 N¹에 비례. sinθ·N이 상수 → sinθ ~ 1/N.")
    print("κ·N²가 상수 → κ ~ 1/N² (과잉 정규화). median H만 N에 무관하게 유지된다.")


def exp3_relog_contrast_tables():
    banner("[실험 3 / M8] 6.2절 표 재실행 (위상 대비 민감도, mean vs median)")
    experiment9_phase_contrast_sensitivity()
    experiment10_mean_bootstrap_confirmation()


# ======================================================================
# 실험 A (M2 + M4): H의 N-불변성 회귀 + 변화율 정정
# ======================================================================
def expA_invariance_regression(n_trials=10, L=100_000, N_list=(4, 8, 12, 16),
                               seed0=1000, n_boot=2000):
    banner("[실험 A / M2+M4] H의 N-불변성: median H ~ a + b*N 회귀 기울기 부트스트랩 검정")
    seeds = [seed0 + i for i in range(n_trials)]
    print(f"사용 시드 리스트(재현용): {seeds}\n")

    pattern = [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)]
    per_trial = np.zeros((n_trials, len(N_list)))
    header = " trial | " + " | ".join(f"{'H(N=%d)' % N:>12}" for N in N_list)
    print(header)
    print("-" * len(header))
    for t, sd in enumerate(seeds):
        x, y = generate_periodic_tact(L, pattern, seed=sd)
        for j, N in enumerate(N_list):
            per_trial[t, j] = np.nanmedian(triangle_metrics(x, y, N)["H"])
        print(f"{t:>6} | " + " | ".join(f"{v:>12.5f}" for v in per_trial[t]))

    med = np.median(per_trial, axis=0)
    print("\nN별 median H (10회 시행의 median): " +
          ", ".join(f"N={N}: {v:.5f}" for N, v in zip(N_list, med)))

    Ns = np.array(N_list, dtype=float)
    b, a = np.polyfit(Ns, med, 1)
    print(f"회귀식: H ≈ {a:.6f} + ({b:.3e})·N")

    rng = np.random.default_rng(12345)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_trials, size=n_trials)
        m2 = np.median(per_trial[idx], axis=0)
        slopes.append(np.polyfit(Ns, m2, 1)[0])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    print(f"기울기 b = {b:.3e}, 부트스트랩 95% CI = [{lo:.3e}, {hi:.3e}]")

    span = float(N_list[-1] - N_list[0])
    mean_H = float(np.mean(med))
    print(f"\n[M4] N={N_list[0]}→{N_list[-1]} 구간(ΔN={span:.0f}) 변화량을 평균 H({mean_H:.6f}) 대비 %로 환산:")
    print(f"  점추정 기울기 기준 : {b*span:+.3e} ({abs(b*span)/mean_H*100:.2f}%)")
    pct_lo, pct_hi = abs(lo * span) / mean_H * 100, abs(hi * span) / mean_H * 100
    print(f"  CI 하한 기준       : {lo*span:+.3e} ({pct_lo:.2f}%)")
    print(f"  CI 상한 기준       : {hi*span:+.3e} ({pct_hi:.2f}%)")
    print(f"  → CI 양끝 중 큰 쪽 = {max(pct_lo, pct_hi):.2f}%")
    print(f"  (v2는 이 값을 0.06%로 잘못 적었다. 정정값은 {max(pct_lo, pct_hi):.2f}%다.)")

    ok = lo <= 0 <= hi
    print(f"\n  [{'PASS' if ok else 'FAIL'}] 기울기 95% CI가 0을 포함 → N-불변성 기각 못함")
    return b, lo, hi, max(pct_lo, pct_hi)


# ======================================================================
# 실험 B: Σ⁻ 근사 / 실험 C: 후보 집합 (v2와 동일, 재로그)
# ======================================================================
def expB_lln(L=100_000, N_list=(1, 2, 4, 8, 16), seed=42):
    banner("[실험 B] 작은 N에서 Σ⁻_N(n) ≈ N·d 근사의 타당성 (변동계수 감소)")
    pattern = [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)]
    x, y = generate_periodic_tact(L, pattern, seed=seed)
    d = float(np.mean(y))
    print(f"tact 평균 d = {d:.4f} (seed={seed}, L={L:,})\n")
    print(f"{'N':>4} | {'mean Σ⁻':>10} | {'N·d':>10} | {'std Σ⁻':>10} | {'std/mean':>10}")
    print("-" * 56)
    for N in N_list:
        n = np.arange(N, len(x) - N)
        sig = x[n] - x[n - N]
        print(f"{N:>4} | {sig.mean():>10.4f} | {N*d:>10.4f} | {sig.std():>10.4f} | "
              f"{sig.std()/sig.mean():>10.4f}")
    print("\nmean Σ⁻가 N·d와 거의 일치하고, 변동계수는 N이 커질수록 감소한다.")


def expC_candidate_set(N_max=20, n_boot=200):
    banner("[실험 C] find_period_direct_phase가 반환 가능한 N 후보 집합 점검")
    reach = reachable_Ns_doubling(N_max)
    print(f"N_max={N_max}일 때 2배 증가 방식이 반환 가능한 N: {reach}")
    print("→ 2의 거듭제곱만 가능하므로 P=5, 6, 7, ... 같은 주기는 원리적으로 반환 불가.\n")
    for P, bigs in [(6, [10.0, 5.0, 7.0]), (5, [10.0, 5.0, 7.0])]:
        pat = make_pattern(bigs, P)
        _, y = generate_periodic_tact(15_000, pat, seed=2000)
        d2 = find_period_direct_phase(y, N_max, n_boot=n_boot, rng=np.random.default_rng(2000))
        dg = find_period_direct_phase_general(y, N_max, n_boot=n_boot, rng=np.random.default_rng(2000))
        print(f"  P={P}(참값 {P}): 2배 방식 → {d2}, 일반화(k=2..) 방식 → {dg}")
    ok = reach == [1, 2, 4, 8, 16]
    print(f"  [{'PASS' if ok else 'FAIL'}] 2배 방식의 후보 집합이 {{1,2,4,8,16}}으로 제한됨")


# ======================================================================
# 실험 D (C3 + C4 + M6): 18개 시나리오 x 30회, 시행 단위 원자료 보존
# ======================================================================
def run_all_methods(x, y, N_max, seed, n_boot=200):
    """한 데이터셋에 6개 방법을 모두 적용해 '반환한 N'을 그대로 돌려준다."""
    rng = np.random.default_rng(seed)
    geo = find_period_two_stage(x, y, N_max=N_max, n_trials=5, confirm_stat_fn=np.mean,
                                confirm_n_boot=n_boot, rng=rng)
    return {
        "기하2단계": geo["N_star"],
        "ACF(첫국소최대)": find_period_acf(y, N_max),
        "ACF(전역최대)": find_period_acf_global(y, N_max),
        "FFT": find_period_fft(y, N_max),
        "직접위상(2배)": find_period_direct_phase(y, N_max, n_boot=n_boot,
                                                  rng=np.random.default_rng(seed)),
        "직접위상(일반화)": find_period_direct_phase_general(y, N_max, n_boot=n_boot,
                                                            rng=np.random.default_rng(seed)),
    }


def make_dataset(P, pat, cond, L, seed, drift_amp=3.0, drift_period=2000):
    if cond == "정상":
        return generate_periodic_tact(L, pat, seed=seed)
    if cond == "사인드리프트":
        return generate_periodic_tact_with_drift(L, pat, drift_amp, drift_period, seed=seed)
    return generate_periodic_tact_with_midshift(L, pat, shift_pattern(pat, delta=2.0), seed=seed)


def expD_success(patterns_by_P, n_reps=30, L=15_000, N_max=20, seed0=2000,
                 n_boot=200, label="18개 시나리오", make_image=None):
    """
    시행 단위로 '반환한 N'을 모두 보존한다(C3의 대응표본 검정에 필요).

    Returns
    -------
    returned : {(P, contrast, cond): {method: [N, ...]}}
    """
    banner(f"[실험 D / C3+C4] {label} x {n_reps}회 반복 (L={L:,}, N_max={N_max})")
    print(f"시드: {seed0}..{seed0 + n_reps - 1} (시나리오마다 동일하게 재사용 → 방법 간 대응표본)")
    for P, d in patterns_by_P.items():
        for c, pat in d.items():
            print(f"  P={P} {c:>7}: C={contrast_index(pat):.3f}  {pat}")

    returned = {}
    for P in patterns_by_P:
        for contrast, pat in patterns_by_P[P].items():
            for cond in ("정상", "사인드리프트", "중간시프트"):
                got = {m: [] for m in METHODS}
                for r in range(n_reps):
                    seed = seed0 + r
                    x, y = make_dataset(P, pat, cond, L, seed)
                    res = run_all_methods(x, y, N_max, seed, n_boot=n_boot)
                    for m in METHODS:
                        got[m].append(res[m])
                returned[(P, contrast, cond)] = got
                strict = {m: sum(1 for v in got[m] if v == P) for m in METHODS}
                print(f"  [P={P}, {contrast}, {cond}] " +
                      ", ".join(f"{m}={strict[m]}/{n_reps}" for m in METHODS))
    if make_image:
        _plot_success_heatmap(returned, n_reps, make_image)
    return returned


def score(returned, mode):
    """strict: N==P / lenient: N이 P의 정배수(N % P == 0). 둘 다 0/1 배열로 반환."""
    out = {m: [] for m in METHODS}
    keys = list(returned.keys())
    for k in keys:
        P = k[0]
        for m in METHODS:
            for v in returned[k][m]:
                ok = (v == P) if mode == "strict" else (v % P == 0)
                out[m].append(1 if ok else 0)
    return {m: np.array(v, dtype=int) for m, v in out.items()}


def report_success_tables(returned, n_reps, tag=""):
    banner(f"[실험 D 요약{tag}] 성공 판정 기준별 성공률 (C4)")
    print("성공 판정 기준")
    print("  strict  : 반환 N == P 만 성공 (주 기준)")
    print("  lenient : 반환 N이 P의 정배수(N % P == 0)면 성공. P의 약수·비배수는 실패")
    n_total = len(returned) * n_reps

    for mode in ("strict", "lenient"):
        sc = score(returned, mode)
        print(f"\n--- {mode} 기준 전체 {n_total}회 시행 ---")
        for m in METHODS:
            s = int(sc[m].sum())
            lo, hi = wilson_ci(s, n_total)
            print(f"  {m:<18}: {s:>4}/{n_total} = {100*s/n_total:5.1f}% "
                  f"[{100*lo:.1f}, {100*hi:.1f}]")
        print(f"  (위 Wilson CI는 각 방법의 marginal 요약이며, 방법 간 비교 근거가 아니다 — C3)")

        print(f"  주기별 소계({mode}):")
        for P in sorted({k[0] for k in returned}):
            cells = []
            for m in METHODS:
                s = sum(1 for k in returned if k[0] == P for v in returned[k][m]
                        if (v == P if mode == "strict" else v % P == 0))
                n = sum(n_reps for k in returned if k[0] == P)
                cells.append(f"{m}={100*s/n:5.1f}%")
            print(f"    P={P}: " + "  ".join(cells))

    # 두 기준에서 순위가 바뀌는지
    rank = {}
    for mode in ("strict", "lenient"):
        sc = score(returned, mode)
        rank[mode] = sorted(METHODS, key=lambda m: -sc[m].sum())
    print(f"\n순위(strict) : {' > '.join(rank['strict'])}")
    print(f"순위(lenient): {' > '.join(rank['lenient'])}")
    print(f"→ 두 기준에서 순위가 {'동일하다' if rank['strict'] == rank['lenient'] else '다르다'}.")
    return rank


def report_scenario_table(returned, n_reps, mode="strict"):
    print(f"\n--- 시나리오별 성공률(%) [Wilson 95% CI], {mode} 기준, n={n_reps} ---")
    header = f"{'시나리오':<24} | " + " | ".join(f"{m:>20}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for k in returned:
        P, contrast, cond = k
        cells = []
        for m in METHODS:
            s = sum(1 for v in returned[k][m] if (v == P if mode == "strict" else v % P == 0))
            lo, hi = wilson_ci(s, n_reps)
            cells.append(f"{100*s/n_reps:5.1f} [{100*lo:4.1f},{100*hi:5.1f}]".rjust(20))
        print(f"{f'P={P} {contrast} {cond}':<24} | " + " | ".join(cells))


def report_mcnemar(returned, mode="strict", tag=""):
    banner(f"[실험 D / C3{tag}] 방법 간 대응표본 비교: McNemar 정확검정 + Holm 보정 ({mode})")
    sc = score(returned, mode)
    n = len(next(iter(sc.values())))
    print(f"동일한 {n}개 데이터셋에서 6개 방법을 모두 평가했으므로 대응표본이다.")
    print("Wilson CI 겹침 판정은 독립 표본을 가정하므로 여기서는 사용하지 않는다.\n")

    pairs, raw = [], []
    for i in range(len(METHODS)):
        for j in range(i + 1, len(METHODS)):
            A, B = METHODS[i], METHODS[j]
            n01, n10, p = mcnemar_exact(sc[A], sc[B])
            pairs.append((A, B, n01, n10, p))
            raw.append(p)
    adj = holm(raw)

    rng = np.random.default_rng(777)
    print(f"{'방법 A':<18} {'방법 B':<18} {'A만성공':>7} {'B만성공':>7} "
          f"{'차이(B-A)':>10} {'95% CI':>22} {'p':>10} {'Holm p':>10} {'유의':>5}")
    print("-" * 120)
    key_rows = []
    for (A, B, n01, n10, p), pa in zip(pairs, adj):
        d, lo, hi = paired_bootstrap_diff(sc[A], sc[B], rng=rng)
        sig = "예" if pa < 0.05 else "아니오"
        print(f"{A:<18} {B:<18} {n01:>7} {n10:>7} {100*d:>+9.1f}%p "
              f"[{100*lo:>+6.1f},{100*hi:>+6.1f}]%p {p:>10.2e} {pa:>10.2e} {sig:>5}")
        key_rows.append((A, B, n01, n10, d, lo, hi, p, pa, pa < 0.05))
    print(f"\n총 {len(pairs)}쌍 비교, Holm-Bonferroni 보정 적용 (α=0.05).")

    print("\n[리뷰 지정 필수 3쌍]")
    must = [("기하2단계", "ACF(첫국소최대)"), ("기하2단계", "직접위상(일반화)"),
            ("ACF(첫국소최대)", "직접위상(일반화)")]
    for A, B in must:
        for row in key_rows:
            if row[0] == A and row[1] == B:
                _, _, n01, n10, d, lo, hi, p, pa, sig = row
                print(f"  {A} vs {B}: 차이 {100*d:+.1f}%p, 95% CI [{100*lo:+.1f}, {100*hi:+.1f}]%p, "
                      f"McNemar p={p:.2e}, Holm p={pa:.2e} → {'유의' if sig else '유의하지 않음'}")
    return key_rows


def _plot_success_heatmap(returned, n_reps, out_name):
    _ensure_image_dir()
    keys = list(returned.keys())
    labels = [f"P={k[0]} {k[1]} {k[2]}" for k in keys]
    grid = np.array([[sum(1 for v in returned[k][m] if v == k[0]) / n_reps for m in METHODS]
                     for k in keys])
    fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(keys) + 2)))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(METHODS)):
            ax.text(j, i, f"{100*grid[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="성공률(strict)")
    ax.set_title(f"{len(keys)}개 시나리오 x {n_reps}회 반복: 방법별 검출 성공률(%) [strict]")
    plt.tight_layout()
    out = IMAGE_DIR / out_name
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out}")


# ======================================================================
# 실험 E: 실패 원인 진단 (v2와 동일, 재로그)
# ======================================================================
def expE_diagnosis(n_reps=15, L=15_000, N_max=20, seed0=2000, n_boot=200):
    banner("[실험 E] 실패 원인 진단: 각 방법이 반환한 N의 분포와 ACF 곡선")
    for P, bigs in [(4, [10.0, 5.0]), (6, [10.0, 5.0, 7.0]), (5, [10.0, 5.0, 7.0])]:
        pat = make_pattern(bigs, P)
        got = {m: Counter() for m in METHODS}
        for r in range(n_reps):
            seed = seed0 + r
            x, y = generate_periodic_tact(L, pat, seed=seed)
            res = run_all_methods(x, y, N_max, seed, n_boot=n_boot)
            for m in METHODS:
                got[m][res[m]] += 1
        print(f"\n  --- P={P} (정상 조건, {n_reps}회): 반환된 N의 분포 ---")
        for m in METHODS:
            print(f"    {m:<18}: {dict(sorted(got[m].items()))}")

    pat6 = make_pattern([10.0, 5.0, 7.0], 6)
    _, y = generate_periodic_tact(L, pat6, seed=seed0)
    curve = acf_curve(y, N_max)
    print(f"\n  --- P=6 데이터의 ACF 곡선 (seed={seed0}) ---")
    print("    " + ", ".join(f"lag{k+1}={curve[k]:+.3f}" for k in range(12)))
    thr = np.percentile(curve, 80)
    print(f"    상위 20% 임계값 = {thr:+.4f}")
    print(f"    참 주기 lag=6의 ACF = {curve[5]:+.4f} (lag=2: {curve[1]:+.4f}, lag=4: {curve[3]:+.4f})")
    print("    → lag=6의 신호는 lag=2,4와 뚜렷이 구분되지만, '상위 20% 이내의 첫")
    print("      국소최댓값' 규칙이 임계값 부근의 lag=2/4를 먼저 채택해 오답을 낸다.")
    print("      즉 실패 원인은 ACF 지표가 아니라 판정 규칙이다.")


# ======================================================================
# 실험 F (M5): 드리프트 진폭 스윕
# ======================================================================
def expF_drift_sweep(amps=(0.0, 1.0, 3.0, 5.0, 10.0), n_reps=30, L=15_000,
                     N_max=20, seed0=2000, n_boot=200, drift_period=2000):
    banner("[실험 F / M5] 사인 드리프트 진폭 스윕: 드리프트가 성능을 올리는가")
    print("v2 로그에서 P=4 high 기하2단계가 정상 24/30 → 드리프트 30/30으로 '상승'했다.")
    print("드리프트 진폭을 바꿔가며 이 역설이 실재하는지, 조건에 따라 갈리는지 확인한다.")
    print(f"진폭: {list(amps)}, 드리프트 주기: {drift_period}, 각 조건 {n_reps}회\n")

    combos = [(4, "high", make_pattern([10.0, 5.0], 4)),
              (4, "low", make_pattern([10.0, 9.0], 4)),
              (6, "high", make_pattern([10.0, 5.0, 7.0], 6)),
              (6, "low", make_pattern([10.0, 9.0, 9.5], 6))]

    results = {}
    for P, contrast, pat in combos:
        print(f"  --- P={P} {contrast} (대비 지표 C={contrast_index(pat):.3f}) ---")
        for amp in amps:
            hits = {m: 0 for m in METHODS}
            for r in range(n_reps):
                seed = seed0 + r
                if amp == 0.0:
                    x, y = generate_periodic_tact(L, pat, seed=seed)
                else:
                    x, y = generate_periodic_tact_with_drift(L, pat, amp, drift_period, seed=seed)
                res = run_all_methods(x, y, N_max, seed, n_boot=n_boot)
                for m in METHODS:
                    if res[m] == P:
                        hits[m] += 1
            results[(P, contrast, amp)] = hits
            print(f"    진폭={amp:>4.1f}: " +
                  ", ".join(f"{m}={100*hits[m]/n_reps:5.1f}%" for m in METHODS))

    print("\n[진폭 0 → 3 변화량(%p), 기하2단계 기준]")
    for P, contrast, _ in combos:
        h0 = results[(P, contrast, 0.0)]["기하2단계"] / n_reps * 100
        h3 = results[(P, contrast, 3.0)]["기하2단계"] / n_reps * 100
        arrow = "상승" if h3 > h0 else ("하락" if h3 < h0 else "동일")
        print(f"  P={P} {contrast:<5}: {h0:5.1f}% → {h3:5.1f}%  ({h3-h0:+5.1f}%p, {arrow})")

    _ensure_image_dir()
    fig, axes = plt.subplots(1, len(combos), figsize=(4.2 * len(combos), 4.2), sharey=True)
    for ax, (P, contrast, pat) in zip(axes, combos):
        for m in METHODS:
            ys = [100 * results[(P, contrast, a)][m] / n_reps for a in amps]
            ax.plot(amps, ys, marker="o", label=m)
        ax.set_title(f"P={P} {contrast} (C={contrast_index(pat):.3f})")
        ax.set_xlabel("사인 드리프트 진폭")
        ax.grid(alpha=0.3)
        ax.set_ylim(-5, 105)
    axes[0].set_ylabel("성공률(%) [strict]")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"드리프트 진폭에 따른 검출 성공률 (L={L:,}, {n_reps}회 반복)")
    plt.tight_layout()
    out = IMAGE_DIR / "이미지_07_드리프트진폭_성공률.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out}")
    return results


# ======================================================================
# 실험 G (M6): 대비 지표와 대비를 맞춘 재실험
# ======================================================================
def expG_contrast_control(n_reps=30, L=15_000, N_max=20, seed0=2000, n_boot=200):
    banner("[실험 G / M6] 위상 대비의 정량화와 대비를 맞춘 패턴으로의 재실험")
    print("대비 지표 C = min|μi−μj| / mean(μ)  (큰 위상들 사이의 최소 상대 간격)")
    print("C가 작을수록 위상 구분이 어렵다.\n")

    orig = {
        4: {"high": make_pattern([10.0, 5.0], 4), "low": make_pattern([10.0, 9.0], 4)},
        6: {"high": make_pattern([10.0, 5.0, 7.0], 6), "low": make_pattern([10.0, 9.0, 9.5], 6)},
        5: {"high": make_pattern([10.0, 5.0, 7.0], 5), "low": make_pattern([10.0, 9.0, 9.5], 5)},
    }
    print("현행 6개 패턴의 대비 지표:")
    print(f"{'패턴':<12} | {'큰 위상 평균들':<22} | {'최소 간격':>9} | {'mean(μ)':>8} | {'C':>7}")
    print("-" * 72)
    for P in (4, 6, 5):
        for c in ("high", "low"):
            pat = orig[P][c]
            mus = np.array([m for m, _ in pat])
            bigs = [float(v) for v in mus[mus > 1.0]]
            gaps = [abs(bigs[i] - bigs[j]) for i in range(len(bigs)) for j in range(i + 1, len(bigs))]
            print(f"P={P} {c:<7} | {str(bigs):<22} | {min(gaps):>9.2f} | "
                  f"{mus.mean():>8.3f} | {contrast_index(pat):>7.3f}")
    print("\n→ 대비 지표가 P마다 크게 다르다. 따라서 v2의 '주기별 성공률 차이'는")
    print("  주기 효과와 패턴 난이도 효과가 뒤섞인 값이다(M6).")

    matched = {P: {"matched": matched_pattern(P)} for P in (4, 6, 5)}
    print("\n대비를 맞춘 패턴 세트 (mean(μ)=6.0, 최소 간격=1.2로 고정):")
    for P in (4, 6, 5):
        pat = matched[P]["matched"]
        print(f"  P={P}: C={contrast_index(pat):.3f}  {pat}")
    print("\n주의: 이 세트는 '큰 값/작은 값 교대' 구조가 없으므로 6.2절의 홀짝 혼동원도")
    print("      함께 제거된다. 대비만 통제한 것이 아니라 위상 구조도 바뀐 세트다.")

    returned = expD_success(matched, n_reps=n_reps, L=L, N_max=N_max, seed0=seed0,
                            n_boot=n_boot, label="대비 통제 3개 주기",
                            make_image="이미지_08_대비통제_성공률.png")
    report_success_tables(returned, n_reps, tag=" (대비 통제)")
    report_scenario_table(returned, n_reps, mode="strict")
    report_mcnemar(returned, mode="strict", tag=" (대비 통제)")
    return returned


# ======================================================================
def main():
    root = OUT_DIR
    log_path = root / "verification_v3_log.txt"
    tee = RelPathTee(log_path, root)
    sys.stdout = tee
    try:
        print("=" * 72)
        print("period_detection_paper_v3.md 개정 근거 검증 로그 (2차 리뷰 대응)")
        print("=" * 72)

        exp1_relog_legacy_tables()
        exp2_scale_decomposition()
        exp3_relog_contrast_tables()
        expA_invariance_regression()
        expB_lln()
        expC_candidate_set()

        base = {
            4: {"high": make_pattern([10.0, 5.0], 4), "low": make_pattern([10.0, 9.0], 4)},
            6: {"high": make_pattern([10.0, 5.0, 7.0], 6), "low": make_pattern([10.0, 9.0, 9.5], 6)},
            5: {"high": make_pattern([10.0, 5.0, 7.0], 5), "low": make_pattern([10.0, 9.0, 9.5], 5)},
        }
        returned = expD_success(base, n_reps=30, label="18개 시나리오",
                                make_image="이미지_06_시나리오별_성공률.png")
        report_scenario_table(returned, 30, mode="strict")
        report_success_tables(returned, 30)
        report_mcnemar(returned, mode="strict")
        report_mcnemar(returned, mode="lenient")

        expE_diagnosis()
        expF_drift_sweep()
        expG_contrast_control()

        print("\n" + "=" * 72)
        print("검증 로그 종료")
        print("=" * 72)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
    print(f"\n완료: {log_path}")


if __name__ == "__main__":
    main()
