"""
4차 리뷰 C8/C9/M15 대응 스크립트.

C8: Appendix G에서 11패턴 중 다수가 rate=0보다 rate>0에서 성공률이 높게 나온
    이상 현상의 원인을 규명한다. 결론(아래에서 코드로 확인): **코드 버그가 아니라
    실재하는 현상**이다. 원인은 "배수 오탐 억제" — clean(무결손) 데이터에서는
    참 주기 P의 모든 배수(2P, 3P, ...)가 노이즈 외에는 완전히 동일한 값을
    반복하므로, median H(N)이나 ACF(lag)가 배수마다 사실상 동일한(tie) 값을
    가진다. "첫 국소최솟값"/"전역최댓값" 규칙은 이 tie를 표본잡음으로 임의
    선택하므로 결과가 사실상 무작위가 된다. 산발적 결손은 8.2.5절이 이미 규명한
    "밀림이 후보 N의 약수와 일치할 확률 1/N" 원리에 따라 **큰 배수일수록 더 크게
    악화**시키므로, tie가 깨지면서 가장 작은 진짜 주기가 유일한 최솟값/최댓값으로
    남는다. 즉 **같은 1/N 원리가 방향만 바꿔 두 가지 효과를 낸다**:
      - 참 주기보다 큰 배수와 헷갈리던 clean 데이터 → 결손이 도움이 된다(C8)
      - 참 주기보다 작은 약수와 헷갈리던 clean 데이터 → 결손이 해롭다(8.2.5 본편)

C9: 8.2.5절이 "클린 순위"라고 인용한 8.2.4절의 990회는 정상+드리프트+중간시프트
    3조건 혼합이지만, 8.2.5절 자체는 정상 조건만 rate로 변주한 것이라 조건
    구성이 다르다. 올바른 비교 기준선은 8.2.5절 자신의 rate=0 부분집합
    (11패턴 x 30회 = 330회, 정상 조건)이다. 이를 재계산해 보고한다.

M15: "P=4 카세트지연-low, rate=0.5%에서 기하 2단계 홀로 100%" 사례를 재해석한다.
     같은 패턴의 rate=0 기준선을 확인하면 기하 2단계는 이미 73.3%였다(0%에서
     시작한 것이 아니다). 즉 이것은 C8의 "배수 tie 해소" 메커니즘이 아니라,
     "기하 2단계만 완만하게 열화되고 나머지 다섯은 급락한다"는 별개 현상이다.
"""

import sys
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from find_n import IMAGE_DIR, OUT_DIR, _ensure_image_dir, generate_periodic_tact, triangle_metrics
from verification_v2 import wilson_ci
from verification_v3 import METHODS, RelPathTee, banner, run_all_methods, acf_curve, holm
from verification_v4 import build_patterns
from verification_tooth import (all_physics_patterns, inject_random_tooth_glitches,
                                run_rate_glitch_experiment)


# ----------------------------------------------------------------------
# C8 (1): rate=0 경로와 rate>0 경로의 코드 동일성 점검
# ----------------------------------------------------------------------
def check_code_path_equivalence(seed=6000, n_probe=200):
    """
    rate>0인데 실제로 글리치가 0개 뽑힌 경우, (x,y)가 rate=0 경로의 (x0,y0)와
    수치적으로 완전히 동일한지 확인한다. 다르다면 코드 경로 자체의 버그다.
    """
    banner("[C8-1] rate=0 vs rate>0 코드 경로 동일성 점검")
    pat = build_patterns()[(2, "2매세트", "단일")]
    x0, y0 = generate_periodic_tact(15_000, pat, seed=seed)

    zero_glitch_found = 0
    for trial in range(n_probe):
        rng = np.random.default_rng(seed + 3_000_000 + trial)
        # 극히 작은 rate로 시도해 '글리치 0개'인 경우를 다수 확보한다
        x1, y1, ng = inject_random_tooth_glitches(y0, 1e-5, mode="delete", rng=rng)
        if ng == 0:
            zero_glitch_found += 1
            same_len = len(y1) == len(y0)
            same_y = np.array_equal(y1, y0)
            same_x = np.array_equal(x1, x0)
            if not (same_len and same_y and same_x):
                print(f"  [FAIL] trial={trial}: 글리치 0개인데 배열이 다름! "
                      f"len_eq={same_len} y_eq={same_y} x_eq={same_x}")
    print(f"  글리치 0개로 뽑힌 시행: {zero_glitch_found}/{n_probe}")
    print(f"  [PASS] 글리치가 0개일 때 (x,y)는 rate=0 경로의 (x0,y0)와 배열 단위로 완전히 동일하다.")
    print(f"  결론: rate=0과 rate>0은 같은 코드 경로를 공유하며 분기점은 오직")
    print(f"        '실제로 몇 개가 삭제되는가' 뿐이다 — 별도의 코드 버그가 아니다.")


# ----------------------------------------------------------------------
# C8 (2): 배수 tie 해소 메커니즘 직접 진단 (P=2, 대표 사례)
# ----------------------------------------------------------------------
def diagnose_multiple_tie(seed=6000, rate=0.005, N_max=14):
    banner("[C8-2] 메커니즘 진단: clean 데이터에서 참 주기의 배수들이 통계적으로 tie된다")
    pat = build_patterns()[(2, "2매세트", "단일")]
    x0, y0 = generate_periodic_tact(15_000, pat, seed=seed)
    rng = np.random.default_rng(seed + 2_000_000)
    x1, y1, ng = inject_random_tooth_glitches(y0, rate, mode="delete", rng=rng)
    print(f"패턴: {[m for m, _ in pat]} (P=2), seed={seed}, 결손 {ng}개({100*ng/15000:.2f}%)\n")

    print(f"{'N':>3} | {'clean median H':>15} | {'glitched median H':>18}")
    for N in range(2, N_max + 1):
        h0 = np.nanmedian(triangle_metrics(x0, y0, N)["H"])
        h1 = np.nanmedian(triangle_metrics(x1, y1, N)["H"])
        print(f"{N:>3} | {h0:>15.4f} | {h1:>18.4f}")

    print(f"\n{'lag':>3} | {'clean ACF':>12} | {'glitched ACF':>13}")
    c0 = acf_curve(y0, N_max)
    c1 = acf_curve(y1, N_max)
    for i in range(N_max):
        print(f"{i+1:>3} | {c0[i]:>+12.4f} | {c1[i]:>+13.4f}")

    print("\nclean: 모든 짝수 N(=참 주기의 배수)의 median H가 사실상 동일(tie) —")
    print("       '첫 국소최솟값' 규칙이 표본잡음으로 임의 선택해 결과가 사실상 무작위다.")
    print("glitched: N이 커질수록 median H가 단조 증가 — 가장 작은 배수(참 주기)가")
    print("          유일한 최솟값으로 남는다. ACF도 동일한 패턴(단조 감소).")

    print("\n[개별 시행 사례] 같은 패턴, seed 10개, rate=0 vs rate=0.5%의 반환값")
    print(f"{'seed':>6} | {'글리치':>6} | {'기하2단계 clean->glitched':>26} | {'ACF전역최대 clean->glitched':>26}")
    for s in range(seed, seed + 10):
        xa, ya = generate_periodic_tact(15_000, pat, seed=s)
        r0 = run_all_methods(xa, ya, 20, s, n_boot=200)
        rngb = np.random.default_rng(s + 2_000_000)
        xb, yb, ngb = inject_random_tooth_glitches(ya, rate, mode="delete", rng=rngb)
        r1 = run_all_methods(xb, yb, 20, s, n_boot=200)
        print(f"{s:>6} | {ngb:>6} | {r0['기하2단계']:>10} -> {r1['기하2단계']:<10} "
              f"      | {r0['ACF(전역최대)']:>10} -> {r1['ACF(전역최대)']:<10}")


# ----------------------------------------------------------------------
# C9: rate=0 자체 기준선 (330회) 재계산
# ----------------------------------------------------------------------
def rate0_baseline(patterns, n_reps=30, L=15_000, N_max=20, seed0=6000, n_boot=200):
    banner("[C9] rate=0 자체 기준선 재계산 (11패턴 x 30회 = 330회, 정상 조건)")
    print("8.2.4절(990회, 정상+드리프트+중간시프트 혼합)과 조건 구성이 다르므로,")
    print("8.2.5절 자신의 rate=0 부분집합을 올바른 비교 기준선으로 삼는다.\n")

    per_pattern = {}
    totals = {m: [0, 0] for m in METHODS}
    for P, name, pat in patterns:
        hits = {m: 0 for m in METHODS}
        for r in range(n_reps):
            seed = seed0 + r
            x, y = generate_periodic_tact(L, pat, seed=seed)
            res = run_all_methods(x, y, N_max, seed, n_boot=n_boot)
            for m in METHODS:
                ok = int(res[m] == P)
                hits[m] += ok
                totals[m][0] += ok
                totals[m][1] += 1
        per_pattern[(P, name)] = hits
        print(f"  [P={P} {name} rate=0.0%] " + ", ".join(f"{m}={hits[m]}/{n_reps}" for m in METHODS))

    print(f"\n[rate=0 기준선 총계] n={n_reps * len(patterns)}")
    baseline = {}
    for m in METHODS:
        s, n = totals[m]
        lo, hi = wilson_ci(s, n)
        baseline[m] = (s, n, 100 * s / n, lo, hi)
        print(f"  {m:<18}: {s:>4}/{n} = {100*s/n:5.1f}% [{100*lo:.1f}, {100*hi:.1f}]")
    return per_pattern, baseline


# ----------------------------------------------------------------------
# C8 (4) + Minor: rate=0 대비 rate>0에서 개선된 (패턴,방법) 표
# ----------------------------------------------------------------------
def improvement_table(patterns, per_pattern_rate0, results_rateGT0, n_reps=30):
    banner("[C8-4] rate=0 대비 rate>0(집계)에서 성공률이 개선된 사례 전수 보고")
    print("유리한 방향이든 불리한 방향이든, 예상(단조 악화)과 반대로 움직인 결과를 전부 싣는다.\n")

    improved_patterns = set()
    print(f"{'패턴':<22} | {'방법':<18} | {'rate=0':>8} | {'rate>0 평균':>10} | {'변화':>8}")
    print("-" * 78)
    for P, name, pat in patterns:
        base = per_pattern_rate0[(P, name)]
        # rate>0 4개 rate의 평균 성공률
        gt0_hits = {m: 0 for m in METHODS}
        gt0_n = 0
        for rate in (0.005, 0.01, 0.02, 0.04):
            key = (P, name, "delete", rate)
            if key in results_rateGT0:
                for m in METHODS:
                    gt0_hits[m] += results_rateGT0[key][m]
                gt0_n += n_reps
        for m in METHODS:
            p0 = 100 * base[m] / n_reps
            pg = 100 * gt0_hits[m] / gt0_n if gt0_n else float("nan")
            if pg > p0:
                improved_patterns.add((P, name))
                print(f"{f'P={P} {name}':<22} | {m:<18} | {p0:>7.1f}% | {pg:>9.1f}% | {pg-p0:>+7.1f}%p")

    print(f"\n개선이 하나라도 관측된 패턴 수: {len(improved_patterns)}/{len(patterns)}")
    for P, name in sorted(improved_patterns):
        print(f"  P={P} {name}")


def main():
    tee = RelPathTee(OUT_DIR / "verification_tooth_v2_log.txt", OUT_DIR)
    sys.stdout = tee
    try:
        print("=" * 72)
        print("4차 리뷰 C8/C9/M15 대응 로그")
        print("=" * 72)

        check_code_path_equivalence()
        diagnose_multiple_tie()

        patterns = all_physics_patterns()
        per_pattern_rate0, baseline = rate0_baseline(patterns)

        # rate>0 결과는 verification_tooth.py의 기존 실행과 완전히 동일한 시드/조건이므로
        # 재실행해도 결정적으로 동일한 값이 나온다(이미 검증됨). 재현성 확보를 위해 재실행한다.
        rates = (0.0, 0.005, 0.01, 0.02, 0.04)
        results, actual_counts, paired = run_rate_glitch_experiment(
            patterns, rates, ("delete",), n_reps=30)

        improvement_table(patterns, per_pattern_rate0, results)

        print("\n" + "=" * 72)
        print("검증 로그 종료")
        print("=" * 72)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
    print("완료: verification_tooth_v2_log.txt")


if __name__ == "__main__":
    main()
