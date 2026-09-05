"""
"삼각형 기하학적 유사성 기반 시계열 반복주기(N) 검출 기법" 논문의 실험 재현 스크립트.
(period_detection_paper.md의 4.2, 6.1~6.3절 실험 및 5.4절 알고리즘을 검증한다.)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent


def check(condition, message):
    """assert 대신 사용하는 소프트 검증: 결과를 출력하고 통과 여부를 bool로 반환한다."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    return condition


# ----------------------------------------------------------------------
# 함수 1. generate_periodic_tact
# ----------------------------------------------------------------------
def generate_periodic_tact(L, pattern_params, seed=None):
    """
    주기 P = len(pattern_params)를 갖는 합성 tact 시계열을 생성한다.

    index i의 tact는 N(pattern_params[i % P][0], pattern_params[i % P][1])에서
    샘플링되며, 간격은 음수가 될 수 없으므로 1e-4로 하한 클리핑한다.

    Parameters
    ----------
    L : int
        생성할 샘플 개수
    pattern_params : list[tuple[float, float]]
        [(mean, std), ...] 길이 P인 리스트 (P = 주기)
    seed : int | None
        난수 시드. None이면 매 호출마다 다른 난수열 사용.

    Returns
    -------
    x : np.ndarray
        누적 시각 (time) = cumsum(y)
    y : np.ndarray
        간격 시간 (tact)
    """
    rng = np.random.default_rng(seed)
    P = len(pattern_params)
    idx = np.arange(L)
    means = np.array([pattern_params[i % P][0] for i in idx])
    stds = np.array([pattern_params[i % P][1] for i in idx])

    y = rng.normal(loc=means, scale=stds)
    y = np.clip(y, 1e-4, None)
    x = np.cumsum(y)
    return x, y


# ----------------------------------------------------------------------
# 함수 2. triangle_metrics
# ----------------------------------------------------------------------
def triangle_metrics(x, y, N):
    """
    인덱스 n을 N..len(x)-N-1 범위에서 벡터화하여 삼각형 ABC의 기하 지표를 계산한다.

    A=(x[n],y[n]), B=(x[n-N],y[n-N]), C=(x[n+N],y[n+N])

    |AB|, |AC|, |BC|      : 각 변의 유클리드 거리
    R  = |AB| / |AC|      : 변 비율
    S  = 0.5*|AB x AC|    : 외적 기반 삼각형 넓이
    H  = 2*S / |BC|       : BC를 밑변으로 하는 높이
    theta                 : A에서의 내각, arccos(AB·AC / (|AB||AC|))
    sin_theta = 2*S/(|AB||AC|) : 완전히 정규화된 대안 지표 (7절 참고)

    Returns
    -------
    dict[str, np.ndarray] : 위 8개 지표를 담은 딕셔너리 (각 길이 = len(x)-2N)
    """
    n = np.arange(N, len(x) - N)
    ax, ay = x[n], y[n]
    bx, by = x[n - N], y[n - N]
    cx, cy = x[n + N], y[n + N]

    ABx, ABy = bx - ax, by - ay
    ACx, ACy = cx - ax, cy - ay
    BCx, BCy = cx - bx, cy - by

    AB = np.hypot(ABx, ABy)
    AC = np.hypot(ACx, ACy)
    BC = np.hypot(BCx, BCy)

    cross = ABx * ACy - ABy * ACx
    S = 0.5 * np.abs(cross)
    H = np.where(BC > 0, 2 * S / BC, np.nan)
    R = np.where(AC > 0, AB / AC, np.nan)

    dot = ABx * ACx + ABy * ACy
    denom = AB * AC
    cos_theta = np.clip(np.where(denom > 0, dot / denom, np.nan), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.where(denom > 0, 2 * S / denom, np.nan)

    return {"AB": AB, "AC": AC, "BC": BC, "R": R, "S": S, "H": H,
            "theta": theta, "sin_theta": sin_theta}


# ----------------------------------------------------------------------
# 함수 3. find_period (5.4절 순차 탐색 알고리즘)
# ----------------------------------------------------------------------
def find_period(x, y, N_max, n_trials=5):
    """
    N=1..N_max를 오름차순으로 스캔하며 mean H(N)을 계산하고,
    "국소 최솟값이면서 전체 H(N) 분포의 하위 20% 이내"인 첫 N을 주기로 판정한다.
    (전역 최솟값을 쓰지 않는 이유는 5.4절 "주의" 참고: 정배수도 비슷한 H값을
    가지므로 노이즈에 의해 배수가 최솟값이 되는 오탐을 피하기 위함.)

    노이즈에 대한 강건성을 높이기 위해 (x, y)를 n_trials개의 연속 구간으로
    나누어 구간별 mean H(N)을 구한 뒤 평균낸다 (일종의 블록 재표본).

    임계값(하위 20%)으로 정합점을 찾지 못하면, 대안으로 "직전 대비 50% 이상
    하락하는 국소 최솟값"을 찾고, 그마저 없으면 전역 최솟값을 반환한다.

    Returns
    -------
    int : 추정 주기 N*
    """
    n = len(x)
    chunk_size = n // n_trials

    Ns, H_of_N = [], []
    for N in range(1, N_max + 1):
        if 2 * N >= chunk_size:
            break
        chunk_H_means = []
        for t in range(n_trials):
            start, end = t * chunk_size, (t + 1) * chunk_size
            xs, ys = x[start:end], y[start:end]
            m = triangle_metrics(xs, ys, N)
            chunk_H_means.append(np.nanmean(m["H"]))
        Ns.append(N)
        H_of_N.append(np.mean(chunk_H_means))

    Ns = np.array(Ns)
    H_of_N = np.array(H_of_N)
    threshold = np.percentile(H_of_N, 20)

    for i in range(len(Ns)):
        is_local_min = (i == 0 or H_of_N[i] < H_of_N[i - 1]) and \
                        (i == len(Ns) - 1 or H_of_N[i] < H_of_N[i + 1])
        if is_local_min and H_of_N[i] <= threshold:
            return int(Ns[i])

    for i in range(1, len(Ns)):
        dropped_50pct = H_of_N[i] < H_of_N[i - 1] * 0.5
        is_local_min = i == len(Ns) - 1 or H_of_N[i] < H_of_N[i + 1]
        if dropped_50pct and is_local_min:
            return int(Ns[i])

    return int(Ns[np.argmin(H_of_N)])


# ----------------------------------------------------------------------
# 실험 1 (4.2절): S의 N-선형성 vs H의 N-불변성 검증
# ----------------------------------------------------------------------
def experiment1_scale_dependency():
    print("\n" + "=" * 70)
    print("실험 1 (4.2절): S는 N에 선형 비례, H는 N-불변인지 검증 (10회 반복, 시드 없음)")
    print("=" * 70)

    pattern_params = [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)]
    L = 100_000
    N_list = [4, 8, 12, 16]
    n_trials = 10

    trial_results = {N: {"S": [], "BC": [], "H": []} for N in N_list}
    for _ in range(n_trials):
        x, y = generate_periodic_tact(L, pattern_params, seed=None)
        for N in N_list:
            m = triangle_metrics(x, y, N)
            trial_results[N]["S"].append(np.nanmean(m["S"]))
            trial_results[N]["BC"].append(np.nanmean(m["BC"]))
            trial_results[N]["H"].append(np.nanmean(m["H"]))

    print(f"\n{'N':>4} | {'mean S':>10} | {'S/N':>8} | {'mean |BC|':>10} | {'|BC|/N':>8} | {'mean H':>10}")
    print("-" * 65)
    row = {}
    for N in N_list:
        mean_S = np.mean(trial_results[N]["S"])
        mean_BC = np.mean(trial_results[N]["BC"])
        mean_H = np.mean(trial_results[N]["H"])
        row[N] = (mean_S, mean_S / N, mean_BC, mean_BC / N, mean_H)
        print(f"{N:>4} | {mean_S:>10.4f} | {mean_S / N:>8.4f} | {mean_BC:>10.4f} | "
              f"{mean_BC / N:>8.4f} | {mean_H:>10.5f}")

    s_over_n = np.array([row[N][1] for N in N_list])
    bc_over_n = np.array([row[N][3] for N in N_list])
    mean_h = np.array([row[N][4] for N in N_list])

    print()
    check((s_over_n.max() / s_over_n.min() - 1) <= 0.01,
          f"S/N이 N에 무관하게 상수 (오차 1% 이내): {s_over_n}")
    check((bc_over_n.max() / bc_over_n.min() - 1) <= 0.01,
          f"|BC|/N이 N에 무관하게 상수 (오차 1% 이내): {bc_over_n}")
    check((mean_h.max() / mean_h.min() - 1) <= 0.01,
          f"mean H가 N에 무관하게 상수 (오차 1% 이내): {mean_h}")


# ----------------------------------------------------------------------
# 실험 2 (6.2절): 주기 정배수 vs 비주기 구간 비교
# ----------------------------------------------------------------------
def experiment2_multiple_vs_nonmultiple():
    print("\n" + "=" * 70)
    print("실험 2 (6.2절): 주기 정배수(4,8,12) vs 비배수(2,6,10)의 H 비교 (seed=42)")
    print("=" * 70)

    pattern_params = [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)]
    L = 100_000
    N_list = [2, 4, 6, 8, 10, 12]

    x, y = generate_periodic_tact(L, pattern_params, seed=42)
    H_results = {N: triangle_metrics(x, y, N)["H"] for N in N_list}

    print(f"\n{'N':>4} | {'mean H':>10} | {'median H':>10} | {'std':>10} | {'max':>10}")
    print("-" * 55)
    mean_h = {}
    for N in N_list:
        h = H_results[N]
        mean_h[N] = np.nanmean(h)
        print(f"{N:>4} | {mean_h[N]:>10.4f} | {np.nanmedian(h):>10.4f} | "
              f"{np.nanstd(h):>10.4f} | {np.nanmax(h):>10.4f}")

    multiples = [4, 8, 12]
    non_multiples = [2, 6, 10]
    mult_mean = np.mean([mean_h[N] for N in multiples])
    nonmult_mean = np.mean([mean_h[N] for N in non_multiples])

    print()
    check(mult_mean * 3 <= nonmult_mean,
          f"정배수 mean H({mult_mean:.4f})가 비배수 mean H({nonmult_mean:.4f})의 1/3 이하")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot([H_results[N] for N in N_list], tick_labels=[f"N={N}" for N in N_list], showfliers=True)
    ax.set_ylabel("Height (H)")
    ax.set_title("Triangle Height (H) by N — synthetic N=4 periodic data (seed=42)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "synth_h_boxplot.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out_path}")

    return x, y


# ----------------------------------------------------------------------
# 실험 3 (5.4절): 순차 탐색 알고리즘 검증
# ----------------------------------------------------------------------
def experiment3_find_period(x, y):
    print("\n" + "=" * 70)
    print("실험 3 (5.4절): find_period 순차 탐색 알고리즘 검증")
    print("=" * 70)

    N_star = find_period(x, y, N_max=20, n_trials=5)
    print(f"\nfind_period(N_max=20, n_trials=5) 반환값: N* = {N_star}")
    check(N_star == 4, "합성 데이터(N=4 반복 패턴)에서 추정 주기가 4로 검출됨")


if __name__ == "__main__":
    experiment1_scale_dependency()
    x_seeded, y_seeded = experiment2_multiple_vs_nonmultiple()
    experiment3_find_period(x_seeded, y_seeded)
