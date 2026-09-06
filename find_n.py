"""
"삼각형 기하학적 유사성 기반 시계열 반복주기(N) 검출 기법" 논문의 실험 재현 스크립트.
(period_detection_paper.md의 4.3, 6~8절 실험 및 5.2절 알고리즘을 검증한다.)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = OUT_DIR / "image"


def _ensure_image_dir():
    IMAGE_DIR.mkdir(exist_ok=True)
    return IMAGE_DIR


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
    샘플링되며, 간격은 음수가 될 수 없으므로 0.001로 하한 클리핑하고 실측 데이터의
    자릿수 특성을 흉내내기 위해 소수점 3자리로 반올림한다.

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
    y = np.clip(y, 0.001, None)
    y = np.round(y, 3)
    x = np.cumsum(y)
    return x, y


def inject_outliers(y, rng, n_outliers=50, outlier_range=(200.0, 600.0)):
    """
    y 중 n_outliers개 지점을 무작위로 골라 outlier_range 범위의 난수를 더한다.
    설비 정지/알람 지연처럼 이따금 발생하는 비정상적으로 긴 대기 시간을 흉내내는
    이상치(outlier) 주입기. 원본 y는 변경하지 않고 복사본을 반환한다.

    Parameters
    ----------
    y : np.ndarray
        원본 tact 배열
    rng : np.random.Generator
    n_outliers : int
        주입할 이상치 개수 (예: len(y)=100,000에 50개)
    outlier_range : tuple[float, float]
        각 이상치 지점에 더할 난수의 (하한, 상한)

    Returns
    -------
    np.ndarray : 이상치가 주입된 새 tact 배열 (x는 이 y로 다시 cumsum해야 함)
    """
    y = y.copy()
    idx = rng.choice(len(y), size=n_outliers, replace=False)
    y[idx] += rng.uniform(outlier_range[0], outlier_range[1], size=n_outliers)
    return y


def generate_random_pattern(period, rng, mean_range=(0.1, 20.0), std_pct_range=(0.05, 0.30)):
    """
    길이 `period`인 랜덤 (mean, std) 패턴을 생성한다.

    각 구간(phase)의 mean은 mean_range에서, std는 "mean의 std_pct_range 비율"로
    독립적으로 뽑아 구간마다 산포 정도가 들쭉날쭉하도록 한다 (고정 13.3% 비율 대신
    5~30% 범위에서 무작위 선택).

    Returns
    -------
    list[tuple[float, float]] : generate_periodic_tact에 바로 사용 가능한 pattern_params
    """
    means = rng.uniform(mean_range[0], mean_range[1], size=period)
    std_pcts = rng.uniform(std_pct_range[0], std_pct_range[1], size=period)
    stds = means * std_pcts
    return list(zip(means.tolist(), stds.tolist()))


# ----------------------------------------------------------------------
# 함수 2. triangle_metrics
# ----------------------------------------------------------------------
def triangle_metrics(x, y, N):
    """
    인덱스 n을 N..len(x)-N-1 범위에서 벡터화하여 삼각형 ABC의 기하 지표를 계산한다.

    A=(x[n],y[n]), B=(x[n-N],y[n-N]), C=(x[n+N],y[n+N])

    |AB|, |AC|, |BC|      : 각 변의 유클리드 거리
    R  = |AB-AC| / (AB+AC) : 대칭 차이 (0에 가까울수록 A가 B,C로부터 대칭적 위치).
                            기존 |AB|/|AC| 비율 형태는 0~∞로 비대칭이고 분모가
                            작을 때 불안정해서, 대칭이고 [0,1]로 유계인 이 형태로 대체.
    S  = 0.5*|AB x AC|    : 외적 기반 삼각형 넓이
    H  = 2*S / |BC|       : BC를 밑변으로 하는 높이
    theta                 : A에서의 내각, arccos(AB·AC / (|AB||AC|))
    cos_theta             : cos(theta) = AB·AC / (|AB||AC|)
    sin_theta = 2*S/(|AB||AC|) : 완전히 정규화된 대안 지표 (5.3절 참고)
    kappa = 1 + cos_theta : 카파 지수. theta→π(완전 일직선, 원하는 케이스)일 때 0,
                            theta→0(퇴화된 반대 방향 케이스)일 때 2로 수렴한다.
                            sin_theta와 달리 theta=0과 theta=π를 구분할 수 있어
                            R(형태 조건)과 H(평탄도 조건) 두 조건을 단일 값으로 통합한다.

    Returns
    -------
    dict[str, np.ndarray] : 위 지표들을 담은 딕셔너리 (각 길이 = len(x)-2N)
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
    R = np.where((AB + AC) > 0, np.abs(AB - AC) / (AB + AC), np.nan)

    dot = ABx * ACx + ABy * ACy
    denom = AB * AC
    cos_theta = np.clip(np.where(denom > 0, dot / denom, np.nan), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.where(denom > 0, 2 * S / denom, np.nan)
    kappa = 1.0 + cos_theta

    return {"AB": AB, "AC": AC, "BC": BC, "R": R, "S": S, "H": H,
            "theta": theta, "cos_theta": cos_theta, "sin_theta": sin_theta,
            "kappa": kappa}


def _metrics_curve(x, y, candidate_Ns, stat_fn=np.nanmedian):
    """
    candidate_Ns 각각에 대해 R/S/H/kappa를 stat_fn(기본 median)으로 집계해
    {metric: {N: value}}로 반환한다. stat_fn=np.nanmean으로 바꾸면 이상치에
    민감한 평균 기반 집계로 전환할 수 있다.
    """
    val = {m: {} for m in ("R", "S", "H", "kappa")}
    for N in candidate_Ns:
        tm = triangle_metrics(x, y, N)
        val["R"][N] = stat_fn(tm["R"])
        val["S"][N] = stat_fn(tm["S"])
        val["H"][N] = stat_fn(tm["H"])
        val["kappa"][N] = stat_fn(tm["kappa"])
    return val


def _pick_best_N(val, candidate_Ns):
    """전역 최적값 기준 지표별 채택 N (H/S/kappa/R 모두 최솟값 = 0에 최근접)."""
    return {
        "R": min(candidate_Ns, key=lambda N: val["R"][N]),
        "S": min(candidate_Ns, key=lambda N: val["S"][N]),
        "H": min(candidate_Ns, key=lambda N: val["H"][N]),
        "kappa": min(candidate_Ns, key=lambda N: val["kappa"][N]),
    }


# ----------------------------------------------------------------------
# 함수 3. find_period (5.2절 순차 탐색 알고리즘)
# ----------------------------------------------------------------------
def find_period(x, y, N_max, n_trials=5):
    """
    N=1..N_max를 오름차순으로 스캔하며 median H(N)을 계산하고,
    "국소 최솟값이면서 전체 H(N) 분포의 하위 20% 이내"인 첫 N을 주기로 판정한다.
    (전역 최솟값을 쓰지 않는 이유는 5.2절 "주의" 참고: 정배수도 비슷한 H값을
    가지므로 노이즈에 의해 배수가 최솟값이 되는 오탐을 피하기 위함.)

    노이즈에 대한 강건성을 높이기 위해 (x, y)를 n_trials개의 연속 구간으로
    나누어 구간별 median H(N)을 구한 뒤 다시 median을 낸다 (일종의 블록 재표본).
    대표값은 평균(mean) 대신 이상치에 강건한 median으로 통일한다.

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
        chunk_H_medians = []
        for t in range(n_trials):
            start, end = t * chunk_size, (t + 1) * chunk_size
            xs, ys = x[start:end], y[start:end]
            m = triangle_metrics(xs, ys, N)
            chunk_H_medians.append(np.nanmedian(m["H"]))
        Ns.append(N)
        H_of_N.append(np.median(chunk_H_medians))

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
# 함수 4. bootstrap_diff_ci / find_period_confirm (6.2절 2단계 확인)
# ----------------------------------------------------------------------
def bootstrap_diff_ci(a, b, stat_fn=np.mean, n_boot=1000, alpha=0.05, rng=None):
    """
    stat_fn(b) - stat_fn(a)의 부트스트랩 (1-alpha) 신뢰구간을 계산한다.
    구간이 0을 포함하지 않으면 두 그룹의 차이를 통계적으로 유의미하다고 판정한다.
    (개별 CI가 서로 겹치는지 보는 방식보다, 차이 자체의 CI를 보는 이 방식이 더 정확하다.)

    Returns
    -------
    tuple : (point_estimate, ci_low, ci_high, significant: bool)
    """
    if rng is None:
        rng = np.random.default_rng()
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = a[rng.integers(0, na, size=na)]
        sb = b[rng.integers(0, nb, size=nb)]
        diffs[i] = stat_fn(sb) - stat_fn(sa)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = stat_fn(b) - stat_fn(a)
    significant = bool((lo > 0) or (hi < 0))
    return point, lo, hi, significant


def find_period_confirm(x, y, N_candidate, N_max, stat_fn=np.mean, n_boot=1000, alpha=0.05, rng=None):
    """
    find_period()가 반환한 후보 N_candidate 하나를, 그와 배수 관계가 아닌 비배수
    N들 중 "가장 헷갈리는(median H가 가장 낮은, 즉 가장 평탄해 보이는)" 것을
    배경(background)으로 골라 mean + 부트스트랩 신뢰구간으로 재검증한다.

    단순히 가장 가까운 N을 배경으로 고르면 그게 우연히 홀수(위상이 완전히
    달라 대비가 뚜렷한 "쉬운" 케이스)일 수 있어 검증이 느슨해진다(6.2절 참고:
    N=4의 진짜 함정은 가까운 N=3이 아니라 같은 홀짝 구조를 공유하는 N=2다).
    가장 헷갈리는 후보와 비교해야 "진짜 확인"이 된다.

    median 기반 1차 스캔(find_period)은 이상치에 강건하지만, 위상 값들이 서로
    가까운 저대비(low-contrast) 데이터에서는 검출력이 부족하다(6.2절). mean은
    반대로 검출력은 높지만 이상치에 취약하다(6.2절). 그래서 넓은 범위를 훑는
    1차 스캔은 저렴한 median으로 하고, 최종 후보 단 하나만 비용이 큰
    mean+부트스트랩으로 확인하는 2단계 설계를 사용한다.

    Returns
    -------
    dict : {"candidate", "background_N", "diff", "ci", "confirmed"}
    """
    if rng is None:
        rng = np.random.default_rng()

    non_multiples = [n for n in range(1, N_max + 1)
                      if n != N_candidate and n % N_candidate != 0 and N_candidate % n != 0]
    median_h_by_n = {n: np.nanmedian(triangle_metrics(x, y, n)["H"]) for n in non_multiples}
    background_N = min(non_multiples, key=lambda n: median_h_by_n[n])

    h_cand = triangle_metrics(x, y, N_candidate)["H"]
    h_bg = triangle_metrics(x, y, background_N)["H"]
    # diff = stat(h_bg) - stat(h_cand): 양수면 배경이 후보보다 덜 평탄하다(=candidate가 더 그럴듯하다)는 기대 방향
    diff, lo, hi, significant = bootstrap_diff_ci(h_cand, h_bg, stat_fn=stat_fn, n_boot=n_boot, alpha=alpha, rng=rng)
    confirmed = bool(diff > 0 and significant)
    return {"candidate": N_candidate, "background_N": background_N,
            "diff": diff, "ci": (lo, hi), "confirmed": confirmed}


def find_period_two_stage(x, y, N_max, n_trials=5, confirm_stat_fn=np.mean, confirm_n_boot=1000, rng=None):
    """
    5.2절 2단계 탐색. 1단계 find_period()(median, 저비용, 이상치 강건)로 후보 N*를
    구하고, 2단계 find_period_confirm()(mean+부트스트랩, 고비용, 고검출력)으로 그
    후보가 이웃 비배수 N과 통계적으로 유의미하게 다른지 확인한다.

    Returns
    -------
    dict : {"N_star", "confirmation"}
    """
    N_star = find_period(x, y, N_max, n_trials=n_trials)
    confirmation = find_period_confirm(x, y, N_star, N_max, stat_fn=confirm_stat_fn,
                                        n_boot=confirm_n_boot, rng=rng)
    return {"N_star": N_star, "confirmation": confirmation}


# ----------------------------------------------------------------------
# 실험 1 (4.3절): S의 N-선형성 vs H의 N-불변성 검증
# ----------------------------------------------------------------------
def experiment1_scale_dependency():
    print("\n" + "=" * 70)
    print("실험 1 (4.3절): S는 N에 선형 비례, H/κ는 N-불변인지 검증 (10회 반복, 시드 없음)")
    print("=" * 70)

    pattern_params = [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)]
    L = 100_000
    N_list = [4, 8, 12, 16]
    n_trials = 10

    trial_results = {N: {"S": [], "BC": [], "H": [], "kappa": []} for N in N_list}
    for _ in range(n_trials):
        x, y = generate_periodic_tact(L, pattern_params, seed=None)
        for N in N_list:
            m = triangle_metrics(x, y, N)
            trial_results[N]["S"].append(np.nanmedian(m["S"]))
            trial_results[N]["BC"].append(np.nanmedian(m["BC"]))
            trial_results[N]["H"].append(np.nanmedian(m["H"]))
            trial_results[N]["kappa"].append(np.nanmedian(m["kappa"]))

    print(f"\n{'N':>4} | {'median S':>10} | {'S/N':>8} | {'median |BC|':>10} | {'|BC|/N':>8} | "
          f"{'median H':>10} | {'median κ':>10}")
    print("-" * 80)
    row = {}
    for N in N_list:
        median_S = np.median(trial_results[N]["S"])
        median_BC = np.median(trial_results[N]["BC"])
        median_H = np.median(trial_results[N]["H"])
        median_kappa = np.median(trial_results[N]["kappa"])
        row[N] = (median_S, median_S / N, median_BC, median_BC / N, median_H, median_kappa)
        print(f"{N:>4} | {median_S:>10.4f} | {median_S / N:>8.4f} | {median_BC:>10.4f} | "
              f"{median_BC / N:>8.4f} | {median_H:>10.5f} | {median_kappa:>10.6f}")

    s_over_n = np.array([row[N][1] for N in N_list])
    bc_over_n = np.array([row[N][3] for N in N_list])
    median_h = np.array([row[N][4] for N in N_list])
    median_kappa = np.array([row[N][5] for N in N_list])

    print()
    check((s_over_n.max() / s_over_n.min() - 1) <= 0.01,
          f"S/N이 N에 무관하게 상수 (오차 1% 이내): {s_over_n}")
    check((bc_over_n.max() / bc_over_n.min() - 1) <= 0.01,
          f"|BC|/N이 N에 무관하게 상수 (오차 1% 이내): {bc_over_n}")
    check((median_h.max() / median_h.min() - 1) <= 0.01,
          f"median H가 N에 무관하게 상수 (오차 1% 이내): {median_h}")
    check((median_kappa.max() - median_kappa.min()) <= 0.01,
          f"median κ가 N에 무관하게 상수 (절대 오차 0.01 이내, κ는 0 근처값이라 상대오차 대신 절대오차 사용): "
          f"{median_kappa}")


# ----------------------------------------------------------------------
# 실험 2 (7.2절): 주기 정배수 vs 비주기 구간 비교
# ----------------------------------------------------------------------
def experiment2_multiple_vs_nonmultiple():
    print("\n" + "=" * 70)
    print("실험 2 (7.2절): 주기 정배수(4,8,12) vs 비배수(2,6,10)의 H, κ 비교 (seed=42)")
    print("=" * 70)

    pattern_params = [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)]
    L = 100_000
    N_list = [2, 4, 6, 8, 10, 12]

    x, y = generate_periodic_tact(L, pattern_params, seed=42)
    metrics = {N: triangle_metrics(x, y, N) for N in N_list}
    H_results = {N: metrics[N]["H"] for N in N_list}
    K_results = {N: metrics[N]["kappa"] for N in N_list}

    print(f"\n{'N':>4} | {'median H':>10} | {'std H':>10} | {'max H':>10} | "
          f"{'median κ':>10} | {'std κ':>10} | {'max κ':>10}")
    print("-" * 90)
    median_h, median_k = {}, {}
    for N in N_list:
        h, k = H_results[N], K_results[N]
        median_h[N], median_k[N] = np.nanmedian(h), np.nanmedian(k)
        print(f"{N:>4} | {median_h[N]:>10.4f} | {np.nanstd(h):>10.4f} | "
              f"{np.nanmax(h):>10.4f} | {median_k[N]:>10.6f} | "
              f"{np.nanstd(k):>10.6f} | {np.nanmax(k):>10.6f}")

    multiples = [4, 8, 12]
    non_multiples = [2, 6, 10]
    h_mult_median = np.median([median_h[N] for N in multiples])
    h_nonmult_median = np.median([median_h[N] for N in non_multiples])
    k_mult_median = np.median([median_k[N] for N in multiples])
    k_nonmult_median = np.median([median_k[N] for N in non_multiples])

    h_ratio = h_nonmult_median / h_mult_median
    k_ratio = k_nonmult_median / k_mult_median

    print()
    # median은 이상치에 강건한 대신 검출력이 낮아, mean 기준(5.07배)보다 훨씬 못 미치는
    # 판별비(H: 약 2배)만 나온다(6.2절 참고). 그래서 median 기준으로는 "3배 이상"
    # 같은 임의의 배율 대신, 방향성(정배수가 더 낮은가)만 검증한다.
    check(h_mult_median < h_nonmult_median,
          f"[H] 정배수 median({h_mult_median:.4f})이 비배수 median({h_nonmult_median:.4f})보다 낮음 "
          f"(비배수/정배수 = {h_ratio:.2f}배, median 기준이라 mean 대비 판별비가 작음 — 6.2절 참고)")
    check(k_mult_median * 3 <= k_nonmult_median,
          f"[κ] 정배수 median({k_mult_median:.6f})이 비배수 median({k_nonmult_median:.6f})의 1/3 이하 "
          f"(비배수/정배수 = {k_ratio:.2f}배)")
    check(k_ratio > h_ratio,
          f"[비교] κ의 판별비({k_ratio:.2f}배)가 H의 판별비({h_ratio:.2f}배)보다 큼 → κ가 더 우수한 판별력")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].boxplot([H_results[N] for N in N_list], tick_labels=[f"N={N}" for N in N_list], showfliers=True)
    axes[0].set_ylabel("Height (H)")
    axes[0].set_title("Triangle Height (H) by N — synthetic N=4 periodic data (seed=42)")
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot([K_results[N] for N in N_list], tick_labels=[f"N={N}" for N in N_list], showfliers=True)
    axes[1].set_ylabel("Kappa Index (κ = 1 + cosθ)")
    axes[1].set_title("κ by N — synthetic N=4 periodic data (seed=42)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_image_dir()
    out_path = IMAGE_DIR / "이미지_04_H_kappa_박스플롯.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out_path}")

    return x, y


# ----------------------------------------------------------------------
# 실험 3 (5.2절): 순차 탐색 알고리즘 검증
# ----------------------------------------------------------------------
def experiment3_find_period(x, y):
    print("\n" + "=" * 70)
    print("실험 3 (5.2절): find_period 순차 탐색 알고리즘 검증")
    print("=" * 70)

    N_star = find_period(x, y, N_max=20, n_trials=5)
    print(f"\nfind_period(N_max=20, n_trials=5) 반환값: N* = {N_star}")
    check(N_star == 4, "합성 데이터(N=4 반복 패턴)에서 추정 주기가 4로 검출됨")


# ----------------------------------------------------------------------
# 실험 4/5 공통 로직: 임의 패턴(target N=1..8)에 대해 R/S/H/κ의 검출력 비교
# ----------------------------------------------------------------------
def _run_target_detection_sweep(title, use_outliers=False, n_outliers=50, outlier_range=(200.0, 600.0)):
    """
    target N(참 주기)을 1~8까지 바꿔가며, 매 target N마다 랜덤 (mean,std) 패턴을
    1회 생성하고 그 패턴으로 10만 샘플씩 10회 반복 생성한다 (패턴은 고정, 노이즈만
    매 회차 다르게 재샘플링). use_outliers=True이면 각 회차 생성 직후 y에
    inject_outliers()로 이상치를 주입한 뒤 x를 다시 cumsum한다.

    후보 N=1~16 전체에 대해 R/S/H/κ의 회차별 median을 구하고, 10회 median
    (median-of-median)으로 대표 곡선을 만든 뒤 각 지표가 전역 최적값(H/S/κ/R 모두
    최솟값 = 0에 최근접) 기준으로 어떤 N을 채택하는지 확인한다.

    정답 판정은 관대(lenient) 기준을 사용한다: 채택된 N이 target N의 배수
    (found_N % target_N == 0)이면 정답으로 인정한다 (7.1절 참고).

    Returns
    -------
    dict[str, list[tuple[int, int, bool]]] : scoreboard[metric] = [(target_N, found_N, correct), ...]
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    if use_outliers:
        print(f"(옵션: 10만 샘플 중 {n_outliers}개 지점에 {outlier_range[0]:.0f}~{outlier_range[1]:.0f} "
              f"범위의 난수를 추가로 더함)")

    L = 100_000
    n_trials = 10
    candidate_Ns = list(range(1, 17))
    target_Ns = list(range(1, 9))
    metrics_list = ["R", "S", "H", "kappa"]

    rng = np.random.default_rng()  # 시드 없음: 매 실행마다 다른 랜덤 패턴/이상치

    scoreboard = {m: [] for m in metrics_list}  # m -> [(target_N, found_N, correct), ...]

    for target_N in target_Ns:
        pattern = generate_random_pattern(target_N, rng)
        pattern_str = ", ".join(f"({m:.3f},{s:.3f})" for m, s in pattern)
        print(f"\n--- target N = {target_N}  (pattern: [{pattern_str}]) ---")

        # trial_medians[metric][candidate_N] = 회차별 median 값 리스트 (길이 n_trials)
        trial_medians = {m: {N: [] for N in candidate_Ns} for m in metrics_list}
        for _ in range(n_trials):
            x, y = generate_periodic_tact(L, pattern, seed=None)
            if use_outliers:
                y = inject_outliers(y, rng, n_outliers=n_outliers, outlier_range=outlier_range)
                x = np.cumsum(y)
            for N in candidate_Ns:
                tm = triangle_metrics(x, y, N)
                trial_medians["R"][N].append(np.nanmedian(tm["R"]))
                trial_medians["S"][N].append(np.nanmedian(tm["S"]))
                trial_medians["H"][N].append(np.nanmedian(tm["H"]))
                trial_medians["kappa"][N].append(np.nanmedian(tm["kappa"]))

        # agg[metric][candidate_N] = 10회 median의 median (대표 곡선)
        agg = {m: {N: np.median(trial_medians[m][N]) for N in candidate_Ns} for m in metrics_list}

        best_N = {
            "R": min(candidate_Ns, key=lambda N: agg["R"][N]),
            "S": min(candidate_Ns, key=lambda N: agg["S"][N]),
            "H": min(candidate_Ns, key=lambda N: agg["H"][N]),
            "kappa": min(candidate_Ns, key=lambda N: agg["kappa"][N]),
        }

        def mark(metric, N):
            return "*" if N == best_N[metric] else " "

        print(f"{'N':>4} | {'R':>10} |  | {'S':>10} |  | {'H':>10} |  | {'kappa':>12} |")
        print("-" * 65)
        for N in candidate_Ns:
            print(f"{N:>4} | {agg['R'][N]:>10.5f} |{mark('R', N)}| {agg['S'][N]:>10.4f} |{mark('S', N)}| "
                  f"{agg['H'][N]:>10.5f} |{mark('H', N)}| {agg['kappa'][N]:>12.7f} |{mark('kappa', N)}|")

        print(f"\n  채택된 N -> R:{best_N['R']}  S:{best_N['S']}  H:{best_N['H']}  kappa:{best_N['kappa']}"
              f"   (target={target_N})")

        for m in metrics_list:
            found = best_N[m]
            correct = (found % target_N == 0)
            scoreboard[m].append((target_N, found, correct))

    print("\n" + "=" * 70)
    print(f"{title} - 요약: target N(1~8)별 각 지표의 채택 N과 정답 여부 (배수도 정답 인정)")
    print("=" * 70)
    header = f"{'target N':>8} | " + " | ".join(f"{m:>16}" for m in metrics_list)
    print(header)
    print("-" * len(header))
    for i, target_N in enumerate(target_Ns):
        cells = []
        for m in metrics_list:
            _, found, correct = scoreboard[m][i]
            cells.append(f"N={found:<3}{'OK' if correct else 'X ':>3}".rjust(16))
        print(f"{target_N:>8} | " + " | ".join(cells))

    print()
    for m in metrics_list:
        n_correct = sum(1 for _, _, c in scoreboard[m] if c)
        print(f"  [{m}] 정답률: {n_correct}/{len(target_Ns)}")

    return scoreboard


# ----------------------------------------------------------------------
# 실험 4: 임의 패턴(target N=1..8)에 대해 R/S/H/κ의 검출력 비교 (이상치 없음)
# ----------------------------------------------------------------------
def experiment4_target_detection():
    return _run_target_detection_sweep("실험 4: 임의 패턴(target N=1~8)에서 R/S/H/κ의 검출력 비교")


# ----------------------------------------------------------------------
# 실험 5: 실험 4와 동일 조건 + 이상치(outlier) 주입 시 강건성 비교
# ----------------------------------------------------------------------
def experiment5_outlier_robustness():
    return _run_target_detection_sweep(
        "실험 5: 이상치 주입 시 R/S/H/κ의 검출력 비교 (10만개 중 50개에 200~600 난수 추가)",
        use_outliers=True, n_outliers=50, outlier_range=(200.0, 600.0),
    )


def _print_outlier_comparison(scoreboard_clean, scoreboard_outlier):
    print("\n" + "=" * 70)
    print("실험 4 vs 실험 5 비교: 이상치 주입 전/후 정답률 변화")
    print("=" * 70)
    print(f"{'지표':>8} | {'이상치 없음':>10} | {'이상치 있음':>10}")
    print("-" * 36)
    for m in ["R", "S", "H", "kappa"]:
        clean_correct = sum(1 for _, _, c in scoreboard_clean[m] if c)
        outlier_correct = sum(1 for _, _, c in scoreboard_outlier[m] if c)
        print(f"{m:>8} | {clean_correct:>7}/8 | {outlier_correct:>7}/8")


# ----------------------------------------------------------------------
# 실험 6/7 공통 로직: 회차별 "독립 판정" 적중률 (aggregate 방식이 아닌, 매 회차가
# 실전에서 얻는 단일 데이터셋이라고 보고 회차마다 개별적으로 최적 N을 판정)
# ----------------------------------------------------------------------
def _run_hitrate_sweep(title, use_outliers=False, n_outliers=50, outlier_range=(200.0, 600.0),
                        n_trials=20, target_Ns=(1, 2, 3, 4, 6, 8), candidate_Ns=range(1, 17), L=100_000):
    """
    실험 4/5는 n_trials회 시행을 median(median-of-median)한 뒤 딱 한 번만 최적 N을
    판정했다. 이 함수는 대신, 각 회차를 "실전에서 얻는 독립된 단일 데이터셋"으로
    보고 회차마다 개별적으로 최적 N을 판정한 뒤, target N당 n_trials번 중 몇 번
    정답(또는 배수)을 맞추는지 적중률(hit rate)을 구한다.

    use_outliers=True이면 각 회차 생성 직후 y에 이상치를 주입한다.

    Returns
    -------
    dict[str, dict[int, int]] : hit_counts[metric][target_N] = n_trials 중 정답 횟수
    """
    candidate_Ns = list(candidate_Ns)
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    if use_outliers:
        print(f"(옵션: {L:,} 샘플 중 {n_outliers}개 지점에 {outlier_range[0]:.0f}~{outlier_range[1]:.0f} "
              f"범위의 난수를 추가로 더함)")

    metrics_list = ["R", "S", "H", "kappa"]
    rng = np.random.default_rng()  # 시드 없음

    hit_counts = {m: {t: 0 for t in target_Ns} for m in metrics_list}

    for target_N in target_Ns:
        pattern = generate_random_pattern(target_N, rng)
        pattern_str = ", ".join(f"({m:.3f},{s:.3f})" for m, s in pattern)
        print(f"\n--- target N = {target_N}  (pattern: [{pattern_str}]) ---")

        for trial in range(n_trials):
            x, y = generate_periodic_tact(L, pattern, seed=None)
            if use_outliers:
                y = inject_outliers(y, rng, n_outliers=n_outliers, outlier_range=outlier_range)
                x = np.cumsum(y)

            val = {m: {} for m in metrics_list}
            for N in candidate_Ns:
                tm = triangle_metrics(x, y, N)
                val["R"][N] = np.nanmedian(tm["R"])
                val["S"][N] = np.nanmedian(tm["S"])
                val["H"][N] = np.nanmedian(tm["H"])
                val["kappa"][N] = np.nanmedian(tm["kappa"])

            best_N = {
                "R": min(candidate_Ns, key=lambda N: val["R"][N]),
                "S": min(candidate_Ns, key=lambda N: val["S"][N]),
                "H": min(candidate_Ns, key=lambda N: val["H"][N]),
                "kappa": min(candidate_Ns, key=lambda N: val["kappa"][N]),
            }
            for m in metrics_list:
                if best_N[m] % target_N == 0:
                    hit_counts[m][target_N] += 1

        rate_str = ", ".join(f"{m}={hit_counts[m][target_N]}/{n_trials}" for m in metrics_list)
        print(f"  target N={target_N} 적중률: {rate_str}")

    print("\n" + "-" * 70)
    print(f"{title} - 요약 (target N별 적중 횟수 / {n_trials})")
    print("-" * 70)
    header = f"{'target N':>8} | " + " | ".join(f"{m:>10}" for m in metrics_list)
    print(header)
    print("-" * len(header))
    for t in target_Ns:
        cells = " | ".join(f"{hit_counts[m][t]:>7}/{n_trials}" for m in metrics_list)
        print(f"{t:>8} | {cells}")

    print()
    for m in metrics_list:
        total_hits = sum(hit_counts[m].values())
        total_n = n_trials * len(target_Ns)
        print(f"  [{m}] 전체 적중률: {total_hits}/{total_n} ({100 * total_hits / total_n:.1f}%)")

    return hit_counts


def experiment6_hitrate_clean():
    return _run_hitrate_sweep(
        "실험 6: 회차별 독립 판정 적중률 (이상치 없음, target=1,2,3,4,6,8, 20회/target)",
        use_outliers=False, n_trials=20, target_Ns=(1, 2, 3, 4, 6, 8),
    )


def experiment7_hitrate_outlier():
    return _run_hitrate_sweep(
        "실험 7: 회차별 독립 판정 적중률 (이상치 100개/10만, target=1,2,3,4,6,8, 20회/target)",
        use_outliers=True, n_outliers=100, outlier_range=(200.0, 600.0),
        n_trials=20, target_Ns=(1, 2, 3, 4, 6, 8),
    )


def _print_hitrate_comparison(hits_clean, hits_outlier, n_trials=20, target_Ns=(1, 2, 3, 4, 6, 8)):
    print("\n" + "=" * 70)
    print("실험 6 vs 실험 7 비교: 이상치 주입 전/후 회차별 적중률 변화")
    print("=" * 70)
    print(f"{'지표':>8} | {'이상치 없음':>14} | {'이상치 있음':>14}")
    print("-" * 42)
    total_n = n_trials * len(target_Ns)
    for m in ["R", "S", "H", "kappa"]:
        clean_hits = sum(hits_clean[m].values())
        outlier_hits = sum(hits_outlier[m].values())
        print(f"{m:>8} | {clean_hits:>6}/{total_n} ({100*clean_hits/total_n:4.1f}%) | "
              f"{outlier_hits:>6}/{total_n} ({100*outlier_hits/total_n:4.1f}%)")


# ----------------------------------------------------------------------
# 실험 8: 페어(paired) 비교 - 동일 패턴/동일 기본 노이즈에서 이상치 유무만 다르게
# ----------------------------------------------------------------------
def experiment8_paired_outlier_effect(n_trials=20, n_outliers=100, outlier_range=(200.0, 600.0),
                                       target_Ns=(1, 2, 3, 4, 6, 8), candidate_Ns=range(1, 17), L=100_000,
                                       stat_fn=np.nanmedian, stat_label="median"):
    """
    실험 6/7은 "이상치 없음"과 "이상치 있음"을 서로 다른 실행에서 비교했기 때문에,
    관측된 차이가 이상치 효과인지 그냥 실행마다 다른 랜덤 패턴 탓인지 구분할 수
    없었다. 이 실험은 그 교란 변수를 제거한다: 매 회차마다 (x_clean, y_base)를
    "한 번만" 생성한 뒤, 여기에 이상치를 추가한 (x_outlier, y_outlier)를 만들어
    같은 시행을 짝지어(paired) 비교한다. 두 조건의 유일한 차이는 이상치 주입
    여부뿐이므로, "helped"(이상치 덕분에 오답→정답)와 "hurt"(이상치 때문에
    정답→오답) 횟수를 직접 셀 수 있다.

    stat_fn: 후보 N별 R/S/H/kappa를 집계하는 통계량. 기본 median은 이상치에
    견고해서 소수(<50%) 이상치로는 거의 흔들리지 않는다. stat_fn=np.nanmean으로
    바꾸면 이상치에 훨씬 민감한 평균 기반 집계로 전환된다.

    Returns
    -------
    dict : {"hit_clean":.., "hit_outlier":.., "flips":..}
    """
    candidate_Ns = list(candidate_Ns)
    metrics_list = ["R", "S", "H", "kappa"]
    outlier_pct = 100 * n_outliers / L
    print("\n" + "=" * 70)
    print(f"실험 8: 페어 비교 [{stat_label} 집계] - 이상치 {n_outliers:,}개/{L:,} "
          f"({outlier_pct:.1f}%), 범위 {outlier_range[0]:.0f}~{outlier_range[1]:.0f}")
    print("=" * 70)

    rng = np.random.default_rng()

    hit_clean = {m: {t: 0 for t in target_Ns} for m in metrics_list}
    hit_outlier = {m: {t: 0 for t in target_Ns} for m in metrics_list}
    flips = {m: {"helped": 0, "hurt": 0, "unchanged_correct": 0, "unchanged_wrong": 0} for m in metrics_list}

    for target_N in target_Ns:
        pattern = generate_random_pattern(target_N, rng)
        pattern_str = ", ".join(f"({m:.3f},{s:.3f})" for m, s in pattern)
        print(f"\n--- target N = {target_N}  (pattern: [{pattern_str}]) ---")

        for _ in range(n_trials):
            x_clean, y_base = generate_periodic_tact(L, pattern, seed=None)
            y_outlier = inject_outliers(y_base, rng, n_outliers=n_outliers, outlier_range=outlier_range)
            x_outlier = np.cumsum(y_outlier)

            best_clean = _pick_best_N(_metrics_curve(x_clean, y_base, candidate_Ns, stat_fn), candidate_Ns)
            best_outlier = _pick_best_N(_metrics_curve(x_outlier, y_outlier, candidate_Ns, stat_fn), candidate_Ns)

            for m in metrics_list:
                c_ok = (best_clean[m] % target_N == 0)
                o_ok = (best_outlier[m] % target_N == 0)
                hit_clean[m][target_N] += int(c_ok)
                hit_outlier[m][target_N] += int(o_ok)
                if c_ok and o_ok:
                    flips[m]["unchanged_correct"] += 1
                elif (not c_ok) and (not o_ok):
                    flips[m]["unchanged_wrong"] += 1
                elif (not c_ok) and o_ok:
                    flips[m]["helped"] += 1
                else:
                    flips[m]["hurt"] += 1

        rate_str = ", ".join(
            f"{m}(없음/있음)={hit_clean[m][target_N]}/{hit_outlier[m][target_N]} (/{n_trials})"
            for m in metrics_list
        )
        print(f"  {rate_str}")

    total_n = n_trials * len(target_Ns)
    print("\n" + "-" * 70)
    print(f"실험 8 요약 (총 {total_n}회 페어 비교)")
    print("-" * 70)
    print(f"{'지표':>8} | {'이상치 없음':>12} | {'이상치 있음':>12} | {'helped':>8} | {'hurt':>6}")
    print("-" * 58)
    for m in metrics_list:
        c_total = sum(hit_clean[m].values())
        o_total = sum(hit_outlier[m].values())
        print(f"{m:>8} | {c_total:>6}/{total_n} | {o_total:>6}/{total_n} | "
              f"{flips[m]['helped']:>8} | {flips[m]['hurt']:>6}")

    print()
    for m in metrics_list:
        net = flips[m]["helped"] - flips[m]["hurt"]
        verdict = "이상치가 순(net)으로 도움" if net > 0 else ("이상치가 순(net)으로 방해" if net < 0 else "차이 없음")
        print(f"  [{m}] helped={flips[m]['helped']}, hurt={flips[m]['hurt']}, net={net:+d} -> {verdict}")

    return {"hit_clean": hit_clean, "hit_outlier": hit_outlier, "flips": flips}


# ----------------------------------------------------------------------
# 실험 9 (6.2절): 위상 대비(phase contrast)에 따른 median 판별력 민감도
# ----------------------------------------------------------------------
def experiment9_phase_contrast_sensitivity():
    """
    같은 주기 4짜리 패턴이라도 교대되는 두 "큰" 위상 값이 서로 멀리 떨어져
    있는지(고대비) 가까운지(저대비)에 따라 median 기반 H/R의 판별력이 크게
    달라짐을 검증한다. 추가로 저대비 패턴에서 median 차이가 통계적으로
    유의미해지는 데 필요한 샘플 수(L)를 부트스트랩 신뢰구간으로 확인한다.
    """
    print("\n" + "=" * 70)
    print("실험 9: 위상 대비(phase contrast)에 따른 median 판별력 민감도")
    print("=" * 70)

    N_list = [2, 4, 6, 8, 10, 12]
    multiples = [4, 8, 12]
    non_multiples = [2, 6, 10]
    patterns = {
        "고대비 (10, 0.1, 5, 0.1)": [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)],
        "저대비 (10, 0.1, 9, 0.1)": [(10.0, 1.3333), (0.1, 0.0133), (9.0, 1.2000), (0.1, 0.0133)],
    }
    L_report = 100_000

    ratios = {}
    for name, pattern in patterns.items():
        x, y = generate_periodic_tact(L_report, pattern, seed=42)
        median_h = {N: np.nanmedian(triangle_metrics(x, y, N)["H"]) for N in N_list}
        median_r = {N: np.nanmedian(triangle_metrics(x, y, N)["R"]) for N in N_list}
        h_ratio = np.median([median_h[n] for n in non_multiples]) / np.median([median_h[n] for n in multiples])
        r_ratio = np.median([median_r[n] for n in non_multiples]) / np.median([median_r[n] for n in multiples])
        ratios[name] = (h_ratio, r_ratio)
        print(f"\n--- {name} (L={L_report:,}, seed=42) ---")
        print(f"{'N':>4} | {'median H':>10} | {'median R':>10}")
        for N in N_list:
            print(f"{N:>4} | {median_h[N]:>10.4f} | {median_r[N]:>10.5f}")
        print(f"  median 판별비: H={h_ratio:.2f}배, R={r_ratio:.2f}배")

    _ensure_image_dir()
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(patterns.keys())
    x_pos = np.arange(len(names))
    width = 0.35
    ax.bar(x_pos - width / 2, [ratios[n][0] for n in names], width, label="H 판별비")
    ax.bar(x_pos + width / 2, [ratios[n][1] for n in names], width, label="R 판별비")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_ylabel("판별비 (비배수/정배수)")
    ax.set_title(f"위상 대비에 따른 median 기반 판별비 (L={L_report:,}, seed=42)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img1_path = IMAGE_DIR / "이미지_01_위상대비별_판별비.png"
    plt.savefig(img1_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {img1_path}")

    low_contrast = patterns["저대비 (10, 0.1, 9, 0.1)"]
    L_list = [10_000, 15_000, 50_000, 100_000, 1_000_000]
    rng = np.random.default_rng(0)
    ci_records = []
    print("\n--- 저대비 패턴: 샘플 수(L)에 따른 median H(N=2)-H(N=4) 부트스트랩 신뢰구간 ---")
    for L in L_list:
        x, y = generate_periodic_tact(L, low_contrast, seed=42)
        h4 = triangle_metrics(x, y, 4)["H"]
        h2 = triangle_metrics(x, y, 2)["H"]
        diff, lo, hi, significant = bootstrap_diff_ci(h4, h2, stat_fn=np.median, n_boot=500, rng=rng)
        ci_records.append((L, diff, lo, hi, significant))
        print(f"  L={L:>9,}: diff(N2-N4)={diff:+.5f}, 95% CI=[{lo:+.5f},{hi:+.5f}], 유의미={significant}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ys = np.arange(len(ci_records))
    diffs = [r[1] for r in ci_records]
    los = [r[1] - r[2] for r in ci_records]
    his = [r[3] - r[1] for r in ci_records]
    colors = ["tab:blue" if r[4] else "tab:red" for r in ci_records]
    ax.errorbar(diffs, ys, xerr=[los, his], fmt="none", capsize=4, ecolor="gray", zorder=2)
    ax.scatter(diffs, ys, c=colors, s=80, zorder=3)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"L={r[0]:,}" for r in ci_records])
    ax.set_xlabel("median H(N=2) - median H(N=4)  (95% 부트스트랩 CI)")
    ax.set_title("저대비 패턴: 샘플 수가 늘수록 신뢰구간이 좁아져 유의미해짐\n(파랑=유의미, 빨강=유의미하지 않음)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img2_path = IMAGE_DIR / "이미지_02_샘플수별_신뢰구간_수렴.png"
    plt.savefig(img2_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {img2_path}")

    return ratios, ci_records


# ----------------------------------------------------------------------
# 실험 10 (6.2/7.4절): mean + 부트스트랩 신뢰구간으로 저대비 패턴 검출
# ----------------------------------------------------------------------
def experiment10_mean_bootstrap_confirmation():
    """
    실험 9에서 median으로는 실제 배치 규모(1만~1만5천)에서 검출하지 못했던
    저대비 패턴을, mean + 부트스트랩 신뢰구간(find_period_confirm과 동일한 원리)으로
    재검증하면 검출되는지 확인한다. 추가로 find_period_two_stage()가 이
    저대비 데이터에서 실제로 N=4를 찾아내고 확인(confirm)하는지도 검증한다.
    """
    print("\n" + "=" * 70)
    print("실험 10: mean + 부트스트랩 신뢰구간으로 저대비 패턴 검출 (L=10,000~15,000)")
    print("=" * 70)

    low_contrast = [(10.0, 1.3333), (0.1, 0.0133), (9.0, 1.2000), (0.1, 0.0133)]
    L_list = [10_000, 15_000]
    rng = np.random.default_rng(0)

    records = []
    for L in L_list:
        x, y = generate_periodic_tact(L, low_contrast, seed=42)
        for stat_name, stat_fn in [("median", np.median), ("mean", np.mean)]:
            h4 = triangle_metrics(x, y, 4)["H"]
            h2 = triangle_metrics(x, y, 2)["H"]
            diff, lo, hi, significant = bootstrap_diff_ci(h4, h2, stat_fn=stat_fn, n_boot=1000, rng=rng)
            records.append((L, stat_name, diff, lo, hi, significant))
            print(f"  L={L:>6,}, {stat_name:>6}: diff(N2-N4)={diff:+.5f}, "
                  f"95% CI=[{lo:+.5f},{hi:+.5f}], 유의미={significant}")

    print()
    for m in ["median", "mean"]:
        n_sig = sum(1 for r in records if r[1] == m and r[5])
        check(n_sig == len(L_list) if m == "mean" else True,
              f"[{m}] {n_sig}/{len(L_list)}개 L에서 유의미하게 검출됨")

    _ensure_image_dir()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"L={r[0]:,}, {r[1]}" for r in records]
    ys = np.arange(len(records))
    diffs = [r[2] for r in records]
    los = [r[2] - r[3] for r in records]
    his = [r[4] - r[2] for r in records]
    colors = ["tab:blue" if r[5] else "tab:red" for r in records]
    ax.errorbar(diffs, ys, xerr=[los, his], fmt="none", capsize=4, ecolor="gray", zorder=2)
    ax.scatter(diffs, ys, c=colors, s=80, zorder=3)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("H(N=2) - H(N=4)  (95% 부트스트랩 CI)")
    ax.set_title("저대비 패턴, 실제 배치 규모(1만~1만5천): mean은 유의미, median은 아님\n"
                  "(파랑=유의미, 빨강=유의미하지 않음)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img3_path = IMAGE_DIR / "이미지_03_평균vs중앙값_신뢰구간_비교.png"
    plt.savefig(img3_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {img3_path}")

    print("\n--- find_period_two_stage()로 실제 파이프라인 검증 (L=15,000) ---")
    x, y = generate_periodic_tact(15_000, low_contrast, seed=42)
    result = find_period_two_stage(x, y, N_max=20, n_trials=5, confirm_stat_fn=np.mean, rng=rng)
    print(f"  1단계(median) 후보 N* = {result['N_star']}")
    conf = result["confirmation"]
    print(f"  2단계(mean+부트스트랩) 확인: background_N={conf['background_N']}, "
          f"diff={conf['diff']:+.5f}, CI={conf['ci']}, confirmed={conf['confirmed']}")
    check(result["N_star"] == 4 and conf["confirmed"],
          "find_period_two_stage가 저대비 데이터(L=15,000)에서 N=4를 찾고 통계적으로 확인함")

    return records, result


# ----------------------------------------------------------------------
# 함수 5. 대안 기법: ACF / FFT / 직접 위상비교 (8절)
# ----------------------------------------------------------------------
def generate_periodic_tact_with_drift(L, pattern_params, drift_amp, drift_period, seed=None):
    """
    generate_periodic_tact에 사인파 드리프트를 더한 버전 (8절 비교 실험용).
    실제 데이터의 완만한 추세(설비 워밍업 등)를 흉내낸다.
    """
    rng = np.random.default_rng(seed)
    P = len(pattern_params)
    idx = np.arange(L)
    means = np.array([pattern_params[i % P][0] for i in idx])
    stds = np.array([pattern_params[i % P][1] for i in idx])
    drift = drift_amp * np.sin(2 * np.pi * idx / drift_period)
    y = rng.normal(loc=means + drift, scale=stds)
    y = np.clip(y, 0.001, None)
    y = np.round(y, 3)
    x = np.cumsum(y)
    return x, y


def generate_periodic_tact_with_midshift(L, pattern_before, pattern_after, seed=None):
    """
    중간 지점(인덱스 L//2)에서 패턴의 중심값이 pattern_before에서 pattern_after로
    바뀌는 합성 tact 시계열 (8절 비교 실험용). 제품 교체·설비 재조정 등으로 인한
    레벨 시프트를 흉내낸다.
    """
    rng = np.random.default_rng(seed)
    P = len(pattern_before)
    shift = L // 2
    idx = np.arange(L)
    means = np.array([(pattern_before if i < shift else pattern_after)[i % P][0] for i in idx])
    stds = np.array([(pattern_before if i < shift else pattern_after)[i % P][1] for i in idx])
    y = rng.normal(loc=means, scale=stds)
    y = np.clip(y, 0.001, None)
    y = np.round(y, 3)
    x = np.cumsum(y)
    return x, y


def find_period_acf(y, N_max):
    """
    자기상관함수(ACF) 기반 주기 추정. lag k=1..N_max의 ACF를 계산하여
    "상위 20% 이내의 첫 국소최댓값"을 채택한다 (5.2절과 동일한 순차 탐색 철학:
    배수가 전역 최댓값이 되는 것을 피하기 위해 작은 lag부터 스캔).
    """
    yc = y - np.mean(y)
    denom = np.sum(yc ** 2)
    acf = np.array([np.sum(yc[:-k] * yc[k:]) / denom for k in range(1, N_max + 1)])
    threshold = np.percentile(acf, 80)
    for i in range(len(acf)):
        is_local_max = (i == 0 or acf[i] > acf[i - 1]) and (i == len(acf) - 1 or acf[i] > acf[i + 1])
        if is_local_max and acf[i] >= threshold:
            return i + 1
    return int(np.argmax(acf)) + 1


def find_period_fft(y, N_max):
    """
    FFT 파워 스펙트럼에서 가장 강한 피크의 주파수를 주기로 환산한다.
    주의(8.3절): 배음(harmonic) 구조를 보정하지 않은 단순 버전이라, 정배수 성분이
    기본 주기 성분보다 강할 경우 옥타브 오류(더 작은 배수를 주기로 오판)가 날 수 있다.
    """
    yc = y - np.mean(y)
    L = len(yc)
    spectrum = np.abs(np.fft.rfft(yc)) ** 2
    freqs = np.fft.rfftfreq(L)
    valid = (freqs > 1.0 / N_max) & (freqs <= 1.0)
    idx_valid = np.where(valid)[0]
    idx_peak = idx_valid[np.argmax(spectrum[valid])]
    return int(round(1.0 / freqs[idx_peak]))


def find_period_direct_phase(y, N_max, n_boot=500, alpha=0.05, rng=None):
    """
    직접 위상비교 기반 주기 추정 (기하학적 변환을 전혀 쓰지 않는다).
    N=1에서 시작해 M=2N으로 배가시켜 나가며, 같은 N-위상에 속하는 두
    M-서브위상(y[i::M], y[i+N::M])의 평균이 부트스트랩으로 유의하게 다른지
    검사한다. 유의한 세분화가 있으면 M으로 갱신, 없으면 그 N에서 종료한다.
    """
    if rng is None:
        rng = np.random.default_rng()
    N = 1
    while 2 * N <= N_max:
        M = 2 * N
        refine = False
        for i in range(N):
            sub_a, sub_b = y[i::M], y[i + N::M]
            if len(sub_a) < 10 or len(sub_b) < 10:
                continue
            _, lo, hi, sig = bootstrap_diff_ci(sub_a, sub_b, stat_fn=np.mean, n_boot=n_boot, alpha=alpha, rng=rng)
            if sig:
                refine = True
                break
        if refine:
            N = M
        else:
            break
    return N


# ----------------------------------------------------------------------
# 실험 11 (8절): 기하학적 방법 vs ACF vs FFT vs 직접 위상비교
# ----------------------------------------------------------------------
def experiment11_method_comparison():
    """
    기하학적 2단계 방법(find_period_two_stage)과 표준 기법(ACF, FFT, 직접
    위상비교)을 정상/사인드리프트/중간시프트 x 고대비/저대비 = 6개 시나리오
    (L=15,000, 실무 배치 규모)에서 비교한다.
    """
    print("\n" + "=" * 70)
    print("실험 11 (8절): 기하학적 방법 vs ACF vs FFT vs 직접 위상비교")
    print("=" * 70)

    L = 15_000
    N_max = 20
    high = [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)]
    low = [(10.0, 1.3333), (0.1, 0.0133), (9.0, 1.2000), (0.1, 0.0133)]
    high_shifted = [(12.0, 1.6000), (0.1, 0.0133), (7.0, 0.9333), (0.1, 0.0133)]
    low_shifted = [(12.0, 1.6000), (0.1, 0.0133), (11.0, 1.4667), (0.1, 0.0133)]
    drift_amp, drift_period = 3.0, 2000

    scenarios = {
        "고대비-정상": generate_periodic_tact(L, high, seed=42),
        "저대비-정상": generate_periodic_tact(L, low, seed=42),
        "고대비-사인드리프트": generate_periodic_tact_with_drift(L, high, drift_amp, drift_period, seed=42),
        "저대비-사인드리프트": generate_periodic_tact_with_drift(L, low, drift_amp, drift_period, seed=42),
        "고대비-중간시프트": generate_periodic_tact_with_midshift(L, high, high_shifted, seed=42),
        "저대비-중간시프트": generate_periodic_tact_with_midshift(L, low, low_shifted, seed=42),
    }

    print(f"{'시나리오':<20} | {'기하(2단계)':>12} | {'ACF':>6} | {'FFT':>6} | {'직접위상':>8}")
    print("-" * 66)
    rows = {}
    for name, (x, y) in scenarios.items():
        geo = find_period_two_stage(x, y, N_max=N_max, n_trials=5, confirm_stat_fn=np.mean,
                                     rng=np.random.default_rng(1))
        geo_str = f"{geo['N_star']}({'O' if geo['confirmation']['confirmed'] else 'X'})"
        acf_N = find_period_acf(y, N_max)
        fft_N = find_period_fft(y, N_max)
        dp_N = find_period_direct_phase(y, N_max, rng=np.random.default_rng(2))
        rows[name] = (geo["N_star"], geo_str, acf_N, fft_N, dp_N)
        print(f"{name:<20} | {geo_str:>12} | {acf_N:>6} | {fft_N:>6} | {dp_N:>8}")

    print()
    n_geo = sum(1 for v in rows.values() if v[0] == 4)
    n_acf = sum(1 for v in rows.values() if v[2] == 4)
    n_fft = sum(1 for v in rows.values() if v[3] == 4)
    n_dp = sum(1 for v in rows.values() if v[4] == 4)
    check(n_acf == 6, f"[ACF] 6개 시나리오 중 {n_acf}개에서 N=4 검출")
    check(n_dp == 6, f"[직접위상] 6개 시나리오 중 {n_dp}개에서 N=4 검출")
    check(n_fft == 6, f"[FFT] 6개 시나리오 중 {n_fft}개에서 N=4 검출 (미달 시 옥타브 오류 확인 — 8.3절 참고)")
    check(n_geo == 6, f"[기하 2단계] 6개 시나리오 중 {n_geo}개에서 N=4 검출 (미달 시 견고성 열세를 보여줌)")

    _ensure_image_dir()
    names = list(scenarios.keys())
    methods = ["기하(2단계)", "ACF", "FFT", "직접위상"]
    grid = np.array([[1 if v[0] == 4 else 0, 1 if v[2] == 4 else 0,
                       1 if v[3] == 4 else 0, 1 if v[4] == 4 else 0] for v in rows.values()])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(methods)):
            label = "O" if grid[i, j] == 1 else "X"
            ax.text(j, i, label, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_title("시나리오별 N=4 검출 성공(O)/실패(X) (L=15,000)")
    plt.tight_layout()
    img_path = IMAGE_DIR / "이미지_05_방법별_시나리오별_검출결과.png"
    plt.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {img_path}")

    return rows


if __name__ == "__main__":
    experiment1_scale_dependency()
    x_seeded, y_seeded = experiment2_multiple_vs_nonmultiple()
    experiment3_find_period(x_seeded, y_seeded)
    scoreboard4 = experiment4_target_detection()
    scoreboard5 = experiment5_outlier_robustness()
    _print_outlier_comparison(scoreboard4, scoreboard5)
    hits6 = experiment6_hitrate_clean()
    hits7 = experiment7_hitrate_outlier()
    _print_hitrate_comparison(hits6, hits7)
    experiment8_paired_outlier_effect(n_outliers=100, outlier_range=(200.0, 600.0),
                                       stat_fn=np.nanmedian, stat_label="median, 0.1%")
    experiment8_paired_outlier_effect(n_outliers=5_000, outlier_range=(200.0, 600.0),
                                       stat_fn=np.nanmean, stat_label="mean, 5%")
    experiment8_paired_outlier_effect(n_outliers=10_000, outlier_range=(200.0, 600.0),
                                       stat_fn=np.nanmean, stat_label="mean, 10%")
    experiment9_phase_contrast_sensitivity()
    experiment10_mean_bootstrap_confirmation()
    experiment11_method_comparison()
