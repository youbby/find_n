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
    sin_theta = 2*S/(|AB||AC|) : 완전히 정규화된 대안 지표 (7절 참고)
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
    print("실험 1 (4.2절): S는 N에 선형 비례, H/κ는 N-불변인지 검증 (10회 반복, 시드 없음)")
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
            trial_results[N]["S"].append(np.nanmean(m["S"]))
            trial_results[N]["BC"].append(np.nanmean(m["BC"]))
            trial_results[N]["H"].append(np.nanmean(m["H"]))
            trial_results[N]["kappa"].append(np.nanmean(m["kappa"]))

    print(f"\n{'N':>4} | {'mean S':>10} | {'S/N':>8} | {'mean |BC|':>10} | {'|BC|/N':>8} | "
          f"{'mean H':>10} | {'mean κ':>10}")
    print("-" * 80)
    row = {}
    for N in N_list:
        mean_S = np.mean(trial_results[N]["S"])
        mean_BC = np.mean(trial_results[N]["BC"])
        mean_H = np.mean(trial_results[N]["H"])
        mean_kappa = np.mean(trial_results[N]["kappa"])
        row[N] = (mean_S, mean_S / N, mean_BC, mean_BC / N, mean_H, mean_kappa)
        print(f"{N:>4} | {mean_S:>10.4f} | {mean_S / N:>8.4f} | {mean_BC:>10.4f} | "
              f"{mean_BC / N:>8.4f} | {mean_H:>10.5f} | {mean_kappa:>10.6f}")

    s_over_n = np.array([row[N][1] for N in N_list])
    bc_over_n = np.array([row[N][3] for N in N_list])
    mean_h = np.array([row[N][4] for N in N_list])
    mean_kappa = np.array([row[N][5] for N in N_list])

    print()
    check((s_over_n.max() / s_over_n.min() - 1) <= 0.01,
          f"S/N이 N에 무관하게 상수 (오차 1% 이내): {s_over_n}")
    check((bc_over_n.max() / bc_over_n.min() - 1) <= 0.01,
          f"|BC|/N이 N에 무관하게 상수 (오차 1% 이내): {bc_over_n}")
    check((mean_h.max() / mean_h.min() - 1) <= 0.01,
          f"mean H가 N에 무관하게 상수 (오차 1% 이내): {mean_h}")
    check((mean_kappa.max() - mean_kappa.min()) <= 0.01,
          f"mean κ가 N에 무관하게 상수 (절대 오차 0.01 이내, κ는 0 근처값이라 상대오차 대신 절대오차 사용): "
          f"{mean_kappa}")


# ----------------------------------------------------------------------
# 실험 2 (6.2절): 주기 정배수 vs 비주기 구간 비교
# ----------------------------------------------------------------------
def experiment2_multiple_vs_nonmultiple():
    print("\n" + "=" * 70)
    print("실험 2 (6.2절): 주기 정배수(4,8,12) vs 비배수(2,6,10)의 H, κ 비교 (seed=42)")
    print("=" * 70)

    pattern_params = [(10.0, 1.3333), (0.1, 0.0133), (5.0, 0.6667), (0.1, 0.0133)]
    L = 100_000
    N_list = [2, 4, 6, 8, 10, 12]

    x, y = generate_periodic_tact(L, pattern_params, seed=42)
    metrics = {N: triangle_metrics(x, y, N) for N in N_list}
    H_results = {N: metrics[N]["H"] for N in N_list}
    K_results = {N: metrics[N]["kappa"] for N in N_list}

    print(f"\n{'N':>4} | {'mean H':>10} | {'median H':>10} | {'std H':>10} | {'max H':>10} | "
          f"{'mean κ':>10} | {'median κ':>10} | {'std κ':>10} | {'max κ':>10}")
    print("-" * 110)
    mean_h, mean_k = {}, {}
    for N in N_list:
        h, k = H_results[N], K_results[N]
        mean_h[N], mean_k[N] = np.nanmean(h), np.nanmean(k)
        print(f"{N:>4} | {mean_h[N]:>10.4f} | {np.nanmedian(h):>10.4f} | {np.nanstd(h):>10.4f} | "
              f"{np.nanmax(h):>10.4f} | {mean_k[N]:>10.6f} | {np.nanmedian(k):>10.6f} | "
              f"{np.nanstd(k):>10.6f} | {np.nanmax(k):>10.6f}")

    multiples = [4, 8, 12]
    non_multiples = [2, 6, 10]
    h_mult_mean = np.mean([mean_h[N] for N in multiples])
    h_nonmult_mean = np.mean([mean_h[N] for N in non_multiples])
    k_mult_mean = np.mean([mean_k[N] for N in multiples])
    k_nonmult_mean = np.mean([mean_k[N] for N in non_multiples])

    h_ratio = h_nonmult_mean / h_mult_mean
    k_ratio = k_nonmult_mean / k_mult_mean

    print()
    check(h_mult_mean * 3 <= h_nonmult_mean,
          f"[H] 정배수 mean({h_mult_mean:.4f})이 비배수 mean({h_nonmult_mean:.4f})의 1/3 이하 "
          f"(비배수/정배수 = {h_ratio:.2f}배)")
    check(k_mult_mean * 3 <= k_nonmult_mean,
          f"[κ] 정배수 mean({k_mult_mean:.6f})이 비배수 mean({k_nonmult_mean:.6f})의 1/3 이하 "
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


# ----------------------------------------------------------------------
# 실험 4/5 공통 로직: 임의 패턴(target N=1..8)에 대해 R/S/H/κ의 검출력 비교
# ----------------------------------------------------------------------
def _run_target_detection_sweep(title, use_outliers=False, n_outliers=50, outlier_range=(200.0, 600.0)):
    """
    target N(참 주기)을 1~8까지 바꿔가며, 매 target N마다 랜덤 (mean,std) 패턴을
    1회 생성하고 그 패턴으로 10만 샘플씩 10회 반복 생성한다 (패턴은 고정, 노이즈만
    매 회차 다르게 재샘플링). use_outliers=True이면 각 회차 생성 직후 y에
    inject_outliers()로 이상치를 주입한 뒤 x를 다시 cumsum한다.

    후보 N=1~16 전체에 대해 R/S/H/κ의 회차별 median을 구하고, 10회 평균
    (mean-of-median)으로 대표 곡선을 만든 뒤 각 지표가 전역 최적값(H/S/κ/R 모두
    최솟값 = 0에 최근접) 기준으로 어떤 N을 채택하는지 확인한다.

    정답 판정은 관대(lenient) 기준을 사용한다: 채택된 N이 target N의 배수
    (found_N % target_N == 0)이면 정답으로 인정한다 (6.3절 참고).

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

        # agg[metric][candidate_N] = 10회 median의 평균 (대표 곡선)
        agg = {m: {N: np.mean(trial_medians[m][N]) for N in candidate_Ns} for m in metrics_list}

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
    실험 4/5는 n_trials회 시행을 평균(mean-of-median)한 뒤 딱 한 번만 최적 N을
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
