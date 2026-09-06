"""
산발적 결손(tooth loss) 강건성 실험.

지금까지의 비정상성 테스트(사인 드리프트, 중간 레벨 시프트)는 값이 바뀔 뿐
인덱스 n과 물리적 위상(n mod P)의 대응관계는 유지되는 경우였다. 이 스크립트는
그 대응관계 자체가 깨지는 경우 — 검출 실패로 로그 한 줄이 통째로 누락되는
경우 — 를 다룬다. 오검출(스퓨리어스 삽입)은 실무 확인 결과 이 설비에서
실재하지 않는 실패 모드이므로 다루지 않는다.

결손이 한 번 일어나면 그 지점 이후 전체 구간에서 인덱스-위상 대응이 영구히
한 칸 밀린다. 실제로 측정해보면 이 밀림은 특정 방법에만 국한된 위협이 아니라,
밀림 양이 후보 주기 N의 약수와 우연히 일치하면(shift mod N = 0) 그 N에게는
"안 보이는" 오차가 된다는 확률적 구조에서 비롯된다 — 이는 알고리즘이 아니라
데이터 자체의 성질이므로 모든 방법에 공통으로 작용한다. 그 결과, 산발적
결손은 검출을 참 주기의 작은 약수 쪽으로 체계적으로 편향시키며, 이 편향은
원래 대비(contrast)가 이미 얕은 시나리오에서 특히 파괴적이다.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from find_n import IMAGE_DIR, OUT_DIR, _ensure_image_dir, generate_periodic_tact
from verification_v2 import wilson_ci
from verification_v3 import METHODS, RelPathTee, banner, run_all_methods
from verification_v4 import build_patterns, MICRO

plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def inject_tooth_glitches(y, positions, mode="delete", insert_value=MICRO):
    """
    y의 지정된 위치에 결손(delete) 또는 삽입(insert) 글리치를 낸다.

    delete: 그 위치의 tact 값을 아예 제거한다 (검출 누락 — 로그 한 줄이 안 찍힘).
    insert: 그 위치에 미세 딜레이 값을 하나 끼워 넣는다 (오검출 — 스퓨리어스 트리거).

    두 경우 모두 그 지점 이후 배열의 인덱스가 원래 위상 대비 한 칸씩 밀리므로,
    n mod P에 의존하는 모든 방법에 구조적 영향을 준다.
    """
    y = np.asarray(y, dtype=float).copy()
    if mode == "delete":
        y = np.delete(y, positions)
    else:
        for pos in sorted(positions, reverse=True):
            y = np.insert(y, pos, insert_value)
    x = np.cumsum(y)
    return x, y


def inject_random_tooth_glitches(y, rate, mode="delete", rng=None, insert_value=MICRO):
    """
    y의 각 위치를 독립적으로 확률 rate로 판정해 결손·삽입 글리치를 낸다
    ("중간중간 랜덤하게 군데군데 이가 빠지는" 시나리오).

    앞선 고정 개수(k) 방식은 글리치 개수가 우연히 참 주기 P의 배수와 겹치면
    (예: k=20은 P=4,5의 배수) 위상이 누적으로 "제자리에 돌아오는" 인공적
    효과가 섞여 결과 해석이 어려웠다. 위치마다 독립 확률로 판정하면 실제
    글리치 개수는 이항분포를 따르는 확률변수가 되어 이 문제가 사라진다.

    Returns
    -------
    x, y, n_glitch : 글리치가 적용된 배열과 실제 발생한 글리치 개수
    """
    if rng is None:
        rng = np.random.default_rng()
    y = np.asarray(y, dtype=float)
    n = len(y)
    mask = rng.random(n) < rate
    positions = np.flatnonzero(mask)
    if mode == "delete":
        y2 = np.delete(y, positions)
    else:
        y2 = y.copy()
        for pos in positions[::-1]:  # 뒤에서부터 삽입해야 앞선 인덱스가 밀리지 않는다
            y2 = np.insert(y2, pos, insert_value)
    x2 = np.cumsum(y2)
    return x2, y2, len(positions)


def run_rate_glitch_experiment(patterns, rates, modes, n_reps=20, L=15_000,
                               N_max=20, seed0=6000, n_boot=200):
    """
    8.2.4절의 물리 기반 패턴 전체(정상 조건)에 결손 글리치를 rate 확률로
    군데군데 뿌려 강건성을 측정한다. rate는 모두 5% 미만으로 둔다.
    """
    banner("[실험 I / 신규] 무작위 산발적 결손(tooth loss) (rate < 5%)")
    print("각 tact 위치를 독립적으로 확률 rate로 판정해 결손시킨다(검출 누락).")
    print("모든 rate는 5% 미만이며, 실제 발생 개수는 평균 rate*L 근방의 확률변수다.\n")

    results = {}
    actual_counts = {}
    paired = {m: [] for m in METHODS}   # 방법별 0/1 성공 벡터. 모든 (P,mode,rate,r) 순서가 동일하므로 대응표본이다.
    for P, name, pat in patterns:
        for mode in modes:
            for rate in rates:
                hits = {m: 0 for m in METHODS}
                counts = []
                for r in range(n_reps):
                    seed = seed0 + r
                    x0, y0 = generate_periodic_tact(L, pat, seed=seed)
                    rng = np.random.default_rng(seed + 2_000_000)
                    if rate > 0:
                        x, y, ng = inject_random_tooth_glitches(y0, rate, mode=mode, rng=rng)
                        counts.append(ng)
                    else:
                        x, y = x0, y0
                        counts.append(0)
                    res = run_all_methods(x, y, N_max, seed, n_boot=n_boot)
                    for m in METHODS:
                        ok = int(res[m] == P)
                        hits[m] += ok
                        if rate > 0:   # McNemar 집계는 rate=0(베이스라인)을 제외하고 결손 조건만 본다
                            paired[m].append(ok)
                results[(P, name, mode, rate)] = hits
                actual_counts[(P, name, mode, rate)] = (np.mean(counts), np.min(counts), np.max(counts))
                mc = actual_counts[(P, name, mode, rate)]
                print(f"  [P={P} {name} {mode} rate={rate*100:4.1f}% "
                      f"(실제 평균 {mc[0]:.1f}개, {mc[1]}~{mc[2]})] " +
                      ", ".join(f"{m}={hits[m]}/{n_reps}" for m in METHODS))
    return results, actual_counts, paired


def report_mcnemar_deletion(paired, baseline="기하2단계"):
    """
    산발적 결손(rate>0) 조건에서 baseline과 나머지 방법의 McNemar 대응표본 검정.
    모든 (패턴,rate>0,반복)이 baseline과 정확히 같은 순서로 쌓였으므로 대응표본이다.
    """
    banner(f"[실험 I / McNemar] {baseline} vs 나머지 (결손 rate>0, 대응표본)")
    a = np.array(paired[baseline])
    n = len(a)
    print(f"n={n} (모든 패턴 x rate>0 x 반복 합산)\n")

    others = [m for m in METHODS if m != baseline]
    raw = []
    for m in others:
        b = np.array(paired[m])
        n01 = int(np.sum((a == 1) & (b == 0)))
        n10 = int(np.sum((a == 0) & (b == 1)))
        k = min(n01, n10)
        from math import comb
        tail = sum(comb(n01 + n10, i) for i in range(k + 1)) * (0.5 ** (n01 + n10)) if (n01 + n10) > 0 else 1.0
        p = min(1.0, 2 * tail)
        raw.append((m, n01, n10, p, float(b.mean() - a.mean())))

    ps = [r[3] for r in raw]
    m_ = len(ps)
    order = sorted(range(m_), key=lambda i: ps[i])
    adj = [0.0] * m_
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m_ - rank) * ps[i])
        adj[i] = min(1.0, running)

    print(f"{'비교(baseline vs B)':<18} {'baseline만성공':>12} {'B만성공':>8} {'차이(B-baseline)':>16} {'Holm p':>10} {'판정':>6}")
    print("-" * 76)
    for (m, n01, n10, p, diff), pa in zip(raw, adj):
        sig = "유의" if pa < 0.05 else "n.s."
        print(f"{m:<18} {n01:>12} {n10:>8} {100*diff:>+15.1f}%p {pa:>10.2e} {sig:>6}")


def report_rate_table(results, actual_counts, n_reps):
    banner("[실험 I 요약] 산발적 글리치 비율(rate)별 성공률(%) [Wilson 95% CI]")
    keys = sorted(results.keys())
    header = f"{'시나리오':<30} | " + " | ".join(f"{m:>20}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for k in keys:
        P, name, mode, rate = k
        cells = []
        for m in METHODS:
            s = results[k][m]
            lo, hi = wilson_ci(s, n_reps)
            cells.append(f"{100*s/n_reps:5.1f} [{100*lo:4.1f},{100*hi:5.1f}]".rjust(20))
        label = f"P={P} {name} {mode} r={rate*100:.1f}%"
        print(f"{label:<30} | " + " | ".join(cells))

    print("\n[전체 방법별 총계 — 모든 패턴·모든 rate>0 합산]")
    Ps_total = {m: [0, 0] for m in METHODS}
    for k in keys:
        if k[3] == 0:
            continue
        for m in METHODS:
            Ps_total[m][0] += results[k][m]
            Ps_total[m][1] += n_reps
    for m in METHODS:
        s, n = Ps_total[m]
        lo, hi = wilson_ci(s, n)
        print(f"  {m:<18}: {s:>4}/{n} = {100*s/n:5.1f}% [{100*lo:.1f}, {100*hi:.1f}]")


def plot_rate_curves(results, n_reps, rates, patterns, modes, out_name):
    """modes는 이제 ("delete",) 단일값이므로 패턴을 격자로 배치한다."""
    _ensure_image_dir()
    mode = modes[0]
    ncols = 3
    nrows = -(-len(patterns) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows), squeeze=False)
    rates_pct = [r * 100 for r in rates]
    for idx, (P, name, pat) in enumerate(patterns):
        ax = axes[idx // ncols][idx % ncols]
        for m in METHODS:
            ys = [100 * results[(P, name, mode, r)][m] / n_reps for r in rates]
            ax.plot(rates_pct, ys, marker="o", label=m)
        ax.set_title(f"P={P} {name}", fontsize=9)
        ax.set_xlabel("결손 비율(%)")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
    for idx in range(len(patterns), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    axes[0][0].set_ylabel("성공률(%) [strict]")
    axes[0][0].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"산발적 결손(tooth loss) 비율에 따른 검출 성공률 "
                f"(L=15,000, {n_reps}회 반복, rate<5%)")
    plt.tight_layout()
    out = IMAGE_DIR / out_name
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out}")


def run_glitch_experiment(patterns, glitch_counts, modes, n_reps=20, L=15_000,
                          N_max=20, seed0=5000, n_boot=200):
    """
    [예비 실험 — 설계 결함으로 본 실험(run_rate_glitch_experiment)으로 대체됨]

    고정 개수(glitch_counts)를 위치를 무작위로 뽑아 적용하는 방식이었으나,
    개수가 우연히 참 주기 P의 배수(예: k=20은 P=4,5의 배수)와 겹치면 누적
    위상 이동이 "제자리로 돌아오는" 인공적 효과가 섞여 P=4/6에서 정반대의
    결론(글리치가 많을수록 성공률이 오히려 오르내림)이 나왔다. 원인 진단
    자체는 유효한 발견이었으므로 함수는 남겨 두되, 본 검증에는 쓰지 않는다.
    실제 검증은 위치별 독립 확률(rate)로 판정하는 run_rate_glitch_experiment를 쓴다.
    """
    banner("[예비 실험 H — 참고용] 이(tooth) 결손·삽입에 대한 강건성 (고정 개수)")
    print("가설: 전역 인덱스 스트라이드(직접위상)·전체 상관(ACF)은 결손/삽입 1회로도")
    print("      그 지점 이후 전체가 위상 오정렬되지만, 기하학적 방법은 국소 이웃만")
    print("      쓰므로 글리치 주변 2N개 포인트만 오염된다는 가설을 측정한다.\n")

    results = {}
    for P, name, pat in patterns:
        for mode in modes:
            for k in glitch_counts:
                hits = {m: 0 for m in METHODS}
                for r in range(n_reps):
                    seed = seed0 + r
                    x0, y0 = generate_periodic_tact(L, pat, seed=seed)
                    rng = np.random.default_rng(seed + 1_000_000)
                    if k > 0:
                        lo, hi = L // 10, len(y0) - L // 10
                        positions = rng.integers(lo, hi, size=k)
                        x, y = inject_tooth_glitches(y0, positions, mode=mode)
                    else:
                        x, y = x0, y0
                    res = run_all_methods(x, y, N_max, seed, n_boot=n_boot)
                    for m in METHODS:
                        if res[m] == P:
                            hits[m] += 1
                results[(P, name, mode, k)] = hits
                print(f"  [P={P} {name} {mode} k={k:>2}] " +
                      ", ".join(f"{m}={hits[m]}/{n_reps}" for m in METHODS))
    return results


def report_table(results, n_reps):
    banner("[실험 H 요약] 글리치 개수별 성공률(%) [Wilson 95% CI]")
    keys = sorted(results.keys())
    header = f"{'시나리오':<28} | " + " | ".join(f"{m:>20}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for k in keys:
        P, name, mode, cnt = k
        cells = []
        for m in METHODS:
            s = results[k][m]
            lo, hi = wilson_ci(s, n_reps)
            cells.append(f"{100*s/n_reps:5.1f} [{100*lo:4.1f},{100*hi:5.1f}]".rjust(20))
        print(f"{f'P={P} {name} {mode} k={cnt}':<28} | " + " | ".join(cells))


def plot_curves(results, n_reps, glitch_counts, patterns, modes, out_name):
    _ensure_image_dir()
    fig, axes = plt.subplots(len(patterns), len(modes),
                             figsize=(6.5 * len(modes), 4.2 * len(patterns)),
                             squeeze=False)
    for i, (P, name, pat) in enumerate(patterns):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            for m in METHODS:
                ys = [100 * results[(P, name, mode, k)][m] / n_reps for k in glitch_counts]
                ax.plot(glitch_counts, ys, marker="o", label=m)
            ax.set_title(f"P={P} {name} / {mode}")
            ax.set_xlabel("글리치(결손·삽입) 개수")
            ax.set_ylim(-5, 105)
            ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("성공률(%) [strict]")
    axes[0][-1].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"이(tooth) 결손·삽입 개수에 따른 검출 성공률 (L=15,000, {n_reps}회 반복)")
    plt.tight_layout()
    out = IMAGE_DIR / out_name
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out}")


def diagnose_shift_mechanism(seed=6000, rate=0.005, N_max=8):
    """
    산발적 결손이 왜 특정 방법이 아니라 데이터 구조 자체의 문제인지 진단한다.

    이가 하나 빠지면 그 지점 이후 인덱스-위상 대응이 영구히 밀린다. 이 밀림
    (shift)이 후보 주기 N의 약수와 우연히 일치하면(shift mod N = 0) 그 N에게는
    "안 보이는" 오차가 된다 — 확률은 1/N이므로 작은 N일수록 원천적으로 안전하다.
    저대비 패턴은 원래 N과 그 절반(N/2) 후보의 median H 격차가 이미 얇으므로,
    이 비대칭 하나만으로 즉시 역전된다. 고대비 패턴은 격차가 넓어 더 버틴다.
    """
    from find_n import triangle_metrics
    banner("[진단] 산발적 결손이 median H(N) 순위를 뒤집는 메커니즘")
    print(f"seed={seed}, rate={rate*100:.1f}%, 참 주기 P=4 패턴 두 종류로 비교\n")

    all_pats = build_patterns()
    for label, key in [("저대비 (10, 0.1, 9, 0.1)", (4, "2매세트", "low")),
                       ("고대비 (10, 0.1, 5, 0.1)", (4, "2매세트", "high"))]:
        pat = all_pats[key]
        x0, y0 = generate_periodic_tact(15_000, pat, seed=seed)
        rng = np.random.default_rng(seed + 2_000_000)
        x1, y1, ng = inject_random_tooth_glitches(y0, rate, mode="delete", rng=rng)
        print(f"--- {label}, 결손 {ng}개({100*ng/15000:.2f}%) ---")
        print(f"  {'N':>3} | {'clean median H':>15} | {'glitched median H':>18} | {'변화율':>8}")
        for N in range(2, N_max + 1):
            h0 = np.nanmedian(triangle_metrics(x0, y0, N)["H"])
            h1 = np.nanmedian(triangle_metrics(x1, y1, N)["H"])
            print(f"  {N:>3} | {h0:>15.4f} | {h1:>18.4f} | {h1/h0:>7.2f}x")
        print()


def all_physics_patterns():
    """8.2.4절의 물리 기반 11개 패턴 전체를 (P, 이름, 패턴) 리스트로 변환."""
    all_pats = build_patterns()
    out = []
    for (P, fam, con), pat in sorted(all_pats.items()):
        out.append((P, f"{fam}-{con}", pat))
    return out


def main():
    tee = RelPathTee(OUT_DIR / "verification_tooth_log.txt", OUT_DIR)
    sys.stdout = tee
    try:
        print("=" * 72)
        print("산발적 결손(tooth loss) 강건성 실험 로그")
        print("=" * 72)
        print("실무 확인 결과, 이 설비에서 오검출(스퓨리어스 삽입)은 실재하지 않는다.")
        print("검출 누락(결손)만 현실적인 실패 모드이므로 삽입(insert) 시나리오는 제외한다.\n")

        diagnose_shift_mechanism()

        patterns = all_physics_patterns()
        print(f"대상 패턴 ({len(patterns)}개, 8.2.4절 물리 기반 세트 전체, 정상 조건):")
        for P, name, _ in patterns:
            print(f"  P={P} {name}")

        rates = (0.0, 0.005, 0.01, 0.02, 0.04)   # 0%, 0.5%, 1%, 2%, 4% — 모두 5% 미만
        modes = ("delete",)
        n_reps = 30   # 8.2.2/8.2.4절과 동일한 반복 수로 통일

        results, actual_counts, paired = run_rate_glitch_experiment(patterns, rates, modes, n_reps=n_reps)
        report_rate_table(results, actual_counts, n_reps)
        report_mcnemar_deletion(paired, baseline="기하2단계")
        plot_rate_curves(results, n_reps, rates, patterns, modes,
                         "이미지_10_산발적결손_강건성.png")

        print("\n" + "=" * 72)
        print("검증 로그 종료")
        print("=" * 72)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
    print("완료: verification_tooth_log.txt")


if __name__ == "__main__":
    main()
