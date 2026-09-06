"""
period_detection_paper_v4.md를 위한 검증 스크립트 — 설비 물리에 근거한 패턴 재설계.

v3까지의 합성 패턴은 "큰 값/작은 값 교대"라는 형태만 보고 임의로 주기를 붙였다.
실제 설비 구조를 반영하면 주기의 홀짝이 곧 설비 유형을 뜻한다.

  2매 세트형 (Family A) : 짝수 주기. 2매를 동시 생산하므로 두 번째 매는 미세
                      딜레이(0.1)로 찍힌다. tact = [b1, 0.1, b2, 0.1, ...], P = 2m
  순차 stage형 (Family B) : 홀수 주기. m개 stage가 1매씩 순차 적재하며, 한 주기에
                      한 번 긴 간격이 끼고 나머지는 균일하다.
                      tact = [big, small, small, ...] (small이 m-1회), P = m
                      예) P=3 → 10, 5, 5   /   P=5 → 10, 5, 5, 5, 5

따라서 짝수 주기는 2매 세트형, 홀수 주기는 순차 stage형에서만 나온다. v3의 P=5
패턴 [10, 0.1, 9, 0.1, 9.5]는 홀수 주기에 0.1 갭이 섞여 있어 어느 구조에도
대응하지 않는다(순환 시 9.5 다음 10이 연속). 본 스크립트는 이를 교체한다.

추가 검증
  - 설비 유형별로 방법 순위가 갈리는지 (v3 8.2.3의 등차 세트가 곧 Family B였음)
  - 오경보/미탐 분석: N=2(정상) ↔ N=4(카세트 교체 지연) 혼동은 이 응용에서
    각각 오경보와 미탐을 뜻하므로, 그 방향을 나눠 집계한다.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from find_n import (IMAGE_DIR, OUT_DIR, _ensure_image_dir, generate_periodic_tact,
                    generate_periodic_tact_with_drift, generate_periodic_tact_with_midshift)
from verification_v2 import STD_RATIO, wilson_ci
from verification_v3 import (METHODS, RelPathTee, banner, holm,
                             mcnemar_exact, paired_bootstrap_diff, run_all_methods)

plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SMALL = (0.1, 0.0133)          # 2매 동시 생산 시의 미세 딜레이
CONDITIONS = ("정상", "사인드리프트", "중간시프트")


# ----------------------------------------------------------------------
# 물리 기반 패턴 생성
# ----------------------------------------------------------------------
MICRO = 0.1        # 2매 동시 생산 시의 미세 딜레이. 이 값만 '동시 생산' 표식으로 쓴다.


def _ph(v):
    return (round(float(v), 4), round(float(v) * STD_RATIO, 4))


def family_A(gaps_after_each_pair):
    """
    2매 세트형(짝수 주기): 세트마다 [세트 간격, 그 세트 안 2매의 미세 딜레이].
    P = 2 * 세트 수.
      family_A([10])            → 10, 0.1                     (1 stage x 2매)
      family_A([10, 5])         → 10, 0.1, 5, 0.1             (2 stage x 2매)
      family_A([10, 0.1, ...])  → 카세트 교체 지연은 두 번째 값으로 표현
    """
    pat = []
    for v in gaps_after_each_pair:
        pat.append(_ph(v))
        pat.append(_ph(MICRO))
    return pat


def family_A_cassette(stage_gap, change_gap):
    """
    카세트 교체 지연형(P=4): 1 stage x 2매를 두 번 하고 교체 시간이 끼어든다.
      정상(교체가 미리 준비됨) → 10, 0.1                (P=2)
      지연 발생               → 10, 0.1, 10, change_gap (P=4)
    change_gap이 미세 딜레이 0.1에 가까울수록 P=2와 구분하기 어렵다(조기경보 구간).
    """
    return [_ph(stage_gap), _ph(MICRO), _ph(stage_gap), _ph(change_gap)]


def family_B(n_stage, big, small):
    """
    순차 stage형(홀수 주기): 한 주기에 긴 간격 1회 + 균일한 간격 (n_stage-1)회.
      family_B(3, 10, 5) → 10, 5, 5
      family_B(5, 10, 5) → 10, 5, 5, 5, 5
    """
    return [_ph(big)] + [_ph(small)] * (n_stage - 1)


def build_patterns():
    """(P, 유형, 대비) -> pattern"""
    pats = {}
    # 짝수 주기 = 2매 세트형
    pats[(2, "2매세트", "단일")] = family_A([10.0])
    pats[(4, "2매세트", "high")] = family_A([10.0, 5.0])
    pats[(4, "2매세트", "low")] = family_A([10.0, 9.0])
    pats[(4, "카세트지연", "high")] = family_A_cassette(10.0, 3.0)   # 지연이 뚜렷
    pats[(4, "카세트지연", "low")] = family_A_cassette(10.0, 0.6)    # 지연 시작(조기경보)
    pats[(6, "2매세트", "high")] = family_A([10.0, 5.0, 7.0])
    pats[(6, "2매세트", "low")] = family_A([10.0, 9.0, 9.5])
    # 홀수 주기 = 순차 stage형
    pats[(3, "순차stage", "high")] = family_B(3, 10.0, 5.0)
    pats[(3, "순차stage", "low")] = family_B(3, 10.0, 9.0)
    pats[(5, "순차stage", "high")] = family_B(5, 10.0, 5.0)
    pats[(5, "순차stage", "low")] = family_B(5, 10.0, 9.0)
    return pats


def contrast_C(pattern):
    """
    위상 대비 지표 C = (서로 다른 위상 값들 사이의 최소 간격) / mean(모든 위상).

    v3은 미세 딜레이(0.1)를 제외하고 계산했으나, 카세트 교체 지연형에서는
    "교체시간이 미세 딜레이에 얼마나 가까운가"가 곧 난이도이므로 제외하면 안 된다.
    v4는 모든 위상 값을 포함하되 중복 값은 하나로 본다(균일 구간은 대비를 만들지 않음).
    """
    mus = [m for m, _ in pattern]
    uniq = sorted(set(round(v, 6) for v in mus))
    if len(uniq) < 2:
        return float("nan")
    gap = min(b - a for a, b in zip(uniq, uniq[1:]))
    return gap / (sum(mus) / len(mus))


def shift_stage_values(pattern, delta=2.0):
    """중간 레벨 시프트: 미세 딜레이(0.1)를 제외한 모든 간격에 +delta."""
    out = []
    for m, s in pattern:
        if abs(m - MICRO) > 1e-9:
            out.append(_ph(m + delta))
        else:
            out.append((m, s))
    return out


def make_dataset(pat, cond, L, seed, drift_amp=3.0, drift_period=2000):
    if cond == "정상":
        return generate_periodic_tact(L, pat, seed=seed)
    if cond == "사인드리프트":
        return generate_periodic_tact_with_drift(L, pat, drift_amp, drift_period, seed=seed)
    return generate_periodic_tact_with_midshift(L, pat, shift_stage_values(pat), seed=seed)


# ----------------------------------------------------------------------
def exp1_pattern_table(pats):
    banner("[실험 1 / v4] 설비 물리에 근거한 패턴 정의")
    print("2매 세트형(짝수 P): 2매 동시 생산 → 두 번째 매가 미세 딜레이 0.1로 찍힘")
    print("                    tact = [b1, 0.1, b2, 0.1, ...],  P = 2 x 세트 수")
    print("카세트 지연형(P=4): 정상은 10, 0.1 (P=2). 교체가 지연되면 10, 0.1, 10, X (P=4)")
    print("순차 stage형(홀수 P): 긴 간격 1회 + 균일 간격 (m-1)회")
    print("                    tact = [10, 5, 5] (P=3),  [10, 5, 5, 5, 5] (P=5)")
    print("→ 짝수 주기는 2매 세트 구조에서, 홀수 주기는 순차 stage 구조에서 발생한다.\n")
    print(f"{'P':>3} | {'유형':<10} | {'대비':<6} | {'C':>7} | 패턴")
    print("-" * 96)
    for (P, fam, con), pat in sorted(pats.items()):
        vals = [m for m, _ in pat]
        print(f"{P:>3} | {fam:<10} | {con:<6} | {contrast_C(pat):>7.3f} | "
              f"{[round(v, 2) for v in vals]}")
    print("\n참고: v3의 P=5 패턴 [10, 0.1, 9, 0.1, 9.5]는 홀수 주기에 0.1 갭이 섞여")
    print("      순환 시 9.5 다음에 10이 연속으로 오므로 2매 동시 구조가 깨진다.")
    print("      어느 설비 유형에도 대응하지 않으므로 순차 stage형 P=5로 교체하였다.")


def exp2_success(pats, n_reps=30, L=15_000, N_max=20, seed0=2000, n_boot=200):
    banner(f"[실험 2 / v4] 물리 기반 {len(pats)}개 패턴 x {len(CONDITIONS)}개 조건 x {n_reps}회")
    print(f"L={L:,}, N_max={N_max}, 시드 {seed0}..{seed0+n_reps-1} (방법 간 대응표본)\n")
    returned = {}
    for (P, fam, con), pat in sorted(pats.items()):
        for cond in CONDITIONS:
            got = {m: [] for m in METHODS}
            for r in range(n_reps):
                seed = seed0 + r
                x, y = make_dataset(pat, cond, L, seed)
                res = run_all_methods(x, y, N_max, seed, n_boot=n_boot)
                for m in METHODS:
                    got[m].append(res[m])
            returned[(P, fam, con, cond)] = got
            hits = {m: sum(1 for v in got[m] if v == P) for m in METHODS}
            print(f"  [P={P} {fam} {con} {cond}] " +
                  ", ".join(f"{m}={hits[m]}/{n_reps}" for m in METHODS))
    return returned


def _rate(returned, keys, m, mode="strict"):
    ok = tot = 0
    for k in keys:
        P = k[0]
        for v in returned[k][m]:
            tot += 1
            ok += 1 if (v == P if mode == "strict" else v % P == 0) else 0
    return ok, tot


def exp3_by_family(returned, n_reps):
    banner("[실험 3 / v4] 설비 유형별 방법 성능")
    for mode in ("strict", "lenient"):
        print(f"\n--- {mode} 기준 ---")
        print(f"{'방법':<18} | {'짝수P(2매세트)':>18} | {'홀수P(순차stage)':>18} | {'전체':>18}")
        print("-" * 82)
        for m in METHODS:
            row = []
            for sel in (lambda k: k[1] in ("2매세트", "카세트지연"), lambda k: k[1] == "순차stage", lambda k: True):
                keys = [k for k in returned if sel(k)]
                ok, tot = _rate(returned, keys, m, mode)
                lo, hi = wilson_ci(ok, tot)
                row.append(f"{100*ok/tot:5.1f}% [{100*lo:4.1f},{100*hi:5.1f}]".rjust(18))
            print(f"{m:<18} | " + " | ".join(row))

    print("\n[주기별 성공률(strict, %)]")
    Ps = sorted({k[0] for k in returned})
    print(f"{'방법':<18} | " + " | ".join(f"P={P:<5}".rjust(8) for P in Ps))
    print("-" * (20 + 11 * len(Ps)))
    for m in METHODS:
        cells = []
        for P in Ps:
            keys = [k for k in returned if k[0] == P]
            ok, tot = _rate(returned, keys, m)
            cells.append(f"{100*ok/tot:6.1f}%".rjust(8))
        print(f"{m:<18} | " + " | ".join(cells))


def exp4_mcnemar(returned, tag=""):
    banner(f"[실험 4 / v4] 방법 간 McNemar + Holm{tag}")
    keys = sorted(returned.keys())
    sc = {}
    for m in METHODS:
        vals = []
        for k in keys:
            P = k[0]
            vals += [1 if v == P else 0 for v in returned[k][m]]
        sc[m] = np.array(vals)
    n = len(sc[METHODS[0]])
    print(f"동일한 {n}개 데이터셋 기반 대응표본, strict 기준\n")

    pairs, raw = [], []
    for i in range(len(METHODS)):
        for j in range(i + 1, len(METHODS)):
            A, B = METHODS[i], METHODS[j]
            n01, n10, p = mcnemar_exact(sc[A], sc[B])
            pairs.append((A, B, n01, n10, p))
            raw.append(p)
    adj = holm(raw)
    rng = np.random.default_rng(777)
    print(f"{'방법 A':<18} {'방법 B':<18} {'차이(B-A)':>10} {'95% CI':>22} {'Holm p':>10} {'유의':>5}")
    print("-" * 92)
    for (A, B, n01, n10, p), pa in zip(pairs, adj):
        d, lo, hi = paired_bootstrap_diff(sc[A], sc[B], rng=rng)
        print(f"{A:<18} {B:<18} {100*d:>+9.1f}%p [{100*lo:>+6.1f},{100*hi:>+6.1f}]%p "
              f"{pa:>10.2e} {'예' if pa < 0.05 else '아니오':>5}")


def exp5_alarm_analysis(returned, n_reps):
    """
    이 응용에서 N은 설비 진단 신호다(정상 P=2 → 교체 지연 시 P=4).
    따라서 2↔4 혼동은 방향에 따라 뜻이 다르다.
      참 P=2 인데 4 이상 반환  → 오경보(정상 설비를 이상으로 신고)
      참 P=4 인데 2 반환       → 미탐(설비 이상을 놓침)
    """
    banner("[실험 5 / v4] 오경보 / 미탐 분석 (N=2 정상 ↔ N=4 교체지연)")
    k2 = [k for k in returned if k[0] == 2]
    k4 = [k for k in returned if k[0] == 4]

    print("참 P=2(정상 설비)에서 반환한 N의 분포:")
    for m in METHODS:
        c = Counter(v for k in k2 for v in returned[k][m])
        n = sum(c.values())
        false_alarm = sum(v for kk, v in c.items() if kk != 2 and kk % 2 == 0)
        print(f"  {m:<18}: {dict(sorted(c.items()))}"
              f"   → 오경보(짝수배수 반환) {false_alarm}/{n} = {100*false_alarm/n:.1f}%")

    print("\n참 P=4(카세트 교체 지연)에서 반환한 N의 분포:")
    for m in METHODS:
        c = Counter(v for k in k4 for v in returned[k][m])
        n = sum(c.values())
        missed = c.get(2, 0) + c.get(1, 0)
        print(f"  {m:<18}: {dict(sorted(c.items()))}"
              f"   → 미탐(1 또는 2 반환) {missed}/{n} = {100*missed/n:.1f}%")


def exp6_plot(returned, n_reps):
    _ensure_image_dir()
    keys = sorted(returned.keys())
    labels = [f"P={k[0]} {k[1]} {k[2]} {k[3]}" for k in keys]
    grid = np.array([[sum(1 for v in returned[k][m] if v == k[0]) / n_reps for m in METHODS]
                     for k in keys])
    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(keys) + 2)))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(METHODS)):
            ax.text(j, i, f"{100*grid[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="성공률(strict)")
    ax.set_title(f"설비 물리 기반 {len(keys)}개 시나리오 x {n_reps}회: 검출 성공률(%)")
    plt.tight_layout()
    out = IMAGE_DIR / "이미지_09_설비유형별_성공률.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out}")


# ======================================================================
# 3차 리뷰 대응: v3의 두 세트를 재실행하고 대조한다
# ======================================================================
def v3_original_patterns():
    """v3 8.2.2절의 18개 시나리오 패턴 (큰 값/작은 값 교대형)."""
    from verification_v2 import make_pattern
    return {
        (4, "원래", "high"): make_pattern([10.0, 5.0], 4),
        (4, "원래", "low"): make_pattern([10.0, 9.0], 4),
        (6, "원래", "high"): make_pattern([10.0, 5.0, 7.0], 6),
        (6, "원래", "low"): make_pattern([10.0, 9.0, 9.5], 6),
        (5, "원래", "high"): make_pattern([10.0, 5.0, 7.0], 5),
        (5, "원래", "low"): make_pattern([10.0, 9.0, 9.5], 5),
    }


def v3_control_patterns():
    """v3 8.2.3절의 대비 통제 패턴 (등차 배치, mean 6.0 / 간격 1.2)."""
    from verification_v3 import matched_pattern
    return {(P, "통제", "matched"): matched_pattern(P) for P in (4, 6, 5)}


def run_set(pats, label, n_reps=30, L=15_000, N_max=20, seed0=2000, n_boot=200):
    banner(f"[재실행] {label}: {len(pats)}개 패턴 x {len(CONDITIONS)}조건 x {n_reps}회")
    returned = {}
    for (P, fam, con), pat in sorted(pats.items()):
        for cond in CONDITIONS:
            got = {m: [] for m in METHODS}
            for r in range(n_reps):
                seed = seed0 + r
                x, y = make_dataset(pat, cond, L, seed)
                res = run_all_methods(x, y, N_max, seed, n_boot=n_boot)
                for m in METHODS:
                    got[m].append(res[m])
            returned[(P, fam, con, cond)] = got
            hits = {m: sum(1 for v in got[m] if v == P) for m in METHODS}
            print(f"  [P={P} {con} {cond}] " +
                  ", ".join(f"{m}={hits[m]}/{n_reps}" for m in METHODS))
    return returned


def report_full(returned, n_reps, label):
    """M9: 원래 세트와 동일한 수준으로 보고한다 (시나리오별 / strict·lenient / 반환분포)."""
    banner(f"[{label}] 시나리오별 성공률(%) [Wilson 95% CI], strict, n={n_reps}")
    print("주의: 아래 구간은 시나리오별 marginal 요약이며, 열 간 겹침으로 우열을 판정하지 말 것(C3).")
    header = f"{'시나리오':<26} | " + " | ".join(f"{m:>20}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for k in sorted(returned):
        P = k[0]
        cells = []
        for m in METHODS:
            s = sum(1 for v in returned[k][m] if v == P)
            lo, hi = wilson_ci(s, n_reps)
            cells.append(f"{100*s/n_reps:5.1f} [{100*lo:4.1f},{100*hi:5.1f}]".rjust(20))
        print(f"{f'P={k[0]} {k[2]} {k[3]}':<26} | " + " | ".join(cells))

    print(f"\n[{label}] strict / lenient 총계 및 주기별 소계")
    for mode in ("strict", "lenient"):
        print(f"\n--- {mode} ---")
        allk = list(returned.keys())
        for m in METHODS:
            ok, tot = _rate(returned, allk, m, mode)
            lo, hi = wilson_ci(ok, tot)
            per = []
            for P in sorted({k[0] for k in returned}):
                o2, t2 = _rate(returned, [k for k in allk if k[0] == P], m, mode)
                per.append(f"P={P}:{100*o2/t2:5.1f}%")
            print(f"  {m:<18}: {ok:>4}/{tot} = {100*ok/tot:5.1f}% "
                  f"[{100*lo:4.1f},{100*hi:5.1f}]   " + "  ".join(per))

    print(f"\n[{label}] 각 방법이 반환한 N의 분포 (주기별, 전 조건 합산)")
    for P in sorted({k[0] for k in returned}):
        print(f"  --- P={P} ---")
        for m in METHODS:
            c = Counter(v for k in returned if k[0] == P for v in returned[k][m])
            print(f"    {m:<18}: {dict(sorted(c.items()))}")


def two_set_mcnemar(orig, ctrl):
    """C5: 원래 세트와 통제 세트의 McNemar 결과를 나란히 놓고 부호 반전을 찾는다."""
    banner("[C5] 원래 세트 vs 통제 세트: 방법 쌍별 McNemar 대조 (strict)")

    def scores(returned):
        out = {}
        for m in METHODS:
            vals = []
            for k in sorted(returned):
                P = k[0]
                vals += [1 if v == P else 0 for v in returned[k][m]]
            out[m] = np.array(vals)
        return out

    so, sc = scores(orig), scores(ctrl)
    pairs = [("기하2단계", b) for b in METHODS if b != "기하2단계"]

    # 각 세트 내부에서 15쌍 전체를 검정한 뒤 Holm 보정 → 해당 쌍의 보정 p를 인용
    def holm_map(s):
        ps, keys = [], []
        for i in range(len(METHODS)):
            for j in range(i + 1, len(METHODS)):
                A, B = METHODS[i], METHODS[j]
                ps.append(mcnemar_exact(s[A], s[B])[2])
                keys.append((A, B))
        adj = holm(ps)
        return {k: (p, a) for k, p, a in zip(keys, ps, adj)}

    ho, hc = holm_map(so), holm_map(sc)
    rng = np.random.default_rng(4242)
    print(f"{'비교쌍 (기하2단계 vs B)':<24} | {'원래 세트':>26} | {'통제 세트':>26} | 부호")
    print("-" * 100)
    flips = []
    for _, B in pairs:
        key = ("기하2단계", B) if ("기하2단계", B) in ho else (B, "기하2단계")
        sign_fix = 1 if key[0] == "기하2단계" else -1
        do, _, _ = paired_bootstrap_diff(so["기하2단계"], so[B], rng=rng)
        dc, _, _ = paired_bootstrap_diff(sc["기하2단계"], sc[B], rng=rng)
        po, pc = ho[key][1], hc[key][1]
        so_s = "유의" if po < 0.05 else "n.s."
        sc_s = "유의" if pc < 0.05 else "n.s."
        flip = "**반전**" if (do > 0) != (dc > 0) else ""
        if flip:
            flips.append((B, do, dc, po, pc))
        print(f"{B:<24} | {100*do:>+7.1f}%p Holm p={po:8.2e} {so_s:<4} | "
              f"{100*dc:>+7.1f}%p Holm p={pc:8.2e} {sc_s:<4} | {flip}")
    print("\n(차이는 B − 기하2단계. 양수면 B가 더 낫고, 음수면 기하2단계가 더 낫다.)")
    if flips:
        print("\n부호가 뒤집힌 쌍:")
        for B, do, dc, po, pc in flips:
            print(f"  기하2단계 vs {B}: 원래 {100*do:+.1f}%p(p={po:.2e}) → "
                  f"통제 {100*dc:+.1f}%p(p={pc:.2e})")
    else:
        print("\n부호가 뒤집힌 쌍: 없음")
    return flips


def diagnose_geo_p6_control(n_reps=30, L=15_000, N_max=20, seed0=2000, n_boot=200):
    """M11: 통제 세트 P=6에서 기하2단계가 0%인 원인을 1단계/2단계로 분해한다."""
    from verification_v3 import matched_pattern
    from find_n import find_period, find_period_confirm, triangle_metrics
    banner("[M11] 통제 세트 P=6에서 기하2단계 0.0%의 원인 분해")
    pat = matched_pattern(6)
    print(f"패턴: {[m for m, _ in pat]}  (C={contrast_C(pat):.3f})\n")

    stage1 = Counter()
    confirm_ok = Counter()
    for cond in CONDITIONS:
        for r in range(n_reps):
            seed = seed0 + r
            x, y = make_dataset(pat, cond, L, seed)
            N1 = find_period(x, y, N_max, n_trials=5)
            conf = find_period_confirm(x, y, N1, N_max, stat_fn=np.mean,
                                       n_boot=n_boot, rng=np.random.default_rng(seed))
            stage1[N1] += 1
            confirm_ok[(N1, bool(conf["confirmed"]))] += 1
    print(f"1단계(median H 스캔)가 고른 N*의 분포: {dict(sorted(stage1.items()))}")
    print(f"2단계 confirm 통과 여부: "
          f"{ {f'N*={k[0]},confirmed={k[1]}': v for k, v in sorted(confirm_ok.items())} }")

    x, y = generate_periodic_tact(L, pat, seed=seed0)
    print(f"\nN별 median H (seed={seed0}, 1단계가 보는 곡선):")
    for N in range(1, N_max + 1):
        h = np.nanmedian(triangle_metrics(x, y, N)["H"])
        mark = "  <- 참 주기" if N == 6 else ("  <- 1단계 선택" if N == stage1.most_common(1)[0][0] else "")
        print(f"  N={N:>2}: {h:.5f}{mark}")


def main():
    tee = RelPathTee(OUT_DIR / "verification_v4_log.txt", OUT_DIR)
    sys.stdout = tee
    try:
        print("=" * 72)
        print("period_detection_paper_v4.md 근거 검증 로그")
        print("  (1) 3차 리뷰 대응: v3 두 세트 재실행·대조 (C5/C6/M9/M11)")
        print("  (2) 설비 물리에 근거한 패턴 재설계")
        print("=" * 72)

        orig = run_set(v3_original_patterns(), "v3 원래 세트 (교대형 18시나리오)")
        report_full(orig, 30, "원래 세트")
        ctrl = run_set(v3_control_patterns(), "v3 통제 세트 (등차 배치 9시나리오)")
        report_full(ctrl, 30, "통제 세트")
        two_set_mcnemar(orig, ctrl)
        diagnose_geo_p6_control()

        pats = build_patterns()
        exp1_pattern_table(pats)
        returned = exp2_success(pats)
        exp3_by_family(returned, 30)
        exp4_mcnemar(returned)
        exp5_alarm_analysis(returned, 30)
        exp6_plot(returned, 30)
        print("\n" + "=" * 72)
        print("검증 로그 종료")
        print("=" * 72)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
    print("완료: verification_v4_log.txt")


if __name__ == "__main__":
    main()
