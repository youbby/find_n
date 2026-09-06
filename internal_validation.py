"""
논문 7.3.1절 "사내 검증 절차"의 실행 스크립트.

사내망에서 실측 생산 로그에 대해 논문과 동일한 검증을 수행하고, 7.3절에
그대로 붙여넣을 수 있는 집계표/그림/판정 결과를 생성한다.

핵심 원칙
---------
1. 원자료(개별 tact 값, 생산 시각)는 어떤 출력물에도 기록하지 않는다.
   출력은 전부 집계 통계(count/mean/median/std, 분위수, 판정 N)뿐이다.
2. find_n.py / verification_v2.py의 지표·탐색 함수는 수정하지 않고 그대로 쓴다.
   합성 데이터 생성 함수 대신 실측 (x, y)를 넣는 것이 유일한 차이다.
3. 데이터 위생 상태(중복·역행·비양수 tact·결측)를 먼저 보고하고, 비생산
   구간 제거 여부에 따라 결과가 어떻게 달라지는지 둘 다 산출한다.

사용법
------
    python internal_validation.py --input production_log.csv
    python internal_validation.py --input log.xlsx --sheet Sheet1 --time-col 완료시각
    python internal_validation.py --input log.csv --max-tact 300   # 300초 초과 구간을 비생산으로 간주

산출물 (기본 out_internal/)
    internal_validation_log.txt   : 콘솔 출력 전문 (7.3절 근거로 인용)
    H_by_N_boxplot.png            : N별 H 분포 Box plot
    section_7_3_draft.md          : 7.3절에 붙여넣을 초안 (표 채워진 상태)

반출 전 확인
    출력 3종에 원시 tact/시각이 포함되지 않는 것은 코드로 보장되지만,
    집계값 자체(평균 tact 등)가 사내 기준상 반출 가능한지는 사람이 판단해야 한다.
    스크립트 말미의 "반출 점검" 절을 참고할 것.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_n import (triangle_metrics, find_period_two_stage,
                    find_period_acf, find_period_fft, find_period_direct_phase)
from verification_v2 import find_period_acf_global, find_period_direct_phase_general, acf_curve

def _configure_korean_font():
    """사내 PC의 OS가 무엇이든 설치된 한글 폰트만 골라 쓴다(없으면 경고 없이 기본값)."""
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    preferred = [n for n in ("Malgun Gothic", "AppleGothic", "NanumGothic",
                             "Noto Sans CJK KR", "Gulim", "Batang") if n in installed]
    plt.rcParams["font.family"] = preferred + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return preferred


_KOREAN_FONTS = _configure_korean_font()

# 논문 8.2절과 동일한 조건
DEFAULT_N_MAX = 20
RECOMMENDED_L = 10_000     # 8.2절 L=15,000 기준. 이보다 훨씬 적으면 예비 관찰로만 취급


class Tee:
    """콘솔과 로그 파일에 동시에 기록한다."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def write(self, text):
        sys.__stdout__.write(text)
        self.file.write(text)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ----------------------------------------------------------------------
# 1. 입력 규격 (7.3.1절 (1))
# ----------------------------------------------------------------------
TIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
]

TIME_COL_HINTS = ["time", "시각", "일시", "완료", "생산", "timestamp", "datetime", "date"]


def _parse_timestamp(text):
    text = text.strip().strip('"')
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"시각 형식을 인식하지 못했습니다: {text!r}\n"
                     f"지원 형식: {', '.join(TIME_FORMATS)}")


def _pick_time_column(header):
    """컬럼명에 힌트 단어가 들어간 첫 컬럼을 시각 컬럼으로 추정한다."""
    lowered = [h.strip().lower() for h in header]
    for hint in TIME_COL_HINTS:
        for i, name in enumerate(lowered):
            if hint in name:
                return i
    return None


def load_timestamps(path, time_col=None, sheet=None):
    """CSV 또는 Excel에서 생산 완료 시각 컬럼만 읽어 datetime 리스트로 반환한다."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except ImportError:
            raise SystemExit("Excel 입력에는 pandas와 openpyxl이 필요합니다. "
                             "CSV로 저장해 다시 시도하거나 pandas를 설치하세요.")
        df = pd.read_excel(path, sheet_name=sheet or 0)
        header = list(df.columns)
        if time_col is None:
            idx = _pick_time_column([str(h) for h in header])
            if idx is None:
                raise SystemExit(f"시각 컬럼을 찾지 못했습니다. --time-col 로 지정하세요. 컬럼: {header}")
            col = header[idx]
        else:
            col = time_col
        values = df[col].astype(str).tolist()
        chosen = str(col)
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if not rows:
            raise SystemExit("입력 파일이 비어 있습니다.")
        header, body = rows[0], rows[1:]
        if time_col is None:
            idx = _pick_time_column(header)
            if idx is None:
                raise SystemExit(f"시각 컬럼을 찾지 못했습니다. --time-col 로 지정하세요. 컬럼: {header}")
        elif time_col.isdigit():
            idx = int(time_col)
        else:
            if time_col not in header:
                raise SystemExit(f"컬럼 {time_col!r}이 없습니다. 컬럼: {header}")
            idx = header.index(time_col)
        values = [r[idx] for r in body if len(r) > idx and r[idx].strip()]
        chosen = header[idx]

    print(f"입력 파일       : {path.name}")
    print(f"시각 컬럼       : {chosen}")
    print(f"읽은 행 수      : {len(values):,}")
    return [_parse_timestamp(v) for v in values], chosen


def to_xy(timestamps, max_tact=None):
    """
    시각 배열에서 x(경과 초), y(tact 초)를 만든다.

    y[n] = time[n] - time[n-1] 을 코드가 직접 계산한다. 원본에 tact 컬럼이
    있더라도 사용하지 않는다(7.3.1절 (1)).

    Returns
    -------
    x, y, report : x[0]=0 기준 경과초, tact, 위생 점검 결과 dict
    """
    ts = sorted(timestamps)
    non_monotonic = sum(1 for a, b in zip(timestamps, timestamps[1:]) if b < a)
    duplicates = sum(1 for a, b in zip(ts, ts[1:]) if a == b)

    t0 = ts[0]
    x_all = np.array([(t - t0).total_seconds() for t in ts], dtype=float)
    y_all = np.diff(x_all)          # 첫 행의 tact는 정의되지 않으므로 자동 제외
    x_all = x_all[1:]

    nonpositive = int(np.sum(y_all <= 0))
    keep = y_all > 0
    removed_gap = 0
    if max_tact is not None:
        gap = y_all > max_tact
        removed_gap = int(np.sum(gap))
        keep = keep & ~gap

    # 구간을 잘라내면 x의 연속성이 깨지므로, 남은 tact로 x를 다시 누적한다.
    # (x는 삼각형의 가로축일 뿐이고 x[n]-x[n-1]=y[n] 관계만 유지되면 된다.)
    y = y_all[keep]
    x = np.cumsum(y)

    report = {
        "rows_read": len(timestamps),
        "non_monotonic": non_monotonic,
        "duplicates": duplicates,
        "nonpositive_tact": nonpositive,
        "removed_as_gap": removed_gap,
        "usable": len(y),
    }
    return x, y, report


def print_hygiene(report, max_tact):
    print()
    print("-" * 72)
    print("[1] 데이터 위생 점검")
    print("-" * 72)
    print(f"  읽은 행 수                    : {report['rows_read']:,}")
    print(f"  시각 역행(정렬로 교정)        : {report['non_monotonic']:,}")
    print(f"  중복 시각                     : {report['duplicates']:,}")
    print(f"  비양수 tact(제외)             : {report['nonpositive_tact']:,}")
    if max_tact is not None:
        print(f"  비생산 구간 tact>{max_tact}초(제외) : {report['removed_as_gap']:,}")
    else:
        print(f"  비생산 구간 제거              : 미적용 (--max-tact 미지정)")
    print(f"  최종 유효 tact 수 (L)         : {report['usable']:,}")

    L = report["usable"]
    if L < RECOMMENDED_L:
        print()
        print(f"  [경고] L={L:,}는 논문 8.2절 검증 규모(L=15,000)에 못 미칩니다.")
        print(f"         6.2절에서 보였듯 저대비 신호는 L이 작으면 통계적으로")
        print(f"         확인되지 않습니다. 이 결과는 결론의 근거가 아니라")
        print(f"         **예비 관찰**로만 취급하고, 7.3절에도 그렇게 표기하십시오.")


# ----------------------------------------------------------------------
# 2. N별 H 집계 (7.3.1절 (3)-1)
# ----------------------------------------------------------------------
def aggregate_H(x, y, N_max):
    """N=1..N_max 각각에 대해 전체 유효 앵커의 H를 집계한다 (앵커 1~2개가 아님)."""
    rows = []
    for N in range(1, N_max + 1):
        if len(x) <= 2 * N + 1:
            break
        h = triangle_metrics(x, y, N)["H"]
        valid = h[~np.isnan(h)]
        if valid.size == 0:
            continue
        rows.append({
            "N": N,
            "count": int(valid.size),
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "std": float(np.std(valid)),
            "p25": float(np.percentile(valid, 25)),
            "p75": float(np.percentile(valid, 75)),
        })
    return rows


def print_H_table(rows):
    print()
    print("-" * 72)
    print("[2] N별 H 집계 (전체 유효 앵커)")
    print("-" * 72)
    print(f"{'N':>4} | {'count':>9} | {'mean':>10} | {'median':>10} | {'std':>10}")
    print("-" * 60)
    for r in rows:
        print(f"{r['N']:>4} | {r['count']:>9,} | {r['mean']:>10.5f} | "
              f"{r['median']:>10.5f} | {r['std']:>10.5f}")
    best = min(rows, key=lambda r: r["median"])
    print(f"\n  median H 최소: N={best['N']} (median={best['median']:.5f})")
    print("  주의: 정배수도 비슷하게 낮은 H를 가지므로(5.2절), 최소값 자체가")
    print("        주기라는 뜻은 아니다. 판정은 [4]의 2단계 결과를 따른다.")


def plot_box(x, y, rows, out_path):
    """N별 H 분포 Box plot (7.3.1절 (3)-2)."""
    Ns = [r["N"] for r in rows]
    data = []
    for N in Ns:
        h = triangle_metrics(x, y, N)["H"]
        data.append(h[~np.isnan(h)])

    fig, ax = plt.subplots(figsize=(max(8, len(Ns) * 0.6), 5))
    ticks = [str(n) for n in Ns]
    try:                                     # matplotlib >= 3.9
        ax.boxplot(data, tick_labels=ticks, showfliers=False)
    except TypeError:                        # matplotlib < 3.9
        ax.boxplot(data, labels=ticks, showfliers=False)
    ax.set_xlabel("N (후보 주기)")
    ax.set_ylabel("H = 2S/|BC|")
    ax.set_title("실측 데이터: N별 H 분포 (이상치 미표시)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\n  saved: {out_path}")


# ----------------------------------------------------------------------
# 3. 판정 (7.3.1절 (3)-3, (3)-4)
# ----------------------------------------------------------------------
def run_two_stage(x, y, N_max, seed):
    print()
    print("-" * 72)
    print("[3] 기하학적 2단계 판정 (6.2절)")
    print("-" * 72)
    n_trials = 5
    while n_trials > 1 and len(x) // n_trials <= 2 * N_max:
        n_trials -= 1
    if len(x) // n_trials <= 2 * N_max:
        print(f"  [건너뜀] L={len(x):,}가 N_max={N_max}에 비해 너무 작습니다.")
        return None
    if n_trials != 5:
        print(f"  [조정] 청크 수를 5 → {n_trials}로 낮춤 (L이 작아 청크당 2*N_max 확보 불가)")

    res = find_period_two_stage(x, y, N_max=N_max, n_trials=n_trials,
                                confirm_stat_fn=np.mean, confirm_n_boot=1000,
                                rng=np.random.default_rng(seed))
    c = res["confirmation"]
    print(f"  1단계 (median 스캔)  N* = {res['N_star']}")
    if c["background_N"] is None:
        print("  2단계 배경 N          = 없음")
        print("  → N*=1은 '반복 구조를 찾지 못했다'는 뜻이며, 비교할 비배수 배경이")
        print("     존재하지 않아 2단계 확인을 수행할 수 없습니다. 이 데이터에서는")
        print("     기하학적 방법이 주기를 검출하지 못했다고 보고하십시오.")
        return res
    print(f"  2단계 배경 N          = {c['background_N']} (N*와 배수 관계가 아닌 것 중 가장 헷갈리는 후보)")
    print(f"  mean H 차이(배경-후보) = {c['diff']:+.5f}")
    print(f"  부트스트랩 95% CI      = [{c['ci'][0]:+.5f}, {c['ci'][1]:+.5f}]")
    print(f"  확정(confirmed)        = {'예' if c['confirmed'] else '아니오'}")
    if not c["confirmed"]:
        print("  → 신뢰구간이 0을 포함하거나 방향이 반대입니다. 이 데이터에서는")
        print("     기하학적 방법이 주기를 확정하지 못했다고 보고해야 합니다.")
    return res


def run_standard_methods(x, y, N_max, seed):
    print()
    print("-" * 72)
    print("[4] 표준 기법 대조 (8절)")
    print("-" * 72)
    results = {}
    results["ACF(첫국소최대)"] = find_period_acf(y, N_max)
    results["ACF(전역최대)"] = find_period_acf_global(y, N_max)
    results["FFT"] = find_period_fft(y, N_max)
    results["직접위상(2배)"] = find_period_direct_phase(
        y, N_max, n_boot=500, rng=np.random.default_rng(seed))
    results["직접위상(일반화)"] = find_period_direct_phase_general(
        y, N_max, n_boot=500, rng=np.random.default_rng(seed))

    for name, val in results.items():
        print(f"  {name:<18}: N = {val}")

    print()
    print("  참고: 540회 합성 검증에서의 성공률은 직접위상(일반화) 91.9% >")
    print("        ACF(첫국소최대) 76.1% > ACF(전역최대) 70.2% > 기하2단계 64.3% >")
    print("        직접위상(2배) 28.9% > FFT 0.0% 였다(8.2절). FFT는 배음 보정이")
    print("        없어 항상 2를 반환하므로 판단에서 제외한다(8.3절).")

    curve = acf_curve(y, N_max)
    print()
    print("  ACF 곡선 (판정 규칙 진단용, 8.3절):")
    for i in range(0, len(curve), 6):
        seg = ", ".join(f"lag{k + 1}={curve[k]:+.3f}" for k in range(i, min(i + 6, len(curve))))
        print(f"    {seg}")
    print(f"    상위 20% 임계값 = {np.percentile(curve, 80):+.4f}")
    return results


# ----------------------------------------------------------------------
# 4. 7.3절 초안 생성
# ----------------------------------------------------------------------
def emit_draft(path, report, rows, two_stage, standard, max_tact, n_max):
    L = report["usable"]
    lines = []
    lines.append("### 7.3 실측 데이터에 대한 검증 *(사내 실행 결과로 교체됨)*\n")
    lines.append("> 이 절은 `internal_validation.py`를 사내 실측 생산 로그에 실행한 결과다.")
    lines.append("> 원자료는 사내 비공개 자료여서 수록하지 않으며, 아래는 원시 tact 값을")
    lines.append("> 복원할 수 없는 집계 통계만을 옮긴 것이다(7.3.1절 (4) 첫 번째 경로).\n")

    lines.append("**데이터 개요**\n")
    lines.append("| 항목 | 값 |")
    lines.append("|---|---|")
    lines.append(f"| 읽은 행 수 | {report['rows_read']:,} |")
    lines.append(f"| 중복 시각 | {report['duplicates']:,} |")
    lines.append(f"| 시각 역행 | {report['non_monotonic']:,} |")
    lines.append(f"| 비양수 tact(제외) | {report['nonpositive_tact']:,} |")
    lines.append(f"| 비생산 구간 제거 | {'tact > ' + str(max_tact) + '초, ' + format(report['removed_as_gap'], ',') + '건' if max_tact is not None else '미적용'} |")
    lines.append(f"| **최종 유효 tact 수 (L)** | **{L:,}** |\n")
    if L < RECOMMENDED_L:
        lines.append(f"> **주의**: L={L:,}는 8.2절 검증 규모(L=15,000)에 못 미친다. 6.2절에서")
        lines.append("> 보였듯 저대비 신호는 표본이 작으면 통계적으로 확인되지 않으므로,")
        lines.append("> 아래 결과는 결론의 근거가 아니라 **예비 관찰**로만 취급한다.\n")

    lines.append("**N별 H 집계 (전체 유효 앵커)**\n")
    lines.append("| N | count | mean | median | std |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['N']} | {r['count']:,} | {r['mean']:.5f} | {r['median']:.5f} | {r['std']:.5f} |")
    lines.append("")
    lines.append("**[이미지 07]** 실측 데이터의 N별 H 분포\n")
    lines.append("![이미지 07](image/이미지_07_실측_N별_H_박스플롯.png)\n")

    lines.append("**기하학적 2단계 판정**\n")
    if two_stage is None:
        lines.append("- 표본이 부족해 2단계 판정을 수행하지 못했다.\n")
    else:
        c = two_stage["confirmation"]
        lines.append(f"- 1단계(median 스캔): N* = **{two_stage['N_star']}**")
        if c["background_N"] is None:
            lines.append("- 2단계: 수행 불가 (N*=1이라 비교할 비배수 배경이 없음)")
            lines.append("- 확정 여부: **미검출** — 기하학적 방법이 반복 구조를 찾지 못했다.\n")
        else:
            lines.append(f"- 2단계(mean + 부트스트랩, 배경 N={c['background_N']}): "
                         f"차이 = {c['diff']:+.5f}, 95% CI = [{c['ci'][0]:+.5f}, {c['ci'][1]:+.5f}]")
            lines.append(f"- 확정 여부: **{'확정' if c['confirmed'] else '미확정'}**\n")

    lines.append("**표준 기법 대조**\n")
    lines.append("| 방법 | 반환 N |")
    lines.append("|---|---|")
    if two_stage is not None:
        lines.append(f"| 기하 2단계 | {two_stage['N_star']} |")
    for name, val in standard.items():
        lines.append(f"| {name} | {val} |")
    lines.append("")
    lines.append("**해석 시 주의**: FFT는 배음 보정이 없어 합성 검증에서 항상 2를 반환했다")
    lines.append("(8.3절). 위 표의 FFT 값은 그 한계를 감안해 읽어야 한다.\n")

    lines.append("<!-- 아래는 작성자가 직접 채울 것 -->")
    lines.append("**결과 해석**: (합성 실험의 순위가 실측에서도 재현되었는지, ")
    lines.append("기하학적 방법의 판정이 표준 기법과 일치했는지를 서술)\n")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  saved: {path}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="사내 실측 로그에 대한 주기 검출 검증 (논문 7.3.1절)")
    ap.add_argument("--input", required=True, help="생산 로그 CSV 또는 Excel 경로")
    ap.add_argument("--time-col", default=None, help="시각 컬럼명 또는 인덱스 (미지정 시 자동 추정)")
    ap.add_argument("--sheet", default=None, help="Excel 시트명")
    ap.add_argument("--max-tact", type=float, default=None,
                    help="이 값을 넘는 tact를 비생산 구간으로 보고 제외 (초). "
                         "지정 시 미지정 실행과 결과를 반드시 둘 다 보고할 것")
    ap.add_argument("--n-max", type=int, default=DEFAULT_N_MAX, help=f"최대 탐색 주기 (기본 {DEFAULT_N_MAX})")
    ap.add_argument("--out-dir", default="out_internal", help="산출물 디렉터리")
    ap.add_argument("--seed", type=int, default=42, help="부트스트랩 시드 (재현용)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(out_dir / "internal_validation_log.txt")
    sys.stdout = tee

    try:
        print("=" * 72)
        print("실측 생산 로그 주기 검출 검증 (논문 7.3.1절 절차)")
        print("=" * 72)
        print(f"실행 시각       : {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"N_max           : {args.n_max}")
        print(f"부트스트랩 시드 : {args.seed}")
        if _KOREAN_FONTS:
            print(f"그래프 한글 폰트: {_KOREAN_FONTS[0]}")
        else:
            print("그래프 한글 폰트: 없음 (그림의 한글이 깨질 수 있습니다. "
                  "나눔고딕 등을 설치하세요)")

        timestamps, _ = load_timestamps(args.input, args.time_col, args.sheet)
        if len(timestamps) < 3:
            raise SystemExit("행이 너무 적습니다 (최소 3행 필요).")

        x, y, report = to_xy(timestamps, max_tact=args.max_tact)
        print_hygiene(report, args.max_tact)
        if report["usable"] < 2 * args.n_max + 2:
            raise SystemExit(f"유효 tact가 {report['usable']}개뿐이라 N_max={args.n_max} 분석이 불가능합니다.")

        rows = aggregate_H(x, y, args.n_max)
        print_H_table(rows)
        img = out_dir / "H_by_N_boxplot.png"
        plot_box(x, y, rows, img)

        two_stage = run_two_stage(x, y, args.n_max, args.seed)
        standard = run_standard_methods(x, y, args.n_max, args.seed)

        emit_draft(out_dir / "section_7_3_draft.md", report, rows, two_stage,
                   standard, args.max_tact, args.n_max)

        print()
        print("=" * 72)
        print("반출 점검 (사람이 확인할 것)")
        print("=" * 72)
        print("  산출물 3종에 개별 생산 시각·개별 tact 값은 기록되지 않는다.")
        print("  다만 아래 집계값이 사내 기준상 반출 가능한지는 사람이 판단해야 한다:")
        print("    - N별 H의 count/mean/median/std (생산량 규모가 count로 드러남)")
        print("    - Box plot의 y축 눈금 (H의 절대 스케일)")
        print("    - 판정된 주기 N (생산 순환 구조가 드러남)")
        print("  반출이 어려우면 7.3.1절 (4)의 두 번째 경로(대리 데이터 공개)를 쓰거나,")
        print("  7.3절을 현재의 '미검증' 상태로 유지한다. 추정치로 채우지 말 것.")
        print("=" * 72)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()

    print(f"\n완료. 산출물: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
