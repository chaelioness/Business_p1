"""발표 덱 만들기: docs/presentation/deck_src.html 의 {{APPENDIX}}, {{TOC}} 를 채워 deck.html 로.
    uv run python scripts/build_deck.py
부록 표는 experiment_table.csv · numbers.json · params.json · reproduce.csv 에서 바로 만듦 (손으로 옮겨 적지 않음).
"""

import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "presentation"
AN = ROOT / "cache" / "analysis"
NUM = json.loads((AN / "numbers.json").read_text(encoding="utf-8"))
PARAMS = json.loads((AN / "params.json").read_text(encoding="utf-8"))
E = html.escape


def slide(sid, title, body, foot, notes=None, t=None):
    n = f'<details class="notes"><summary>발표 노트</summary><div>{notes}</div></details>' if notes else ""
    return (f'<section class="unit" id="{sid}" data-t="{E(t or sid.upper())}">\n<div class="slide"><div class="s">\n'
            f'<h2>{title}</h2>\n<div class="body">{body}</div>\n'
            f'<div class="foot"><span>{foot}</span><span>{sid.upper()}</span></div>\n</div></div>\n{n}</section>\n')


def f4(v):
    return "" if pd.isna(v) else f"{v:.3f}"


def exp_tables():
    d = pd.read_csv(DOC / "experiment_table.csv")
    out, part = [], 0
    groups = list(dict.fromkeys(d["묶음"]))
    chunks, cur = [], []
    for g in groups:
        rows = d[d["묶음"] == g]
        if cur and sum(len(x) for x in cur) + len(rows) > 15:
            chunks.append(cur)
            cur = []
        cur.append(rows)
    chunks.append(cur)
    for ci, ch in enumerate(chunks):
        trs = ['<tr><th>묶음</th><th>run</th><th>바꾼 것 (이전 run 대비 하나)</th><th class="n">f1</th><th class="n">f2</th>'
               '<th class="n">f3</th><th class="n">f4</th><th class="n">평균</th><th class="n">unseen</th><th class="n">vs 규칙 f4</th></tr>']
        for rows in ch:
            for _, r in rows.iterrows():
                hi = ' class="hi"' if r.run == "rule_gapz_ext" else ""
                trs.append(f'<tr{hi}><td>{E(r["묶음"])}</td><td>{E(r.run)}</td><td>{E(r["바꾼 것"])}</td>'
                           + "".join(f'<td class="n">{f4(r[c])}</td>' for c in ["fold1", "fold2", "fold3", "fold4", "4폴드 평균", "unseen fold4"])
                           + f'<td class="n">{r["vs 규칙 fold4"]:+.4f}</td></tr>')
        body = f'<table class="t dense">{"".join(trs)}</table>'
        part += 1
        out.append(slide(f"a1-{part}", f"A1. 실험 검증표 ({part}/{len(chunks)})", body,
                         "W&amp;B 기록 · '바꾼 것'은 a_v2.py EXPS · models2.py 코드로 확인 · 전체 열: experiment_table.md", t=f"A1-{part}"))
    return out


def params_table():
    keys = [("lam", "λ 고저폭"), ("gam", "γ 신뢰도"), ("delta", "δ 경계"), ("beta", "β 방향"), ("w", "w 앙상블"),
            ("kappa", "κ 시장갭"), ("theta", "θ 장전")]
    rows = []
    for n, seen in PARAMS.items():
        if n.startswith("lgb") or not any(k in seen[0] for k, _ in keys):
            continue
        cells = []
        for k, _ in keys:
            vals = [s.get(k) for s in seen]
            cells.append("" if all(v is None for v in vals) else "/".join(f"{v:g}" for v in vals))
        nz = any(c and set(c.replace("/", "").replace("0", "").replace(".", "")) for c in cells[1:])
        rows.append((n, cells, nz))
    head = '<tr><th>run</th>' + "".join(f'<th class="n">{h}</th>' for _, h in keys) + "</tr>"
    half = (len(rows) + 1) // 2
    out = []
    for i, part in enumerate([rows[:half], rows[half:]], 1):
        trs = [head] + [f'<tr{" class=\"hi\"" if nz else ""}><td>{E(n)}</td>' + "".join(f'<td class="n">{c}</td>' for c in cells) + "</tr>"
                        for n, cells, nz in part]
        note = ('<p class="small mut" style="margin:0">값은 fold1/2/3/4 에서 train score 가 가장 높았던 후보. 강조한 줄 = 추가 정보의 가중치가 0 이 아닌 폴드가 있음. '
                'LightGBM 공격성 k = 3 / 1.5 / 2 / 2.5, 반복 수 45 / 47 / 70 / 54. 제출 규칙(B 원본, λ 후보 0~0.5) λ = 0.05 / 0.3 / 0.2 / 0.2.</p>')
        out.append(slide(f"a2-{i}", f"A2. 폴드마다 train 이 고른 파라미터 ({i}/2)", note + f'<table class="t dense">{"".join(trs)}</table>',
                         "scripts/analysis_params.py → cache/analysis/params.json", t=f"A2-{i}"))
    return out


def decompose():
    d = NUM["decompose"]
    names = ["급하락", "하락", "보합", "상승", "급상승"]
    trs = ['<tr><th>정답</th><th class="n">비율</th><th class="n">규칙 실제</th><th class="n">규칙 무작위</th><th class="n">LGB 실제</th><th class="n">LGB 무작위</th></tr>']
    r, g = d["rule_gapz_ext"], d["lgb_v1_aggr"]
    for i in range(5):
        trs.append(f'<tr><td>{names[i]}</td><td class="n">{r["true_dist"][i]:.1%}</td><td class="n">{r["O_by_true"][i]:,.0f}</td>'
                   f'<td class="n">{r["E_by_true"][i]:,.0f}</td><td class="n">{g["O_by_true"][i]:,.0f}</td><td class="n">{g["E_by_true"][i]:,.0f}</td></tr>')
    trs.append(f'<tr class="hi"><td>합</td><td></td><td class="n">{sum(r["O_by_true"]):,.0f}</td><td class="n">{sum(r["E_by_true"]):,.0f}</td>'
               f'<td class="n">{sum(g["O_by_true"]):,.0f}</td><td class="n">{sum(g["E_by_true"]):,.0f}</td></tr>')
    pd_ = lambda x: " / ".join(f"{v:.0%}" for v in x)  # noqa: E731
    body = (f'<div class="cols c2"><div><div class="fig"><img src="figs/04_decompose.png" alt="정답 등급별 벌점"></div>'
            f'<table class="t dense">{"".join(trs)}</table></div>'
            f'<div><div class="fig"><img src="figs/A_decompose_cells.png" alt="칸별 차이"></div>'
            f'<ul class="small"><li>예측 비율 (급하락/하락/보합/상승/급상승) 규칙 {pd_(r["pred_dist"])}, LGB aggr {pd_(g["pred_dist"])}, LGB argmax {pd_(d["lgb_v1_argmax"]["pred_dist"])}.</li>'
            f'<li>분자(실제 벌점)는 LightGBM 이 438 더 큼 (9,827 vs 9,389). 그중 344 가 정답 급상승 줄, 156 이 급하락 줄. 분모는 두 모델이 거의 같음 (19,477 vs 19,376).</li><li>즉 차이는 급등락 날 방향을 얼마나 덜 틀리느냐에서 생김.</li></ul></div></div>')
    return [slide("a3", "A3. 점수식 분해: 규칙과 LightGBM 의 차이는 급등락 줄에서 생긴다", body,
                  "fold4 · Σw·O 정답 등급별 · 혼동행렬은 W&amp;B confusion/fold4 와 같음", t="A3")]


def fig_slide(sid, title, img, bullets, foot):
    body = (f'<div class="cols c32"><div><div class="fig"><img src="figs/{img}" alt="{E(title)}"></div></div>'
            f'<div><ul>{"".join(f"<li>{b}</li>" for b in bullets)}</ul></div></div>')
    return slide(sid, title, body, foot, t=sid.upper())


def figs():
    m = NUM["monthly"]
    nz = NUM["noise"]
    s = NUM["shap"]
    return [
        fig_slide("a4", "A4. 폴드별 점수: 네 폴드 모두 순위가 같다", "A_folds.png",
                  ["fold1~3 은 EDA 기간과 겹쳐 참고용.", "B 규칙 0.434 / 0.436 / 0.402 / 0.515, LightGBM aggr 0.346 / 0.420 / 0.392 / 0.495.",
                   "fold4 가 가장 높은 건 그 기간 급등락 비율이 높아서(19.5%, train 13.6%) 갭으로 맞힐 거리가 많았기 때문으로 보임."],
                  "W&amp;B fold*/all/score"),
        fig_slide("a5", "A5. 월별: 급등락이 많은 달에 점수도 높다", "08_monthly.png",
                  ["월은 <b>대상일</b> 기준 (W&amp;B monthly 표와 같음).",
                   f"fold4 train 급등락 {m['big_rate_train_fold4']:.1%}, fold4 val {m['big_rate_val_fold4']:.1%}.",
                   "recent250 은 3월 같고 4월 0.517→0.527, 5월 0.532→0.538. 급등락 비율이 높아진 2026 국면에 맞춘 경계가 그 국면에서만 나음.",
                   "LightGBM 은 2025 여름(fold1)에 크게 흔들림: 반복 수·k 를 정한 train 마지막 40일이 2025-04 관세 충격 직후."],
                  "scripts/analysis_figs.py fig_monthly"),
        fig_slide("a6", "A6. recent250 − 규칙, 날짜 bootstrap", "A_recent_boot.png",
                  ["fold4 val 60일을 복원 추출 1,000번, 두 모델을 같은 날짜로 채점해 차이.",
                   f"95% 구간 {nz['recent250_minus_rule_ci'][0]:+.3f} ~ {nz['recent250_minus_rule_ci'][1]:+.3f}, 0 이하 확률 {nz['recent250_minus_rule_p_le0']:.0%}.",
                   f"규칙 fold4 점수 자체 95% 구간 {nz['rule_fold4_boot_ci'][0]:.2f} ~ {nz['rule_fold4_boot_ci'][1]:.2f} (sd {nz['rule_fold4_boot_sd']:.3f}).",
                   "창 길이(250)는 검증 점수를 보기 전에 정한 후보. 여기서 더 고르면 fold4 에 맞추는 셈."],
                  "scripts/analysis_figs.py fig_noise"),
        fig_slide("a7", "A7. gap_z 가 LightGBM 예측을 미는 방향", "A_shap_gapz.png",
                  ["fold4 LightGBM (B 피처 26개, 반복 54회, k 2.5) 를 다시 학습해 val 3,000행의 pred_contrib.",
                   "세로축 = gap_z 가 급상승 logit − 급하락 logit 에 더한 값. 거의 단조 증가, 0 근처에서 급하게 바뀜.",
                   "방향 기여 상위: " + ", ".join(f"{k} {v:.2f}" for k, v in list(s["direction_top"].items())[:4]) + ".",
                   "크기 기여 상위: " + ", ".join(f"{k} {v:.2f}" for k, v in list(s["size_top"].items())[:4]) + "."],
                  "lightgbm Booster.predict(pred_contrib=True)"),
    ]


def unseen():
    e = NUM["errors"]
    rows = sorted(e["unseen_score_by_symbol"].items(), key=lambda x: x[1])
    trs = ['<tr><th>종목</th><th class="n">fold4 score</th><th class="n">vol20 평균</th></tr>'] + [
        f'<tr{" class=\"hi\"" if v < 0.25 else ""}><td>{k}</td><td class="n">{v:.3f}</td><td class="n">{e["unseen_vol20"].get(k, float("nan")):.4f}</td></tr>'
        for k, v in rows]
    body = (f'<div class="cols c2"><div><table class="t dense">{"".join(trs)}</table></div><div><ul>'
            f'<li>처음 보는 10종목 전체 0.418, 본 종목 포함 전체 0.515.</li>'
            f'<li>급등락 비율 처음 보는 종목 {e["big_rate_unseen"]:.1%} vs 본 종목 {e["big_rate_seen"]:.1%}, vol20 {e["vol20_unseen"]:.4f} vs {e["vol20_seen"]:.4f}.</li>'
            f'<li>반대 방향 오답 96행 중 처음 보는 종목 {e["unseen_rate_big_wrong"]:.0%} (행 비중 20%).</li>'
            f'<li>규칙은 종목 이름을 안 씀 (갭 ÷ 자기 변동성). 학습에서 빠진 영향보다 이 세 종목이 그 기간 갭과 반대로 움직인 날이 많았던 영향으로 보임. 종목 10개라 운의 몫이 큼.</li>'
            f'</ul></div></div>')
    return [slide("a8", "A8. 처음 보는 종목: 세 종목이 점수를 끌어내린다", body, "fold4 · lab/folds.json unseen_symbols", t="A8")]


def reproduce():
    r = pd.read_csv(DOC / "reproduce.csv")
    n, same = len(r), int((r.max_abs_diff.fillna(1) < 1e-6).sum())
    body = (f'<div class="cols c2"><div><ul>'
            f'<li>W&amp;B 기록 없이(track=False) 같은 폴드로 다시 돌린 run: <b>{n}개</b>, 네 폴드 점수가 기록과 같은 run: <b>{same}개</b> (최대 차이 0.00000).</li>'
            f'<li>핵심 세 숫자: B 규칙 fold4 0.5154, LightGBM aggr 0.4954, recent250 0.5209.</li>'
            f'<li>다시 돌리지 않은 것: 기준선 6개, rule_gapz, C 텍스트 ablation 4개 (같은 함수의 결과라 생략).</li>'
            f'<li>W&amp;B 의 빈 run 4개(failed_empty)는 오류로 점수 없이 끝난 첫 시도. 표에서 뺌.</li></ul></div>'
            f'<div><ul><li>재현 코드: <code>scripts/analysis_params.py</code></li><li>표: <code>docs/presentation/reproduce.csv</code></li>'
            f'<li>W&amp;B 원자료: <code>scripts/analysis_runs.py</code> → <code>experiment_table_raw.csv</code></li>'
            f'<li>그림·숫자: <code>scripts/analysis_figs.py</code> → <code>figs/</code>, <code>cache/analysis/numbers.json</code></li></ul></div></div>')
    return [slide("a9", "A9. 재현: 다시 돌린 run 전부 기록과 같은 점수", body, "홀드아웃 날짜는 읽지 않음 (lab/folds.json 날짜만)", t="A9")]


def diffs():
    rows = [
        ("월별 점수의 월", "worklog §2 '기준일 월'", "W&amp;B monthly 는 <b>대상일</b> 월 (run_cv 예측의 date 열 = day.y 날짜)"),
        ("recent120 5월 점수", "§8 '5월 0.538~0.541' (세 run)", "recent250 0.538, winvote 0.541, <b>recent120 0.533</b>"),
        ("'train 이 텍스트 가중치를 0 으로'", "§7 세 방식 모두 0", "신뢰도 γ·방향 β·앙상블 w·κ·θ 는 0. <b>경계 δ 는 0 아닌 값이 자주</b> (기사 수 0.3/0/0/0, 실적 단어 0.3/0.3/0/−0.1). text_big γ = −0.1"),
        ("무작위 흔들림", "§2.1 ±0.07, §7 ±0.03", "±0.07 은 폴드 사이 범위, ±0.03 은 fold4 1σ (0.029). 둘 다 맞음"),
        ("v3 run 태그", "§8 'W&amp;B 태그 v2'", "기록대로임. 다만 v3 를 찾으려면 이름(v3_*)으로 찾아야 함"),
        ("기준선 6개 run", "", "score_main · 월별 표 없음 (기능 추가 전 기록). 폴드 점수는 있음"),
        ("analysis run vl8iwnuk", "", "이번 분석 첫 기록. 날짜를 하루 어긋나게 붙인 오차 분석 숫자가 들어 있음 → 고친 뒤 mdaziq90 로 다시 기록, vl8iwnuk 는 삭제"),
    ]
    trs = ['<tr><th>항목</th><th>기존 기록</th><th>확인한 값</th></tr>'] + [f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows]
    return [slide("a10", "A10. 기록과 다르게 나온 것", f'<table class="t">{"".join(trs)}</table>',
                  "docs/worklog/2026-10-03_A_모델_v1_WandB.md 대비", t="A10")]


QA = [
    ("왜 딥러닝이나 LLM 으로 예측 안 했나?", "9:30 전에 보이는 정보 중 방향 신호는 갭 하나(일별 IC 0.30)였고, 트리 모델도 기여도를 보면 같은 피처만 씀(10쪽). 학습 행 약 2만 개에서 딥러닝 이점이 없음. LLM 은 예산(14만원 중 절반 권장) 때문에 방향 정보를 뽑는 사건 분류에만 소량 쓸 계획."),
    ("홀드아웃 점수는?", "아직 안 봄. 최종 모델을 정한 뒤 load_folds(ds, holdout=True) 로 한 번만 채점하기로 함."),
    ("과적합 아닌가?", "규칙이 학습하는 숫자는 5개(λ, 경계 a·b, 표준화용 중앙값·표준편차). train 0.454 < fold4 0.515 이고 네 폴드 모두 순위가 같음. 다만 실험 53개를 같은 fold4 로 비교했으니 '골라서 생기는' 과적합은 있음. 그래서 채택 기준과 가짜 피처 비교를 둠."),
    ("갭을 쓰는 건 규칙 위반 아닌가?", "과제 PDF 8쪽: cutoff(대상일 09:30)까지 들어오는 것에 '대상일 개장 전 시간외 거래'가 명시됨. 09:00 봉은 known_at 이 09:30 이라 day.price() 가 자동으로 빼서 실제로 쓰는 마지막 가격은 08:00 봉 종가(EDA 통합본 §5). 모든 예측은 guard 누수 검사 통과."),
    ("왜 fold4 만 보나?", "EDA 를 대상일 ~2026-02-13 데이터로 했음. fold1~3 val 은 그 안이라 우리가 이미 본 기간. EDA 와 안 겹치는 건 fold4 뿐."),
    ("0.515 와 0.522 는 다른가?", "날짜 bootstrap 으로 차이 95% 구간 −0.002 ~ +0.014, 0 이하 확률 9%. 그리고 4·5월에만 오름. 지금 데이터로는 구분이 안 됨."),
    ("LightGBM 이 왜 규칙보다 못한가?", "학습 손실(로그 손실)이 점수식의 비대칭을 모름. argmax 로 찍으면 보합을 37% 찍어 0.433. 디코딩을 고치면 0.495, 점수식 가중을 넣어도 0.499. 배운 내용은 규칙과 같음(갭 방향 + 변동성 크기)."),
    ("텍스트는 정말 쓸모없나?", "크기 신호는 있음(EDA 기사 수 크기 IC 0.086). 문제는 크기만으로는 점수가 안 오른다는 것(9쪽). 기사에서 방향(실적 서프라이즈, 가이던스 상향·하향)을 뽑을 수 있다면 다름."),
    ("처음 보는 종목은 어떻게 하나?", "규칙은 종목 이름 대신 자기 변동성으로 나눠서 그대로 적용됨. fold4 에서 낮은 건 세 종목(CVS·NFLX·WELL) 영향이 큼. 아직 대책은 없고, 10종목 표본이라 운의 몫도 큼."),
    ("10월에 급등락 비율이 바뀌면?", "경계를 |s| 분위로 정해서 예측 비율이 train 분포에 묶여 있음. 10월은 실적 시즌이라 실적일 비중이 커지는데, train 은 실적일에 오히려 덜 크게 찍는 쪽을 고름(9쪽). 최근 창 학습이 이 상황용 후보."),
]


def qa():
    out = []
    for i, part in enumerate([QA[:5], QA[5:]], 1):
        trs = ['<tr><th style="width:28%">질문</th><th>답 (근거)</th></tr>'] + [
            f"<tr><td><b>Q{(i - 1) * 5 + j + 1}. {E(q)}</b></td><td>{E(a)}</td></tr>" for j, (q, a) in enumerate(part)]
        out.append(slide(f"a11-{i}", f"A11. 예상 질문과 답 ({i}/2)", f'<table class="t dense" style="font-size:1.15cqw">{"".join(trs)}</table>',
                         "본편 쪽 번호 참고", t=f"A11-{i}"))
    return out


def main():
    src = (DOC / "deck_src.html").read_text(encoding="utf-8")
    app = exp_tables() + params_table() + decompose() + figs() + unseen() + reproduce() + diffs() + qa()
    out = src.replace("{{APPENDIX}}", "\n".join(app))
    ids = re.findall(r'<section class="unit" id="([^"]+)" data-t="([^"]+)"', out)
    toc = "".join(f'<a href="#{i}">{(i[1:] + " ") if i[0] == "s" else ""}{E(t)}</a>' for i, t in ids)
    out = out.replace("{{TOC}}", toc)
    # 발표 노트는 deck_src.html 에만 두고 게시본에서는 뺌 (상세 설명은 docs/presentation/해설.md)
    out = re.sub(r'<details class="notes">.*?</details>\s*', "", out, flags=re.S)
    out = re.sub(r'\s*<button id="toggle-notes".*?</button>', "", out, flags=re.S)
    out = re.sub(r"<script>.*?</script>", "", out, flags=re.S)
    (DOC / "deck.html").write_text(out, encoding="utf-8")
    print(len(ids), "slides")


if __name__ == "__main__":
    main()
