"""B 추가 피처 후보 실험 — A 의 폴드(lab/folds.json)와 run_cv 로.

    uv run python work/b/exp/r01_features.py table      # 피처 표 만들기 (cache/b_v2.parquet, 약 4분)
    uv run python work/b/exp/r01_features.py evidence   # 1·2단계: 근거(날짜별 IC) + 겹침
    uv run python work/b/exp/r01_features.py ablation   # 3단계: V1 + 후보 1개씩, 폴드 4개 × 시드 3개
    uv run python work/b/exp/r01_features.py final      # 채택 묶음 확인

채택 기준 (실험 전에 정함)
  1. 근거: fold1 학습 구간(대상일 ~2025-05-27, 어느 폴드 val 과도 안 겹침)에서
     날짜별 순위 상관(방향=ret, 크기=|ret|) |t| >= 3, 4개 기간 중 3개 이상 부호 유지
  2. 겹침: V1 26개와 순위 상관 |rho| 최대값 보고. 가장 비슷한 V1 피처 3개로 설명되는 몫을 빼도
     남은 몫(잔차)의 날짜별 순위 상관 |t| >= 2
  3. 점수: V1 기본 모델 대비 폴드 4개 평균 score +0.005 이상, 폴드 3개 이상에서 상승,
     fold4(2026-03~05, EDA 와 안 겹침)에서 하락하지 않음. 시드 3개 평균으로 비교.
모델: LightGBM 다중분류(v1_check 기본값) + 분포맞춤 디코딩. 처음 보는 10종목은 학습에서 뺌.
점수: src.score (run_cv 그대로), all / seen / unseen.
"""

import sys

import numpy as np
import pandas as pd

import common  # noqa: E402,F401  (경로 설정)

from lab.cv import run_cv  # noqa: E402
from features_b import B_EXTRA, B_FEATURES_V1  # noqa: E402
from common import BASE, OUT, SEEDS, VALID_DAYS, load_table  # noqa: E402


ROLE = {  # 후보가 맞히려는 것: dir = 방향(ret), mag = 크기(|ret|)
    "earn_hist_absr": "mag", "earn_timing": "mag", "absret1_z": "mag", "big_yday": "mag",
    "bigfreq20": "mag", "mkt_absgap": "mag", "gap_disp": "mag",
    "gap_onz": "dir", "gap_rank": "dir", "pre_late": "dir", "ext_pos": "dir",
}


# ---------------------------------------------------------------- 1·2단계

def date_ic(df, x, y):
    r = df[["date", x, y]].dropna()
    a, b = r.groupby("date")[x].rank(), r.groupby("date")[y].rank()
    t = pd.DataFrame({"date": r.date, "a": a, "b": b})
    ic = t.groupby("date").apply(lambda g: g.a.corr(g.b) if g.a.nunique() > 1 else np.nan, include_groups=False).dropna()
    return ic


def ic_stats(df, x, y, n_per=4):
    ic = date_ic(df, x, y)
    if len(ic) < 20:
        return dict(ic=np.nan, t=np.nan, same=np.nan, n=len(ic))
    per = pd.qcut(np.arange(len(ic)), n_per, labels=False)
    pm = ic.groupby(per).mean().values
    return dict(ic=ic.mean(), t=ic.mean() / ic.std() * np.sqrt(len(ic)),
                same=int((np.sign(pm) == np.sign(ic.mean())).sum()), n=len(ic))


def evidence():
    t, folds = load_table()
    tr_dates = {d.date for d in folds[0].train}
    e = t[t.date.isin(tr_dates)].copy()
    e["absret"] = e.ret_pct.abs()
    print(f"근거 구간: fold1 학습 {e.date.min():%Y-%m-%d} ~ {e.date.max():%Y-%m-%d} ({e.date.nunique()}일, {len(e):,}행)\n")

    # 날짜 안에서 값이 같은 후보(mkt_absgap, gap_disp)는 날짜별 순위 상관이 없음 → 날짜 단위 상관으로 대신
    rows = []
    for f in B_EXTRA:
        role = ROLE[f]
        y = "ret_pct" if role == "dir" else "absret"
        if e.groupby("date")[f].nunique().max() <= 1:
            dm = e.groupby("date").agg(x=(f, "first"), big=("absret", lambda s: (s > 2.5).mean()))
            per = pd.qcut(np.arange(len(dm)), 4, labels=False)
            rho = dm.x.corr(dm.big, method="spearman")
            pr = [g.x.corr(g.big, method="spearman") for _, g in dm.groupby(per)]
            rows.append({"후보": f, "역할": "크기(날짜)", "IC": rho, "t": rho * np.sqrt(len(dm) - 2) / np.sqrt(1 - rho ** 2),
                         "유지": int((np.sign(pr) == np.sign(rho)).sum()), "비고": "날짜 단위: 그날 급등락 종목 비율과 상관"})
            continue
        s = ic_stats(e, f, y)
        rows.append({"후보": f, "역할": "방향" if role == "dir" else "크기", "IC": s["ic"], "t": s["t"], "유지": s["same"], "비고": ""})
    ev = pd.DataFrame(rows).set_index("후보")

    # 겹침: V1 과의 순위 상관, 가장 비슷한 3개로 설명한 뒤 남은 몫의 IC
    rk = e[B_FEATURES_V1 + B_EXTRA].rank(pct=True)
    corr = rk.corr()
    red = []
    for f in B_EXTRA:
        c = corr.loc[f, B_FEATURES_V1].abs().sort_values(ascending=False)
        top = list(c.index[:3])
        sub = rk[[f, *top]].dropna()
        X = np.column_stack([np.ones(len(sub)), sub[top].values])
        beta, *_ = np.linalg.lstsq(X, sub[f].values, rcond=None)
        e.loc[sub.index, "_res"] = sub[f].values - X @ beta
        y = "ret_pct" if ROLE[f] == "dir" else "absret"
        if ev.loc[f, "역할"] == "크기(날짜)":
            dm = e.groupby("date").agg(x=("_res", "mean"), big=("absret", lambda s: (s > 2.5).mean()))
            rr = dm.x.corr(dm.big, method="spearman")
            res_ic, res_t = rr, rr * np.sqrt(len(dm) - 2) / np.sqrt(1 - rr ** 2)
        else:
            s = ic_stats(e, "_res", y)
            res_ic, res_t = s["ic"], s["t"]
        red.append({"후보": f, "가장 비슷한 V1": f"{top[0]} ({c.iloc[0]:.2f})", "2·3번째": f"{top[1]} ({c.iloc[1]:.2f}), {top[2]} ({c.iloc[2]:.2f})",
                    "겹침 뺀 IC": res_ic, "겹침 뺀 t": res_t})
        e.drop(columns="_res", inplace=True)
    red = pd.DataFrame(red).set_index("후보")
    out = ev.join(red)
    out["1단계"] = (out.t.abs() >= 3) & (out["유지"] >= 3)
    out["2단계"] = out["겹침 뺀 t"].abs() >= 2
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
    print(out.round(3).to_string())
    out.to_csv(OUT / "exp_evidence.csv")

    # V1 안에서도 많이 겹치는 쌍 (참고)
    cv1 = corr.loc[B_FEATURES_V1, B_FEATURES_V1].abs()
    pairs = [(a, b, cv1.loc[a, b]) for i, a in enumerate(B_FEATURES_V1) for b in B_FEATURES_V1[i + 1:] if cv1.loc[a, b] >= 0.8]
    print("\nV1 안에서 순위 상관 0.8 이상인 쌍:", [(a, b, round(v, 2)) for a, b, v in sorted(pairs, key=lambda p: -p[2])])


# ---------------------------------------------------------------- 3단계

def decode_fit(p_tr, y_tr, how):
    """train 확률로 경계를 정함. E = 기대 등급 분위, DS = 방향 × 크기 (급등락 여부와 방향을 따로)."""
    pi = np.bincount(y_tr, minlength=5) / len(y_tr)
    if how == "E":
        return ("E", np.quantile(p_tr @ np.arange(5), np.cumsum(pi)[:-1]))
    d = p_tr[:, 3] + p_tr[:, 4] - p_tr[:, 0] - p_tr[:, 1]
    m = p_tr[:, 0] + p_tr[:, 4]
    t_big = np.quantile(m, 1 - (pi[0] + pi[4]))
    mid = pi[1:4] / pi[1:4].sum()
    d_mid = d[m <= t_big]
    return ("DS", (t_big, np.quantile(d_mid, [mid[0], mid[0] + mid[1]])))


def decode_apply(p, rule):
    how, c = rule
    if how == "E":
        return np.searchsorted(c, p @ np.arange(5), side="right")
    t_big, (q1, q2) = c
    d = p[:, 3] + p[:, 4] - p[:, 0] - p[:, 1]
    m = p[:, 0] + p[:, 4]
    lab = np.where(d < q1, 1, np.where(d < q2, 2, 3))
    return np.where(m > t_big, np.where(d > 0, 4, 0), lab)


class Model:
    def __init__(self, booster, feats, rule, table):
        self.b, self.feats, self.rule, self.table = booster, feats, rule, table

    def predict(self, day):
        x = self.table[self.table.date == day.date].set_index("symbol").reindex(day.symbols)
        lab = decode_apply(self.b.predict(x[self.feats]), self.rule)
        return pd.DataFrame({"symbol": day.symbols, "label": lab.astype(int)})


def make_fit(table, feats, seed, decode="E", params=None):
    import lightgbm as lgb
    p = {**BASE, "seed": seed, **(params or {})}

    def fit(train_days, exclude):
        dates = {d.date for d in train_days}
        tr = table[table.date.isin(dates) & ~table.symbol.isin(exclude)]
        cut = np.sort(tr.date.unique())[-VALID_DAYS]
        a, v = tr[tr.date < cut], tr[tr.date >= cut]
        m = lgb.train(p, lgb.Dataset(a[feats], a.label), 2000, valid_sets=[lgb.Dataset(v[feats], v.label)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        b = lgb.train(p, lgb.Dataset(tr[feats], tr.label), max(m.best_iteration, 20))
        return Model(b, feats, decode_fit(b.predict(tr[feats]), tr.label.values, decode), table)
    return fit


def cv(table, folds, feats, seed, decode="E", params=None):
    res = run_cv(make_fit(table, feats, seed, decode, params), folds=folds, verbose=False)
    t = res.table.pivot(index="fold", columns="group", values="score")
    big = res.table[res.table.group == "all"].set_index("fold")[["big_recall", "big_prec"]]
    return t, big


EARN = ["earn_now", "earn_timing", "earn_hist_absr"]


def round2():
    """크기 피처가 반영될 수 있게: 방향×크기 디코딩, 실적 묶음은 잎 최소 100 으로."""
    v = {"V1": (B_FEATURES_V1, "DS", None)}
    for c in ["earn_hist_absr", "big_yday", "mkt_absgap", "gap_disp", "gap_rank", "ext_pos", "pre_late"]:
        v[f"+{c}"] = (B_FEATURES_V1 + [c], "DS", None)
    v["+실적 묶음"] = (B_FEATURES_V1 + EARN, "DS", None)
    ablation(v, tag="round2_DS")
    leaf = {"min_data_in_leaf": 100}
    ablation({"V1": (B_FEATURES_V1, "DS", leaf), "+실적 묶음": (B_FEATURES_V1 + EARN, "DS", leaf),
              "+earn_hist_absr": (B_FEATURES_V1 + ["earn_hist_absr"], "DS", leaf)}, tag="round2_DS_leaf100")
    ablation({"V1 (E 디코딩)": (B_FEATURES_V1, "E", None), "V1": (B_FEATURES_V1, "DS", None)}, tag="decode_compare")


def ablation(variants=None, tag="ablation"):
    table, folds = load_table()
    variants = variants or {"V1": B_FEATURES_V1, **{f"+{c}": B_FEATURES_V1 + [c] for c in B_EXTRA}}
    rows = []
    for name, spec in variants.items():
        feats, decode, params = spec if isinstance(spec, tuple) else (spec, "E", None)
        for s in SEEDS:
            t, big = cv(table, folds, feats, s, decode, params)
            for fold in t.index:
                rows.append({"variant": name, "seed": s, "fold": fold, **t.loc[fold].to_dict(), **big.loc[fold].to_dict()})
        print(f"  {name} 완료", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"exp_{tag}.csv", index=False)
    report(r)


def report(r, base="V1"):
    m = r.groupby(["variant", "fold"])[["all", "seen", "unseen", "big_recall", "big_prec"]].mean()
    sd_seed = r[r.variant == base].groupby("seed")["all"].mean().std()
    b = m.loc[base]
    rows = []
    for v in r.variant.unique():
        x = m.loc[v]
        d = x["all"] - b["all"]
        rows.append({"variant": v, "score 평균": x["all"].mean(), "Δ평균": d.mean(), "Δfold4": d.get("fold4", np.nan),
                     "상승 폴드": int((d > 0).sum()), "unseen 평균": x["unseen"].mean(), "Δunseen": (x["unseen"] - b["unseen"]).mean(),
                     "급등락 포착": x["big_recall"].mean(), "급등락 적중": x["big_prec"].mean(),
                     **{f: x.loc[f, "all"] for f in x.index}})
    out = pd.DataFrame(rows).set_index("variant")
    out["3단계"] = (out["Δ평균"] >= 0.005) & (out["상승 폴드"] >= 3) & (out["Δfold4"] >= 0)
    pd.set_option("display.width", 250)
    print(f"\n시드 3개 평균. 기준 {base} 의 시드 간 표준편차(폴드 평균 score) = {sd_seed:.4f}")
    print(out.round(4).to_string())


def final():
    ev = pd.read_csv(OUT / "exp_evidence.csv", index_col=0)
    ab = pd.read_csv(OUT / "exp_ablation.csv")
    ok = [c for c in B_EXTRA if ev.loc[c, "1단계"] and ev.loc[c, "2단계"]]
    m = ab.groupby(["variant", "fold"])["all"].mean().unstack()
    d = m.sub(m.loc["V1"], axis=1)
    passed = [c for c in ok if d.loc[f"+{c}"].mean() >= 0.005 and (d.loc[f"+{c}"] > 0).sum() >= 3 and d.loc[f"+{c}", "fold4"] >= 0]
    print("1·2단계 통과:", ok)
    print("3단계까지 통과:", passed)
    if passed:
        ablation({"V1": B_FEATURES_V1, "V1+통과 전부": B_FEATURES_V1 + passed}, tag="final")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    {"table": lambda: print(load_table()[0].shape), "evidence": evidence, "ablation": ablation, "round2": round2, "final": final}[sys.argv[1]]()
