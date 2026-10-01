"""폴드별 교차검증.

    from lab.cv import run_cv, feature_table

    def fit(train_days, exclude):          # exclude: 학습에서 뺄 unseen 종목
        ...
        return model                       # predict(day) 만 있으면 됨

    res = run_cv(fit)                      # 폴드마다 학습 -> val 예측 -> 채점
    res.table                              # 폴드별 score (all / seen / unseen)

val 예측은 전부 guard(day) 로 돌려서, 누수가 있으면 점수 대신 LeakError 가 남.
점수는 src.score (교수님 채점 함수) 를 그대로 씀.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import Dataset, check_output, score
from src.paths import ROOT

from .folds import load_folds, unseen_symbols
from .leak import guard

CACHE = ROOT / "cache"


def feature_table(days, build, name=None, verbose=True):
    """build(day) 를 날마다 돌려 쌓고 정답을 붙임.

    build 는 guard 를 씌운 day 를 받아 symbol + feature 열을 돌려줘야 함.
    name 을 주면 cache/<name>.parquet 에 저장해 두고 다음엔 거기서 읽음.
    """
    path = CACHE / f"{name}.parquet" if name else None
    if path is not None and path.exists():
        cached = pd.read_parquet(path)
        need = {d.date for d in days}
        if need <= set(cached.date):
            return cached[cached.date.isin(need)].reset_index(drop=True)

    rows = []
    for i, day in enumerate(days, 1):
        x = build(guard(day))
        if len(x):
            x = x.merge(day.y[["symbol", "label", "ret_pct"]], on="symbol")
            x.insert(0, "date", day.date)
            rows.append(x)
        if verbose and i % 100 == 0:
            print(f"  {i}/{len(days)}일", flush=True)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    if path is not None:
        CACHE.mkdir(exist_ok=True)
        if path.exists():
            old = pd.read_parquet(path)
            out = (pd.concat([old[~old.date.isin(out.date)], out])
                   .sort_values(["date", "symbol"]).reset_index(drop=True))
        out.to_parquet(path, index=False)
        out = out[out.date.isin({d.date for d in days})].reset_index(drop=True)
    return out


def _scores(r):
    s = score(r.label, r.label_pred)
    return {k: s[k] for k in ("score", "accuracy", "big_recall", "big_prec", "n")}


@dataclass
class CVResult:
    table: pd.DataFrame          # 폴드 x (all / seen / unseen)
    preds: pd.DataFrame          # fold, date, symbol, label, label_pred, ret_pct

    def summary(self):
        t = self.table
        piv = t.pivot(index="fold", columns="group", values="score")[["all", "seen", "unseen"]]
        piv.loc["mean"] = piv.mean()
        piv.loc["sd"] = piv.iloc[:-1].std()
        return piv.round(4)


def run_cv(fit, ds=None, folds=None, use_unseen=True, verbose=True):
    """폴드마다 fit(train_days, exclude) 로 모델을 만들고 val 을 채점함."""
    ds = ds or Dataset()
    folds = folds or load_folds(ds)
    unseen = set(unseen_symbols()) if use_unseen else set()

    rows, preds = [], []
    for f in folds:
        model = fit(f.train, sorted(unseen))
        out = []
        for day in f.val:
            p = check_output(model.predict(guard(day)), day.symbols)
            out.append(day.y[["symbol", "date", "ret_pct", "label"]]
                       .merge(p, on="symbol", suffixes=("", "_pred")))
        r = pd.concat(out, ignore_index=True)
        r["label"] = r.label.astype(int)
        r.insert(0, "fold", f.name)
        preds.append(r)
        for group, part in (("all", r), ("seen", r[~r.symbol.isin(unseen)]),
                            ("unseen", r[r.symbol.isin(unseen)])):
            if len(part):
                rows.append({"fold": f.name, "group": group, **_scores(part)})
        if verbose:
            s = _scores(r)
            print(f"  {f.name}  score {s['score']:+.4f}  급등락 포착 {s['big_recall']:.3f}"
                  f"  ({f.val[0].date:%Y-%m-%d}~{f.val[-1].date:%Y-%m-%d})", flush=True)
    return CVResult(pd.DataFrame(rows), pd.concat(preds, ignore_index=True))
