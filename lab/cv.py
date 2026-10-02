"""폴드별 교차검증.

    from lab.cv import run_cv, feature_table

    def fit(train_days, exclude):          # exclude: 학습에서 뺄 unseen 종목
        ...
        return model                       # predict(day) 만 있으면 됨

    res = run_cv(fit)                      # 폴드마다 학습 -> val 예측 -> 채점
    res.table                              # 폴드별 score (all / seen / unseen)

    # W&B 에 남기기 (키가 없으면 저절로 꺼짐, lab/tracking.py)
    res = run_cv(fit, name="lgb_v1", config={"features": FEATS, "decode": "match"},
                 tags=["lgb"], log_model=True)

val 예측은 전부 guard(day) 로 돌려서, 누수가 있으면 점수 대신 LeakError 가 남.
점수는 src.score (교수님 채점 함수) 를 그대로 씀.
"""

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import Dataset, check_output, score
from src.paths import ROOT

from . import tracking
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


def baseline_path(folds_hash):
    return CACHE / "baselines" / f"{folds_hash}.json"


def save_baseline(fit=None, key="gap", ds=None, folds=None):
    """규칙 기준선 점수를 폴드 해시별 파일에 key 로 저장 (같은 파일에 여러 규칙).
    run_cv 는 이 파일이 있으면 같은 run 에 baseline/<key>/* 와 vs_<key> 를 붙임.
        uv run python -m lab.cv baseline           # 갭 규칙 (eda.cv_check_ovn_gap)
    """
    if fit is None:
        from eda.cv_check_ovn_gap import fit_ovn_gap as fit

    ds = ds or Dataset()
    folds = folds or load_folds(ds)
    res = run_cv(fit, ds, folds, track=False)
    t = res.table[res.table.group == "all"].set_index("fold").score
    out = {**{f"{key}/{k}": float(v) for k, v in t.items()},
           f"{key}/mean": float(t.mean()), f"{key}/sd": float(t.std())}
    for g in ("seen", "unseen"):
        out[f"{key}/mean_{g}"] = float(res.table[res.table.group == g].score.mean())
    path = baseline_path(tracking.folds_info(folds)["folds_hash"])
    path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old = {k: v for k, v in old.items() if not k.startswith(f"{key}/")}
    path.write_text(json.dumps({**old, **out}, indent=2), encoding="utf-8")
    return out


save_gap_baseline = save_baseline


def run_cv(fit, ds=None, folds=None, use_unseen=True, verbose=True,
           name=None, config=None, tags=None, log_model=False, method=None, track=True):
    """폴드마다 fit(train_days, exclude) 로 모델을 만들고 val 을 채점함.

    name / config / tags 는 W&B run 이름·설정·태그. config 에는 피처 목록, 하이퍼파라미터,
    디코딩 방식 같은 것을 넣음 (실행자·커밋·폴드 정보는 자동으로 붙음).
    method 는 검증 방식 이름 (기본: 폴드 개수로 "A-folds4" 같은 이름).
    log_model=True 면 폴드별 모델을 W&B Artifact 로 올림. track=False 면 기록 안 함.
    모델에 feature_importance (dict / Series / 그걸 돌려주는 함수) 가 있으면 같이 기록.
    """
    ds = ds or Dataset()
    folds = folds or load_folds(ds)
    unseen = set(unseen_symbols()) if use_unseen else set()

    tr = tracking.Tracker(name=name, tags=tags, enabled=track, config=lambda: {
        "git": tracking.git_info(), **tracking.folds_info(folds),
        "method": method or f"A-folds{len(folds)}",
        "fit": f"{getattr(fit, '__module__', '?')}.{getattr(fit, '__qualname__', repr(fit))}",
        "unseen_symbols": sorted(unseen), "use_unseen": use_unseen, **(config or {})})
    if verbose and tr.on:
        print(f"  W&B run: {tr.run.url}", flush=True)

    rows, preds, models = [], [], {}
    t0 = time.time()
    try:
        for i, f in enumerate(folds):
            model = fit(f.train, sorted(unseen))
            if log_model:
                models[f.name] = model
            out = []
            for day in f.val:
                p = check_output(model.predict(guard(day)), day.symbols)
                out.append(day.y[["symbol", "date", "ret_pct", "label"]]
                           .merge(p, on="symbol", suffixes=("", "_pred")))
            r = pd.concat(out, ignore_index=True)
            r["label"] = r.label.astype(int)
            r.insert(0, "fold", f.name)
            preds.append(r)
            fold_metrics = {}
            for group, part in (("all", r), ("seen", r[~r.symbol.isin(unseen)]),
                                ("unseen", r[r.symbol.isin(unseen)])):
                if len(part):
                    fold_metrics[group] = _scores(part)
                    rows.append({"fold": f.name, "group": group, **fold_metrics[group]})
            tr.fold(i, f.name, fold_metrics, r.label.to_numpy(), r.label_pred.to_numpy(),
                    getattr(model, "feature_importance", None))
            if verbose:
                s = fold_metrics["all"]
                print(f"  {f.name}  score {s['score']:+.4f}  급등락 포착 {s['big_recall']:.3f}"
                      f"  ({f.val[0].date:%Y-%m-%d}~{f.val[-1].date:%Y-%m-%d})", flush=True)
        res = CVResult(pd.DataFrame(rows), pd.concat(preds, ignore_index=True))
        if tr.on:
            bp = baseline_path(tracking.folds_info(folds)["folds_hash"])
            base = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else None
            tr.run.summary["seconds"] = time.time() - t0
            tr.finish_cv(res.table, res.preds, base)
            if log_model:
                summ = res.summary()
                tr.model(models, name, {"config": dict(tr.run.config),
                                        "score": summ.to_dict()})
    finally:
        tr.finish()
    return res


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["baseline"]:
        print(save_baseline())
