"""재현 + 폴드별 학습된 파라미터 (발표 분석 b). W&B 에 기록하지 않음 (track=False).
    uv run python scripts/analysis_params.py [이름 ...]
모든 v2/v3 실험(scripts/a_v2.py EXPS)과 B 규칙, LightGBM 4종을 같은 폴드로 다시 돌려
  - 폴드별 score 를 W&B 기록과 비교 (재현)
  - 폴드마다 train 이 고른 λ, γ, δ, β, w, κ, θ, a, b, LightGBM 반복 수·공격성 k 를 저장
  - fold4 예측 행 (오차 분석용) 을 저장
출력: cache/analysis/params.json, cache/analysis/preds_<이름>.parquet, docs/presentation/reproduce.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from a_v2 import EXPS  # noqa: E402
from lab.cv import run_cv  # noqa: E402
from lab.folds import HOLDOUT  # noqa: E402
from src import Dataset  # noqa: E402
from work.a import exp2 as E  # noqa: E402
from work.a import lgb as L  # noqa: E402
from work.a.models2 import make_fit  # noqa: E402
from work.a.rules import fit_gapz_ext  # noqa: E402

OUT = ROOT / "cache" / "analysis"
DOC = ROOT / "docs" / "presentation"


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def capture(fit, info_of):
    seen = []

    def wrapped(train_days, exclude):
        m = fit(train_days, exclude)
        seen.append(_plain(info_of(m)))
        return m
    wrapped.__qualname__ = getattr(fit, "__qualname__", "fit")
    return wrapped, seen


def jobs():
    out = {"rule_gapz_ext": (fit_gapz_ext, lambda m: m.params)}
    for dec in ["argmax", "match_train", "match_recent", "aggr"]:
        out[f"lgb_v1_{dec}"] = (L.make_fit(dec), lambda m: {**m.info, "th": m.th})
    for n, (kind, feats, text) in EXPS.items():
        out[n] = (make_fit(kind, feats, text), lambda m: m.info)
    return out


def main(names):
    ds = Dataset()
    E.master(ds.split(HOLDOUT)[0])
    wb = pd.read_csv(DOC / "experiment_table_raw.csv")
    wb = wb.drop_duplicates("run", keep="last").set_index("run")
    allj = jobs()
    names = names or list(allj)
    pfile = OUT / "params.json"
    params = json.loads(pfile.read_text(encoding="utf-8")) if pfile.exists() else {}
    rows = []
    for n in names:
        fit, info_of = allj[n]
        f, seen = capture(fit, info_of)
        res = run_cv(f, ds, track=False, verbose=False)
        t = res.table[res.table.group == "all"].set_index("fold").score
        u = res.table[res.table.group == "unseen"].set_index("fold").score
        params[n] = seen
        res.preds[res.preds.fold == "fold4"].to_parquet(OUT / f"preds_{n}.parquet", index=False)
        row = {"run": n, **{f: t[f] for f in t.index}, "fold4_unseen": u["fold4"]}
        if n in wb.index:
            for f in t.index:
                row[f"wandb_{f}"] = wb.loc[n, f]
            row["max_abs_diff"] = max(abs(t[f] - wb.loc[n, f]) for f in t.index)
        rows.append(row)
        print(f"{n:30s} " + " ".join(f"{v:.4f}" for v in t) + f"  diff {row.get('max_abs_diff', float('nan')):.5f}",
              flush=True)
        pfile.write_text(json.dumps(params, indent=1, default=str), encoding="utf-8")
    rep = pd.DataFrame(rows)
    path = DOC / "reproduce.csv"
    if path.exists() and names != list(allj):
        old = pd.read_csv(path)
        rep = pd.concat([old[~old.run.isin(rep.run)], rep])
    rep.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main(sys.argv[1:])
