"""W&B finance-direction run 전체를 읽어 실험 검증표의 원자료를 만듦 (발표 준비, 읽기 전용).
    uv run python scripts/analysis_runs.py
출력
    cache/analysis/runs.json         run 별 config · summary · 혼동행렬 · 중요도 · 월별 표 (W&B 원본 그대로)
    docs/presentation/experiment_table_raw.csv    run 당 한 줄 (점수 열)
W&B 에 새로 쓰지 않음. failed_empty 태그 run 은 표에서 빼고 목록만 남김.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "cache" / "analysis"
DOC = ROOT / "docs" / "presentation"
PROJECT = "BI_finance_project/finance-direction"
FOLDS = ["fold1", "fold2", "fold3", "fold4"]


def _table(run, key):
    """run 에 log 된 wandb.Table 하나를 DataFrame 으로 (없으면 None)."""
    for f in run.files():
        if f.name.startswith(f"media/table/{key.replace('/', '/')}_") and f.name.endswith(".table.json"):
            p = f.download(root=str(OUT / "files" / run.id), replace=True)
            d = json.load(open(p.name, encoding="utf-8"))
            return pd.DataFrame(d["data"], columns=d["columns"])
    return None


def main():
    import wandb

    api = wandb.Api()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in api.runs(PROJECT, order="+created_at"):
        s = {k: v for k, v in r.summary._json_dict.items() if not isinstance(v, dict)}
        item = {"id": r.id, "name": r.name, "tags": r.tags, "state": r.state,
                "created": r.created_at, "config": dict(r.config), "summary": s, "tables": {}}
        if "failed_empty" not in r.tags and "submit" not in r.tags and "analysis" not in r.tags:
            for key in [*(f"confusion/{f}" for f in FOLDS), *(f"importance/{f}" for f in FOLDS),
                        "importance/mean", "monthly", "dist/pred_by_fold", "cv_table"]:
                t = _table(r, key)
                if t is not None:
                    item["tables"][key] = t.to_dict("list")
        runs.append(item)
        print(r.name, len(item["tables"]), flush=True)
    (OUT / "runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    rows = []
    for it in runs:
        if "failed_empty" in it["tags"] or "submit" in it["tags"] or "analysis" in it["tags"]:
            continue
        s, row = it["summary"], {"run": it["name"], "id": it["id"], "tags": " ".join(it["tags"])}
        for f in FOLDS:
            row[f] = s.get(f"{f}/all/score")
        row["mean4"] = s.get("mean/all/score")
        row["fold4_unseen"] = s.get("fold4/unseen/score")
        row["mean_unseen"] = s.get("mean/unseen/score")
        for k, v in s.items():
            if k.startswith("month/"):
                row[k] = v
        row["vs_gapz_ext_fold4"] = s.get("vs_gapz_ext_fold4")
        row["vs_gapz_ext"] = s.get("vs_gapz_ext")
        rows.append(row)
    df = pd.DataFrame(rows)
    DOC.mkdir(parents=True, exist_ok=True)
    df.to_csv(DOC / "experiment_table_raw.csv", index=False, encoding="utf-8-sig")
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
