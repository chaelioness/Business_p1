"""W&B Models 기록. run_cv 가 부르므로 직접 쓸 일은 거의 없음.

    res = run_cv(fit, name="lgb_v1", config={"features": FEATS, "decode": "match"},
                 tags=["lgb"], log_model=True)

프로젝트는 lab/wandb.json (entity / project) 을 읽음. 환경변수 WANDB_ENTITY / WANDB_PROJECT 가
있으면 그게 우선 (잠깐 바꿀 때만). 키는 각자 `uv run wandb login` 또는 WANDB_API_KEY.

아래 경우엔 조용히 꺼지고 run_cv 는 예전처럼 돌아감.
    - wandb 가 설치 안 됨
    - WANDB_MODE=disabled
    - 키가 없음 (WANDB_MODE=offline 이면 키 없이 로컬에만 남김)
기록 중 오류가 나도 경고만 찍고 실험은 계속함.

올리지 않는 것: 원본 데이터, 예측 행 전체(res.preds), API 키.
"""

import hashlib
import json
import netrc
import os
import pickle
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import score
from src.paths import ROOT

CONFIG = Path(__file__).with_name("wandb.json")
HOST = "api.wandb.ai"


def settings() -> dict:
    s = json.loads(CONFIG.read_text(encoding="utf-8"))
    s["entity"] = os.environ.get("WANDB_ENTITY") or s["entity"]
    s["project"] = os.environ.get("WANDB_PROJECT") or s["project"]
    return s


def has_key() -> bool:
    """키가 있는지만 봄. 키 값은 읽어서 어디에도 쓰지 않음."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    home = Path.home()
    for name in (".netrc", "_netrc"):
        try:
            if netrc.netrc(home / name).authenticators(HOST):
                return True
        except (FileNotFoundError, netrc.NetrcParseError, OSError):
            pass
    return False


def _wandb():
    """쓸 수 있으면 wandb 모듈, 아니면 None."""
    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode in ("disabled", "dryrun"):
        return None
    try:
        import wandb
    except ImportError:
        return None
    if mode != "offline" and not has_key():
        return None
    return wandb


def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info() -> dict:
    status = _git("status", "--porcelain")
    return {"user": _git("config", "user.name"), "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None}


def folds_info(folds) -> dict:
    """폴드 날짜와 그 해시. 같은 해시면 같은 검증."""
    dates = {f.name: {"train": [f"{f.train[0].date:%Y-%m-%d}", f"{f.train[-1].date:%Y-%m-%d}"],
                      "val": [f"{f.val[0].date:%Y-%m-%d}", f"{f.val[-1].date:%Y-%m-%d}"]}
             for f in folds}
    h = hashlib.sha1(json.dumps(dates, sort_keys=True).encode()).hexdigest()[:10]
    return {"folds_hash": h, "folds": dates}


def _safe(fn):
    def wrap(self, *a, **k):
        if self.run is None:
            return None
        try:
            return fn(self, *a, **k)
        except Exception as e:                       # 기록 실패로 실험이 멈추면 안 됨
            warnings.warn(f"W&B 기록 실패 ({fn.__name__}): {type(e).__name__}: {e}")
            return None
    return wrap


class Tracker:
    """run 하나 = run_cv 한 번. 꺼져 있으면 모든 메서드가 아무것도 안 함."""

    def __init__(self, name=None, config=None, tags=None, job_type="cv", enabled=True):
        """config 는 dict 또는 dict 를 돌려주는 함수 (꺼져 있으면 부르지 않음)."""
        self.run = None
        wandb = _wandb() if enabled else None
        if wandb is None:
            return
        try:
            s = settings()
            config = config() if callable(config) else config
            self.wandb = wandb
            self.run = wandb.init(entity=s["entity"], project=s["project"], name=name,
                                  config=config or {}, tags=list(tags or []), job_type=job_type,
                                  dir=str(ROOT), reinit="finish_previous")
        except Exception as e:
            warnings.warn(f"W&B 시작 실패, 기록 없이 계속함: {type(e).__name__}: {e}")
            self.run = None

    @property
    def on(self):
        return self.run is not None

    @_safe
    def fold(self, step, fold, metrics, y, pred, importance=None):
        """폴드 하나 끝날 때. metrics = {group: {score, accuracy, ...}}."""
        log = {f"fold/{g}/{k}": v for g, m in metrics.items() for k, v in m.items()}
        log["fold/index"] = step
        self.run.log(log, step=step)
        for g, m in metrics.items():
            for k, v in m.items():
                self.run.summary[f"{fold}/{g}/{k}"] = v
        cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred"))
        cm = cm.reindex(index=range(5), columns=range(5), fill_value=0)
        self.run.log({f"confusion/{fold}": self.wandb.Table(
            columns=["true", *[f"pred_{j}" for j in range(5)]],
            data=[[i, *map(int, cm.loc[i])] for i in range(5)])}, step=step)
        if importance is not None:
            imp = _importance(importance)
            if imp is not None:
                self.run.log({f"importance/{fold}": self.wandb.Table(
                    columns=["feature", "importance"],
                    data=[[str(k), float(v)] for k, v in imp.items()])}, step=step)
                self._imp = getattr(self, "_imp", []) + [imp]

    @_safe
    def finish_cv(self, table, preds, baseline=None):
        """평균·표준편차, 등급 분포, 기준선, 결과 표."""
        stats = table.drop(columns=["fold"]).groupby("group").agg(["mean", "std"])
        for k, agg in stats.columns:
            for g in stats.index:
                self.run.summary[f"{agg}/{g}/{k}"] = float(stats.loc[g, (k, agg)])
        self.run.summary["score"] = float(stats.loc["all", ("score", "mean")])
        dist = pd.DataFrame({
            "pred": preds.label_pred.value_counts(normalize=True),
            "true": preds.label.value_counts(normalize=True)}).reindex(range(5)).fillna(0)
        for i in range(5):
            self.run.summary[f"dist/pred_{i}"] = float(dist.pred[i])
            self.run.summary[f"dist/true_{i}"] = float(dist.true[i])
        by_fold = (preds.groupby("fold").label_pred.value_counts(normalize=True)
                   .unstack(fill_value=0).reindex(columns=range(5), fill_value=0))
        self.run.log({"dist/pred_by_fold": self.wandb.Table(
            columns=["fold", *[f"pred_{j}" for j in range(5)]],
            data=[[f, *map(float, r)] for f, r in by_fold.iterrows()])})
        self.run.log({"cv_table": self.wandb.Table(dataframe=table.round(6))})
        if getattr(self, "_imp", None):
            imp = pd.concat(self._imp, axis=1).mean(axis=1).sort_values(ascending=False)
            self.run.log({"importance/mean": self.wandb.Table(
                columns=["feature", "importance"],
                data=[[str(k), float(v)] for k, v in imp.items()])})
        months = preds.assign(month=pd.to_datetime(preds.date).dt.strftime("%Y-%m"))
        mrows = [[m, int(len(g)), float(score(g.label, g.label_pred)["score"])]
                 for m, g in months.groupby("month")]
        self.run.log({"monthly": self.wandb.Table(columns=["month", "n", "score"], data=mrows)})
        for m, n, sc in mrows:
            self.run.summary[f"month/{m}"] = sc
        main = table[(table.group == "all")].set_index("fold").score
        last = main.index[-1]
        self.run.summary["score_main"] = float(main.iloc[-1])      # 마지막 폴드 = 주 지표 (안 2)
        if baseline:
            for k, v in baseline.items():
                self.run.summary[f"baseline/{k}"] = v
            for key in {k.split("/")[0] for k in baseline}:
                if f"{key}/mean" in baseline:
                    self.run.summary[f"vs_{key}"] = self.run.summary["score"] - baseline[f"{key}/mean"]
                if f"{key}/{last}" in baseline:
                    self.run.summary[f"vs_{key}_{last}"] = float(main.iloc[-1]) - baseline[f"{key}/{last}"]

    @_safe
    def model(self, models, name, metadata):
        """폴드별 모델을 Artifact(type=model) 하나로. save(path) 가 있으면 그걸, 없으면 pickle."""
        art = self.wandb.Artifact(name=_slug(name or self.run.name), type="model",
                                  metadata=_jsonable(metadata))
        with tempfile.TemporaryDirectory() as d:
            for fold, m in models.items():
                p = Path(d) / fold
                if hasattr(m, "save"):
                    m.save(p)
                else:
                    with open(p.with_suffix(".pkl"), "wb") as f:
                        pickle.dump(m, f)
            art.add_dir(d)
            self.run.log_artifact(art)
            art.wait()

    def finish(self):
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as e:
                warnings.warn(f"W&B 종료 실패: {type(e).__name__}: {e}")
            self.run = None


def _importance(x):
    """model.feature_importance 를 Series 로. dict / Series / 호출 가능한 것 다 받음."""
    if callable(x):
        x = x()
    if isinstance(x, pd.Series):
        s = x
    elif isinstance(x, dict):
        s = pd.Series(x)
    else:
        return None
    s = s.astype(float)
    return s / s.sum() if s.sum() else s


def _slug(s):
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in str(s))[:120] or "model"


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
