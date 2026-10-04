"""W&B 연동이 run_cv 결과를 바꾸지 않고, wandb 가 없거나 고장 나도 실험이 멈추지 않는지 확인.
서버에는 아무것도 보내지 않음 (가짜 wandb 모듈을 씀). dataset/ 이 있어야 돌아감."""
import sys
import types

import pandas as pd
import pytest

from src import DATASET, Dataset, score
from lab import tracking
from lab.baselines import fit_flat, fit_yesterday
from lab.cv import run_cv
from lab.folds import Fold, load_folds, unseen_symbols

pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="dataset/ 없음")


@pytest.fixture(scope="module")
def ds():
    return Dataset()


@pytest.fixture(scope="module")
def folds(ds):
    """빨리 돌도록 폴드 2개, val 각 5일만."""
    return [Fold(f.name, f.train, f.val[:5]) for f in load_folds(ds)[-2:]]


def manual(fit, folds):
    """연동 전 run_cv 와 같은 계산을 직접 함."""
    unseen = set(unseen_symbols())
    rows = []
    for f in folds:
        m = fit(f.train, sorted(unseen))
        r = pd.concat([d.y[["symbol", "label"]].merge(m.predict(d), on="symbol",
                                                      suffixes=("", "_pred")) for d in f.val])
        for g, part in (("all", r), ("seen", r[~r.symbol.isin(unseen)]),
                        ("unseen", r[r.symbol.isin(unseen)])):
            rows.append((f.name, g, score(part.label.astype(int), part.label_pred)["score"]))
    return rows


def as_rows(res):
    return [(r.fold, r.group, r.score) for r in res.table.itertuples()]


def test_disabled_same_result(ds, folds, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    res = run_cv(fit_yesterday, ds, folds, verbose=False, name="t", config={"a": 1})
    got, want = as_rows(res), manual(fit_yesterday, folds)
    assert [r[:2] for r in got] == [r[:2] for r in want]
    assert [r[2] for r in got] == pytest.approx([r[2] for r in want])


def test_without_wandb_installed(ds, folds, monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    monkeypatch.setitem(sys.modules, "wandb", None)          # import wandb -> ImportError
    res = run_cv(fit_flat, ds, folds, verbose=False, log_model=True)
    assert (res.table.score == 0).all()


def test_no_key_is_off(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking.Path, "home", lambda: tmp_path)
    assert tracking.Tracker(name="x").on is False


# ---------------------------------------------------------------- 가짜 wandb

class FakeRun:
    def __init__(self, **kw):
        self.init = kw
        self.config = kw.get("config", {})
        self.name, self.url = kw.get("name") or "run", "local://fake"
        self.summary, self.logs, self.artifacts = {}, [], []

    def log(self, d, step=None):
        self.logs.append(d)

    def log_artifact(self, a):
        self.artifacts.append(a)

    def finish(self):
        self.finished = True


def fake_wandb(fail_init=False):
    m = types.ModuleType("wandb")
    m.runs = []

    def init(**kw):
        if fail_init:
            raise RuntimeError("network down")
        m.runs.append(FakeRun(**kw))
        return m.runs[-1]

    class Table:
        def __init__(self, columns=None, data=None, dataframe=None):
            self.columns, self.data, self.dataframe = columns, data, dataframe

    class Artifact:
        def __init__(self, name, type, metadata=None):
            self.name, self.type, self.metadata, self.dirs = name, type, metadata, []

        def add_dir(self, d):
            import os
            self.dirs.append(sorted(os.listdir(d)))

        def wait(self):
            pass

    m.init, m.Table, m.Artifact = init, Table, Artifact
    return m


class FlatWithImp:
    feature_importance = {"gap": 3.0, "atr14": 1.0}

    def predict(self, day):
        return pd.DataFrame({"symbol": day.symbols, "label": 2})


def test_logs_with_fake_wandb(ds, folds, monkeypatch):
    w = fake_wandb()
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    monkeypatch.setitem(sys.modules, "wandb", w)
    res = run_cv(lambda tr, ex: FlatWithImp(), ds, folds, verbose=False, name="flat",
                 config={"decode": "argmax"}, tags=["test"], log_model=True)
    run = w.runs[0]
    assert run.init["entity"] == "BI_finance_project" and run.init["project"] == "finance-direction"
    cfg = run.init["config"]
    assert cfg["decode"] == "argmax" and cfg["method"] == "A-folds2"
    assert {"git", "folds_hash", "folds", "fit", "unseen_symbols"} <= set(cfg)
    assert "not-a-real-key" not in repr(cfg) + repr(run.summary)
    assert run.summary["score"] == 0 and run.summary["dist/pred_2"] == 1.0
    assert any(k.startswith("confusion/") for d in run.logs for k in d)
    assert any("importance/mean" in d for d in run.logs)
    assert run.artifacts[0].type == "model" and len(run.artifacts[0].dirs[0]) == 2
    assert run.finished
    assert len(res.table) == 6


def test_fake_wandb_init_failure(ds, folds, monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb(fail_init=True))
    with pytest.warns(UserWarning, match="W&B 시작 실패"):
        res = run_cv(fit_flat, ds, folds, verbose=False)
    assert len(res.table) == 6


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "scratch")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    s = tracking.settings()
    assert s["project"] == "scratch" and s["entity"] == "BI_finance_project"
    assert s["weave_project"] == "finance-llm"
