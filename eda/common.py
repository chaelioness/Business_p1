"""EDA 공통: 경로, EDA 구간 컷오프, 그림·표 저장, 통계 도우미.

EDA 는 팀 규칙대로 fold4 train 구간(기준일 ~ 2026-02-12)만 봄. 데이터 폴더에 dev·전체
데이터가 풀려 있어도 여기서 강제로 자름. 날짜는 lab/folds.json 에서 읽음.

    대상일(target) T  : EDA_FIRST_TARGET ~ EDA_LAST_TARGET
    원시 데이터        : known_at < EDA_LAST_TARGET 09:30  (마지막 대상일의 cutoff)
    정답(label)        : T 의 daily ret 만 사용 (16:00 확정)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from src.data import LABELS, RET_CUTS, START, WEIGHT, label_of, score
from src.paths import DATASET, ROOT

EDA = ROOT / "eda"
OUT = EDA / "out"
CACHE = ROOT / "cache"

_folds = json.loads((ROOT / "lab" / "folds.json").read_text(encoding="utf-8"))
TRAIN_END = pd.Timestamp(_folds["folds"][-1]["train"][1])     # 마지막 기준일 2026-02-12
UNSEEN = list(_folds["unseen_symbols"])
FOLDS = _folds["folds"]


def _trading_dates():
    d = pq.read_table(DATASET / "daily.parquet", columns=["date_et"]).to_pandas()
    return np.sort(d.date_et.unique())


_DATES = pd.DatetimeIndex(_trading_dates())
EDA_FIRST_TARGET = _DATES[_DATES > START][0]
EDA_LAST_TARGET = _DATES[_DATES > TRAIN_END][0]                # 2026-02-13
RAW_CUTOFF = EDA_LAST_TARGET + pd.Timedelta(hours=9, minutes=30)

LABEL_NAMES = list(LABELS)
LABEL_COLORS = ["#b2182b", "#ef8a62", "#bdbdbd", "#67a9cf", "#2166ac"]


# ---------------------------------------------------------------- 읽기

def read(name, columns=None, until=None):
    """원시 표를 읽되 known_at < RAW_CUTOFF 로 자름 (EDA 구간 밖은 절대 안 보임).

    until 은 정답을 붙일 때만 씀 (일봉을 마지막 대상일 16:00 까지)."""
    cols = None if columns is None else list(dict.fromkeys([*columns, "known_at"]))
    df = pq.read_table(DATASET / f"{name}.parquet", columns=cols,
                       filters=[("known_at", "<", until or RAW_CUTOFF)]).to_pandas()
    return df


def read_reddit(sub, kind, columns):
    cols = list(dict.fromkeys([*columns, "created_et"]))
    df = pq.read_table(DATASET / "reddit" / f"{sub}.{kind}.parquet", columns=cols).to_pandas()
    return df[df.created_et < RAW_CUTOFF].reset_index(drop=True)


def subreddits():
    return sorted({p.name.split(".")[0] for p in (DATASET / "reddit").glob("*.parquet")})


def panel():
    """build_panel.py 가 만든 (symbol, target) 패널."""
    path = CACHE / "eda_panel.parquet"
    if not path.exists():
        from eda.build_panel import build
        build()
    return pd.read_parquet(path)


# ---------------------------------------------------------------- 저장

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 9,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Malgun Gothic",
    "mathtext.default": "regular",
})


def _plain(v):
    """json 에 넣을 수 있게 numpy 값을 파이썬 값으로 (소수 4자리)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    return v


class Section:
    """섹션 하나의 산출물 폴더. fig/tab 저장과 핵심 수치(findings) 기록."""

    def __init__(self, key):
        self.key = key
        self.dir = OUT / key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.findings = {}

    def fig(self, fig, name):
        path = self.dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig] {path.relative_to(ROOT)}")

    def tab(self, df, name, index=True):
        path = self.dir / f"{name}.csv"
        df.to_csv(path, index=index, encoding="utf-8-sig", float_format="%.5g")
        print(f"  [tab] {path.relative_to(ROOT)}")
        return df

    def note(self, key, value):
        value = _plain(value)
        self.findings[key] = value
        print(f"  [note] {key} = {value}")

    def save(self):
        path = self.dir / "findings.json"
        path.write_text(json.dumps(self.findings, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")


# ---------------------------------------------------------------- 통계

def daily_ic(df, x, y="ret_pct", by="target", method="spearman", min_n=20):
    """날짜별 횡단면 상관(IC). 평균, 표준편차, t값(=mean/sd*sqrt(n)), 양수 비율."""
    def one(g):
        g = g[[x, y]].dropna()
        if len(g) < min_n or g[x].nunique() < 3:
            return np.nan
        return g[x].corr(g[y], method=method)
    ic = df.groupby(by).apply(one, include_groups=False).dropna()
    n = len(ic)
    return {
        "ic_mean": ic.mean(), "ic_sd": ic.std(),
        "t": ic.mean() / ic.std() * np.sqrt(n) if n > 1 and ic.std() > 0 else np.nan,
        "hit": (ic > 0).mean(), "n_days": n,
    }, ic


def pooled_corr(df, x, y="ret_pct"):
    g = df[[x, y]].dropna()
    return stats.spearmanr(g[x], g[y]).statistic if len(g) > 10 else np.nan


def bucket_table(df, x, q=10, y="ret_pct"):
    """x 를 분위로 나눠 평균 수익률, 급하락·급상승 확률, |ret| 평균을 봄."""
    g = df[[x, y, "label"]].dropna().copy()
    g["bucket"] = pd.qcut(g[x].rank(method="first"), q, labels=False) + 1
    lab = g.label.astype(int)
    out = g.assign(crash=lab.eq(0), surge=lab.eq(4), big=lab.isin([0, 4]),
                   down=lab.le(1), up=lab.ge(3), absret=g[y].abs()).groupby("bucket").agg(
        x_mean=(x, "mean"), x_min=(x, "min"), x_max=(x, "max"), n=(y, "size"),
        ret_mean=(y, "mean"), absret_mean=("absret", "mean"),
        p_crash=("crash", "mean"), p_surge=("surge", "mean"), p_big=("big", "mean"),
        p_down=("down", "mean"), p_up=("up", "mean"))
    return out


def label_dist(s):
    v = s.astype(int).value_counts(normalize=True).reindex(range(5), fill_value=0)
    v.index = LABEL_NAMES
    return v


def ols_r2(x, y):
    g = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(g) < 3:
        return np.nan
    return g.x.corr(g.y) ** 2


__all__ = [
    "EDA", "OUT", "CACHE", "TRAIN_END", "UNSEEN", "FOLDS", "EDA_FIRST_TARGET",
    "EDA_LAST_TARGET", "RAW_CUTOFF", "LABEL_NAMES", "LABEL_COLORS", "RET_CUTS", "WEIGHT",
    "read", "read_reddit", "subreddits", "panel", "Section", "daily_ic", "pooled_corr",
    "bucket_table", "label_dist", "ols_r2", "label_of", "score", "plt", "np", "pd", "stats",
]
