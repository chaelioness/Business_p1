"""팀 공용 train / val 나누기. 모든 실험은 이 파일 하나로 나눈다.

    from folds import split, folds, TEST_START, VAL_START, EMBARGO

    tr_mask, val_mask = split(targets)          # 행마다 예측 대상일(target)
    for name, tr_mask, val_mask in folds(targets): ...   # 같은 나누기를 (이름, mask, mask) 로

기준 (2026-10-01 팀 합의)
- 대상일 2026-06-01 이후(73일)는 최종 테스트. 팀 데이터 zip 에도 없고, 여기서도 어디에도 안 씀.
- 그 앞 449일을 train / val 로 한 번 나눔 (홀드아웃).
  val   = 대상일 2026-03-01 ~ 2026-05-31 (최근 3개월, 약 63일). 올해 급등락 비율이 크게 올라 최근 구간으로 평가.
  train = val 시작 직전 EMBARGO 거래일을 비운 나머지 전부 (대상일 ~2026-02-12).
- embargo: 실제 채점은 학습 데이터가 9월 중순에 끝나고 10월을 평가해 2주 남짓이 빈다. 그 간격을 흉내 냄.
- 점수는 val 전체를 한 번에 낸 값이 주 지표. 월별(3·4·5월) 점수를 같이 보고 일관성을 확인한다.
- 제출용 모델은 마지막에 train + val (449일 전부) 로 다시 학습한다.
- EDA·피처 선정처럼 데이터를 보고 "정하는" 작업은 train 구간(대상일 <= train_end) 안에서만.

경계는 모두 예측 대상일(target) 기준이다 (Dataset.split 과 같음). 거래일은 targets 에 나오는 날짜로 센다.
"""

import numpy as np
import pandas as pd

TEST_START = "2026-06-01"
VAL_START = "2026-03-01"
EMBARGO = 10                # 거래일


def _t(targets):
    return pd.to_datetime(pd.Series(targets)).reset_index(drop=True)


def train_end(targets, val_start=VAL_START, embargo=EMBARGO):
    """train 에 들어가는 마지막 대상일."""
    days = np.sort(_t(targets).unique())
    k = np.searchsorted(days, np.datetime64(pd.Timestamp(val_start)))
    return pd.Timestamp(days[max(k - embargo - 1, 0)])


def split(targets, val_start=VAL_START, end=TEST_START, embargo=EMBARGO):
    """(train mask, val mask). TEST_START 이후 행은 둘 다 False."""
    t = _t(targets)
    te = train_end(t, val_start, embargo)
    return (t <= te).values, ((t >= pd.Timestamp(val_start)) & (t < pd.Timestamp(end))).values


def folds(targets=None, dates=None, **kw):
    """split 결과를 [(이름, train mask, val mask)] 로. 여러 폴드를 돌던 코드와 모양을 맞춤."""
    t = targets if targets is not None else dates
    tr, va = split(t, **kw)
    return [(f"VAL {VAL_START[:7]}~", tr, va)]


def val_months(targets):
    """val 행을 월별로 나눠 보고할 때 쓰는 라벨 (예: '2026-03')."""
    return _t(targets).dt.strftime("%Y-%m").values


# ---------------------------------------------------------------- 파일을 나누지 않고 보기

TRAIN_UNTIL_HOUR = pd.Timedelta(hours=16, minutes=1)    # train 마지막 대상일의 정답(16:00 일봉)까지


def train_dataset(folder=None, symbols=None):
    """팀 데이터(449일) 폴더 하나로 train 구간만 보이는 Dataset.

        ds = train_dataset()          # EDA · 피처 선정은 이걸로
        ds.days()                     # 대상일 ~2026-02-12 까지만

    모든 표를 train 마지막 대상일 16:00 까지만 돌려준다 (그날 정답은 보이고, 그날 밤 시간외부터는 안 보임).
    make_split_data.py 로 만든 dataset_train/ 폴더와 같은 결과. 파일을 따로 만들 필요 없음.
    """
    from src import Dataset

    class _TrainView(Dataset):
        cap = None

        def table(self, name, since=None, until=None, **kw):
            hi = self.cap if until is None else min(pd.Timestamp(until), self.cap)
            return super().table(name, since, hi, **kw)

        def reddit(self, subreddit="stocks", since=None, until=None, kind="comments"):
            hi = self.cap if until is None else min(pd.Timestamp(until), self.cap)
            return super().reddit(subreddit, since, hi, kind)

    full = Dataset(folder, symbols=symbols)
    days = pd.Series(full.dates())
    ds = _TrainView(folder, symbols=symbols)
    ds.cap = train_end(days[days >= pd.Timestamp("2024-08-14")]) + TRAIN_UNTIL_HOUR
    return ds


def val_days(ds, val_start=VAL_START, end=TEST_START):
    """평가할 기준일 목록 (대상일이 val 구간인 날). ds 는 팀 데이터 전체(449일) Dataset.

    val 날짜의 피처는 그 전 과거(일봉 130일, 실적 400일 등)가 필요하므로 전체 Dataset 에서 꺼낸다.
    Day 는 그날 cutoff 이후를 보여 주지 않으므로 val 정답이 피처로 새지 않음.
    """
    lo, hi = pd.Timestamp(val_start), pd.Timestamp(end)
    return [d for d in ds.days() if lo <= d.target < hi]
