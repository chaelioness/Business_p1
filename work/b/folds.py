"""walk-forward 폴드. A 의 폴드 파일이 오면 FOLD_STARTS 한 줄만 바꾸면 된다.

팀 기준: 대상일 2026-06-01 이후(73일)는 최종 테스트용으로 빼 둔다. 여기 폴드는 전부 그 앞
449일 안에서만 만들고, TEST_START 이후 행은 학습·평가 어디에도 들어가지 않는다.

각 폴드: 평가 = [시작일, 다음 시작일), 마지막은 TEST_START 전까지.
학습 = 평가 시작일 앞에서 EMBARGO 거래일을 비운 나머지 전부.
Dataset.split 과 같이 경계는 예측 대상일(target) 기준이다.

embargo: 실제 채점은 학습 데이터가 9월 중순에 끝나고 10월을 평가해 2주 남짓이 빈다.
그 간격을 흉내 내려고 평가 시작 직전 EMBARGO 거래일의 대상일을 학습에서 뺀다.

임시 폴드(A 폴드 전까지): 올해 급등락 비율이 크게 올라서(작년 10% 안팎 → 올해 20% 안팎)
평가를 최근 구간 위주로 둠. 2026-01 ~ 05 를 두 달씩 (마지막은 데이터가 5/29 에서 끝나 한 달).
폴드 크기가 달라서 점수는 폴드 평균보다 전체 평가 행을 합쳐 한 번 낸 값(pooled)을 주 지표로 본다.
"""

import numpy as np
import pandas as pd

TEST_START = "2026-06-01"
FOLD_STARTS = ["2026-01-01", "2026-03-01", "2026-05-01"]
EMBARGO = 10                # 거래일


def folds(dates=None, targets=None, starts=FOLD_STARTS, end=TEST_START, embargo=EMBARGO):
    """(이름, 학습 mask, 평가 mask) 를 돌려준다.

    targets: 행마다 예측 대상일(Series). 없으면 dates 를 쓴다.
    거래일은 targets 에 나오는 날짜들로 센다.
    """
    t = pd.to_datetime(pd.Series(targets if targets is not None else dates)).reset_index(drop=True)
    days = np.sort(t.unique())
    edges = [pd.Timestamp(s) for s in starts] + [pd.Timestamp(end)]
    out = []
    for i, lo in enumerate(edges[:-1]):
        hi = edges[i + 1]
        k = np.searchsorted(days, np.datetime64(lo))          # 평가 첫 거래일의 위치
        train_end = pd.Timestamp(days[max(k - embargo, 0)])  # 이 날 전까지만 학습
        out.append((f"F{i + 1} {lo:%Y-%m}", (t < train_end).values, ((t >= lo) & (t < hi)).values))
    return out
