"""walk-forward 폴드. A 의 폴드 파일이 오면 FOLD_STARTS 한 줄만 바꾸면 된다.

팀 기준: 대상일 2026-06-01 이후(73일)는 최종 테스트용으로 빼 둔다. 여기 폴드는 전부 그 앞
449일 안에서만 만들고, TEST_START 이후 행은 학습·평가 어디에도 들어가지 않는다.

각 폴드: 평가 = [시작일, 다음 시작일), 마지막은 TEST_START 전까지. 학습 = 평가 시작일 앞 전부.
Dataset.split 과 같이 경계는 예측 대상일(target) 기준이다.

임시 폴드(A 폴드 전까지): 올해 급등락 비율이 크게 올라서(작년 10% 안팎 → 올해 20% 안팎)
평가를 최근 구간 위주로 둠. 2026-01 ~ 05 를 두 달씩.
"""

import pandas as pd

TEST_START = "2026-06-01"
FOLD_STARTS = ["2026-01-01", "2026-03-01", "2026-05-01"]


def folds(dates=None, targets=None, starts=FOLD_STARTS, end=TEST_START):
    """(이름, 학습 mask, 평가 mask) 를 돌려준다.

    targets: 행마다 예측 대상일(Series). 없으면 dates 를 쓴다.
    """
    t = pd.to_datetime(pd.Series(targets if targets is not None else dates)).reset_index(drop=True)
    edges = [pd.Timestamp(s) for s in starts] + [pd.Timestamp(end)]
    out = []
    for i, lo in enumerate(edges[:-1]):
        hi = edges[i + 1]
        out.append((f"F{i + 1} {lo:%Y-%m}", (t < lo).values, ((t >= lo) & (t < hi)).values))
    return out
