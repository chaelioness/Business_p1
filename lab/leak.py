"""누수 검사.

    from lab.leak import guard, check_no_leak

    x = build(guard(day))                    # 피처 하나 뽑을 때
    check_no_leak(model.predict, days)       # 여러 날 돌려 보고 실패하면 LeakError

guard(day) 는 Day 를 감싸서 아래를 하면 바로 LeakError 를 냄.

    - cutoff(대상일 09:30) 이후까지 표를 읽음. day.ds.table("news") 처럼 구간 없이
      전체를 읽는 것도 포함
    - day.y 나 기준일 이후의 정답을 봄
    - Reddit score / n_comments 를 씀 (36시간 뒤 값). guard 에서는 이 열이 아예 빠져서
      쓰려고 하면 KeyError 가 나고, check_no_leak 이 그걸 LeakError 로 바꿔 줌

day.daily(), day.news() 처럼 Day 메서드만 쓰면 걸릴 일이 없음.
피처를 표로 미리 계산해 둘 때는 assert_known_before 로 known_at 을 확인함.
"""

import copy

import pandas as pd

from src.data import Dataset, Day, _ts

BANNED_REDDIT = ("score", "n_comments")


class LeakError(AssertionError):
    pass


def _guarded_dataset(ds, day):
    ds.labels()                          # 정답 캐시를 먼저 채워 두고 복사본과 같이 씀
    g = copy.copy(ds)                    # 읽어 둔 파일 핸들과 정답 캐시는 같이 씀
    g.__class__ = _GuardDataset
    g._cutoff = day.cutoff
    g._label_until = day.date + pd.Timedelta(days=1)    # 기준일 종가(16:00)까지만
    return g


class _GuardDataset(Dataset):
    """원래 Dataset 을 복사한 뒤 클래스만 바꿔 씀. 읽기마다 cutoff 를 확인함."""

    def _check(self, what, until):
        if until is None:
            raise LeakError(f"{what}: 구간 끝 없이 전체를 읽음. cutoff {self._cutoff} 까지만 읽어야 함")
        if _ts(until) > self._cutoff:
            raise LeakError(f"{what}: {_ts(until)} 까지 읽음. cutoff 는 {self._cutoff}")

    def table(self, name, since=None, until=None, **kw):
        self._check(f"table('{name}')", until)
        return super().table(name, since, until, **kw)

    def reddit(self, subreddit="stocks", since=None, until=None, kind="comments"):
        self._check(f"reddit('{subreddit}')", until)
        df = super().reddit(subreddit, since, until, kind)
        return df.drop(columns=[c for c in BANNED_REDDIT if c in df.columns])

    def labels(self, since=None, until=None):
        if until is None or _ts(until) > self._label_until:
            raise LeakError(f"labels(): {until} 까지 정답을 봄. "
                            f"{self._label_until:%Y-%m-%d} 전(기준일까지)만 볼 수 있음")
        return super().labels(since, until)

    def next_trading_day(self, date):
        raise LeakError("next_trading_day(): 미래 거래일 목록을 봄")


class GuardDay(Day):
    """정답과 cutoff 이후를 막은 Day."""

    def __init__(self, day):
        self._symbols = list(day.symbols)
        self.ds = _guarded_dataset(day.ds, day)
        self.date, self.target, self.cutoff = day.date, day.target, day.cutoff

    @property
    def symbols(self):
        return list(self._symbols)

    @property
    def y(self):
        raise LeakError("day.y 를 씀. 정답은 학습할 때만 볼 수 있음")


def guard(day) -> GuardDay:
    return day if isinstance(day, GuardDay) else GuardDay(day)


def check_no_leak(fn, days, verbose=True):
    """fn(day) 를 guard 를 씌워 날마다 돌림. 누수가 있으면 LeakError."""
    for day in days:
        try:
            fn(guard(day))
        except KeyError as e:
            if any(c in str(e) for c in BANNED_REDDIT):
                raise LeakError(f"{day.date:%Y-%m-%d}: Reddit {e} 를 씀. 36시간 뒤 값이라 금지") from e
            raise
        except LeakError as e:
            raise LeakError(f"{day.date:%Y-%m-%d}: {e}") from e
    if verbose:
        print(f"누수 검사 통과 {len(days)}일")
    return True


def assert_known_before(df, cutoff, col="known_at"):
    """미리 계산한 표에 cutoff 이후 행이 섞였으면 LeakError."""
    if col not in df.columns or df.empty:
        return
    late = df[df[col] >= _ts(cutoff)]
    if len(late):
        raise LeakError(f"{col} 가 cutoff({cutoff}) 이후인 행 {len(late)}개. "
                        f"가장 늦은 값 {late[col].max()}")

