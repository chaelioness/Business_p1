"""누수 검사가 걸러야 할 것은 거르고, 정상 코드는 통과시키는지 확인. dataset/ 이 있어야 돌아감."""
import pandas as pd
import pytest

from src import Dataset, DATASET
from lab.leak import LeakError, assert_known_before, check_no_leak

pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="dataset/ 없음")


@pytest.fixture(scope="module")
def days():
    ds = Dataset()
    return [ds.day("2025-11-14"), ds.day("2026-02-20")]     # 금요일 포함


def honest(day):
    """Day 메서드만 쓰는 정상 피처."""
    day.daily(days=60); day.price(hours=20); day.news(days=1)
    day.earnings(days=90); day.analyst(days=30)
    day.load(days=3, reddit=["stocks"])
    r = day.reddit("stocks", "posts", hours=24)
    assert "score" not in r.columns
    return pd.DataFrame({"symbol": day.symbols, "label": 2})


def test_honest_passes(days):
    assert check_no_leak(honest, days, verbose=False)


@pytest.mark.parametrize("leaky", [
    lambda day: day.y,                                         # 정답
    lambda day: day.ds.table("news"),                          # 구간 없이 전체
    lambda day: day.ds.table("daily", until=day.target + pd.Timedelta(hours=16)),  # 대상일 종가
    lambda day: day.ds.labels(),                               # 전체 정답
    lambda day: day.ds.next_trading_day(day.target),
    lambda day: day.reddit("stocks", "posts", hours=24)["score"].mean(),
    lambda day: day.reddit("stocks", "comments", hours=24)["n_comments"],
])
def test_leaks_are_caught(days, leaky):
    with pytest.raises(LeakError):
        check_no_leak(leaky, days, verbose=False)


def test_known_before():
    df = pd.DataFrame({"known_at": pd.to_datetime(["2026-02-20 16:00", "2026-02-23 09:30"])})
    assert_known_before(df.iloc[:1], "2026-02-23 09:30")
    with pytest.raises(LeakError):
        assert_known_before(df, "2026-02-23 09:30")
