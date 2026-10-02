"""제출 모델(src/model.py): artifacts 만으로 로드, 누수 없음, B 원본 규칙과 같은 예측."""
import pytest

from src import DATASET, Dataset
from src.model import load_model
from lab.leak import check_no_leak
from work.a import gapz_rule

pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="dataset/ 없음")


@pytest.fixture(scope="module")
def days():
    ds = Dataset()
    return [ds.day("2025-11-14"), ds.day("2026-01-16"), ds.day("2026-01-30")]   # 1/30 → 2/2 는 갭 결측일


def test_no_leak(days):
    assert check_no_leak(load_model().predict, days, verbose=False)


def test_same_as_b(days):
    m = load_model()
    for d in days:
        a = m.predict(d).sort_values("symbol").reset_index(drop=True)
        b = gapz_rule.predict(d, m.params).sort_values("symbol").reset_index(drop=True)
        assert a.equals(b)


def test_missing_gap_is_flat(days):
    out = load_model().predict(days[2])
    assert set(out.symbol) == set(days[2].symbols) and (out.label == 2).all()
