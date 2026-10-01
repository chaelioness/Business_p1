"""채점 함수가 슬라이드·README 설명과 같은지 확인."""
import numpy as np
import pytest

from src import WEIGHT, score

# 슬라이드 10쪽 가중치 표 (행 정답, 열 예측)
SLIDE_W = [[0, 4, 16, 36, 64],
           [1, 0, 1, 4, 9],
           [0, 0, 0, 0, 0],
           [9, 4, 1, 0, 1],
           [64, 36, 16, 4, 0]]


def test_weight_matches_slide():
    assert (WEIGHT == np.array(SLIDE_W)).all()


def test_slide_example():
    # 슬라이드 11~12쪽: 행 합 0,2,5,0,3 / 열 합 0,0,6,2,2, 벌점 2x4 + 1x16 = 24
    t = [1, 1, 2, 2, 2, 2, 2, 4, 4, 4]
    p = [3, 3, 2, 2, 2, 2, 2, 2, 4, 4]
    t, p = np.array(t), np.array(p)
    o = np.zeros((5, 5)); np.add.at(o, (t, p), 1)
    e = np.outer(o.sum(1), o.sum(0)) / len(t)
    assert (WEIGHT * o).sum() == 24
    assert (WEIGHT * e).sum() == pytest.approx(37.6)
    assert score(t, p)["score"] == pytest.approx(1 - 24 / 37.6)


@pytest.mark.parametrize("const", range(5))
def test_constant_prediction_is_zero(const):
    t = np.random.default_rng(0).choice(5, 1000, p=np.array([.074, .241, .329, .274, .083]) / 1.001)
    assert score(t, np.full_like(t, const))["score"] == pytest.approx(0)


def test_perfect_is_one():
    t = np.random.default_rng(1).integers(0, 5, 500)
    assert score(t, t)["score"] == pytest.approx(1)
