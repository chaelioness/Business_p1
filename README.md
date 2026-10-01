# Business_p1

Business Analytics Finance Project — 다음 거래일 방향 예측. 과제 설명은 [docs/project_overview.md](docs/project_overview.md),
교수님 원본 README 는 [docs/course_README.md](docs/course_README.md).

## 시작

```bash
uv sync
uv run python main.py check --split 2026-06-01    # 형식 검사
uv run pytest -q                                  # 채점 함수·누수 검사 테스트
```

`dataset/` 은 공유 드라이브의 zip 을 저장소 루트에 풀어서 씀 (git 에는 안 올림).

| zip | 기준일 | 용도 |
|---|---|---|
| `dataset_train_until_2026-02-12.zip` | 2024-08-13 ~ 2026-02-12 (377일) | EDA. fold4 의 train 과 같음 |
| `dataset_dev_until_2026-05-29.zip` | 2024-08-13 ~ 2026-05-28 (449일) | 피처 만들고 폴드로 검증할 때 |

홀드아웃(2026-06-01 이후)은 공유하지 않음. 모델이 다 정해진 뒤 한 번만 채점함.

## 폴드

[lab/folds.json](lab/folds.json) 하나를 모두 같이 씀. 날짜를 코드에 따로 적지 말 것.

| | train (기준일) | val (기준일) |
|---|---|---|
| fold1 | 2024-08-13 ~ 2025-05-27 | 2025-06-13 ~ 2025-09-09 |
| fold2 | 2024-08-13 ~ 2025-08-21 | 2025-09-10 ~ 2025-12-03 |
| fold3 | 2024-08-13 ~ 2025-11-14 | 2025-12-04 ~ 2026-03-03 |
| fold4 | 2024-08-13 ~ 2026-02-12 | 2026-03-04 ~ 2026-05-28 |
| holdout | 2024-08-13 ~ 2026-05-11 | 2026-05-29 ~ 2026-09-11 |

- train 과 val 사이에 12거래일을 비움. 실제로도 데이터 마지막 정답(9/14)과 평가 시작(10/1) 사이가 12거래일임
- 경계는 예측 대상일(target) 기준으로 셈. 정답이 target 16:00 에 확정되기 때문
- 평가 종목 절반이 처음 보는 종목이라, 10개(`unseen_symbols`)는 학습에서 빼고 val 점수를 따로 봄

```python
from src import Dataset
from lab.folds import load_folds
from lab.cv import run_cv

def fit(train_days, exclude):      # exclude = 학습에서 뺄 종목
    ...
    return model                   # predict(day) 만 있으면 됨

res = run_cv(fit)
print(res.summary())               # 폴드별 score, all / seen / unseen
```

기준선 (`uv run python scripts/run_baselines.py`)

| | fold1 | fold2 | fold3 | fold4 | 평균 |
|---|---|---|---|---|---|
| 전부 보합 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 어제 label 따라가기 | -0.132 | -0.116 | -0.069 | -0.011 | -0.082 |
| 개장 전 갭 하나로 구간 나누기 | 0.181 | 0.199 | 0.147 | 0.287 | 0.204 |

## 피처 규칙 (AI 에 코드 맡길 때도 이 규칙을 같이 붙여 줄 것)

1. 데이터는 `day` 로만 꺼냄 (`day.daily()`, `day.news()` …). `ds.table()` 로 전체를 읽지 않음
2. 피처 함수 모양은 `build(day) -> DataFrame[symbol, 피처...]`, 종목당 한 줄
3. `day.y` 는 predict 안에서 쓰지 않음
4. Reddit `score`, `n_comments` 는 쓰지 않음 (36시간 뒤 값)
5. 종목 이름·ID 를 피처로 쓰지 않음. 처음 보는 종목에서 깨짐. 수익률·비율·롤링 값처럼 종목과 무관한 형태로
6. 정규화·임계값은 train 구간에서만 fit
7. 최종 피처 코드는 `src/model.py` 안으로 들어가야 함. 채점 때 나머지 `src/` 는 원본으로 덮어써짐
8. 만든 피처는 누수 검사를 통과시킬 것

```python
from lab.leak import check_no_leak
check_no_leak(build, days)        # cutoff 이후 조회, day.y, Reddit score 쓰면 LeakError
```
