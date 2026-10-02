# work/b — B 파트 (정형 데이터: daily · price · earnings · analyst)

담당 민영. 평가는 재은님 폴드(`lab/folds.json`) 기준.

## 팀원이 볼 것 (이것만)

| 파일 | 내용 |
|---|---|
| **[`minyoung_features.md`](minyoung_features.md)** | 고른 피처, 추출 방식, 결측, 선택 이유, 시도한 피처 102개 |
| **[`gapz_rule.py`](gapz_rule.py)** | 전달용 코드. 피처 계산 · 학습 · 예측 한 파일 (numpy, pandas 만) |
| [`check_gapz_rule.py`](check_gapz_rule.py) | `gapz_rule.py` 확인 (피처 값, 누수, A 폴드 점수 재현) |

## B 결론

- **고른 피처: `gap_z`, `ext_range_z` 2개**
  - `gap_z` = 시간외 갭 ÷ 20일 변동성 (방향 정보의 거의 전부)
  - `ext_range_z` = 시간외 고저 폭 ÷ 20일 변동성 (많이 출렁인 갭은 덜 믿음)
- 결측률 둘 다 0.22% (2026-01-30 하루, 원본 시간외 봉 누락)
- **기준 점수: gap_z 규칙 + ext_range_z = 4폴드 평균 0.444** (fold4 0.4995, 처음 보는 종목 0.352)
  - 베이스라인 모델이 이 점수를 넘는지 비교해 주세요.

## 쓰는 법

### 피처만 (모델에 넣을 때)
```python
import sys
sys.path.insert(0, "work/b")                     # work/b 는 패키지가 아니라서 경로로 추가
from gapz_rule import build_gapz
from lab.cv import feature_table

t = feature_table(days, build_gapz, name="gapz_rule")   # date, symbol, gap_z, ext_range_z, label, ret_pct
```
`build_gapz(day)` 는 `symbol, gap_z, ext_range_z` 를 돌려줌. `day.*` 로만 읽어서 cutoff(09:30) 전 데이터만 씀.

### 규칙 그대로 (모델 없이)
```python
import gapz_rule as G

params = G.fit(train_days)              # {"lam", "med", "sd", "a", "b", "train_score"}
G.save(params, "artifacts/gapz_rule.json")

pred = G.predict(day, G.load("artifacts/gapz_rule.json"))   # symbol, label (0~4), day.symbols 전부
```
`src/model.py` 에 붙일 때의 `Model` / `load_model()` 예시는 `gapz_rule.py` 맨 아래 주석에 있음.

### 확인
```bash
uv run python work/b/check_gapz_rule.py
```
피처 값이 실험 표와 같은지(차이 1e-13 이하), score 가 `src.score` 와 같은지, A 폴드 4폴드 평균 0.4437 이 나오는지 출력함.

## 그 밖의 파일 (B 내부용)

| 파일 | 내용 |
|---|---|
| `features_b.py` | B 피처 61개 전체 `build_b(day)`, 추가 후보 11개 `build_b_extra(day)` |
| `exp/` | 피처 실험 코드 r01 ~ r12 와 공용 코드 (`exp/README.md`) |
| `exp/results.md` | 피처 실험 1~12차 전체 기록 |
| `b_baseline.py`, `v1_check.py`, `score_compare.py`, `feature_candidates.md` | 초기 베이스라인·V1 점검·규칙 비교·추가 후보 목록 |
| `test_leak.py`, `test_paste.py` | 누수 검사, `src/model.py` 붙여 넣기 검사 |
| `01~03_b_eda*.ipynb`, `figs/`, `REPORT_b_eda.md` | EDA |
| `cache/` | 실험 중간 결과 (다시 만들 수 있음) |
