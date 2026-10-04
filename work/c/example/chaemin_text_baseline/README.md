# 채민 → 재은: API 없는 텍스트 피처 baseline v1

이 패키지는 **뉴스·Reddit을 종목별 숫자 피처로 변환하는 코드**입니다. 최종 예측 모델이나 검증된 성능 개선 결과가 아닙니다. API 호출·키·과금이 없습니다. 현재 TRAIN 전체에서 실행한 표를 함께 제공합니다.

## 먼저 볼 파일

- `features.py`: 실제 추출 로직. 최종 제출 시 필요한 로직은 재은님이 `src/model.py` 내부로 통합합니다.
- `run.py`: TRAIN 전체 표 생성. 기본 입력 경로를 두지 않아 원본 dataset을 실수로 읽지 않습니다.
- `artifacts/text_baseline_v1/train_text_features.parquet`: 이미 계산한 **label 없는** 피처 표.
- `feature_dictionary.csv`: 정의, 선택 이유, 결측률. `baseline_features.json`: 기본 모델 입력 7개.
- `RESULTS.md`: 실제 생성·진단 결과와 한계.
- `test_features.py`, `validate_real.py`: 시점·공백·신규종목·배치/단일일 일치 검사.

## 채민이 다시 실행하는 방법

현재 컴퓨터에서는 `Finance` 폴더에서 아래 한 줄이면 됩니다. 첫 실행은 TRAIN 원시 파일을 한 번 읽고 작은 이벤트 인덱스를 만들며, 다음 실행은 일치하는 캐시를 재사용합니다.

```bash
.eda-venv/bin/python example/chaemin_text_baseline/run.py --data ../dataset_train --diagnostic
```

`--diagnostic`은 **TRAIN 내부 임시 시간순 진단**만 실행합니다. 빼면 피처 표만 생성합니다. 모델을 저장하거나 최종 제출 파일을 수정하지 않습니다.

다른 환경에서는 교수님 프로젝트 의존성을 설치한 뒤, 추가로 로컬 집계 패키지 `duckdb==1.5.6`이 필요합니다. 유료 서비스가 아닙니다. `requirements.txt`에 필요한 패키지를 적었습니다. 팀에서 uv를 쓴다면 재은님이 `uv add duckdb==1.5.6`으로 의존성을 추가한 뒤 다음처럼 실행할 수 있습니다.

```bash
uv run python example/chaemin_text_baseline/run.py --data 실제_TRAIN_폴더 --diagnostic
uv run python example/chaemin_text_baseline/test_features.py
uv run python example/chaemin_text_baseline/validate_real.py --data 실제_TRAIN_폴더
```

입력 파일·종목 목록·버전이 바뀌면 캐시가 자동으로 혼용되지 않고 오류를 냅니다. 새 데이터는 `--out artifacts/text_baseline_new_data`처럼 **새 출력 경로**를 지정해 다시 계산하세요. 배포 zip에는 큰 로컬 캐시를 넣지 않았습니다.

## 기본 모델 입력: 7개

| 이름 | 역할 | EDA 기반 판단 |
|---|---|---|
| `news_log_count_overnight` | 장 마감 후 뉴스 수의 log1p | 활동량 관련성은 있으나 회사별 coverage 편향 존재 |
| `news_spike20` | 자기 과거 평균 대비 뉴스 증가 | raw count보다 비교 가능; 수집 공백을 먼저 제외 |
| `reddit_log_count_overnight` | 장 마감 후 티커 포함 문서 수의 log1p | 방향보다 extreme 가능성과 관련 |
| `reddit_spike20` | 자기 과거 평균 대비 언급 증가 | 인기 종목의 절대 규모 차이 완화 |
| `reddit_persistence3` | 최근3시점의 관심 급증 횟수 | 단발성과 지속 관심 구별 |
| `text_both_spike` | 뉴스·Reddit 모두 급증했는지 | EDA 결합 패턴을 작은 binary 피처로 표현 |
| `text_overnight_hours` | 장 마감~개장까지 창 길이 | 주말/휴일 노출 시간 통제, 예측 신호 자체로 선정한 것 아님 |

단일 피처·후처리 성능을 보고 여러 차례 튜닝해 고른 목록이 아닙니다. 의미·TRAIN 안정성·중복·측정 가능성으로 정한 **작은 시작 후보**입니다. 공동 validation에서 정형 모델에 추가했을 때 이득이 없으면 제외해야 합니다.

기본7개 외 `news_z20`, `reddit_z20`, `news_tone_mean_overnight`, `news_tone_negative_ratio_overnight`, `text_spike_product`는 옵션5개입니다. z-score는 spike의 **대체군**, 논조는 약한 방향 관계를 확인할 **저우선 대조군**입니다. `qc_*`7개는 품질 확인용이며 **기본 모델 입력에 넣지 않습니다**. 수집 중단 패턴을 시장 예측 신호처럼 학습하는 것을 경계합니다.

## 정확한 계산·결측 규칙

- 모든 창은 `[start, cutoff)`이며 `cutoff = day.target 09:30 ET`입니다. 09:30 자료는 제외합니다.
- Overnight는 `day.date 16:00`부터 cutoff까지입니다. 조기폐장 13시는 별도로 반영하지 않은 정상장 기준 proxy입니다. 실제 경과시간 대신 교수님 코드와 같은 ET wall-clock 계산입니다.
- 뉴스는 `(URL, symbol)`별 최초 `known_at`, Reddit은 파일별 `(문서 ID, symbol)`별 최초 `created_et`만 셉니다. 같은 문서의 반복 티커는1회입니다. 뉴스 URL이 없으면 제목으로 대체하고 둘 다 없으면 개별 행으로 셉니다.
- 원시 이벤트를30분 bin으로 집계하지만, 지원하는 모든 창 경계(00:00/16:00/09:30/24h)는 이 격자에 맞기 때문에 count와 tone 합에 시간 경계 오차가 없습니다. 일반적인 임의 시각 창 API가 아닙니다.
- 뉴스 `symbols`의 개별 태그를 사용하며, 새로운 종목도 인덱스를 그 종목 목록으로 다시 만들면 같은 공식으로 계산됩니다. 종목ID 자체·고정50종목 rank는 입력하지 않습니다.
- 전체 제공 소스에 연속6시간 이상 기록이 없으면 **coverage 불확실**로 판단하고 해당 창의 활동량을NaN 처리합니다. 실제 장애를 확정하는 규칙은 아니며6시간은 고정 엔지니어링 기본값입니다. 짧거나 일부 매체/subreddit만의 공백은 놓칠 수 있습니다.
- 해당 cutoff까지 뉴스에 한 번도 태그가 없던 종목도NaN입니다. AMZN/JNJ/V를 하드코딩하지 않았습니다. 미래에 첫 태그가 생겨도 과거 feature는 달라지지 않습니다.
- 정상적인 소스에서 기존 태그 종목의 뉴스가 없는 것은0입니다. Reddit 미언급도0입니다. 기업명 alias·소문자·문맥을 전부 포착하지는 못하므로0은 관측된 규칙상 언급0입니다.
- spike는 `(현재24h count+1)/(직전20거래일 cutoff의 유효 count 평균+1)`. **현재를 제외**하고 공백/미관측 태그인 과거 창은 평균에서 빼며, **유효15개 미만이면NaN**입니다. 공백이 끝났다고 바로 큰 spike를 출력하지 않습니다.
- z-score는 같은 유효 이력의 표본표준편차(ddof=1)를 씁니다. 분산0이면NaN입니다.
- Reddit 급증은 spike≥2이면서 count≥3, 뉴스 급증은 spike≥2이면서 count≥5입니다. persistence는 현재 포함3개 cutoff 모두 spike가 유효할 때만 계산합니다.
- `tone`은 기사 전체 논조입니다. 기업별 감성으로 확정해서 해석하지 않습니다. tone 통계는 기사0건이면NaN입니다.
- `score`, `n_comments`, `num_comments_reported`, 수집 메타데이터는 원시 projection부터 읽지 않습니다. `day.y`는 피처 함수에서 읽지 않습니다. runner가 모든 피처를 저장한 **후** 학습 정답을 별도 파일에 저장합니다.

### 기존 EDA와 다른 점

EDA 표를 복사한 것이 아닙니다. 이번 버전은 (1) 뉴스 공백/태그 미관측을NaN 처리, (2) 과거20 중 유효15개 이상만 사용, (3) 최초 TRAIN target 이전에 제공된 합법적인 과거 이력도 사용, (4) raw overnight count에 log1p, (5) Reddit 지속성에 최소3건 조건을 추가했습니다. 따라서 기존 EDA 보고서의 수치를 이 피처 표의 성능이라고 그대로 인용하면 안 됩니다.

## 재은: 이미 생성된 TRAIN 표를 정형 피처에 붙이기

```python
import json
import pandas as pd

text = pd.read_parquet('artifacts/text_baseline_v1/train_text_features.parquet')
text_cols = json.load(open('artifacts/text_baseline_v1/baseline_features.json'))

# structured_table과 text 모두 target은 예측 대상일이어야 합니다.
# date는 기준일이므로 이름만 보고 date==target으로 합치지 마세요.
merged = structured_table.merge(
    text[['symbol', 'target', *text_cols]],
    on=['symbol', 'target'], how='left', validate='one_to_one', indicator=True,
)
assert merged['_merge'].eq('both').all(), '피처 표에 없는 대상일/종목: 해당 데이터에서 다시 추출 필요'
merged = merged.drop(columns='_merge')
model_features = structured_feature_names + text_cols
```

- `label`, `ret_pct`, `date`, `target`, `cutoff`, `symbol`, `qc_*`, 옵션 피처를 자동으로 전부 모델에 넣지 마세요. **명시적인 feature list**를 사용합니다.
- NaN을 무조건0으로 바꾸지 않습니다. LightGBM 등의 결측 처리를 쓰거나, imputer/scaler를 **각 fold의 학습 구간에서만 fit**합니다.
- 공통 fold에서 **정형-only → +news → +reddit → +news+reddit+joint**를 같은 모델/행/평가지표로 비교합니다. 마지막 항목에만7개 전부를 사용하면 됩니다. context hours는 모든 비교군에 공통으로 넣는 방식으로 공정하게 비교하세요.
- 공식 fold/정형 피처가 아직 제공되지 않아 이 추가 가치는 이번에 측정하지 않았습니다. 현재 동봉된 diagnostic은 텍스트-only입니다.

## 재은: 단일일 추론 인터페이스

```python
from example.chaemin_text_baseline.features import (
    prepare_index, build_text_features, BASELINE_FEATURES,
)

# 첫 Day에서, 그 평가 환경이 제공하는 dataset으로 한 번만 준비합니다.
# 실제 새 종목을 포함한 universe를 지정하세요.
builder = prepare_index(day.ds.dir, day.ds.symbols)

# 이후 매일 같은 builder를 재사용합니다.
x = build_text_features(day, builder)
X_text = x[BASELINE_FEATURES]
```

`builder.build(day)`는 항상 `day.symbols`에 해당하는 행을 반환합니다. API 요청을 보내지 않습니다. 내부 인덱스에 이후 시각 행이 들어 있더라도 strict cutoff와 과거 이력만 조회합니다. 이를 미래 행 추가/제거 테스트로 확인했습니다. 인덱스에 없는 새 종목·다른 데이터 폴더·원시 파일 변경을 조용히0으로 처리하지 않고 오류를 냅니다.

최종 제출 시에는 `example/`가 남는다고 가정하면 안 됩니다. 재은님이 **features.py의 클래스·상수·보조함수**를 `src/model.py` 안으로 통합하고 의존성도 포함해야 합니다. `load_model()`은 모델 상태만 읽고, 인덱스는 실제 `day`가 들어온 뒤 준비하세요. TRAIN의 날짜별 캐시를 미래 평가일에 조회하는 방식은 금지입니다.

## 비용과 제한

API 비용은0원이며 LLM 호출은 없습니다. 원시 Reddit을 처음 인덱싱하는 CPU·메모리·디스크 비용은 있습니다. DuckDB 집계 메모리 한도는1.2GB이며 이후에는 작은 시각/종목 집계표만 사용합니다. 캐시에는 본문·작성자·금지 인기 지표·정답을 저장하지 않습니다.

이 버전의 목표는 재현 가능한 추가 후보와 안전한 통합 인터페이스입니다. TRAIN에서 극단 움직임과 관련되어도 방향성이 약하므로 score 개선은 보장되지 않습니다. 실제 성능 판단은 팀 공통 validation에서 합니다.
