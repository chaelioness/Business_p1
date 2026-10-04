# 실행 결과: API 없는 텍스트 baseline v1

**최종 유효 피처 확정이 아니라, 재은님 모델에 붙여 비교할 후보 baseline입니다.**

- TRAIN target: 2024-08-14~2026-02-12, 376일, 50종목, 18,800행.
- API 호출0회/비용0원. 기존 src/model.py·교수님 파일 수정 없음.
- 기본 모델 입력7개, 대체 실험용5개, 품질 확인용7개. 모델 입력은 baseline_features.json만 사용합니다.
- 기본 입력의 결측/0 비율은 다음과 같습니다. 결측은 데이터 가용성 또는 과거 이력 부족이며0으로 일괄 대치하지 않습니다.

| feature_name | missing_rate | zero_rate |
| --- | --- | --- |
| news_log_count_overnight | 0.0957 | 0.0752 |
| news_spike20 | 0.1730 | 0.0000 |
| reddit_log_count_overnight | 0.0000 | 0.4880 |
| reddit_spike20 | 0.0000 | 0.0000 |
| reddit_persistence3 | 0.0000 | 0.8582 |
| text_both_spike | 0.1730 | 0.8149 |
| text_overnight_hours | 0.0000 | 0.0000 |

## EDA 이후 반영한 핵심 수정

기존 EDA의 전체 뉴스 공백과 태그 미관측 문제를 반영했습니다. 연속6시간 이상 전체 소스 기록이 없는 창은NaN이며, 종목 태그가 아직 관측되지 않은 경우도NaN입니다. 미래 데이터나 특정 날짜/종목을 하드코딩하지 않았습니다.

뉴스 spike는 직전20 cutoff 중 유효15개 이상에서만 계산합니다. 현재를 baseline에서 제외하며, 공백을0으로 평균에 넣지 않습니다. 2025년6월의 무기록 구간과 재개 직후를 정상 관심 감소/폭증으로 취급하지 않습니다. 이 품질 규칙은 원인 확정이 아닌 보수적 proxy이며 threshold는 validation에서 별도로 민감도 확인해야 합니다.

Reddit은 일반어 오탐 방지를 위해 uppercase ticker/cashtag 규칙을 적용했고, 최소문서수 조건으로1건의 우연한 언급이 지속 급증으로 분류되는 것을 줄였습니다. 기업명·소문자 언급을 일부 놓치는 한계가 있습니다.

## 수정된 피처의 TRAIN 관계

| feature_name | n_valid | spearman_extreme | spearman_return |
| --- | --- | --- | --- |
| news_log_count_overnight | 17001 | 0.0829 | 0.0009 |
| news_spike20 | 15548 | 0.0521 | -0.0114 |
| reddit_log_count_overnight | 18800 | 0.1449 | 0.0036 |
| reddit_spike20 | 18800 | 0.0437 | -0.0041 |
| reddit_persistence3 | 18800 | 0.1119 | -0.0109 |
| text_both_spike | 15548 | 0.0929 | -0.0143 |
| text_overnight_hours | 18800 | -0.0072 | 0.0256 |

활동량과 extreme 관계는 남지만 부호 있는 수익률과의 관계는 약합니다. 상관만으로 입력 피처의 유용성을 확정하지 않았습니다.

| both_spike | n | extreme_rate | crash_rate | surge_rate |
| --- | --- | --- | --- | --- |
| 0 | 15320 | 0.1430 | 0.0689 | 0.0740 |
| 1 | 228 | 0.4167 | 0.2456 | 0.1711 |

위 both_spike=0은 **둘 다 급증하지 않은 모든 경우(한쪽만 급증 포함)**입니다. 과거 EDA의 “둘 다 spike 아님” 집단과 비교 정의가 다릅니다. 현재 TRAIN 탐색 결과이며 인과/validation 성능이 아닙니다.

## 시간순 임시 진단: 좋은 결과만 골라 보고하지 않음

학습 2024-08-14~2025-10-23(15,000행), 이후 TRAIN holdout 2025-10-24~2026-02-12(3,800행). 공식 공통 validation이 아니고 이미 EDA에 사용한 구간입니다. 임의 랜덤 분할/내부 랜덤 early stopping/하이퍼파라미터 탐색은 하지 않았습니다. 기존 src.data.score를 그대로 사용했습니다.

| model | feature_group | score | accuracy | big_recall |
| --- | --- | --- | --- | --- |
| Neutral | none | 0.0000 | 0.3374 | 0.0000 |
| LogisticRegression | news | 0.0003 | 0.3371 | 0.0000 |
| HistGradientBoosting | news | -0.0143 | 0.3258 | 0.0017 |
| LogisticRegression | reddit | -0.0546 | 0.3389 | 0.0103 |
| HistGradientBoosting | reddit | -0.0962 | 0.3347 | 0.0120 |
| LogisticRegression | combined | -0.0225 | 0.3397 | 0.0239 |
| HistGradientBoosting | combined | -0.0723 | 0.3332 | 0.0342 |

**텍스트 단독으로 상수 보합 score=0을 안정적으로 넘지 못했습니다.** 이 결과를 보고 피처나 모델을 재튜닝하지 않았습니다. EDA의 극단 움직임 관련성이5등급 방향 예측 성공으로 이어진다고 주장하지 않습니다. 이 피처를 반드시 최종 모델에 넣어야 한다는 근거도 아닙니다.

## 재은님이 해야 할 최종 확인

같은 공식 시간순 fold, 같은 정형 모델과 같은 표본에서 (1) 정형-only (2) +뉴스 (3) +Reddit (4) +둘+결합을 비교하세요. 전체 score뿐 아니라 기간별/종목별 score와 label0/4 recall을 함께 봅니다. 개선이 일관적이지 않으면 해당 그룹을 제외합니다. 현재 정형 피처/공통 fold가 없어 **정형 대비 추가 가치와 신규종목 성능은 미측정**입니다.

## 검증

- full_batch_single_day_parity: 376
- raw_api_count_checks: 183
- feature_rows: 18800
- strict_cutoff_raw_api_checks: 4
- no_labels_or_engagement_in_feature_table: True
- source_gap_rows: 650
- unseen_tag_rows: 1235
- gap_and_insufficient_history_masking: True
- original_files_unchanged: 17
- api_calls: 0

별도로 합성 데이터9개 테스트(미래행 불변성, cutoff 경계, 공백 재개, 현재 제외, 새 종목/부분 universe, 금지 컬럼, 캐시 일치)를 실행했습니다. tests.log에 결과가 있습니다.

## 전달 방법

chaemin_text_baseline_handoff.zip을 전달하면 됩니다. 저장소 Finance 루트에 풀면 example/와 artifacts/ 아래 전용 파일만 생깁니다. 원시 데이터·캐시·정답 파일·학습 모델은 zip에 포함하지 않습니다. 최종 제출에서는 features.py의 로직을 model.py 안으로 통합하고 새 평가 데이터로 인덱스를 구성해야 합니다.
