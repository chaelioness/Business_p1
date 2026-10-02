# work/b/exp — B 피처 실험 (A 폴드 기준)

결과 전체는 [`results.md`](results.md), 최종 정리는 [`../features/minyoung_features.md`](../features/minyoung_features.md).
실험은 경계 후보 분위 .5~.99 로 돌림 (gap_z 규칙 0.438). 최종 전달값은 재은님 기준선과 같은 .1~.99 후보로 다시 낸 것 (0.447).
실행은 저장소 루트에서 `uv run python work/b/exp/<파일>.py`. 결과 csv 는 `work/b/cache/` 에 저장됨.

## 공용 코드

| 파일 | 내용 |
|---|---|
| `common.py` | 경로 설정, `load_table`(b_v2 표), `full_table`(모든 후보 피처), `split`(폴드 train/val), 점수 `fscore`, 규칙 `cut` / `fit_ab`, 붙이는 방식 `apply`(더하기·곱하기·같은방향), `prep` / `one` / `judge`(피처 하나 평가), LightGBM 도우미 |
| `builders.py` | 원본에서 새로 만드는 피처: `build_more`(b_v3), `past_feats`(지난 결과로 만든 피처), `build_raw`(b_v4) |

## 실험 (차수 순)

| 파일 | 차수 | 무엇을 | 결과 |
|---|---|---|---|
| `r01_features.py` | 1차 | LightGBM 에 B_EXTRA 후보 11개를 하나씩 (근거 → 겹침 → 점수) | 채택 0 |
| `r02_decode.py` | 2차 | 급등락·보합을 몇 % 찍을지 (디코딩 격자) | 급등락 34% 가 최적, 0.404 → 0.417 |
| `r03_twostage.py` | 3차 | 방향 모델 + 크기 모델 2단계 | 0.411 ~ 0.419, 갭 규칙보다 낮음 |
| `r04_gapplus.py` | 4차 | 갭 규칙 + α: **gap_z 경계**, 변동성 구간별, 급등락 내리기, 실적일 경계 | **gap_z 규칙 0.438** (4폴드 모두 ↑) |
| `r05_gapz2.py` | 5차 | gap_z 분모 바꾸기, 규칙 × 모델 섞기 | vol20 최고, 섞으면 하락 |
| `r06_gapfeat.py` | 6차 | gap_z 에 피처 하나씩 더하기 (26개) | 채택 0 |
| `r07_gapfeat_all.py` | 7차 | 피처 75개 × 3방식 | 3개 통과, 우연 수준 |
| `r07b_check.py` | 7차 점검 | 섞은 가짜 피처 기준선, 통과한 셋 함께 | 가짜도 통과 → 우연 |
| `r08_multi.py` | 8차 | 여러 개 함께: 차례로 추가 / Ridge / LightGBM 회귀 | 0.385 ~ 0.428 |
| `r09_more.py` | 9차 | 규칙 모양 7가지 + 새 피처 13개 (b_v3, 과거 결과) | 효과 없음 |
| `r10_prior.py` | 10차 | 근거로 미리 정한 6개를 하나씩 쌓기 | **ext_range_z 하나 0.444** |
| `r10b_solo.py` | – | gap_z 없이 피처 하나만 (단독 규칙, 방향·크기 IC) | 방향은 갭 계열뿐 |
| `r11_raw.py` | 11차 | 원본에서 안 쓴 정보 14개 (b_v4: 장중·거래량, 애널리스트·실적 세부) | 효과 없음 |
| `r12_combo.py` | 12차 | 2개 조합 66가지, 3개 조합 220가지 전부 | 0.438 / 0.425 |

## 만드는 순서 (캐시가 없을 때)

1. `r01_features.py table` → 저장소 루트 `cache/b_v2.parquet` (약 4분)
2. `r09_more.py` → 루트 `cache/b_v3.parquet` (원본 피처 표를 만들고 실험)
3. `r11_raw.py` → 루트 `cache/b_v4.parquet` (약 10~15분)
4. 그다음 `r10b_solo.py`, `r12_combo.py` 처럼 `full_table()` 을 쓰는 실험

피처 표는 `lab.cv.feature_table` 로 만들어서 `guard`(누수 방지) 아래에서 계산됨.
