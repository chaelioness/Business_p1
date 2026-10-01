# eda/ — 금융 데이터 EDA

결론과 해석은 [docs/worklog/2026-10-01_EDA_금융데이터_탐색.md](../docs/worklog/2026-10-01_EDA_금융데이터_탐색.md) 에 있음.
이 폴더는 그 숫자와 그림을 다시 만드는 코드와 산출물.

## 실행

```bash
uv run python -m eda.run_all            # 전체 (약 2분). 패널이 없으면 먼저 만듦
uv run python -m eda.run_all --rebuild  # 패널부터 다시
uv run python -m eda.s03_overnight_gap  # 섹션 하나만
uv run python -m eda.cv_check_ovn_gap   # 핵심 발견의 폴드 CV 확인 (val 구간 필요)
```

## 구간

`common.py` 가 `lab/folds.json` 의 fold4 train 끝(기준일 2026-02-12)을 읽어 **대상일 2024-08-14 ~ 2026-02-13**
만 남기고, 원시 표는 `known_at < 2026-02-13 09:30` 으로 자름. `dataset/` 에 dev·전체 데이터가 풀려 있어도
EDA 에는 val·홀드아웃 구간이 섞이지 않음. (`cv_check_ovn_gap.py` 만 예외로 팀 폴드 CV 절차를 그대로 씀)

## 파일

| 파일 | 내용 |
|---|---|
| `common.py` | 경로, EDA 구간 컷오프, `read()` (구간 밖 차단), 그림·표 저장(`Section`), IC·분위 분석 도우미 |
| `build_panel.py` | (종목, 대상일) 패널 → `cache/eda_panel.parquet`. 일봉·시간외·실적·애널리스트·뉴스 피처 + 결과(y_*) |
| `s01_data_audit.py` | 커버리지, 일봉 vs 시간봉 정합성, 이상 수익률, 시간봉 배치(어느 봉을 cutoff 에 쓸 수 있나), 정보 도착 시각 |
| `s02_target_returns.py` | label 분포(국면·종목·요일), 두꺼운 꼬리, 변동성 군집, 시장 요인, 상관 구조 |
| `s03_overnight_gap.py` | 수익률 = 갭 + 장중 분해, 갭 정의 비교, 갭 구간별 label, 장중 지속/반전 |
| `s04_technical.py` | 일봉 피처 25개의 방향·장중·갭 잔차·크기 IC, 반기별 안정성, 분위 그림 |
| `s05_events_text.py` | 실적(타이밍·서프라이즈·발표 주기), 애널리스트, 뉴스(양·논조), Reddit(활동량·티커 언급) |
| `s06_metric_unseen.py` | 점수식 벌점 분해, 공격성·임계값 지형, 월별 점수 변동, unseen 분포 이동 |
| `cv_check_ovn_gap.py` | 기존 gap 기준선 vs 전일 종가 기준 갭을 같은 절차로 폴드 CV |
| `run_all.py` | 전체 실행 + `out/findings.json` 취합 |

## 산출물 규칙

```
eda/out/<섹션>/fig_*.png      그림
eda/out/<섹션>/tab_*.csv      표 (utf-8-sig, 엑셀에서 바로 열림)
eda/out/<섹션>/findings.json  보고서에 인용한 핵심 수치
eda/out/findings.json         전체 취합
```

패널(`cache/eda_panel.parquet`)은 git 에 안 올림. 코드로 다시 만들 수 있음.

## 주의

- 패널은 EDA 용으로 벡터화해서 만든 것. 모델 피처는 반드시 `build(day)` 로 다시 만들고 `check_no_leak` 를 통과시킬 것.
  `ovn_gap` 은 `cv_check_ovn_gap.build_ovn_gap` 이 그 형태이고, 패널 값과 오차 0 으로 일치하는 것을 확인함.
- `s06` 의 임계값은 EDA 구간 in-sample 값. 그대로 가져다 쓰지 말고 폴드마다 train 에서 다시 고를 것.
