"""Build the complete TRAIN feature handoff; an explicit --data is mandatory."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import Dataset
from features import (prepare_index, BASELINE_FEATURES, OPTIONAL_FEATURES, QUALITY_COLUMNS,
                      NEWS_FEATURES, REDDIT_FEATURES, JOINT_FEATURES, CONTEXT_FEATURES, VERSION)

DEFINITIONS = {
 'news_log_count_overnight': ('news','log1p(고유 URL-symbol 수), [기준일16:00, target09:30). 창에 전체 뉴스 >=6h 공백 또는 cutoff까지 해당 태그 미관측이면 NaN.','큰 종목간 규모 차이 완화; EDA overnight량과 extreme 관련'),
 'news_spike20': ('news','(현재24h count+1)/(이전20거래일 cutoff의 유효24h count 평균+1). 현재 제외, 유효15개 미만 NaN. 전체feed 공백/태그 미관측 창은 baseline에서 제외.','EDA 자기평균 대비 증가 후보; 공백뒤 가짜 급증 억제'),
 'reddit_log_count_overnight': ('reddit','log1p(티커 포함 문서-ID 수), [기준일16:00, cutoff). 같은 문서-symbol 1건; 전체 Reddit >=6h 무기록이면 NaN.','EDA overnight 언급량과 extreme 관련; 0과 source 공백 구별'),
 'reddit_spike20': ('reddit','(현재24h 문서수+1)/(이전20cutoff 중 유효 문서수 평균+1). 현재 제외, 유효15개 이상.','자기 과거 기준 관심 증가, 가격 방향은 미확정'),
 'reddit_persistence3': ('reddit','현재 포함 최근3cutoff에서 spike20>=2 AND count24h>=3인 횟수. 3개 spike 모두 유효해야 계산.','지속 관심; 단1건으로 spike인 희소분모 문제 완화'),
 'text_both_spike': ('joint','news_spike20>=2 & news_count24h>=5 & reddit_spike20>=2 & reddit_count24h>=3이면1, 아니면0. 양쪽 spike가 유효해야 함.','EDA 동시급증 집단 관련; 인과/추가 성능 보장 아님'),
 'text_overnight_hours': ('calendar','기준일16:00부터 target09:30까지의 ET wall-clock 시간. 조기폐장 미반영.','주말/휴일 때문에 count 창 길이가 달라지는 것을 통제'),
 'news_z20': ('news','(현재24h count-유효 과거20평균)/표본표준편차(ddof=1). 유효15개 미만/분산0이면NaN.','spike와 대체 비교용, 기본7개에는 제외'),
 'reddit_z20': ('reddit','(현재24h 문서수-유효 과거20평균)/표본표준편차. 유효15개 미만/분산0이면NaN.','spike와 중복 가능; 옵션'),
 'news_tone_mean_overnight': ('news','가용 overnight 기사들의 유효 tone 평균. 기사0건/피드 공백/미관측태그이면NaN.','방향 관련성이 약해 기본 baseline 제외, 저우선 대조군'),
 'news_tone_negative_ratio_overnight': ('news','유효 overnight tone<0 기사 수 / 유효 tone 기사 수. 분모0/공백이면NaN.','강한 방향 가정하지 않음; 기본 제외'),
 'text_spike_product': ('joint','news_spike20 * reddit_spike20. 둘 중 하나가NaN이면NaN.','text_both_spike와 대체 비교; 기본 제외'),
 'qc_news_gap_24h': ('quality','현재24h 전체 제공 뉴스파일에 연속6h 이상 기록이 없는30분bin run이면1.','수집 누락 확정이 아닌 proxy. 모델 기본입력 제외'),
 'qc_news_gap_overnight': ('quality','overnight 창의 전체feed 연속6h 이상 무기록 여부.','개별종목0건과 구별; 기본입력 제외'),
 'qc_news_symbol_unseen': ('quality','cutoff 이전에 news.symbols에 해당 티커가 한 번도 없으면1.','실제로 뉴스가 없었다는 뜻 아님; 미래에 등장해도 과거값은 바뀌지 않음'),
 'qc_news_history_fraction20': ('quality','직전20 cutoff 중 feed 양호 AND 그 시점까지 symbol 태그 관측된 24h 창 수 /20.','초기/공백복구 상태 확인; 현재 제외'),
 'qc_reddit_gap_24h': ('quality','현재24h 전체 제공 Reddit 문서에 >=6h 연속 무기록이면1.','전체 소스 proxy, 개별 subreddit 중단까지 감지하는 것은 아님'),
 'qc_reddit_gap_overnight': ('quality','overnight 전체 Reddit에서 연속6h 이상 무기록 여부.','모델 기본입력 제외'),
 'qc_reddit_history_fraction20': ('quality','직전20 cutoff 중 전체 Reddit feed 양호인 24h 창 수 /20.','유효15개 미만 spike/z는NaN')
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path, help='TRAIN dataset folder; no default original dataset/')
    p.add_argument('--out', type=Path, default=ROOT/'artifacts/text_baseline_v1')
    p.add_argument('--diagnostic', action='store_true', help='One fixed TRAIN-only temporal holdout, NOT official validation')
    a = p.parse_args()
    data, out = a.data.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ds = Dataset(data)
    days = ds.days()
    if not days:
        raise ValueError('No usable prediction days')
    builder = prepare_index(data, ds.symbols, out/'cache', log=lambda s: print(s, flush=True))
    rows = []
    for i, day in enumerate(days):
        x = builder.build(day, include_optional=True)
        # Labels are deliberately absent from this table.
        rows.append(x.assign(date=day.date, target=day.target, cutoff=day.cutoff))
        if (i+1) % 50 == 0:
            print(f'Features {i+1}/{len(days)} days', flush=True)
    table = pd.concat(rows, ignore_index=True)
    keys = ['symbol','date','target','cutoff']
    table = table[keys+BASELINE_FEATURES+QUALITY_COLUMNS+OPTIONAL_FEATURES]
    assert not table.duplicated(['symbol','target']).any()
    table.to_parquet(out/'train_text_features.parquet', index=False)
    table.to_csv(out/'train_text_features.csv', index=False)
    # y joins only here, AFTER all features have been calculated and saved.
    labels = pd.concat([d.y[['symbol','date','label','ret_pct']].rename(columns={'date':'target'}) for d in days], ignore_index=True)
    labels.to_parquet(out/'train_labels.parquet', index=False)
    dictionary = []
    for name in BASELINE_FEATURES+QUALITY_COLUMNS+OPTIONAL_FEATURES:
        source, definition, reason = DEFINITIONS[name]
        dictionary.append(dict(feature_name=name,group='baseline' if name in BASELINE_FEATURES else 'quality_not_model_input' if name in QUALITY_COLUMNS else 'optional_not_default',source=source,definition=definition,reason=reason,train_missing_rate=table[name].isna().mean()))
    pd.DataFrame(dictionary).to_csv(out/'feature_dictionary.csv', index=False)
    pd.DataFrame([dict(feature_name=n,missing_rate=table[n].isna().mean(),zero_rate=table[n].eq(0).mean(),minimum=table[n].min(),median=table[n].median(),p99=table[n].quantile(.99),maximum=table[n].max()) for n in BASELINE_FEATURES+QUALITY_COLUMNS+OPTIONAL_FEATURES]).to_csv(out/'feature_quality.csv',index=False)
    quality_by_day = table.groupby('target')[QUALITY_COLUMNS].mean().reset_index()
    quality_by_day.to_csv(out/'quality_by_target.csv',index=False)
    table.groupby('symbol')[['qc_news_symbol_unseen','news_spike20','reddit_spike20']].agg({'qc_news_symbol_unseen':'mean','news_spike20':lambda s:s.isna().mean(),'reddit_spike20':lambda s:s.isna().mean()}).rename(columns={'news_spike20':'news_spike20_missing_rate','reddit_spike20':'reddit_spike20_missing_rate'}).reset_index().to_csv(out/'quality_by_symbol.csv',index=False)
    manifest = dict(version=VERSION,scope='TRAIN only; candidate feature baseline, not validated final model',data=str(data),
        target_start=str(table.target.min().date()),target_end=str(table.target.max().date()),reference_start=str(table.date.min().date()),reference_end=str(table.date.max().date()),n_rows=len(table),n_symbols=table.symbol.nunique(),n_days=len(days),baseline_features=BASELINE_FEATURES,optional_features=OPTIONAL_FEATURES,quality_columns_not_model_input=QUALITY_COLUMNS,label_file='train_labels.parquet',feature_file='train_text_features.parquet',join_keys=['symbol','target'],api_calls=0,seconds=round(time.time()-start,2),dependency_note='duckdb for raw indexing; numpy/pandas/pyarrow for cached feature generation')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (out/'baseline_features.json').write_text(json.dumps(BASELINE_FEATURES,indent=2))
    if a.diagnostic:
        from diagnostic import run_diagnostic
        run_diagnostic(table,labels,out)
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
    from package_handoff import write_results, make_handoff
    write_results(out)
    make_handoff(out)


if __name__ == '__main__':
    main()
