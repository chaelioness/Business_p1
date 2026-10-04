"""Validate the generated TRAIN artifact against live single-day extraction and raw API."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src import Dataset
from features import TextFeatureBuilder,BASELINE_FEATURES,OPTIONAL_FEATURES,QUALITY_COLUMNS,FORBIDDEN_COLUMNS


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'artifacts/text_baseline_v1');a=p.parse_args()
    ds=Dataset(a.data.resolve());days=ds.days();out=a.out.resolve()
    b=TextFeatureBuilder.from_index(out/'cache',data_dir=a.data)
    table=pd.read_parquet(out/'train_text_features.parquet')
    assert len(table)==sum(len(d.symbols) for d in days)
    assert not table.duplicated(['symbol','target']).any()
    assert not set(table.columns)&FORBIDDEN_COLUMNS
    assert not np.isinf(table[BASELINE_FEATURES+OPTIONAL_FEATURES].to_numpy()).any()
    # Full batch vs the exact same single-day inference function (not a parallel implementation).
    for day in days:
        expected=table[table.target.eq(day.target)].set_index('symbol').sort_index()
        actual=b.build(day,include_optional=True).set_index('symbol').sort_index()
        assert_frame_equal(expected[actual.columns],actual,check_dtype=False,rtol=1e-12,atol=1e-12)
    direct=0
    chosen=[days[0],days[20],days[len(days)//2],days[-1]]
    for day in chosen:
        raw=day.news(since=day.date+pd.Timedelta(hours=16))
        assert (raw.known_at<day.cutoff).all()
        long=raw.assign(symbol=raw.symbols.str.split(',')).explode('symbol')
        current=table[table.target.eq(day.target)].set_index('symbol')
        for symbol,row in current.iterrows():
            if not row.qc_news_gap_overnight and not row.qc_news_symbol_unseen:
                assert np.isclose(row.news_log_count_overnight,np.log1p(long.symbol.eq(symbol).sum()))
                direct+=1
    gap=table.qc_news_gap_24h.eq(1)
    unseen=table.qc_news_symbol_unseen.eq(1)
    assert table.loc[gap|unseen,'news_spike20'].isna().all()
    assert table.loc[table.qc_news_history_fraction20.lt(.75),'news_spike20'].isna().all()
    assert table.loc[unseen,'news_log_count_overnight'].isna().all()
    hashes=ROOT/'artifacts/eda/original_file_hashes.json'
    original_count=None
    if hashes.exists():
        known=json.loads(hashes.read_text());changed=[p for p,h in known.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
        assert not changed,changed
        original_count=len(known)
    result=dict(full_batch_single_day_parity=len(days),raw_api_count_checks=direct,feature_rows=len(table),strict_cutoff_raw_api_checks=len(chosen),no_labels_or_engagement_in_feature_table=True,source_gap_rows=int(gap.sum()),unseen_tag_rows=int(unseen.sum()),gap_and_insufficient_history_masking=True,original_files_unchanged=original_count,api_calls=0)
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
