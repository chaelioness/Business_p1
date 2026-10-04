"""Read-only handoff demonstration; no final model is trained or saved."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src import Dataset
from features import TextFeatureBuilder,build_text_features,BASELINE_FEATURES


def demo(data_dir, cache_dir):
    ds=Dataset(data_dir)
    builder=TextFeatureBuilder.from_index(cache_dir,data_dir=data_dir)
    day=ds.days()[-1]
    features=build_text_features(day,builder)
    print(f'Reference={day.date.date()}, target={day.target.date()}, cutoff={day.cutoff}')
    print(features[['symbol',*BASELINE_FEATURES]].head().to_string(index=False))
    return features


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=ROOT/'artifacts/text_baseline_v1/cache')
    args=parser.parse_args()
    demo(args.data,args.cache)
