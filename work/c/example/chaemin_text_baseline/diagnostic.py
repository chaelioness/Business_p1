"""Fixed TRAIN-only diagnostic, not team validation and not model selection."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src import score
from features import NEWS_FEATURES,REDDIT_FEATURES,JOINT_FEATURES,CONTEXT_FEATURES,BASELINE_FEATURES


def run_diagnostic(features, labels, out):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    out=Path(out)
    data=features.merge(labels,on=['symbol','target'],validate='one_to_one').sort_values(['target','symbol'])
    dates=data.target.drop_duplicates().sort_values()
    if len(dates)<60:
        raise ValueError('Diagnostic requires at least 60 target dates')
    boundary=dates.iloc[int(len(dates)*.8)]
    train=data[data.target<boundary]
    holdout=data[data.target>=boundary]
    assert train.target.max()<holdout.target.min()
    groups={'news':NEWS_FEATURES+CONTEXT_FEATURES,'reddit':REDDIT_FEATURES+CONTEXT_FEATURES,'combined':BASELINE_FEATURES}
    rows=[dict(model='Neutral',feature_group='none',**score(holdout.label,np.full(len(holdout),2)))]
    predictions=[]
    for name,columns in groups.items():
        models={
            'LogisticRegression':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=1,solver='lbfgs',max_iter=1500)),
            'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1,learning_rate=.05,early_stopping=False,random_state=0)
        }
        for model_name,model in models.items():
            print(f'TRAIN-only diagnostic: {model_name} / {name}',flush=True)
            with threadpool_limits(limits=4):
                model.fit(train[columns],train.label.astype(int))
                pred=model.predict(holdout[columns]).astype(int)
            rows.append(dict(model=model_name,feature_group=name,**score(holdout.label,pred)))
            predictions.append(holdout[['symbol','target','label']].assign(model=model_name,feature_group=name,pred=pred))
    pd.DataFrame(rows).to_csv(out/'diagnostic_scores.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'diagnostic_predictions.csv',index=False)
    config=dict(scope='TRAIN-only diagnostic holdout; the period has already been seen in EDA, NOT independent validation',train_start=str(train.target.min().date()),train_end=str(train.target.max().date()),holdout_start=str(holdout.target.min().date()),holdout_end=str(holdout.target.max().date()),train_rows=len(train),holdout_rows=len(holdout),groups=groups,score='src.data.score',decoding='argmax only',tuning=False,structured_features_used=False,no_model_artifact_saved=True,selection_note='Default feature list fixed BEFORE this diagnostic; do not choose best model from this table')
    (out/'diagnostic_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    # Descriptive TRAIN relationships after missing-coverage treatment.
    data['extreme']=data.label.isin([0,4]).astype(int)
    records=[]
    for n in BASELINE_FEATURES:
        g=data.dropna(subset=[n])
        records.append(dict(feature_name=n,n_valid=len(g),missing_rate=data[n].isna().mean(),spearman_extreme=g[n].corr(g.extreme,method='spearman'),spearman_return=g[n].corr(g.ret_pct,method='spearman')))
    pd.DataFrame(records).to_csv(out/'train_feature_evidence.csv',index=False)
    joint=[]
    for flag,g in data.dropna(subset=['text_both_spike']).groupby('text_both_spike'):
        joint.append(dict(both_spike=int(flag),n=len(g),extreme_rate=g.extreme.mean(),crash_rate=g.label.eq(0).mean(),surge_rate=g.label.eq(4).mean()))
    pd.DataFrame(joint).to_csv(out/'train_joint_rates.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'artifacts/text_baseline_v1');a=p.parse_args()
    run_diagnostic(pd.read_parquet(a.out/'train_text_features.parquet'),pd.read_parquet(a.out/'train_labels.parquet'),a.out)
