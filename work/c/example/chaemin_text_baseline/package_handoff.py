"""Make a small handoff ZIP, excluding raw data, event cache and model artifacts."""
from pathlib import Path
import json
import zipfile
import hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent


def markdown_table(df):
    rows=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for r in df.itertuples(index=False,name=None):
        rows.append('| '+' | '.join('—' if pd.isna(v) else f'{v:.4f}' if isinstance(v,float) else str(v) for v in r)+' |')
    return '\n'.join(rows)


def write_results(out):
    out=Path(out);m=json.loads((out/'manifest.json').read_text())
    q=pd.read_csv(out/'feature_quality.csv')
    q=q[q.feature_name.isin(m['baseline_features'])][['feature_name','missing_rate','zero_rate']]
    text=f'''# 실행 결과: API 없는 텍스트 baseline v1

**최종 유효 피처 확정이 아니라, 재은님 모델에 붙여 비교할 후보 baseline입니다.**

- TRAIN target: {m['target_start']}~{m['target_end']}, {m['n_days']}일, {m['n_symbols']}종목, {m['n_rows']:,}행.
- API 호출0회/비용0원. 기존 src/model.py·교수님 파일 수정 없음.
- 기본 모델 입력7개, 대체 실험용5개, 품질 확인용7개. 모델 입력은 baseline_features.json만 사용합니다.
- 기본 입력의 결측/0 비율은 다음과 같습니다. 결측은 데이터 가용성 또는 과거 이력 부족이며0으로 일괄 대치하지 않습니다.

{markdown_table(q)}

## EDA 이후 반영한 핵심 수정

기존 EDA의 전체 뉴스 공백과 태그 미관측 문제를 반영했습니다. 연속6시간 이상 전체 소스 기록이 없는 창은NaN이며, 종목 태그가 아직 관측되지 않은 경우도NaN입니다. 미래 데이터나 특정 날짜/종목을 하드코딩하지 않았습니다.

뉴스 spike는 직전20 cutoff 중 유효15개 이상에서만 계산합니다. 현재를 baseline에서 제외하며, 공백을0으로 평균에 넣지 않습니다. 2025년6월의 무기록 구간과 재개 직후를 정상 관심 감소/폭증으로 취급하지 않습니다. 이 품질 규칙은 원인 확정이 아닌 보수적 proxy이며 threshold는 validation에서 별도로 민감도 확인해야 합니다.

Reddit은 일반어 오탐 방지를 위해 uppercase ticker/cashtag 규칙을 적용했고, 최소문서수 조건으로1건의 우연한 언급이 지속 급증으로 분류되는 것을 줄였습니다. 기업명·소문자 언급을 일부 놓치는 한계가 있습니다.
'''
    if (out/'train_feature_evidence.csv').exists():
        e=pd.read_csv(out/'train_feature_evidence.csv')
        text+='\n## 수정된 피처의 TRAIN 관계\n\n'+markdown_table(e[['feature_name','n_valid','spearman_extreme','spearman_return']])+'\n\n활동량과 extreme 관계는 남지만 부호 있는 수익률과의 관계는 약합니다. 상관만으로 입력 피처의 유용성을 확정하지 않았습니다.\n'
        text+='\n'+markdown_table(pd.read_csv(out/'train_joint_rates.csv'))+'\n\n위 both_spike=0은 **둘 다 급증하지 않은 모든 경우(한쪽만 급증 포함)**입니다. 과거 EDA의 “둘 다 spike 아님” 집단과 비교 정의가 다릅니다. 현재 TRAIN 탐색 결과이며 인과/validation 성능이 아닙니다.\n'
    if (out/'diagnostic_scores.csv').exists():
        c=json.loads((out/'diagnostic_config.json').read_text())
        s=pd.read_csv(out/'diagnostic_scores.csv')
        text+=f'''\n## 시간순 임시 진단: 좋은 결과만 골라 보고하지 않음

학습 {c['train_start']}~{c['train_end']}({c['train_rows']:,}행), 이후 TRAIN holdout {c['holdout_start']}~{c['holdout_end']}({c['holdout_rows']:,}행). 공식 공통 validation이 아니고 이미 EDA에 사용한 구간입니다. 임의 랜덤 분할/내부 랜덤 early stopping/하이퍼파라미터 탐색은 하지 않았습니다. 기존 src.data.score를 그대로 사용했습니다.

{markdown_table(s[['model','feature_group','score','accuracy','big_recall']])}

**텍스트 단독으로 상수 보합 score=0을 안정적으로 넘지 못했습니다.** 이 결과를 보고 피처나 모델을 재튜닝하지 않았습니다. EDA의 극단 움직임 관련성이5등급 방향 예측 성공으로 이어진다고 주장하지 않습니다. 이 피처를 반드시 최종 모델에 넣어야 한다는 근거도 아닙니다.

## 재은님이 해야 할 최종 확인

같은 공식 시간순 fold, 같은 정형 모델과 같은 표본에서 (1) 정형-only (2) +뉴스 (3) +Reddit (4) +둘+결합을 비교하세요. 전체 score뿐 아니라 기간별/종목별 score와 label0/4 recall을 함께 봅니다. 개선이 일관적이지 않으면 해당 그룹을 제외합니다. 현재 정형 피처/공통 fold가 없어 **정형 대비 추가 가치와 신규종목 성능은 미측정**입니다.
'''
    else:
        text+='\nTRAIN-only diagnostic은 이 출력 폴더에서 아직 실행하지 않았습니다. `run.py --data TRAIN경로 --diagnostic`을 사용하면 됩니다. 공식 validation과 구별하세요.\n'
    if (out/'verification.json').exists():
        v=json.loads((out/'verification.json').read_text())
        text+='\n## 검증\n\n'+''.join(f'- {k}: {val}\n' for k,val in v.items())+'\n별도로 합성 데이터9개 테스트(미래행 불변성, cutoff 경계, 공백 재개, 현재 제외, 새 종목/부분 universe, 금지 컬럼, 캐시 일치)를 실행했습니다. tests.log에 결과가 있습니다.\n'
    text+='\n## 전달 방법\n\nchaemin_text_baseline_handoff.zip을 전달하면 됩니다. 저장소 Finance 루트에 풀면 example/와 artifacts/ 아래 전용 파일만 생깁니다. 원시 데이터·캐시·정답 파일·학습 모델은 zip에 포함하지 않습니다. 최종 제출에서는 features.py의 로직을 model.py 안으로 통합하고 새 평가 데이터로 인덱스를 구성해야 합니다.\n'
    (out/'RESULTS.md').write_text(text,encoding='utf-8')


def make_handoff(out):
    out=Path(out)
    wanted=['train_text_features.parquet','train_text_features.csv','feature_dictionary.csv',
            'baseline_features.json','manifest.json','feature_quality.csv','quality_by_target.csv',
            'quality_by_symbol.csv','diagnostic_scores.csv','diagnostic_config.json',
            'train_feature_evidence.csv','train_joint_rates.csv','verification.json','tests.log',
            'RESULTS.md','integration_example.log']
    files=sorted(p for p in CODE.iterdir() if p.is_file() and p.suffix in {'.py','.md','.txt'})
    files += [out/name for name in wanted if (out/name).exists()]
    dest=out/'chaemin_text_baseline_handoff.zip'
    def archive_name(p):
        return str(Path('example/chaemin_text_baseline')/p.name) if p.parent==CODE else str(Path('artifacts/text_baseline_v1')/p.name)
    info={archive_name(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'handoff_file_hashes.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,archive_name(p))
        z.write(out/'handoff_file_hashes.json','artifacts/text_baseline_v1/handoff_file_hashes.json')
        z.writestr('CHAEMIN_HANDOFF.md', '# 채민 텍스트 baseline v1\n\nFinance 저장소 루트에 풀어주세요.\n\n1. example/chaemin_text_baseline/README.md: 사용법과 통합 방법\n2. artifacts/text_baseline_v1/RESULTS.md: 실제 결과와 한계\n3. artifacts/text_baseline_v1/train_text_features.parquet: TRAIN 피처 표\n4. baseline_features.json의7개만 기본 모델 입력에 사용\n\n공식 validation에서 정형 모델 대비 추가 가치 확인이 필요합니다.\n')
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        assert not any('/cache/' in n or n.endswith('train_labels.parquet') for n in z.namelist())
    print(f'Handoff ZIP: {dest} ({dest.stat().st_size:,} bytes)',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'artifacts/text_baseline_v1');a=p.parse_args()
    write_results(a.out);make_handoff(a.out)
