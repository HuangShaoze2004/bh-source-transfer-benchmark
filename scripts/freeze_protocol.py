#!/usr/bin/env python3
"""Freeze a complete protocol based on cost profiles, never outer performance."""
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target=ROOT/'FINAL_EVAL_LOCK.json'
    if target.exists():raise SystemExit('Existing evaluation lock is immutable; inspect/resume instead of refreezing.')
    feasibility=json.loads((ROOT/'artifacts/FEASIBILITY.json').read_text())
    if feasibility['small_fit']['training_fit_diagnostic']['auroc']<.9:raise ValueError('Fit gate failed')
    token=json.loads((ROOT/'data/audit/token_audit.json').read_text())
    if token['full_reaction']['n_invalid']:raise ValueError('Unresolved tokenizer failures')
    data=pd.read_parquet(ROOT/'data/processed/records.parquet').set_index('row_id',drop=False)
    splits=json.loads((ROOT/'data/processed/splits.json').read_text())
    from bh_transfer.data import assert_split_integrity
    assert_split_integrity(data.reset_index(drop=True),splits)
    profile_fold=splits['folds'][feasibility['diagnostic_fold']]
    base_groups=data.loc[profile_fold['inner'][0]['train_ids']].source.nunique()
    profiles=feasibility['throughput']
    step_seconds={k:v['elapsed_seconds']/v['total_steps'] for k,v in profiles.items()}
    inference_seconds=max(v['inference_seconds']/v['inference_rows'] for v in profiles.values())
    estimates=[]
    for epochs in [10,6,3]:
        dev=0.;formal=0.;dev_jobs=0;final_jobs=0
        for part in splits['folds'].values():
            if not part['test_ids']:continue
            for inner in part['inner']:
                ng=data.loc[inner['train_ids']].source.nunique()
                steps=epochs*math.ceil(len(inner['train_ids'])/(8*ng))
                for objective in ['erm','bal','dro']:
                    seconds=steps*step_seconds['dro' if objective=='dro' else 'erm']*ng/base_groups
                    seconds+=(epochs+3)*len(inner['val_ids'])*inference_seconds+8
                    dev+=4*seconds;dev_jobs+=4
            ng=data.loc[part['train_ids']].source.nunique()
            steps=epochs*math.ceil(len(part['train_ids'])/(8*ng))
            for objective in ['erm','bal','dro']:
                seconds=steps*step_seconds['dro' if objective=='dro' else 'erm']*ng/base_groups
                seconds+=2*len(part['test_ids'])*inference_seconds+15
                formal+=3*seconds;final_jobs+=3
        estimates.append({'epochs':epochs,'development_gpu_hours':1.5*dev/3600,
                          'formal_gpu_hours':1.5*formal/3600,'safety_multiplier':1.5,
                          'development_trajectories':dev_jobs,'formal_trajectories':final_jobs})
    selected=next((e for e in estimates if e['development_gpu_hours']<=45 and e['formal_gpu_hours']<=80),None)
    if selected is None:raise RuntimeError('Smallest complete matrix does not fit reserved budget; record real budget limitation')
    artifact_paths=['data/raw/BH_HTE_Curated_Dataset_v20260219.csv','data/processed/records.parquet',
                    'data/processed/features.npz','data/processed/splits.json','data/processed/feature_manifest.json',
                    'data/audit/token_audit.json','requirements.lock.txt','weights/rxnfp/manifest.json',
                    'weights/rxnfp/tokenizer_extension_manifest.json','weights/rxnfp/pytorch_model.bin',
                    'weights/rxnfp/config.json','weights/rxnfp/vocab.txt','weights/rxnfp/extended_vocab.txt',
                    'src/bh_transfer/neural.py','src/bh_transfer/tokenizer.py','src/bh_transfer/data.py',
                    'src/bh_transfer/tabular.py','src/bh_transfer/evaluate.py',
                    'scripts/pipeline.py','scripts/run_neural_job.py','scripts/predict_job.py',
                    'EXPERIMENT_PROTOCOL.md']
    artifact_hashes={name:digest(ROOT/name) for name in artifact_paths}
    models=['R0_RF','R0_HGB','R1_substitute','B','B_S','B_bal','B_bal_S','B_A','B_A_S']
    contrasts=[]
    for candidate in ['B_S','B_A','B_A_S']:
        for comparator in ['B','B_bal','R0_RF','R0_HGB','R1_substitute']:
            contrasts.append({'candidate':candidate,'comparator':comparator})
    contrasts += [{'candidate':'B_A_S','comparator':m} for m in ['B_A','B_S','B_bal_S']]
    contrasts += [{'candidate':'B_bal_S','comparator':'B_bal'},{'candidate':'B_bal','comparator':'B'}]
    lock={'schema_version':1,'status':'FROZEN','evaluation_allowed':True,
          'release_policy':'Evaluator refuses incomplete model/seed/row coverage; only combine after all predictions completed',
          'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'source_commit':'3a44e398d9fed3f08cf396f1153be2960193f581',
          'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'artifact_hashes':artifact_hashes,'models':models,'seeds':[17,42,2026],'minimum_seeds':3,
          'score_type':'probability','primary_label':'Yield>=10',
          'primary_metric':'eligible-source equal-weight macro AUROC after averaging individual-seed AUROCs',
          'folds':[{'fold':f,'source':p['source'],'eligible':p['eligible'],'test_row_ids':p['test_ids']} for f,p in splits['folds'].items()],
          'neural_grid':[{'lr':lr,'weight_decay':wd} for lr,wd in itertools.product([1e-5,3e-5],[.01,.1])],
          'default_grid_index':2,
          'neural_base_config':{'epochs':selected['epochs'],'batch_per_group':8,'microbatch_size':8,
                               'device':'cuda','dro_eta':.01,'dropout':.1,'grad_clip':1.,
                               'eval_batch_size':32,'checkpoint_every_steps':100,'amp':True,
                               'swa_start_fractions':[.5,.75]},
          'tuning_rule':'Each fold independent; shared ordinary/SWA hyperparameters maximize ordinary mean inner-source AUROC; fixed grid order tie; SWA start then selected internally, tie75%; seed42',
          'ordinary_checkpoint_rule':'Mean inner best-step fraction rounded to final outer epoch; run all fixed updates',
          'calibration':'source-equal weighted internal-validation temperature scaling, logT bounded[-3,3]; threshold .5',
          'development_exhaustion_rule':'Only configurations completed for every objective and inner source in a fold enter selection; discard partial configuration for selection. No common complete configuration -> defaults for all objectives. Stop search and continue preplanned formal runs.',
          'budget_estimates':estimates,'selected_budget_estimate':selected,
          'budgets':{'gpu_total':160,'gpu_development':50,'gpu_feasibility':12,'gpu_formal_reserved':80,'cpu_core_hours':500},
          'bootstrap_replicates':5000,'sensitivity_replicates':1000,'bootstrap_seed':20261110,
          'comparisons':contrasts,'acceptance_candidate':'B_A_S',
          'acceptance_comparators':['B','B_bal','R0_RF','R0_HGB','R1_substitute'],
          'combination_comparators':['B_A','B_S','B_bal_S'],
          'secondary_candidates':['B_A','B_S'],'secondary_inference':'Prespecified Holm-corrected contrasts; no uncorrected winner selection',
          'acceptance_thresholds':{'macro_auroc_improvement':.02,'max_any_source_regression':.03,'paired_ci_lower_above':0},
          'sensitivity_plan':{'author_labels':'same primary rows/splits, original author label for training and testing',
                              'dd_original':'add original unresolved-additive source, recompute strict splits; explicitly incomplete conditions',
                              'models':['R1_substitute','B','B_S','B_A','B_A_S'],'seeds':[17,42,2026],
                              'selection':'reuse primary internal configurations, no new hyperparameter search; refit calibration internally if affordable; never choose based on outer performance',
                              'reserved_gpu_hours':30},
          'limitations':['No product input permission established; products excluded',
                         'R1 is author-inspired substitute, not exact author production model',
                         'Heterogeneous yield measurements','Unlabeled pretrained overlap cannot be excluded',
                         'Confidence intervals condition on observed sources; shared-substrate sensitivity mandatory']}
    target.write_text(json.dumps(lock,indent=2,ensure_ascii=False));target.chmod(0o444)
    (ROOT/'artifacts/BUDGET_ESTIMATE.json').write_text(json.dumps({'estimates':estimates,'selected':selected},indent=2))
    state=json.loads((ROOT/'STATE.json').read_text());state['phase']='locked_nested_development';state['evaluation_lock_sha256']=digest(target)
    state['completed']+=['data_audit','input_and_split_checks','cpu_correctness_tests','gpu_fit_and_throughput','final_protocol_lock']
    (ROOT/'STATE.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    print(json.dumps({'frozen':str(target),'sha256':digest(target),'selected':selected},indent=2))


if __name__=='__main__':main()
