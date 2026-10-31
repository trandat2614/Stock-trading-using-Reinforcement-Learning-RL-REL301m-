"""Train 10 seed, khóa model theo validation, test và đóng gói báo cáo."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent import fit_model, make_env
from baseline import rollout, run_baseline
from data import ROOT, clone_config, file_hash, load_config, metrics, training_source_hash, write_json
from environment import audit_flexible

SOURCE_FILES=('agent.py','environment.py','data.py','baseline.py','main.py','report.py')
ASSET_COLUMNS=('stage','rank','episode','step','date','decision','action','guard','price','close',
               'price_change_pct','cash','shares','assets','asset_change','fee','penalty',
               'reward','cumulative_reward','is_peak')


def active_config(base, seed=1):
    cfg=clone_config(base)
    # All ten trials share the previously used risk profile. Only the seed varies.
    cfg['RISK']=clone_config(base['RISK_PROFILES'][1])
    cfg['RUN_STAGE']='final_train'
    cfg['TRAIN']['TRACK_ALL_ASSETS']=True
    if cfg.get('BUY_PRICE_MODE')!='low' or cfg.get('FIXED_SELL_PRICE')!='high':
        raise ValueError('Phải giữ mua Low, bán High')
    return cfg


def frozen_history(directory,cfg,prices,seed,start,end,model_dir=None):
    """Load/predict only; model files and running statistics must remain unchanged."""
    model_dir=Path(model_dir) if model_dir is not None else directory/'models'
    model_path=model_dir/'agent.zip';norm_path=model_dir/'normalizer.pkl'
    hashes=[file_hash(model_path),file_hash(norm_path)]
    vector=DummyVecEnv([lambda:make_env(prices,cfg,start,end,evaluation=True)])
    norm=VecNormalize.load(norm_path,vector);norm.training=False;norm.norm_reward=False
    model=MaskablePPO.load(model_path,device=cfg['TRAIN']['DEVICE'])
    before=[np.array(x,copy=True) for x in (norm.obs_rms.mean,norm.obs_rms.var,norm.obs_rms.count,
        norm.ret_rms.mean,norm.ret_rms.var,norm.ret_rms.count)]
    env=make_env(prices,cfg,start,end,evaluation=True)
    def predict(obs,state):
        a,_=model.predict(norm.normalize_obs(obs[None,:]),deterministic=True,
            action_masks=state.action_masks()[None,:])
        return int(np.asarray(a).item())
    try:
        history=rollout(env,predict,seed)
        after=(norm.obs_rms.mean,norm.obs_rms.var,norm.obs_rms.count,
            norm.ret_rms.mean,norm.ret_rms.var,norm.ret_rms.count)
        assert all(np.array_equal(a,b) for a,b in zip(before,after))
        assert hashes==[file_hash(model_path),file_hash(norm_path)]
    finally:
        env.close();norm.close()
    audit_cfg=clone_config(cfg);audit_cfg['DATA'].update(TEST_START=start,TEST_END=end)
    audit=audit_flexible(history,prices,audit_cfg)
    return history,{'training':False,'deterministic':True,'normalizer_unchanged':True,
        'model_sha256':hashes[0],'normalizer_sha256':hashes[1],'audit':audit}


def compact_history(h,prices,stage):
    h=h.copy();h['stage']=stage;h['rank']=-1;h['episode']=1
    h['episode_step']=np.arange(len(h))
    indices=prices.index.get_indexer(pd.to_datetime(h.date));indices[0]-=1
    h['close']=prices.Close.iloc[indices].to_numpy()
    h['close_change_pct']=(h.close.to_numpy()/prices.Close.shift(1).iloc[indices].to_numpy()-1)*100
    h['asset_change']=h.total_assets.diff().fillna(0)
    h['new_episode_peak']=h.total_assets.gt(h.total_assets.cummax().shift(1).fillna(-np.inf))
    return export_columns(h)


def export_columns(h):
    return h.rename(columns={'episode_step':'step','requested_action':'decision','guard_applied':'guard',
        'close_change_pct':'price_change_pct','total_assets':'assets','cum_reward':'cumulative_reward',
        'new_episode_peak':'is_peak'}).reindex(columns=ASSET_COLUMNS)


def peak(h):
    value=h.loc[h.total_assets.idxmax()]
    return {'assets':float(value.total_assets),'date':str(value.date)}


def collect_training(directory):
    frames=[pd.read_csv(p) for p in sorted(directory.glob('train_peak_events_rank_*.csv'))]
    if not frames:raise RuntimeError('Missing actual PPO training log')
    h=pd.concat(frames,ignore_index=True)
    np.testing.assert_allclose(h.total_assets,h.cash+h.shares*h.close,rtol=1e-10,atol=1e-6)
    p=h.loc[h.total_assets.idxmax()]
    return h,{'assets':float(p.total_assets),'date':str(p.date),'rank':int(p['rank']),
        'episode':int(p.episode),'transition':int(p.global_transition)}


def campaign_config(base,prices):
    """Use the existing historical 8-year fit and 2-year validation framework."""
    cfg=active_config(base)
    val_start=base['VALIDATION']['START']
    fit_end=str(prices.index[prices.index<pd.Timestamp(val_start)][-1].date())
    cfg['DATA'].update(TRAIN_START=base['VALIDATION']['FIT_START'],TRAIN_END=fit_end)
    protocol={'fit_start':cfg['DATA']['TRAIN_START'],'fit_end':fit_end,
        'validation_start':val_start,'validation_end':base['DATA']['TRAIN_END'],
        'test_start':base['DATA']['TEST_START'],'test_end':base['DATA']['TEST_END']}
    assert pd.Timestamp(protocol['fit_end'])<pd.Timestamp(val_start)<=pd.Timestamp(protocol['validation_end'])<pd.Timestamp(protocol['test_start'])
    return cfg,protocol


def select_validation_run(runs):
    """Selection never reads the test metric."""
    return max(runs,key=lambda row:(row['validation']['final_assets_close'],
        -row['validation']['max_drawdown'],-row['seed']))


def selection_digest(selection):
    """Canonical JSON checksum, independent of Windows/Linux text newlines."""
    return hashlib.sha256(json.dumps(selection,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def add_run(frame,run):
    frame=frame.copy();frame.insert(0,'run',run)
    return frame


def run(work=ROOT/'_run/optimize10',trials=10):
    from data import read_csv,validate_prices
    work.mkdir(parents=True,exist_ok=True)
    if (work/'logs/result.json').exists():raise FileExistsError('Use a fresh --work directory')
    reference_path=work/'reference.json'
    previous=json.loads((reference_path if reference_path.exists() else ROOT/'logs/result.json').read_text(encoding='utf-8'))
    base=load_config();assert file_hash(ROOT/'config.yaml')==previous['config_sha256']
    prices=read_csv(ROOT/'data/market.csv');validate_prices(prices,base)
    assert file_hash(ROOT/'data/market.csv')==previous['market_sha256']
    cfg,protocol=campaign_config(base,prices)
    cfg['RUN_STAGE']='optimization'
    for folder in ('models','logs'):(work/folder).mkdir(exist_ok=True)
    runs=[];train_frames=[]
    pretest=prices.loc[:protocol['validation_end']].copy()
    for seed in range(trials):
        number=seed+1;name=f'run_{number:02d}'
        directory=work/'fits'/name;model_dir=work/'models'/name;model_dir.mkdir()
        print(f'{datetime.now(timezone.utc).isoformat()} TRAIN {number}/{trials} seed={seed}',flush=True)
        fitted=fit_model(pretest,cfg,protocol['fit_start'],protocol['fit_end'],seed,
            cfg['TRAIN']['TOTAL_TIMESTEPS'],directory,
            validation=(protocol['validation_start'],protocol['validation_end']),oracle_cache=work/'cache')
        assert fitted['actual_timesteps']==cfg['TRAIN']['TOTAL_TIMESTEPS']
        for old,new in [('best.zip','agent.zip'),('best_vecnormalize.pkl','normalizer.pkl')]:
            shutil.copy2(directory/old,model_dir/new)
        validation,val_proof=frozen_history(work,cfg,pretest,seed,
            protocol['validation_start'],protocol['validation_end'],model_dir)
        val_metrics=metrics(validation,cfg)
        np.testing.assert_allclose(val_metrics['final_assets_close'],fitted['validation_metrics']['final_assets_close'],rtol=1e-12)
        training,training_peak=collect_training(directory)
        assert (training.event=='step').sum()==fitted['actual_timesteps']
        train_frames.extend([add_run(export_columns(training),number),
            add_run(compact_history(validation,prices,'validation'),number)])
        bc=json.loads((directory/'bc_provenance.json').read_text(encoding='utf-8'))
        assert all(pd.Timestamp(b)<=pd.Timestamp(protocol['fit_end']) for a,b in bc['trajectories'])
        runs.append({'run':number,'seed':seed,'model_dir':name,'training':fitted,'behavior_cloning':bc,
            'train_optimization_peak':training_peak,'validation':val_metrics,'validation_peak':peak(validation),
            'validation_proof':val_proof})
        write_json(work/'progress.json',{'trained':number,'target':trials,'runs':runs})
    selected=select_validation_run(runs)
    model_hashes={str(p.relative_to(work/'models').as_posix()):file_hash(p) for p in (work/'models').rglob('*') if p.is_file()}
    selection={'run':selected['run'],'seed':selected['seed'],
        'criterion':'validation final assets descending; drawdown ascending; seed ascending',
        'locked_at':datetime.now(timezone.utc).isoformat(),'model_hashes':model_hashes}
    write_json(work/'selection_lock.json',selection)
    selection_file_hash=file_hash(work/'selection_lock.json')
    selection_hash=selection_digest(selection)
    print(f'LOCKED selected run={selected["run"]} before any test; validation={selected["validation"]["final_assets_close"]:.2f}',flush=True)
    train,train_proof=frozen_history(work,cfg,prices.loc[:protocol['fit_end']],selected['seed'],
        protocol['fit_start'],protocol['fit_end'],work/'models'/selected['model_dir'])
    train_frames.append(add_run(compact_history(train,prices,'train_replay'),selected['run']))
    pd.concat(train_frames,ignore_index=True).to_csv(work/'logs/assets_train.csv',index=False)
    del train_frames
    test_frames=[];test_started_at=datetime.now(timezone.utc).isoformat()
    for row in runs:
        print(f'{datetime.now(timezone.utc).isoformat()} TEST {row["run"]}/{trials} load/predict only',flush=True)
        history,proof=frozen_history(work,cfg,prices,row['seed'],protocol['test_start'],protocol['test_end'],
            work/'models'/row['model_dir'])
        row.update(test=metrics(history,cfg),test_peak=peak(history),test_proof=proof)
        test_frames.append(add_run(compact_history(history,prices,'test'),row['run']))
        print(f'RESULT run={row["run"]} assets={row["test"]["final_assets_close"]:.2f}',flush=True)
    pd.concat(test_frames,ignore_index=True).to_csv(work/'logs/assets_test.csv',index=False)
    assert file_hash(work/'selection_lock.json')==selection_file_hash
    assert all(file_hash(work/'models'/name)==digest for name,digest in model_hashes.items())
    assert selected['run']==select_validation_run(runs)['run']
    best_test=max(runs,key=lambda row:row['test']['final_assets_close'])
    baselines={name:metrics(run_baseline(name,prices,cfg,protocol['test_start'],protocol['test_end']),cfg)
        for name in ('buy_hold','sma','momentum')}
    result={'status':'complete','completed_at':datetime.now(timezone.utc).isoformat(),'seed':selected['seed'],
        'selected_run':selected['run'],'best_observed_test_run':best_test['run'],'risk':cfg['RISK'],
        'runs':runs,'selection':selection,'selection_sha256':selection_hash,
        'selection_file_sha256':selection_file_hash,'effective_config':cfg,
        'protocol':dict(protocol,fresh_training_runs=trials,test_performance_runs=trials,
            training_timesteps_per_run=cfg['TRAIN']['TOTAL_TIMESTEPS'],
            total_training_timesteps=sum(x['training']['actual_timesteps'] for x in runs),
            test_started_at=test_started_at,test_previously_used=True,
            test_best_is_exploratory=True,checkpoint_selection='validation only at EVAL_FREQ and final update'),
        'business_rules_unchanged':True,'config_sha256':file_hash(ROOT/'config.yaml'),
        'training_source_hash':training_source_hash(),'market_sha256':file_hash(ROOT/'data/market.csv'),
        'training':selected['training'],'behavior_cloning':selected['behavior_cloning'],
        'train_optimization_peak':selected['train_optimization_peak'],
        'train_replay':metrics(train,cfg),'train_replay_peak':peak(train),'train_proof':train_proof,
        'validation':selected['validation'],'test':selected['test'],'test_peak':selected['test_peak'],
        'test_proof':selected['test_proof'],'baselines':baselines,
        'model_hashes':model_hashes,'asset_logs':{p.name:file_hash(p) for p in (work/'logs').glob('*.csv')},
        'algorithm_changes':['temporal summary: recent mean and direction',
            'side-balanced expert sampling using existing BC_CLASS_POWER',
            'frozen validation checkpoint selection after PPO updates'],
        'training_log_scope':'all transitions of all 10 runs/8 ranks; validation all runs; frozen train replay selected run'}
    write_json(work/'logs/result.json',result)
    print(f'COMPLETE selected run={selected["run"]} test={selected["test"]["final_assets_close"]:.2f}; observed test best run={best_test["run"]} assets={best_test["test"]["final_assets_close"]:.2f}',flush=True)
    return result


def reproduce(directory=ROOT):
    from data import read_csv
    result=json.loads((directory/'logs/result.json').read_text(encoding='utf-8'))
    assert file_hash(ROOT/'config.yaml')==result['config_sha256']
    assert training_source_hash()==result['training_source_hash']
    assert file_hash(ROOT/'data/market.csv')==result['market_sha256']
    for name,digest in result['model_hashes'].items():assert file_hash(directory/'models'/name)==digest
    for name,digest in result['asset_logs'].items():assert file_hash(directory/'logs'/name)==digest
    assert selection_digest(result['selection'])==result['selection_sha256']
    assert result['selected_run']==result['selection']['run']==select_validation_run(result['runs'])['run']
    assert pd.Timestamp(result['selection']['locked_at'])<pd.Timestamp(result['protocol']['test_started_at'])
    cfg=result['effective_config'];prices=read_csv(ROOT/'data/market.csv');p=result['protocol']
    frames={name:pd.read_csv(directory/'logs'/name) for name in ('assets_train.csv','assets_test.csv')}
    for row in result['runs']:
        optimization=frames['assets_train.csv']
        optimization=optimization.loc[(optimization.run==row['run'])&(optimization.stage=='optimization')]
        assert (optimization.step>0).sum()==row['training']['actual_timesteps']
        assert pd.to_datetime(optimization.date).between(p['fit_start'],p['fit_end']).all()
        stages=[('validation','assets_train.csv',p['validation_start'],p['validation_end']),
            ('test','assets_test.csv',p['test_start'],p['test_end'])]
        if row['run']==result['selected_run']:stages.append(('train_replay','assets_train.csv',p['fit_start'],p['fit_end']))
        for stage,file,start,end in stages:
            h,_=frozen_history(directory,cfg,prices,row['seed'],start,end,directory/'models'/row['model_dir'])
            expected=frames[file];expected=expected.loc[(expected.run==row['run'])&(expected.stage==stage)].reset_index(drop=True)
            pd.testing.assert_frame_equal(add_run(compact_history(h,prices,stage),row['run']),expected,
                check_dtype=False,rtol=1e-12,atol=1e-8)
            print(f'PASS run={row["run"]} {stage}: same assets/actions, no learning',flush=True)


def archive_previous(names=('models','logs','output'),label=None):
    """Move previous deliverables outside the project before publishing a rerun."""
    label=label or datetime.now(timezone.utc).strftime('rerun_%Y%m%d_%H%M%S')
    archive=(ROOT.parent/'amzn_rl_legacy_20261031'/label).resolve()
    archive.mkdir(parents=True,exist_ok=False)
    for name in names:
        source=ROOT/name;target=archive/name
        if not source.exists():continue
        assert source.resolve().is_relative_to(ROOT.resolve()) and source.resolve()!=ROOT.resolve()
        assert target.resolve().is_relative_to(archive)
        if target.exists():raise FileExistsError(target)
        shutil.move(str(source),str(target))
    return archive


def publish(work):
    for folder in ('models','logs'):
        target=ROOT/folder
        if target.exists() and any(target.iterdir()):raise FileExistsError(target)
        target.mkdir(exist_ok=True)
    shutil.copytree(work/'models',ROOT/'models',dirs_exist_ok=True)
    for name in ('assets_train.csv','assets_test.csv','result.json','run.log','checks.log'):
        if (work/'logs'/name).exists():shutil.copy2(work/'logs'/name,ROOT/'logs'/name)


def finish(work):
    from report import build
    archive_previous()
    publish(work)
    build(ROOT)
    return package()


def package():
    result=json.loads((ROOT/'logs/result.json').read_text(encoding='utf-8'))
    assert training_source_hash()==result['training_source_hash']
    for name,digest in result['model_hashes'].items():assert file_hash(ROOT/'models'/name)==digest
    for name,digest in result['asset_logs'].items():assert file_hash(ROOT/'logs'/name)==digest
    assert file_hash(ROOT/'config.yaml')==result['config_sha256']
    for name,digest in result.get('source_files',{}).items():assert file_hash(ROOT/name)==digest
    if 'report' in result:assert file_hash(ROOT/result['report']['path'])==result['report']['sha256']
    paths=[ROOT/name for name in (*SOURCE_FILES,'README.md','config.yaml','requirements.txt','pyproject.toml','THIRD_PARTY_NOTICES.md')]
    for folder in ('data','models','logs','output/pdf','tests'):
        paths.extend(sorted(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts))
    destination=ROOT/'output/AMZN_NOP_BAI.zip';destination.parent.mkdir(exist_ok=True)
    expected={f'AMZN_NOP_BAI/{p.relative_to(ROOT).as_posix()}':file_hash(p) for p in paths}
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,arcname=f'AMZN_NOP_BAI/{p.relative_to(ROOT).as_posix()}')
    with zipfile.ZipFile(destination) as z:
        import hashlib
        assert z.testzip() is None and set(z.namelist())==set(expected)
        assert all(hashlib.sha256(z.read(name)).hexdigest()==digest for name,digest in expected.items())
    print(f'PACKAGE {len(paths)} files; {destination.stat().st_size/1024**2:.2f} MB',flush=True)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=ROOT/'_run/optimize10')
    parser.add_argument('--reproduce',action='store_true')
    parser.add_argument('--package',action='store_true')
    parser.add_argument('--run-only',action='store_true',help='Keep results inside --work without replacing the submission')
    args=parser.parse_args()
    if args.reproduce:reproduce(ROOT if args.work==ROOT/'_run/optimize10' else args.work)
    elif args.package:package()
    else:
        logdir=args.work/'logs';logdir.mkdir(parents=True,exist_ok=True)
        if (logdir/'result.json').exists():raise FileExistsError('Use a fresh --work directory')
        print('Ten fresh train/test runs; progress: '+str(logdir/'run.log'),flush=True)
        with (logdir/'run.log').open('w',encoding='utf-8') as stream:
            with redirect_stdout(stream),redirect_stderr(stream):run(args.work)
        if not args.run_only:finish(args.work)
