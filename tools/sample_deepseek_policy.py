"""Three paired real DeepSeek runs on a single withheld historical window per card.

Previous means the old two-model menu under the same current controller, NOT an
exact dev5 image replay. The API sees only the training prefix and dated eligible
text. No parameter search, no component ablations, no hidden outcomes.
"""
from pathlib import Path
import argparse, hashlib, json, os, signal, subprocess, sys, threading, time, tomllib
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
UNITS=['t2-F3-conundrum-joint-2005','t2-F2-whatever-it-takes-2012','t2-F1-cpi-glidepath-2023']


def run_case(uid,version,out):
    from numeric_agent import engine, commander, agent
    from numeric_agent.__main__ import build_model
    from numeric_agent.evaluation import strict_truth,fold_reference,score
    unit=ROOT/'track2-forecasting-public/units'/uid
    card=tomllib.loads((unit/'card.toml').read_text())
    s=engine.ForecastSession(unit,unit/'card.toml',card['provenance']['data_cutoff'],out/'output/forecast.parquet',draws=500,audit_root=out/'audit')
    H=int(s.steps.max());origin=len(s.raw)-H
    future=s.raw.iloc[origin:origin+H].copy()
    s.raw=s.raw.iloc[:origin].copy();s.panel=s.panel.loc[s.panel.index<=s.raw.index[-1]].copy()
    last=s.raw.index[-1];s.asof=str(last.to_timestamp(how='end').date() if isinstance(last,pd.Period) else last.date())
    truth=strict_truth(future,s.kind,s.steps);reference,scales=fold_reference(s.raw,s.target,s.steps,card)
    if version=='previous':
        menu=('random_walk','gbm');s.allowed_methods=menu
        commander.METHODS=menu;agent.METHODS=menu
    boundary={'origin':origin,'history_end':str(last),'text_cutoff':s.asof,'test_start':str(future.index[0]),'test_end':str(future.index[-1]),'horizon_steps':s.steps.tolist()}
    (out/'boundary.json').write_text(json.dumps(boundary,indent=2))
    row={'unit':uid,'version':version,'status':'failed','normalized_composite':8.,'boundary':boundary}
    t=time.monotonic()
    try:
        if not os.environ.get('DEEPSEEK_API_KEY'):raise RuntimeError('DeepSeek credential unavailable')
        if os.environ.get('MODEL_ENDPOINT') or os.environ.get('MODEL_TOKEN'):raise RuntimeError('House environment must be absent for this experiment')
        model,label=build_model(SimpleNamespace(thinking=False,seed=42,prompt_key=False,model='deepseek-flash'))
        commander.run_commander(s,model,unit/'text',commander_steps=4,use_tools=True)
        executor=agent.build_agent(s,model,max_steps=8)
        try:
            executor.run('Produce a forecast using only the available numeric tools. The commander stage is complete. Inspect diagnostics, compare every allowed model, review evidence, export and finish.')
        except Exception as error:
            if not (s.exported and type(error).__name__ in ('AgentMaxStepsError','AgentParsingError','AgentGenerationError')):raise
        if not s.exported:raise RuntimeError('agent did not export')
        frame=pd.read_parquet(s.out)
        samples=frame.value.to_numpy().reshape(s.draws,*s.steps.shape)
        meta=json.loads((s.out.parent/'forecast_meta.json').read_text())
        row.update(score(samples,truth,scales,card),status='ok',selected=meta['selected_method'],
                   model=label,commander=s.commander,model_audit=meta.get('model_audit'))
    except Exception as error:
        # Same class of operational fallback as deployment, using ONLY the prefix.
        # Never call export_m0 on the original unit: that would see the withheld tail.
        row['primary_failure_type']=type(error).__name__
        try:
            draws=reference.draws(f'{uid}:{origin}',500).reshape(500,*s.steps.shape)
            row.update(score(draws,truth,scales,card),status='fallback',selected='m0_reconstruction',commander=s.commander)
        except Exception as fallback:row['fallback_failure_type']=type(fallback).__name__
    row.update(model_calls=s.model_calls,total_tokens=s.total_tokens,elapsed_seconds=time.monotonic()-t)
    (out/'result.json').write_text(json.dumps(row,indent=2))
    print(f'DONE {uid} {version} status={row["status"]} selected={row.get("selected")} loss={row["normalized_composite"]:.5f}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--unit');p.add_argument('--version',choices=['previous','current'])
    p.add_argument('--prompt-key',action='store_true',help='Read a local API credential without echo or persistence')
    a=p.parse_args()
    if a.prompt_key:
        import getpass
        os.environ['DEEPSEEK_API_KEY']=getpass.getpass('DeepSeek API key (not saved): ')
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    if a.unit:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('case deadline')))
        signal.alarm(300)
        try:run_case(a.unit,a.version,a.out)
        finally:signal.alarm(0)
        return
    cfg={'units':UNITS,'versions':['previous','current'],'model':'deepseek-flash','seed':42,'draws':500,
         'scope':'One outer historical window/card; paired real LLM controllers; previous is two-model menu under current controller, not exact dev5; no independent research holdout, no historical macro release vintages.',
         'failure_policy':'prefix-only M0 operational fallback, explicitly labelled; unrecoverable failure=8',
         'hashes':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'numeric_agent').glob('*.py'))+[Path(__file__).resolve()]}}
    (a.out/'config.json').write_text(json.dumps(cfg,indent=2));rows=[];start=time.monotonic()
    # Fail before launching any paired runs when credentials / route are unusable.
    # Never log API exception bodies: authentication errors can echo key fragments.
    try:
        from openai import OpenAI
        key=os.environ.get('DEEPSEEK_API_KEY')
        if not key: raise RuntimeError('missing credential')
        client=OpenAI(api_key=key,base_url='https://api.deepseek.com',timeout=20.,max_retries=0)
        client.models.list()
    except Exception as error:
        health={'valid_real_deepseek_comparison':False,'stage':'authentication_preflight',
                'error_type':type(error).__name__,'http_status':getattr(error,'status_code',None)}
        (a.out/'summary.json').write_text(json.dumps(health,indent=2))
        print(json.dumps(health),flush=True);return
    for uid in UNITS:
        for version in ('previous','current'):
            case=a.out/f'{uid}_{version}';log=a.out/f'{uid}_{version}.log'
            print(f'START {uid} {version}',flush=True);t=time.monotonic()
            with log.open('w') as stream:
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--unit',uid,'--version',version,'--out',str(case)],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
                while process.poll() is None:
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:print(f'RUNNING {uid} {version} elapsed={time.monotonic()-t:.0f}s total={time.monotonic()-start:.0f}s',flush=True)
                    if time.monotonic()-t>330:process.kill();process.wait();break
            result=case/'result.json'
            row=json.loads(result.read_text()) if result.exists() else {'unit':uid,'version':version,'status':'failed','normalized_composite':8.,'primary_failure_type':'process_failure'}
            rows.append(row);(a.out/'results.json').write_text(json.dumps(rows,indent=2))
            print(f'RESULT {uid} {version} status={row["status"]} loss={row["normalized_composite"]:.5f}',flush=True)
    summary={'valid_real_deepseek_comparison':all(r['status']=='ok' and (r.get('commander') or {}).get('status')=='accepted' for r in rows),
             'means':{v:float(np.mean([r['normalized_composite'] for r in rows if r['version']==v])) for v in ('previous','current')},
             'statuses':{v:{s:sum(r['version']==v and r['status']==s for r in rows) for s in ('ok','fallback','failed')} for v in ('previous','current')},
             'model_calls':sum(r.get('model_calls',0) for r in rows),'total_tokens':sum(r.get('total_tokens',0) for r in rows),'elapsed_seconds':time.monotonic()-start}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
