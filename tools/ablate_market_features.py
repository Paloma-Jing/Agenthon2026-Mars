"""Small predeclared feature ablation; no whole-suite run or leaderboard claims."""
from pathlib import Path
import sys,json,time,threading,hashlib,tomllib,signal
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from numeric_agent.engine import ForecastSession,simulate
from numeric_agent.tree_forecast import simulate_tree
from numeric_agent.evaluation import strict_truth,fold_reference,score,aggregate
from tools.run_backtest import _origins


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    units=['t2-F3-conundrum-joint-2005','t2-F2-whatever-it-takes-2012','t2-F1-cpi-glidepath-2023']
    methods=['m0','random_walk','core','market','core_gap','market_gap']
    config={'units':units,'methods':methods,'max_folds':3,'seed':42,'draws':500,'gap':'one extra maximum horizon between training labels and calibration origins','scope':'Reused development sample, not independent holdout','source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'numeric_agent').glob('*.py'))}}
    (a.out/'config.json').write_text(json.dumps(config,indent=2))
    stop=threading.Event();start=time.monotonic()
    def heartbeat():
        while not stop.wait(10):print(f'RUNNING elapsed={time.monotonic()-start:.0f}s',flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def timeout(*_):raise TimeoutError('model exceeded 90s')
    signal.signal(signal.SIGALRM,timeout);rows=[];excluded=[]
    try:
        with (a.out/'metrics.jsonl').open('w') as f:
            for uid in units:
                u=ROOT/'track2-forecasting-public/units'/uid;c=tomllib.loads((u/'card.toml').read_text())
                s=ForecastSession(u,u/'card.toml',c['provenance']['data_cutoff'],a.out/uid/'unused.parquet')
                H=int(s.steps.max())
                for origin in _origins(len(s.raw),H,max_folds=3):
                    train=s.raw.iloc[:origin];future=s.raw.iloc[origin:origin+H];key=f'{uid}:{origin}'
                    try:
                        truth=strict_truth(future,s.kind,s.steps);d,scales=fold_reference(train,s.target,s.steps,c)
                    except ValueError as e:excluded.append({'key':key,'reason':str(e)});continue
                    panel=s.panel.loc[s.panel.index<=train.index[-1]]
                    for method in methods:
                        t=time.monotonic();print(f'START {key} {method}',flush=True)
                        r={'unit':uid,'key':key,'method':method,'status':'failed','normalized_composite':8.,'scales':scales}
                        signal.alarm(90)
                        try:
                            if method=='m0':draws=d.draws(key)
                            elif method=='random_walk':draws=simulate(train,s.kind,method,s.steps,500,42,s.window,panel,s.context_families)
                            else:
                                details={};draws=simulate_tree(train,s.kind,s.steps,500,42,s.window,panel,s.context_families,
                                    profile=method.split('_')[0],validation_gap=H if method.endswith('_gap') else 0,audit=details)
                                r['fit_audit']=details
                            r.update(score(draws,truth,scales,c),status='ok')
                        except Exception as e:r['error']=str(e)
                        finally:signal.alarm(0)
                        r['elapsed']=round(time.monotonic()-t,3);rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
                        print(f'DONE {key} {method} loss={r["normalized_composite"]:.4f} elapsed={r["elapsed"]}',flush=True)
        summary={uid:{m:aggregate([r for r in rows if r['unit']==uid and r['method']==m]) for m in methods} for uid in units}
        overall={m:float(np.mean([summary[u][m]['mean_normalized_composite'] for u in units])) for m in methods}
        result={'per_unit':summary,'equal_unit_mean':overall,'excluded':excluded,'elapsed':time.monotonic()-start}
        (a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:stop.set()
if __name__=='__main__':main()
