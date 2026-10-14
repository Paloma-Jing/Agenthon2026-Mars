"""Fixed compact/replay/distribution-mixture recipes; separate development and confirmation cards."""
from pathlib import Path
import sys,json,time,threading,hashlib,tomllib,signal
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from numeric_agent.engine import ForecastSession,simulate
from numeric_agent.tree_forecast import simulate_tree,mix_joint_samples
from numeric_agent.evaluation import strict_truth,fold_reference,score,aggregate
from tools.run_backtest import _origins


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--confirmation',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    units=['t2-F3-conundrum-joint-2005','t2-F2-whatever-it-takes-2012','t2-F1-cpi-glidepath-2023']
    methods=['m0','random_walk','core','compact','replay','blend']
    candidate=None
    if a.confirmation:
        candidate=json.loads((a.confirmation/'summary.json').read_text())['candidate']
        units=['t2-F3-hiking-year-curve-2018','t2-F3-brexit-joint-2016','t2-F4-covid-nfp-2020']
        methods=list(dict.fromkeys(['m0','random_walk','core',candidate]))
    config={'units':units,'methods':methods,'max_folds':3,'seed':42,'draws':500,'gap':'one extra maximum horizon between training labels and calibration origins','phase':'confirmation' if a.confirmation else 'development','promotion_rule':'Lower equal-card mean, no new failures, no card >10% worse than core; confirm frozen candidate once on three predeclared other cards','confirmation_units':['t2-F3-hiking-year-curve-2018','t2-F3-brexit-joint-2016','t2-F4-covid-nfp-2020'],'scope':'Development cards reused; confirmation cards predeclared, histories checked for overlap; no sealed outcomes','source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'numeric_agent').glob('*.py'))}}
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
                    cache={}
                    for method in methods:
                        t=time.monotonic();print(f'START {key} {method}',flush=True)
                        r={'unit':uid,'key':key,'train_end':str(train.index[-1]),'test_start':str(future.index[0]),'test_end':str(future.index[-1]),'method':method,'status':'failed','normalized_composite':8.,'scales':scales}
                        signal.alarm(90)
                        try:
                            if method=='m0':draws=d.draws(key)
                            elif method=='random_walk':draws=simulate(train,s.kind,method,s.steps,500,42,s.window,panel,s.context_families)
                            elif method=='blend':draws=mix_joint_samples(cache['core'],cache['random_walk'],42)
                            else:
                                details={};draws=simulate_tree(train,s.kind,s.steps,500,42,s.window,panel,s.context_families,
                                    profile='compact' if method in ('compact','replay') else 'core',validation_gap=H,audit=details,replay=method=='replay')
                                r['fit_audit']=details
                            if draws.ndim==2:draws=draws.reshape(500,*s.steps.shape)
                            cache[method]=draws
                            r.update(score(draws,truth,scales,c),status='ok')
                        except Exception as e:r['error']=str(e)
                        finally:signal.alarm(0)
                        r['elapsed']=round(time.monotonic()-t,3);rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
                        print(f'DONE {key} {method} loss={r["normalized_composite"]:.4f} elapsed={r["elapsed"]}',flush=True)
        summary={uid:{m:aggregate([r for r in rows if r['unit']==uid and r['method']==m]) for m in methods} for uid in units}
        overall={m:float(np.mean([summary[u][m]['mean_normalized_composite'] for u in units])) for m in methods}
        eligible=[m for m in methods if m not in ('m0','random_walk') and all(summary[u][m]['failed']<=summary[u]['core']['failed'] and summary[u][m]['mean_normalized_composite']<=1.1*summary[u]['core']['mean_normalized_composite'] for u in units)]
        best=min(eligible,key=overall.get)
        result={'candidate':best if candidate is None else candidate,'accepted':candidate is not None and candidate in eligible and overall[candidate]<overall['core'],'per_unit':summary,'equal_unit_mean':overall,'excluded':excluded,'elapsed':time.monotonic()-start}
        (a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:stop.set()
if __name__=='__main__':main()
