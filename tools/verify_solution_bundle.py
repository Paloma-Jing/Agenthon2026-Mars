"""One small bundle comparison, plus output-contract checks; no ablation sweep."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import signal
import subprocess
import sys
import threading
import time
import tomllib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from numeric_agent.engine import ForecastSession, METHODS, simulate
from numeric_agent.evaluation import strict_truth, fold_reference, score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    units=['t2-F3-conundrum-joint-2005','t2-F2-whatever-it-takes-2012','t2-F1-cpi-glidepath-2023']
    config={'units':units,'outer_windows_per_card':1,'inner_selection_folds':3,'draws':500,
            'comparison':'previous two-candidate selector vs current three-candidate selector',
            'scope':'Reused historical sample, numerical policy only, no independent gain claim; no ablation or hyperparameter search.',
            'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'numeric_agent').glob('*.py'))}}
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    start=time.monotonic();stop=threading.Event();records=[]
    def heartbeat():
        while not stop.wait(10):print(f'RUNNING elapsed={time.monotonic()-start:.0f}s',flush=True)
    def timeout(*_):raise TimeoutError('verification card exceeded 240 seconds')
    threading.Thread(target=heartbeat,daemon=True).start();previous=signal.signal(signal.SIGALRM,timeout)
    try:
        for uid in units:
            signal.alarm(240);print(f'START {uid}',flush=True)
            unit=ROOT/'track2-forecasting-public/units'/uid
            card=tomllib.loads((unit/'card.toml').read_text())
            full=ForecastSession(unit,unit/'card.toml',card['provenance']['data_cutoff'],args.out/uid/'forecast.parquet',draws=500)
            H=int(full.steps.max());origin=len(full.raw)-H
            inner=copy.copy(full);inner.raw=full.raw.iloc[:origin].copy()
            inner.panel=full.panel.loc[full.panel.index<=inner.raw.index[-1]].copy()
            inner.cache={};inner.events=[];inner.audit=None
            truth=strict_truth(full.raw.iloc[origin:],full.kind,full.steps)
            _,scales=fold_reference(inner.raw,full.target,full.steps,card)
            for method in METHODS:inner.evaluate(method)
            results={}
            for label,menu in [('previous',METHODS[:2]),('current',METHODS)]:
                good=[m for m in menu if inner.cache[m]['status']=='ok']
                if not good:raise ValueError('no historical selectable candidates')
                method=min(good,key=lambda m:inner.cache[m]['normalized_composite'])
                samples=simulate(inner.raw,inner.kind,method,inner.steps,500,42,inner.window,inner.panel,inner.context_families)
                results[label]={'selected':method,**score(samples,truth,scales,card)}
            (args.out/f'{uid}_historical_selection.json').write_text(json.dumps(inner.cache,indent=2))
            # Final export is separately evaluated on full as-of history: never reuse
            # the truncated selection cache as evidence for a different forecast origin.
            full.diagnose()
            for method in ('random_walk','solution_ensemble'): full.evaluate(method)
            if full.cache['solution_ensemble']['status']!='ok':raise ValueError('bundle cannot export')
            full.export('solution_ensemble','Integration check of the combined recipe; not a superiority claim.')
            gate=subprocess.run([sys.executable,str(ROOT/'track2-forecasting-public/scoring/scoring.py'),'score',
                                 '--card',str(unit/'card.toml'),'--forecast',str(full.out)],capture_output=True,text=True,timeout=30)
            if gate.returncode:raise RuntimeError(gate.stderr)
            verdict=json.loads(gate.stdout);assert verdict['admissible'],verdict
            assert sorted(p.name for p in full.out.parent.iterdir())==['forecast.parquet','forecast_meta.json','forecast_rationale.md']
            (args.out/f'{uid}_gates.json').write_text(json.dumps(verdict,indent=2))
            row={'unit':uid,'outer_origin':origin,'results':results,'admissible':True}
            records.append(row);(args.out/'results.json').write_text(json.dumps(records,indent=2))
            print(f'PASS {uid} previous={results["previous"]["normalized_composite"]:.4f} current={results["current"]["normalized_composite"]:.4f}',flush=True)
            signal.alarm(0)
        summary={'records':records,'mean':{label:float(np.mean([r['results'][label]['normalized_composite'] for r in records])) for label in ('previous','current')},'elapsed_seconds':time.monotonic()-start}
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous);stop.set()


if __name__=='__main__':main()
