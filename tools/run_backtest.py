"""Historical fixed-model evaluation with public M0 expected-error scales.

Official component functions, failure-inclusive arithmetic means, common data exclusions,
progress/timeout logs and hash-checked resume. Not an independent Agent-policy holdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_common.scoring import crps
from qfbench2_track_forecasting.tail import tail_pinball, DEFAULT_TAIL_METRIC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO = ROOT / 'track2-forecasting-public'

#: The methods evaluated by default. random_walk is always present as the frozen reference.
METHODS = ('random_walk', 'gbm', 'solution_ensemble')


def _component_losses(samples, truth, *, joint='variogram', tail_levels=(0.01, 0.05, 0.95, 0.99),
                      tail_metric='pinball'):
    """Official marginal / joint / tail on flattened (draws, cells) samples and (cells,) truth."""
    s = np.asarray(samples, dtype=np.float64).reshape(samples.shape[0], -1)
    y = np.asarray(truth, dtype=np.float64).ravel()
    marg = crps.crps_marginal(s, y)
    if joint == 'energy':
        jnt = crps.energy_score(s, y)
    else:
        jnt = crps.variogram_score(s, y, p=0.5)
    if tail_metric == 'pinball':
        tail = tail_pinball(s, y, levels=tuple(tail_levels))
    else:
        from qfbench2_track_forecasting.tail import tail_coverage
        tail = tail_coverage(s, y, levels=tuple(tail_levels))
    return marg, jnt, tail


def _coverage_width(samples, truth, levels=(0.5, 0.8, 0.9, 0.95)):
    """Per-level interval coverage and mean width, averaged over cells."""
    s = np.asarray(samples, dtype=np.float64).reshape(samples.shape[0], -1)
    y = np.asarray(truth, dtype=np.float64).ravel()
    out = {}
    for a in levels:
        lo = np.quantile(s, (1 - a) / 2, axis=0)
        hi = np.quantile(s, (1 + a) / 2, axis=0)
        out[f'coverage_{int(a*100)}'] = float(np.mean((y >= lo) & (y <= hi)))
        out[f'width_{int(a*100)}'] = float(np.mean(hi - lo))
    return out


def _origins(n, H, min_origin=70, max_folds=8):
    """Non-overlapping outer folds, newest first, each H steps apart and non-empty."""
    origins = list(range(n - H, min_origin - 1, -H))[::-1]
    return origins[-max_folds:] if len(origins) > max_folds else origins


def _truth(future, kind, steps):
    from numeric_agent.evaluation import strict_truth
    return strict_truth(future,kind,steps)


def backtest_unit(session, methods, *, draws=200, seed=42, max_folds=8, timeout_per_model=120., time_varying=True):
    import signal
    from numeric_agent.engine import simulate
    from numeric_agent.evaluation import fold_reference, score
    H=int(session.steps.max())
    def timeout(*_):raise TimeoutError('model time budget exceeded')
    old_handler=signal.signal(signal.SIGALRM,timeout)
    try:
        for origin in _origins(len(session.raw),H,max_folds=max_folds):
            train=session.raw.iloc[:origin];future=session.raw.iloc[origin:origin+H]
            key=f'{session.card["task"]["id"]}:{origin}'
            try:
                truth=_truth(future,session.kind,session.steps)
                reference,scales=fold_reference(train,session.target,session.steps,session.card)
            except ValueError as e:
                # Common data exclusions are retained separately, never silently dropped.
                yield {'key':key,'origin':origin,'method':'__data__','status':'excluded','reason':str(e)}
                continue
            for method in methods:
                t=time.monotonic();print(f'START origin={origin} method={method}',flush=True)
                rec={'key':key,'origin':origin,'method':method,'status':'failed','normalized_composite':8.,'scales':scales,'cell_count':int(session.steps.size)}
                signal.setitimer(signal.ITIMER_REAL,timeout_per_model)
                try:
                    samples=reference.draws(key,draws).reshape(draws,*session.steps.shape) if method=='m0' else simulate(
                        train,session.kind,method,session.steps,draws=draws,seed=seed,window=session.window,
                        panel=session.panel.loc[session.panel.index<=train.index[-1]],context_families=session.context_families,time_varying=time_varying)
                    rec.update(score(samples,truth,scales,session.card),status='ok')
                    rec.update(_coverage_width(samples,truth))
                except Exception as e:rec.update(error_type=type(e).__name__,error=str(e)[:200])
                finally:signal.setitimer(signal.ITIMER_REAL,0)
                rec['elapsed']=round(time.monotonic()-t,3)
                print(f'DONE origin={origin} method={method} status={rec["status"]} elapsed={rec["elapsed"]}',flush=True)
                yield rec
    finally:signal.signal(signal.SIGALRM,old_handler)


def main():
    import hashlib,threading,tomllib
    p=argparse.ArgumentParser(description='Historical fixed-model comparison with reconstructed M0 scales; not an independent policy evaluation.')
    p.add_argument('--unit',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--draws',type=int,default=500)
    p.add_argument('--max-folds',type=int,default=8);p.add_argument('--methods',nargs='+',default=['m0',*METHODS])
    p.add_argument('--time-varying',choices=['on','off'],default='on');p.add_argument('--resume',action='store_true')
    a=p.parse_args();unit=REPO/'units'/a.unit
    from numeric_agent.engine import ForecastSession
    card=tomllib.loads((unit/'card.toml').read_text());a.out.mkdir(parents=True,exist_ok=True)
    session=ForecastSession(unit,unit/'card.toml',card['provenance']['data_cutoff'],a.out/'scratch/forecast.parquet',draws=a.draws,seed=a.seed)
    files=list((ROOT/'numeric_agent').glob('*.py'))+[Path(__file__)]+list(unit.glob('*.parquet'))+[unit/'card.toml']
    files+=list(unit.glob('forecast_spec.json'))
    config={'unit':a.unit,'seed':a.seed,'draws':a.draws,'max_folds':a.max_folds,'methods':a.methods,'time_varying':a.time_varying,
            'source_input_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)},
            'normalization':'public_M0_expected_error_reconstruction','aggregation':'arithmetic_mean_failure_and_clip_8'}
    config['planned_keys']=[f'{a.unit}:{o}' for o in _origins(len(session.raw),int(session.steps.max()),max_folds=a.max_folds)]
    cp=a.out/'config.json';mp=a.out/'metrics.jsonl';done=set()
    if cp.exists():
        if not a.resume:raise SystemExit('fresh output required, or --resume with identical configuration')
        if json.loads(cp.read_text())!=config:raise SystemExit('resume configuration/source/input mismatch')
        for line in mp.read_text().splitlines() if mp.exists() else []:
            r=json.loads(line);k=(r['key'],r['method'])
            if k in done:raise SystemExit('duplicate existing evaluation key')
            done.add(k)
    else:cp.write_text(json.dumps(config,indent=2))
    stop=threading.Event();start=time.monotonic()
    def heartbeat():
        while not stop.wait(10):print(f'RUNNING backtest elapsed={time.monotonic()-start:.0f}s',flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        with mp.open('a') as f:
            for rec in backtest_unit(session,a.methods,draws=a.draws,seed=a.seed,max_folds=a.max_folds,time_varying=a.time_varying=='on'):
                k=(rec['key'],rec['method'])
                if k in done:continue
                f.write(json.dumps(rec)+'\n');f.flush();done.add(k)
    finally:stop.set()
    print(f'COMPLETE records={len(done)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
