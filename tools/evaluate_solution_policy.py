"""Frozen nested walk-forward comparison on a small declared card sample.

Outer outcomes never enter selection. Reused public histories are NOT a pristine
research holdout, nor a historical replay of the text commander / House model.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import signal
import sys
import threading
import time
import tomllib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from numeric_agent.engine import ForecastSession, simulate
from numeric_agent.evaluation import strict_truth, fold_reference, score, aggregate

UNITS = ['t2-F3-conundrum-joint-2005', 't2-F2-whatever-it-takes-2012',
         't2-F1-cpi-glidepath-2023', 't2-F1-ai-mom-2024']
BASE = ('random_walk', 'gbm')
EXPANDED = BASE + ('filtered_volatility',)


def select_prefix(session, origin):
    inner = copy.copy(session)
    inner.raw = session.raw.iloc[:origin].copy()
    inner.panel = session.panel.loc[session.panel.index <= inner.raw.index[-1]].copy()
    inner.audit = None
    inner.cache = {}; inner.events = []; inner._raw_losses = {}
    inner.allowed_methods = EXPANDED
    for method in EXPANDED:
        inner.evaluate(method)
    def choose(methods):
        eligible = [m for m in methods if inner.cache[m]['status'] == 'ok']
        return min(eligible, key=lambda m: inner.cache[m]['normalized_composite']) if eligible else 'm0'
    return inner, {'baseline_policy': choose(BASE), 'expanded_policy': choose(EXPANDED)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    config = {'units': UNITS, 'outer_folds': 2, 'inner_folds': 3, 'draws': 500, 'seed': 42,
              'candidate': 'filtered_volatility', 'recipe': 'zero drift, EWMA .97, warmup 35, standardized joint shocks',
              'promotion_rule': 'Expanded policy lower equal-card mean, no new failures, no card >10% worse; otherwise experimental only.',
              'scope': 'Nested numerical selector, fixed default window and families; reused public histories; no text replay or independent research holdout.',
              'source_sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in [Path(__file__)] + sorted((ROOT/'numeric_agent').glob('*.py'))}}
    (a.out/'config.json').write_text(json.dumps(config, indent=2))
    start = time.monotonic(); stop = threading.Event(); rows = []; excluded = []
    def heartbeat():
        while not stop.wait(10): print(f'RUNNING elapsed={time.monotonic()-start:.0f}s', flush=True)
    def timeout(*_): raise TimeoutError('outer fold exceeded 180s')
    threading.Thread(target=heartbeat, daemon=True).start()
    old = signal.signal(signal.SIGALRM, timeout)
    try:
        with (a.out/'metrics.jsonl').open('w') as output:
            for uid in UNITS:
                u = ROOT/'track2-forecasting-public/units'/uid
                c = tomllib.loads((u/'card.toml').read_text())
                s = ForecastSession(u, u/'card.toml', c['provenance']['data_cutoff'], a.out/uid/'unused.parquet', audit_root=None)
                H = int(s.steps.max()); n = len(s.raw)
                for origin in (n-2*H, n-H):
                    key = f'{uid}:{origin}'; print(f'START {key}', flush=True)
                    train = s.raw.iloc[:origin]; future = s.raw.iloc[origin:origin+H]
                    try:
                        truth = strict_truth(future, s.kind, s.steps)
                        reference, scales = fold_reference(train, s.target, s.steps, c)
                    except ValueError as e:
                        excluded.append({'key': key, 'reason': str(e)}); continue
                    signal.alarm(180)
                    try:
                        inner, choices = select_prefix(s, origin)
                        (a.out/f'{key.replace(":", "_")}_selection.json').write_text(json.dumps(
                            {'choices': choices, 'cache': inner.cache, 'outer_train_end': str(train.index[-1]),
                             'outer_test_start': str(future.index[0]), 'outer_test_end': str(future.index[-1])}, indent=2))
                        predictions = {}
                        for method in dict.fromkeys(['m0', 'random_walk', 'filtered_volatility'] + list(choices.values())):
                            try:
                                draws = reference.draws(key, 500).reshape(500, *s.steps.shape) if method == 'm0' else simulate(
                                    train, s.kind, method, s.steps, 500, 42, s.window, inner.panel, s.context_families)
                                predictions[method] = dict(score(draws, truth, scales, c), status='ok')
                            except Exception as e: predictions[method] = {'status': 'failed', 'normalized_composite': 8., 'error': str(e)}
                        for label, method in dict(m0='m0', random_walk='random_walk', filtered_volatility='filtered_volatility', **choices).items():
                            row = dict(predictions[method], unit=uid, key=key, method=label, selected=method)
                            rows.append(row); output.write(json.dumps(row)+'\n'); output.flush()
                        print(f'DONE {key} choices={choices}', flush=True)
                    finally: signal.alarm(0)
        labels = ['m0', 'random_walk', 'filtered_volatility', 'baseline_policy', 'expanded_policy']
        per_unit = {u: {m: aggregate([r for r in rows if r['unit']==u and r['method']==m])
                        for m in labels} for u in UNITS if any(r['unit']==u for r in rows)}
        overall = {m: float(np.mean([v[m]['mean_normalized_composite'] for v in per_unit.values()])) for m in labels}
        accepted = len(per_unit)==len(UNITS) and not excluded and overall['expanded_policy'] < overall['baseline_policy'] and all(
            v['expanded_policy']['failed'] <= v['baseline_policy']['failed'] and
            v['expanded_policy']['mean_normalized_composite'] <= 1.1*v['baseline_policy']['mean_normalized_composite'] for v in per_unit.values())
        summary = {'accepted': accepted, 'equal_unit_mean': overall, 'per_unit': per_unit, 'excluded': excluded,
                   'elapsed_seconds': time.monotonic()-start}
        (a.out/'summary.json').write_text(json.dumps(summary, indent=2)); print(json.dumps(summary), flush=True)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old); stop.set()


if __name__ == '__main__': main()
