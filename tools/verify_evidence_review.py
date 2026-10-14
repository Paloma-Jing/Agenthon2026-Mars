"""Three-card compatibility sample with evidence review, official gates and heartbeat."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from numeric_agent.engine import ForecastSession


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    units = ['t2-F3-conundrum-joint-2005', 't2-F2-whatever-it-takes-2012', 't2-F1-cpi-glidepath-2023']
    config = {'units': units, 'scope': 'Reused selection cards; compatibility only, no independent gain claim.',
              'source_sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in sorted((ROOT / 'numeric_agent').glob('*.py'))}}
    (a.out / 'config.json').write_text(json.dumps(config, indent=2))
    start = time.monotonic(); stop = threading.Event(); results = []
    def heartbeat():
        while not stop.wait(10): print(f'RUNNING elapsed={time.monotonic()-start:.0f}s', flush=True)
    def timeout(*_): raise TimeoutError('card exceeded 120s')
    threading.Thread(target=heartbeat, daemon=True).start()
    old = signal.signal(signal.SIGALRM, timeout)
    try:
        for uid in units:
            print(f'START {uid}', flush=True); signal.alarm(120)
            u = ROOT / 'track2-forecasting-public/units' / uid
            out = a.out / uid; out.mkdir()
            c = tomllib.loads((u / 'card.toml').read_text())
            s = ForecastSession(u, u / 'card.toml', c['provenance']['data_cutoff'], out / 'forecast.parquet')
            s.diagnose()
            for m in s.allowed_methods: s.evaluate(m)
            report = s.review_evidence()
            assert report['comparable'], report
            s.export(report['recommended_method'], 'Selected by the existing mean-loss criterion after paired evidence review.')
            env = dict(os.environ); env.pop('QFBENCH_AUDIT_DIR', None)
            gate = subprocess.run([sys.executable, str(ROOT / 'track2-forecasting-public/scoring/scoring.py'),
                                   'score', '--card', str(u / 'card.toml'), '--forecast', str(s.out)],
                                  capture_output=True, text=True, env=env, timeout=30)
            assert gate.returncode == 0, gate.stderr
            gates = json.loads(gate.stdout); assert gates['admissible'], gates
            (a.out / f'{uid}_gates.json').write_text(json.dumps(gates, indent=2))
            assert sorted(f.name for f in out.iterdir()) == ['forecast.parquet', 'forecast_meta.json', 'forecast_rationale.md']
            results.append({'unit': uid, 'review': report, 'admissible': True})
            signal.alarm(0); print(f'PASS {uid}', flush=True)
        (a.out / 'summary.json').write_text(json.dumps(results, indent=2))
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old); stop.set()


if __name__ == '__main__': main()
