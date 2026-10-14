"""Structural smoke test: stage a unit the way the harness does, then run and gate it.

## Executive summary (read this first)

The evaluation container sees a STAGED unit, not the repository copy: `/input/card.toml`,
`/input/panels/*.parquet`, `/input/text/`. This script rebuilds that shape in a temporary
directory, runs `python -m numeric_agent forecast --panels <staged>/panels/ ...`, validates
`forecast_meta.json` against the official JSON schema, runs the official g0-g3 gates, and
checks that `forecast_rationale.md` is present and non-empty.

It does NOT prove predictive accuracy: no realized outcome exists locally. It proves the
interface, the schema, the gates and the output-directory discipline.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'track2-forecasting-public'


def schema_path():
    import qfbench2_common
    return Path(qfbench2_common.__file__).parent / 'schemas/forecast.schema.json'


def stage(unit, dest):
    """Rebuild the harness layout: card at the root, panels/ and text/ as subdirectories."""
    card = tomllib.loads((unit / 'card.toml').read_text())
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'panels').mkdir(exist_ok=True)
    shutil.copy2(unit / 'card.toml', dest / 'card.toml')
    # Monthly tasks carry their observation-period mapping in the spec beside card.toml;
    # without it the official horizon resolver refuses the unit.
    for extra in ('forecast_spec.json', 'forecast_card.md', 'manifest.json'):
        if (unit / extra).exists():
            shutil.copy2(unit / extra, dest / extra)
    for panel_id in card['panels']['panel_ids']:
        src = unit / f'{panel_id}.parquet'
        if not src.exists():
            raise SystemExit(f'panel {src} is missing')
        shutil.copy2(src, dest / 'panels' / src.name)
    (dest / 'text').mkdir(exist_ok=True)
    if (unit / 'text').is_dir():
        for doc in (unit / 'text').iterdir():
            if doc.is_file():
                shutil.copy2(doc, dest / 'text' / doc.name)
    return card


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit', default='t2-F1-conflicting-texts-2024')
    p.add_argument('--out', type=Path, default=ROOT / 'runs/numeric_agent/staged_smoke')
    p.add_argument('--mode', choices=['offline', 'agent'], default='offline')
    a = p.parse_args()

    unit = REPO / 'units' / a.unit
    if not unit.is_dir():
        raise SystemExit(f'no such unit: {unit}')
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; remove it or pick a fresh --out')
    staged = a.out / 'staged'
    output = a.out / 'output'
    card = stage(unit, staged)
    asof = card['provenance']['data_cutoff']
    output.mkdir(parents=True)

    argv = [sys.executable, '-m', 'numeric_agent', 'forecast',
            '--panels', str(staged / 'panels'), '--text', str(staged / 'text'),
            '--asof', asof, '--out', str(output / 'forecast.parquet'), '--mode', a.mode]
    print('RUN', ' '.join(argv), flush=True)
    start = time.monotonic()
    proc = subprocess.run(argv, cwd=ROOT)
    if proc.returncode:
        raise SystemExit(f'forecast exited {proc.returncode}')

    # Gate g0 compares this listing as a SET against exactly these three names, so the audit trail
    # must never appear here (it is written to `<out>_audit`, a sibling of this directory).
    produced = sorted(f.name for f in output.iterdir())
    assert produced == ['forecast.parquet', 'forecast_meta.json', 'forecast_rationale.md'], \
        f'output directory holds extra or missing files: {produced}'
    assert (output / 'forecast_rationale.md').stat().st_size > 0, 'rationale is empty'

    meta = json.loads((output / 'forecast_meta.json').read_text())
    import jsonschema
    jsonschema.validate(meta, json.loads(schema_path().read_text()))
    print(f'SCHEMA ok: {meta["selected_method"]}, {meta["n_draws"]} draws, target={meta["target"]}', flush=True)

    gates = subprocess.run([sys.executable, str(REPO / 'scoring/scoring.py'), 'score',
                            '--card', str(staged / 'card.toml'),
                            '--forecast', str(output / 'forecast.parquet')],
                           cwd=ROOT, capture_output=True, text=True)
    print(gates.stdout.strip(), flush=True)
    verdict = json.loads(gates.stdout)
    assert verdict['admissible'], verdict
    print(f'PASS {a.unit}: gates ok in {time.monotonic()-start:.1f}s, mode={a.mode}, no score (no realized outcome)', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
