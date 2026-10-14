"""Arithmetic comparison under common pre-outcome M0 scales; failures stay in denominator."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from numeric_agent.evaluation import aggregate


def compare(metrics, methods, planned_keys=None):
    groups=defaultdict(list);excluded=[]
    for r in metrics:
        if r['status']=='excluded':excluded.append(r);continue
        groups[r['method']].append(r)
    roster=set(planned_keys) if planned_keys is not None else {r['key'] for rows in groups.values() for r in rows}
    excluded_keys={r['key'] for r in excluded}
    if any(r['key'] in excluded_keys for rows in groups.values() for r in rows):raise ValueError('inconsistent data exclusion')
    roster-=excluded_keys
    summary={}
    for method in methods:
        rows=groups[method]
        keys=[r['key'] for r in rows]
        if not set(keys)<=roster:raise ValueError('unplanned evaluation key')
        if len(keys)!=len(set(keys)):raise ValueError('duplicate method/task key')
        # A stopped run cannot improve by silently dropping unattempted method/task pairs.
        rows=rows+[{'key':key,'status':'not_attempted','normalized_composite':8.} for key in sorted(roster-set(keys))]
        summary[method]=aggregate(rows) if rows else {'count':0,'mean_normalized_composite':None,'failed':0}
    # Verify every method uses the same pre-outcome scale on a shared fold.
    scales={}
    for rows in groups.values():
        for r in rows:
            if r['key'] in scales and scales[r['key']]!=r['scales']:raise ValueError('inconsistent reference scale')
            scales[r['key']]=r['scales']
    return {'normalization':'public_M0_expected_error_reconstruction','aggregation':'equal_weight_arithmetic_mean','methods':summary,'excluded_data':excluded,'limitation':'Historical fixed-model diagnostics; not hidden score or complete policy holdout.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    config=json.loads((a.run/'config.json').read_text())
    if config.get('normalization')!='public_M0_expected_error_reconstruction':raise SystemExit('legacy proxy run; regenerate with current evaluator')
    metrics=[json.loads(l) for l in (a.run/'metrics.jsonl').read_text().splitlines()]
    result=compare(metrics,config['methods'],config['planned_keys']);(a.run/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
