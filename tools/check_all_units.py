"""All public cards, fresh per-card outputs, fixed denominator, heartbeat and hard timeout."""
import argparse, hashlib, json, os, subprocess, sys, time, tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'track2-forecasting-public'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['offline','m0'],default='offline')
    p.add_argument('--timeout',type=int,default=180)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    cards=sorted((REPO/'units').glob('*/card.toml'))
    hashes={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted((ROOT/'numeric_agent').glob('*.py'))}
    (a.out/'manifest.json').write_text(json.dumps({'mode':a.mode,'cards':[str(c.relative_to(ROOT)) for c in cards],'source_sha256':hashes,'scope':'Local public cards; no live House or platform execution'},indent=2))
    rows=[];start=time.monotonic()
    for i,c in enumerate(cards,1):
        uid=c.parent.name; d=a.out/uid;d.mkdir(); output=d/'output'; audit=d/'audit'
        card=tomllib.loads(c.read_text());t=time.monotonic()
        print(f'START {i}/{len(cards)} {uid}',flush=True)
        env=dict(os.environ,QFBENCH_AUDIT_DIR=str(audit),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        if a.mode=='m0':
            cmd=[sys.executable,'-m','numeric_agent.m0','--card',str(c),'--out',str(output)]
        else:
            cmd=[sys.executable,'-m','numeric_agent','--panels',str(c.parent),'--asof',card['provenance']['data_cutoff'],'--out',str(output/'forecast.parquet'),'--mode','offline','--timeout',str(a.timeout)]
        row={'unit':uid,'status':'failed','admissible':False}
        with (d/'run.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f'RUNNING {i}/{len(cards)} {uid} elapsed={time.monotonic()-t:.0f}s',flush=True)
                if time.monotonic()-t>a.timeout+10:
                    import signal
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait();row['timeout']=True
            row['exit_code']=proc.returncode
        if proc.returncode==0:
            try:
                gate=subprocess.run([sys.executable,str(REPO/'scoring/scoring.py'),'score','--card',str(c),'--forecast',str(output/'forecast.parquet')],cwd=ROOT,env=env,capture_output=True,text=True,timeout=40)
                (d/'gates.json').write_text(gate.stdout); (d/'gates.stderr.log').write_text(gate.stderr)
                result=json.loads(gate.stdout);row['admissible']=bool(result.get('admissible'))
                row['status']='ok' if row['admissible'] else 'gate_failed'
                row['method']=json.loads((output/'forecast_meta.json').read_text()).get('selected_method',a.mode)
            except Exception as e:row['error']=str(e)
        else:row['error_tail']=(d/'run.log').read_text()[-1800:]
        row['elapsed']=round(time.monotonic()-t,2);rows.append(row)
        with (a.out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        summary={'total':len(cards),'completed':len(rows),'passed':sum(x['admissible'] for x in rows),'failed':sum(not x['admissible'] for x in rows),'elapsed':round(time.monotonic()-start,2),'rows':rows}
        temp=a.out/'summary.tmp';temp.write_text(json.dumps(summary,indent=2));temp.replace(a.out/'summary.json')
        print(f'DONE {i}/{len(cards)} {row["status"]} pass={summary["passed"]} fail={summary["failed"]}',flush=True)
if __name__=='__main__':main()
