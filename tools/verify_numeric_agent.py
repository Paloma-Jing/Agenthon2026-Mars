"""Run four numerical tool pipelines and official gates, with heartbeat and timeouts."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,tomllib
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'track2-forecasting-public'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=ROOT/'runs/numeric_agent'/time.strftime('verification_%Y%m%d_%H%M%S'))
OUT=parser.parse_args().out;OUT.mkdir(parents=True,exist_ok=True)
UNITS=['t2-F3-conundrum-joint-2005','t2-F2-whatever-it-takes-2012','t2-F1-ai-mom-2024','t2-F1-cpi-glidepath-2023']
def run(args,path,env=None):
    start=time.monotonic()
    with path.open('w') as log, path.with_suffix('.stderr.log').open('w') as err:
        proc=subprocess.Popen(args,stdout=log,stderr=err,cwd=ROOT,env=env)
        while proc.poll() is None:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:print(f'RUNNING {path.name}: {time.monotonic()-start:.0f}s',flush=True)
            if time.monotonic()-start>300:proc.kill();proc.wait();raise RuntimeError('timeout')
    if proc.returncode:raise RuntimeError(f'exit {proc.returncode}; see {path}')
results=[]
for i,uid in enumerate(UNITS,1):
    print(f'START {i}/4 {uid}',flush=True);u=REPO/'units'/uid;o=OUT/uid;o.mkdir(exist_ok=True)
    audit=OUT/(uid+'_audit');audit.mkdir(parents=True,exist_ok=True)
    card=tomllib.loads((u/'card.toml').read_text())
    # Gate g0 wants /output to hold exactly the three deliverables, so keep our logs and the
    # agent's trail OUTSIDE the graded output directory, and tell the agent where its trail belongs.
    env=dict(os.environ);env['QFBENCH_AUDIT_DIR']=str(audit/'agent')
    run([sys.executable,'-m','numeric_agent','--panels',str(u),'--asof',card['provenance']['data_cutoff'],'--out',str(o/'forecast.parquet'),'--mode','offline'],audit/'run.log',env)
    run([sys.executable,str(REPO/'scoring/scoring.py'),'score','--card',str(u/'card.toml'),'--forecast',str(o/'forecast.parquet')],audit/'gates.json',env)
    gates=json.loads((audit/'gates.json').read_text());assert gates['admissible'],gates
    results.append({'unit':uid,'admissible':gates['admissible'],'method':json.loads((o/'forecast_meta.json').read_text())['selected_method']});print('PASS',results[-1],flush=True)
(OUT/'summary.json').write_text(json.dumps(results,indent=2))
