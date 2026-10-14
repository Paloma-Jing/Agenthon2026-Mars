"""Run the house-model controller, or an explicitly labelled deterministic tool smoke test.

Competition invocation (the harness passes the verb as the first argument):

  forecast --panels /input/panels/ --text /input/text/ --asof <YYYY-MM-DD> \
           --out /output/forecast.parquet

Inside --network=qfb2-eval with QFBENCH_NETWORK=restricted the only reachable model is
the organizer route; numeric_agent.house reads MODEL_ENDPOINT/MODEL_TOKEN/MODEL_NAME
from the environment. The commander reads dated --text before numerical execution.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import signal
import threading
import time
from .engine import ForecastSession, METHODS, write_audit
from . import house


def build_model(a):
    """Return (model, label) for the requested controller backend."""
    restricted = os.environ.get('QFBENCH_NETWORK') in ('restricted', 'none')
    if house.available():
        return house.build_model(thinking=a.thinking, seed=a.seed), house.model_id()
    if restricted:
        raise RuntimeError('QFBENCH_NETWORK is restricted and MODEL_ENDPOINT/MODEL_TOKEN are unset; '
                           'the house route is the only reachable model')
    # Local development only. Never reachable in the competition image.
    from smolagents import OpenAIModel
    key = getpass.getpass('Local debug API key (not saved): ') if a.prompt_key else os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        raise RuntimeError('set MODEL_ENDPOINT/MODEL_TOKEN, or DEEPSEEK_API_KEY for local debugging')
    model = OpenAIModel(model_id=a.model, api_base='https://api.deepseek.com', api_key=key,
                        client_kwargs={'timeout': 45.0, 'max_retries': 0}, max_tokens=1800,
                        extra_body={'thinking': {'type': 'disabled'}})
    return model, a.model


def audit_dir(out):
    """Where the run trail goes, kept OUT of the graded output directory.

    Gate g0 requires the output tree to hold exactly `forecast.parquet`, `forecast_meta.json` and
    `forecast_rationale.md`; it compares the directory listing as a set, so one extra entry --
    file or directory -- refuses the unit outright
    (`extra_count=1, expected_count=3`, measured against the official scorer). A sibling of the
    output directory is therefore only safe when that parent is writable, which is true locally
    and false on the platform, where the root filesystem is read-only and the only writable
    mounts are /output and a 64 MiB /tmp.
    """
    override=os.environ.get('QFBENCH_AUDIT_DIR')
    target=Path(override) if override else out.parent.with_name(out.parent.name+'_audit')
    try:
        target.mkdir(parents=True,exist_ok=True)
        return target
    except OSError:
        # Sealed platform: the parent is the read-only root, and /output must stay exactly the
        # three graded files, so there is nowhere to put a trail. Continuing without one is fine;
        # failing the unit over a log file is not.
        print(f'WARN no writable audit directory ({target}); continuing without a trail',flush=True)
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('verb',nargs='?',default='forecast',choices=['forecast'])
    p.add_argument('--panels',type=Path,required=True);p.add_argument('--card',type=Path)
    p.add_argument('--text',type=Path,help='Frozen task corpus for the commander')
    p.add_argument('--commander',choices=['on','off'],default='on',help='Pre-task text planning; off retains the numeric baseline')
    p.add_argument('--asof',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['agent','offline'],default='agent')
    p.add_argument('--model',default='deepseek-flash',help='local debug model id only')
    p.add_argument('--prompt-key',action='store_true')
    p.add_argument('--thinking',action='store_true',help='enable house model thinking; off by default')
    p.add_argument('--draws',type=int,default=1000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-steps',type=int,default=house.DEFAULT_MAX_STEPS);p.add_argument('--timeout',type=int,default=300)
    p.add_argument('--commander-steps',type=int,default=4,help='tool-calling commander step budget')
    a=p.parse_args()
    if not 1<=a.max_steps<=20 or not 10<=a.timeout<=1800 or not 1<=a.commander_steps<=8:p.error('invalid runtime budget')
    # Every agent step is at least one admitted request; keep the route's per-unit ceiling in reach.
    # Commander steps (tool loop) + executor steps + one final-answer call must fit under 25.
    if a.mode=='agent' and a.max_steps+1+(a.commander_steps if a.commander=='on' else 0)>house.REQUEST_BUDGET:p.error('max_steps exceeds the per-unit request budget')
    card=a.card or next((v for v in [a.panels/'card.toml',a.panels.parent/'card.toml'] if v.exists()),None)
    if not card:p.error('card.toml not found')
    # Prevent stale outputs from masquerading as a successful new run.
    if a.out.exists():
        p.error('use a fresh output directory for each run')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    audit=audit_dir(a.out)
    if audit is not None and (audit/'run_status.json').exists():
        p.error('use a fresh output directory for each run')
    status={'status':'running','mode':a.mode,'model':None,'text_used':False}
    # The trail is diagnostic only: a full disk or an unwritable path must never turn a scored
    # unit into a failure, so every audit write is attempt-and-continue.
    status_path=(audit/'run_status.json') if audit is not None else None
    if status_path is not None:write_audit(status_path,json.dumps(status,indent=2))
    start=time.monotonic();stop=threading.Event()
    def heartbeat():
        while not stop.wait(10):print(f'RUNNING elapsed={time.monotonic()-start:.0f}s',flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def timeout_handler(*_):raise TimeoutError('run time budget exceeded')
    signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(a.timeout)
    session=None
    try:
        session=ForecastSession(a.panels,card,a.asof,a.out,a.draws,a.seed,controller_status=status,audit_root=audit)
        if a.mode=='offline':
            session.diagnose()
            for method in METHODS:session.evaluate(method)
            choices=[r for r in session.cache.values() if r['status']=='ok']
            if not choices:raise ValueError('no valid candidates; see backtests.json')
            chosen=min(choices,key=lambda r:r['normalized_composite'])['method']
            session.export(chosen,'Deterministic offline selection: arithmetic mean of reconstructed M0-normalized composite; failures included at 8. No LLM called.')
        else:
            model,label=build_model(a);status['model']=label
            if status_path is not None:write_audit(status_path,json.dumps(status,indent=2))
            session.log('controller',**house.describe() if house.available() else {'backend':'local-debug','model_id':label},
                        thinking=bool(a.thinking))
            if a.commander=='on':
                from .commander import run_commander
                run_commander(session,model,a.text,commander_steps=a.commander_steps,use_tools=True)
            from .agent import build_agent
            agent=build_agent(session,model,a.max_steps)
            try:
                agent.run('Produce a forecast using only the available numeric tools. The commander stage is complete. Inspect the data and diagnostics, compare every allowed model, export the chosen forecast and finish.')
            except Exception as e:
                # A controller that exports and then fails to emit final_answer still produced the
                # deliverable. The contract is judged on the three output files, so treat an
                # exhausted step budget as success ONLY when the export already happened.
                if session.exported and type(e).__name__ in ('AgentMaxStepsError','AgentParsingError','AgentGenerationError'):
                    session.log('agent_stopped_after_export',stopped_by=type(e).__name__)
                else: raise
            if not session.exported:raise RuntimeError('agent stopped without exporting forecast')
            session.log('agent_finished',model_calls=session.model_calls,total_tokens=session.total_tokens,
                        request_budget=house.REQUEST_BUDGET)
        meta_path=a.out.parent/'forecast_meta.json';meta=json.loads(meta_path.read_text());meta['controller_mode']=a.mode;meta['controller_model']=status['model'];meta_path.write_text(json.dumps(meta,indent=2))
        status['status']='success'
    except Exception as e:
        # API exceptions may include request details. Log only type, never credentials or raw bodies.
        status.update(status='failed',error_type=type(e).__name__)
        if type(e).__module__.startswith('openai'):print(f'FAILED API: {type(e).__name__}',flush=True)
        else:print(f'FAILED: {type(e).__name__}: {str(e)[:250]}',flush=True)
        if session is not None:
            # Reserve a bounded fallback budget even when the controller exhausts its timer.
            signal.alarm(30)
            try:
                from .m0 import export as export_m0
                details=export_m0(card,a.out.parent,a.panels)
                status.update(status='success',fallback='m0_reconstruction',fallback_trigger=type(e).__name__)
                session.trail('m0_fallback.json',json.dumps(details,indent=2))
                print('FALLBACK completed public M0 reconstruction',flush=True)
            except Exception as fallback_error:
                status.update(status='failed',fallback_error_type=type(fallback_error).__name__)

    finally:
        signal.alarm(0);stop.set();status['elapsed_seconds']=round(time.monotonic()-start,2)
        if status_path is not None:write_audit(status_path,json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    return 0 if status['status']=='success' else 1

if __name__=='__main__':raise SystemExit(main())
