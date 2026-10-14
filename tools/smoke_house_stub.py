"""Stub organizer route: rehearse the house adapter end to end before the real endpoint.

## Executive summary (read this first)

The competition image can only call `MODEL_ENDPOINT/v1/chat/completions` with the
per-unit `MODEL_TOKEN` bearer. This script stands up a local OpenAI-compatible stub that
plays the same protocol, drives a real agent run against it, and then asserts the things
that would otherwise fail for the first time inside the sealed environment:

1. the adapter sends exactly one POST per step to `/v1/chat/completions` carrying the bearer;
2. every request stays inside the 4000 output-token ceiling and the 25-request budget;
3. the tool-calling protocol works against a non-DeepSeek model id read from `MODEL_NAME`;
4. the run writes the three contract files and the audit call counter matches.

It is NOT a model quality test. The stub's answers are fixed, so a pass proves the
plumbing, not the forecasting.
"""
from pathlib import Path
import argparse
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'track2-forecasting-public'

#: Fixed executor script: inspect, diagnose, compare, export, finish. The commander is a separate
#: tool-calling phase (read-only factor/data tools) that ends with a plan JSON as its final answer.
EXECUTOR_TOOLS = {'inspect_data', 'diagnose_series', 'compute_factors', 'data_quality', 'compare_models', 'export_forecast'}
SCRIPT = [
    [('inspect_data', {})],
    [('diagnose_series', {}), ('build_features', {})],
    [('compare_models', {'methods': ['random_walk', 'gbm']})],
    [('review_model_evidence', {})],
    [('export_forecast', {'method': 'random_walk', 'reason': 'Stub rehearsal: Validated random-walk reference on the common proxy scale.'})],
    [('final_answer', {'answer': 'Stub run complete: forecast exported.'})],
]


class State:
    def __init__(self):
        self.requests = []
        self.step = 0
        self.commander_turn = 0
        self.commander_done = False


STATE = State()


def _completion(messages, step):
    """One scripted assistant turn: a batch of tool calls, until the script runs out."""
    if step < len(SCRIPT):
        calls = SCRIPT[step]
        return {'role': 'assistant', 'content': '',
                'tool_calls': [{'id': f'call_{step}_{i}', 'type': 'function',
                                'function': {'name': name, 'arguments': json.dumps(args)}}
                               for i, (name, args) in enumerate(calls)]}
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': f'call_{step}_0', 'type': 'function',
         'function': {'name': 'final_answer', 'arguments': json.dumps({'answer': 'stub exhausted'})}}]}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_POST(self):
        try:
            self._handle()
        except Exception:
            import traceback
            traceback.print_exc()
            self.send_response(500)
            self.send_header('Content-Length', '0')
            self.end_headers()

    def _handle(self):
        length = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(length) or b'{}')
        auth = self.headers.get('Authorization') or ''
        STATE.requests.append({'path': self.path, 'auth': auth, 'body': body,
                               'time': time.time()})
        tool_names = {t['function']['name'] for t in (body.get('tools') or [])}
        # The commander exposes only read-only tools (inspect_data/compute_factors/data_quality);
        # the executor additionally exposes compare_models and export_forecast.
        is_commander = not ({'compare_models', 'export_forecast'} & tool_names)
        if is_commander:
            # Tool-calling commander: one read-only tool call, then the plan as final_answer.
            turn = STATE.commander_turn
            STATE.commander_turn += 1
            if turn == 0:
                message = {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'id': f'cmd_{turn}', 'type': 'function',
                     'function': {'name': 'inspect_data', 'arguments': '{}'}}]}
            else:
                # Recover the task payload (documents + baseline) from the user message text.
                context = None
                for m in body.get('messages', []):
                    content = m.get('content')
                    if isinstance(content, list):
                        text = '\n'.join(b.get('text', '') for b in content if isinstance(b, dict))
                    elif isinstance(content, str):
                        text = content
                    else:
                        continue
                    i = text.find('{')
                    if i >= 0:
                        try:
                            context = json.loads(text[i:text.rfind('}') + 1])
                            if isinstance(context, dict) and 'baseline' in context:
                                break
                            context = None
                        except ValueError:
                            pass
                docs = (context or {}).get('documents', [])
                plan = dict((context or {}).get('baseline') or {
                    'decision': 'keep',
                    'methods': ['random_walk', 'gbm'],
                    'window': 756, 'reason': 'Stub baseline.', 'evidence': []})
                STATE.default_window = plan['window']
                if getattr(STATE, 'commander_weights', None) is not None:
                    plan.update(decision='modify', ensemble_weights=STATE.commander_weights,
                                analysis_reason='Use the supplied monthly multi-step numerical analysis to exercise a trend-only commander decision without a random-walk allocation.')
                if STATE.plan != 'keep' and docs:
                    plan.update(decision='modify', methods=['random_walk', 'gbm'], window=252,
                                reason='Stub rehearsal of text-based workflow configuration.',
                                evidence=[{'doc_id': docs[0]['doc_id'], 'quote': docs[0]['text'][:80]}])
                elif STATE.plan != 'keep':
                    plan.update(decision='modify', methods=['random_walk', 'gbm'], window=252,
                                reason='Stub rehearsal of text-based workflow configuration.',
                                evidence=[])
                if STATE.plan == 'invalid':
                    plan['window'] = -1
                message = {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'id': f'cmd_{turn}', 'type': 'function',
                     'function': {'name': 'final_answer', 'arguments': json.dumps({'answer': json.dumps(plan)})}}]}
                STATE.commander_done = True
        else:
            step = STATE.step
            STATE.step += 1
            message = _completion(body.get('messages') or [], step)
        payload = {'id': f'stub-{STATE.step}-{STATE.commander_turn}', 'object': 'chat.completion', 'created': int(time.time()),
                   'model': body.get('model'), 'choices': [{'index': 0, 'finish_reason': 'tool_calls',
                                                            'message': message}],
                   'usage': {'prompt_tokens': 900, 'completion_tokens': 60, 'total_tokens': 960}}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit', default='t2-F1-conflicting-texts-2024')
    p.add_argument('--out', type=Path, default=ROOT / 'runs/numeric_agent/house_stub')
    p.add_argument('--port', type=int, default=8931)
    p.add_argument('--plan', choices=['keep', 'modify', 'invalid'], default='modify')
    p.add_argument('--commander-weights', help='JSON weights for a numerical commander decision in bundle smoke')
    p.add_argument('--bundle', action='store_true', help='Exercise solution_ensemble with the keep plan')
    p.add_argument('--serve-only', action='store_true',
                   help='stand the stub up on 0.0.0.0 and block, so an already-sealed '
                        'container can be driven against it (used by the container rehearsal)')
    p.add_argument('--host', default='127.0.0.1',
                   help='bind address for --serve-only (default 0.0.0.0 in that mode)')
    a = p.parse_args()
    if a.bundle and a.plan != 'keep': p.error('--bundle requires --plan keep')
    if a.out.exists() and not a.serve_only:
        raise SystemExit(f'{a.out} exists; remove it or pick a fresh --out')

    STATE.commander_weights = json.loads(a.commander_weights) if a.commander_weights else None
    if a.commander_weights and not a.bundle: p.error('--commander-weights requires --bundle')
    STATE.plan = a.plan
    if a.bundle:
        sys.path.insert(0, str(ROOT))
        from numeric_agent.engine import METHODS
        SCRIPT[2] = [('compare_models', {'methods': list(METHODS)})]
        SCRIPT[4] = [('export_forecast', {'method': 'solution_ensemble', 'reason': 'Stub integration of the bounded evidence-weighted joint mixture.'})]
    if a.plan == 'modify':
        SCRIPT[2] = [('compare_models', {'methods': ['random_walk', 'gbm']})]
    if a.serve_only:
        # Same Handler, no local harness: the caller points a sealed container at this port via
        # --add-host=host.docker.internal:host-gateway and MODEL_ENDPOINT=http://host.docker.internal:<port>.
        host = a.host if a.host != '127.0.0.1' else '0.0.0.0'
        server = http.server.ThreadingHTTPServer((host, a.port), Handler)
        print(f'serving stub on {host}:{a.port} (plan={a.plan}); requests will be counted', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            print(f'served {len(STATE.requests)} requests', flush=True)
        return 0
    server = http.server.HTTPServer(('127.0.0.1', a.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    unit = REPO / 'units' / a.unit
    card = tomllib.loads((unit / 'card.toml').read_text())
    staged = a.out / 'staged'
    (staged / 'panels').mkdir(parents=True)
    import shutil
    shutil.copy2(unit / 'card.toml', staged / 'card.toml')
    if (unit / 'forecast_spec.json').exists():
        shutil.copy2(unit / 'forecast_spec.json', staged / 'forecast_spec.json')
    for panel_id in card['panels']['panel_ids']:
        shutil.copy2(unit / f'{panel_id}.parquet', staged / 'panels' / f'{panel_id}.parquet')
    shutil.copytree(unit / 'text', staged / 'text')
    output = a.out / 'output'
    output.mkdir(parents=True)

    env = dict(os.environ)
    env.update({'MODEL_ENDPOINT': f'http://127.0.0.1:{a.port}', 'MODEL_TOKEN': 'stub-token-not-a-credential',
                'MODEL_NAME': 'nvidia/nemotron-3-super-120b-a12b', 'QFBENCH_NETWORK': 'restricted',
                'no_proxy': '127.0.0.1,localhost', 'NO_PROXY': '127.0.0.1,localhost',
                # Gate g0 requires /output to hold exactly the three deliverables, so the trail
                # is routed to a sibling that the checks below read.
                'QFBENCH_AUDIT_DIR': str(a.out / 'output_audit')})
    for var in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
        env.pop(var, None)
    argv = [sys.executable, '-m', 'numeric_agent', 'forecast', '--panels', str(staged / 'panels'),
            '--text', str(staged / 'text'), '--asof', card['provenance']['data_cutoff'],
            '--out', str(output / 'forecast.parquet'), '--mode', 'agent', '--timeout', '300']
    print('RUN', ' '.join(argv), flush=True)
    proc = subprocess.run(argv, cwd=ROOT, env=env)
    server.shutdown()

    status = json.loads((a.out / 'output_audit/run_status.json').read_text())
    progress = [json.loads(line) for line in (a.out / 'output_audit/progress.jsonl').read_text().splitlines()]
    print('run_status:', json.dumps(status), flush=True)

    assert proc.returncode == 0, f'agent exited {proc.returncode}'
    assert STATE.requests, 'the stub route received no request'
    for r in STATE.requests:
        assert r['path'] == '/v1/chat/completions', f'wrong path: {r["path"]}'
        assert r['auth'] == 'Bearer stub-token-not-a-credential', 'bearer missing from a request'
        assert r['body'].get('max_tokens', 0) <= 4000, f'max_tokens over the ceiling: {r["body"].get("max_tokens")}'
        assert r['body'].get('model') == 'nvidia/nemotron-3-super-120b-a12b', 'MODEL_NAME was not used'
    # The commander now sends a read-only tool schema (inspect_data/compute_factors/data_quality);
    # only the executor exposes compare_models/export_forecast, so the first request must lack them.
    first_tools = {t['function']['name'] for t in STATE.requests[0]['body'].get('tools', [])}
    assert first_tools and not ({'compare_models', 'export_forecast'} & first_tools), 'commander must run first'
    assert len(STATE.requests) <= 25, f'request budget exceeded: {len(STATE.requests)}'

    finished = [e for e in progress if e['event'] == 'agent_finished']
    assert finished and finished[0]['model_calls'] == len(STATE.requests), \
        f'audit counted {finished[0]["model_calls"] if finished else None} calls, route saw {len(STATE.requests)}'
    files = sorted(f.name for f in output.iterdir())
    assert files == ['forecast.parquet', 'forecast_meta.json', 'forecast_rationale.md'], files
    meta = json.loads((output / 'forecast_meta.json').read_text())
    if a.commander_weights:
        expected={k:v for k,v in STATE.commander_weights.items() if v>0}
        assert meta['model_audit']['weights']==expected
        assert meta['model_audit']['weight_source']=='commander'
    assert meta['text_used']
    assert meta['commander']['status'] == ('fallback' if a.plan == 'invalid' else 'accepted')
    assert meta['training_window'] == (252 if a.plan == 'modify' else STATE.default_window)
    import jsonschema
    from smoke_staged_unit import schema_path
    jsonschema.validate(meta, json.loads(schema_path().read_text()))
    gates = subprocess.run([sys.executable, str(REPO / 'scoring/scoring.py'), 'score',
                            '--card', str(staged / 'card.toml'), '--forecast', str(output / 'forecast.parquet')],
                           cwd=ROOT, capture_output=True, text=True, check=True)
    verdict = json.loads(gates.stdout)
    assert verdict['admissible'], verdict
    (a.out / 'gates.json').write_text(json.dumps(verdict, indent=2))
    print(f'PASS stub rehearsal: {len(STATE.requests)} requests, all on /v1/chat/completions with bearer, '
          f'total_tokens={finished[0]["total_tokens"]}, files={files}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
