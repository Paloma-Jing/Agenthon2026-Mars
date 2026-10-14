"""Evidence, time boundaries, enforced plans and safe planning fallbacks."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from numeric_agent.commander import baseline_plan, read_corpus, run_commander, validate_plan, _extract_plan_json, FAMILY_WHITELIST
from numeric_agent.engine import ForecastSession, METHODS

ROOT = Path(__file__).resolve().parents[1]


def make_session(tmp_path):
    unit = ROOT / 'track2-forecasting-public/units/t2-F1-ai-mom-2024'
    return ForecastSession(unit, unit / 'card.toml', '2024-05-31', tmp_path / 'output/forecast.parquet')


def corpus(tmp_path):
    root = tmp_path / 'text'
    root.mkdir()
    (root / 'valid.txt').write_text('Inflation uncertainty remains elevated. ' * 100)
    (root / 'corpus_index.json').write_text(json.dumps({'documents': [
        {'doc_id': 'valid', 'timestamp': '2024-05-01', 'file': 'valid.txt'}]}))
    return root


def modified_plan():
    return {'decision': 'modify', 'methods': ['random_walk'], 'window': 252,
            'reason': 'Prioritize recent volatility given the dated uncertainty statement.',
            'evidence': [{'doc_id': 'valid', 'quote': 'Inflation uncertainty remains elevated.'}]}


class Model:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=json.dumps(self.plan), token_usage=SimpleNamespace(input_tokens=10, output_tokens=20))


def test_reader_rejects_future_unindexed_escape_and_symlink(tmp_path):
    root = corpus(tmp_path)
    outside = tmp_path / 'outside.txt';outside.write_text('do not read')
    (root / 'link.txt').symlink_to(outside)
    (root / 'unindexed.txt').write_text('do not read')
    docs = json.loads((root / 'corpus_index.json').read_text())
    docs['documents'] += [
        {'doc_id': 'future', 'timestamp': '2024-06-01', 'file': 'valid.txt'},
        {'doc_id': 'escape', 'timestamp': '2024-05-01', 'file': '../outside.txt'},
        {'doc_id': 'link', 'timestamp': '2024-05-01', 'file': 'link.txt'},
        {'doc_id': 'undated', 'file': 'valid.txt'}]
    (root / 'corpus_index.json').write_text(json.dumps(docs))
    report = read_corpus(root, '2024-05-31')
    assert [d['doc_id'] for d in report['documents']] == ['valid']
    assert report['excluded'] == 4 and report['truncated'] == 1
    assert len(report['documents'][0]['text']) == 3000


def test_modified_plan_enforced_before_tools_and_export(tmp_path):
    session = make_session(tmp_path)
    model = Model(modified_plan())
    record = run_commander(session, model, corpus(tmp_path), use_tools=False)
    assert record['status'] == 'accepted' and record['text_used']
    assert session.model_calls == 1 and session.total_tokens == 30
    assert session.window == 252 and session.allowed_methods == ('random_walk',)
    assert not session.diagnosed and not session.cache
    with pytest.raises(ValueError, match='excluded'):
        session.evaluate('ridge')
    session.diagnose()
    with pytest.raises(ValueError, match='once'):
        session.apply_workflow(record)
    # Isolate plan enforcement with complete synthetic returns; real holiday gaps are
    # intentionally rejected by the strict label policy and covered in backtest tests.
    session.raw = session.raw.fillna(0.)
    for method in session.allowed_methods:
        assert session.evaluate(method)['status'] == 'ok'
    session.export('random_walk', 'Test execution of the accepted plan.')
    meta = json.loads((session.out.parent / 'forecast_meta.json').read_text())
    assert meta['text_used'] and meta['training_window'] == 252
    assert meta['commander']['plan'] == modified_plan()
    assert 'Commander plan' in (session.out.parent / 'forecast_rationale.md').read_text()
    assert len(list(session.out.parent.iterdir())) == 3


@pytest.mark.parametrize('change', [
    {'window': 1}, {'window': True}, {'methods': ['ewma', 'garch']},
    {'methods': ['random_walk', 'shell']}, {'decision': 'keep'}, {'evidence': []},
    {'evidence': [{'doc_id': 'valid', 'quote': 'invented quote that is not in text'}]},
    {'evidence': [{'doc_id': 'other', 'quote': 'Inflation uncertainty remains elevated.'}]},
    {'code': 'print(1)'},
])
def test_invalid_plan_falls_back(tmp_path, change):
    session = make_session(tmp_path)
    plan = modified_plan();plan.update(change)
    record = run_commander(session, Model(plan), corpus(tmp_path), use_tools=False)
    assert record['status'] == 'fallback'
    assert session.allowed_methods == METHODS and session.window == 756
    assert record['text_used']  # text was sent even though the plan was rejected


def test_keep_plan_and_no_text(tmp_path):
    session = make_session(tmp_path)
    model = Model(baseline_plan(756))
    record = run_commander(session, model, None, use_tools=False)
    assert record['status'] == 'accepted' and model.calls
    assert not record['text_used']
    docs = read_corpus(corpus(tmp_path), session.asof)['documents']
    assert validate_plan(baseline_plan(756), docs, 756, False)['decision'] == 'keep'


def test_model_error_and_deadline(tmp_path):
    class Broken:
        def generate(self, **kwargs):
            raise RuntimeError('not logged')
    session = make_session(tmp_path)
    text = corpus(tmp_path)
    record = run_commander(session, Broken(), text, use_tools=False)
    assert record['status'] == 'fallback' and record['error_type'] == 'RuntimeError'
    assert session.model_calls == 1
    class Deadline:
        def generate(self, **kwargs):
            raise TimeoutError('deadline')
    session = make_session(tmp_path / 'deadline')
    with pytest.raises(TimeoutError):
        run_commander(session, Deadline(), text, use_tools=False)


def test_extract_plan_json_forms():
    plan = {'decision': 'keep', 'methods': list(METHODS), 'window': 756, 'reason': 'r', 'evidence': []}
    import json as _json
    assert _extract_plan_json(_json.dumps(plan)) == plan
    assert _extract_plan_json('```json\n' + _json.dumps(plan) + '\n```') == plan
    with pytest.raises(ValueError):
        _extract_plan_json('not json at all')
    with pytest.raises(ValueError):
        _extract_plan_json(123)


def test_validate_plan_features_whitelist():
    docs = [{'doc_id': 'valid', 'text': 'Inflation uncertainty remains elevated.'}]
    # a keep plan carries the full allowed feature set
    keep = baseline_plan(756, features=['factors', 'rates'])
    assert validate_plan(keep, docs, 756, False, allowed_families=['factors', 'rates'])['decision'] == 'keep'
    # unknown family rejected
    bad = dict(keep); bad['features'] = ['not_a_family']
    with pytest.raises(ValueError, match='invalid features'):
        validate_plan(bad, docs, 756, False, allowed_families=['factors', 'rates'])
    # feature not present in panel rejected
    with pytest.raises(ValueError, match='features not present'):
        validate_plan(baseline_plan(756, features=['rates']), docs, 756, False, allowed_families=['factors'])
    # changing features without a modify decision is rejected
    mod = baseline_plan(756, features=['factors', 'rates'])
    mod.update({'features': ['factors']})
    with pytest.raises(ValueError, match='decision does not match'):
        validate_plan(mod, docs, 756, False, allowed_families=['factors', 'rates'])
    # a modify decision with evidence and a feature change is accepted
    good = dict(keep)
    good.update({'decision': 'modify', 'features': ['factors'],
                 'reason': 'Drop rates context given conflicting statements.',
                 'evidence': [{'doc_id': 'valid', 'quote': 'Inflation uncertainty remains elevated.'}]})
    assert validate_plan(good, docs, 756, False, allowed_families=['factors', 'rates'])['decision'] == 'modify'


def test_factor_and_quality_tools(tmp_path):
    session = make_session(tmp_path)
    report = session.factor_report('MOM')
    assert report['families'] == ['factors']
    assert report['factors']['MOM']['family'] == 'factors'
    assert 'cs_momentum_z' in report['factors']['MOM']['factors']
    q = session.data_quality_report()
    assert q['n_assets'] == 6 and 'missing_rate' in q and 'pca_var_ratios' in q


def test_numerical_commander_weights_without_text_are_applied(tmp_path):
    session=make_session(tmp_path)
    plan=baseline_plan(756)
    plan.update(decision='modify',ensemble_weights={'damped_trend':1.,'random_walk':0.},
                analysis_reason='Use supplied numerical trend and missingness analysis to choose a trend-only mixture.')
    model=Model(plan)
    record=run_commander(session,model,None,use_tools=False)
    assert record['status']=='accepted' and not record['text_used']
    assert session.ensemble_weights==plan['ensemble_weights']
    assert record['numeric_analysis']['factors']
    assert 'numeric_analysis' in model.calls[0]['messages'][1]['content']
