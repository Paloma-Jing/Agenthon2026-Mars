"""Paired evidence failures must not turn into claims of model superiority."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from numeric_agent.evidence import review_candidates


def result(losses, statuses=None):
    return {'status': 'ok', 'normalization': 'public_M0', 'aggregation': 'arithmetic',
            'fold_records': [{'key': str(i), 'status': status, 'normalized_composite': loss,
                              'scales': {'marginal': 1.}}
                             for i, (loss, status) in enumerate(zip(losses, statuses or ['ok'] * len(losses)))]}


def test_paired_review_reports_failure_and_local_degradation():
    report = review_candidates({'random_walk': result([2, 2, 2]),
                                'gbm': result([1, 3, 0], ['ok', 'ok', 'failed'])})
    pair = report['paired']['gbm']
    assert (pair['wins'], pair['losses'], pair['ties']) == (1, 2, 0)
    assert pair['new_failure_keys'] == ['2']
    assert pair['worst_loss_increase'] == 6
    assert report['recommended_method'] == 'random_walk'
    assert report['experiments']['gbm']['metrics']['mean_normalized_composite'] == 4


@pytest.mark.parametrize('change', ['key', 'scale', 'normalization', 'aggregation'])
def test_incomparable_evidence_has_no_recommendation(change):
    base = result([2, 2, 2]); alt = deepcopy(base)
    if change == 'key': alt['fold_records'][0]['key'] = 'other'
    elif change == 'scale': alt['fold_records'][0]['scales']['marginal'] = 10.
    else: alt[change] = 'other'
    report = review_candidates({'random_walk': base, 'gbm': alt})
    assert not report['comparable']
    assert report['recommended_method'] is None


def test_missing_reference_and_duplicate_fold_are_not_evidence():
    assert not review_candidates({'gbm': result([1])})['comparable']
    bad = result([1, 1]); bad['fold_records'][1]['key'] = '0'
    with pytest.raises(ValueError, match='duplicate'):
        review_candidates({'random_walk': bad})


def test_mean_improvement_retains_degradation_information_and_baseline_ties():
    report = review_candidates({'random_walk': result([2, 2, 2]), 'gbm': result([1, 1, 3])})
    assert report['recommended_method'] == 'gbm'
    assert report['paired']['gbm']['worst_loss_increase'] == 1
    report = review_candidates({'gbm': result([2, 2, 2]), 'random_walk': result([2, 2, 2])})
    assert report['recommended_method'] == 'random_walk'


def test_export_blocks_mismatched_comparison_before_writing(tmp_path):
    from numeric_agent.engine import ForecastSession
    s = object.__new__(ForecastSession)
    s.diagnosed = True; s.audit = None
    s.cache = {'random_walk': result([2]), 'gbm': result([1])}
    for method, r in s.cache.items(): r['method'] = method
    s.cache['gbm']['fold_records'][0]['key'] = 'other'
    with pytest.raises(ValueError, match='incomparable'):
        s.export('gbm', 'invalid comparison')
    assert list(tmp_path.iterdir()) == []


def test_read_only_review_tool_registered(monkeypatch):
    import numeric_agent.agent as module
    monkeypatch.setattr(module, 'ToolCallingAgent', lambda **kwargs: SimpleNamespace(**kwargs))
    s = SimpleNamespace(review_evidence=lambda: {'recommended_method': 'random_walk'})
    agent = module.build_agent(s, None)
    import json
    tool = next(t for t in agent.tools if t.name == 'review_model_evidence')
    assert json.loads(tool())['recommended_method'] == 'random_walk'
