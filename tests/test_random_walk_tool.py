"""Random-walk tool integration, common score scale and order-independent evaluation."""
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from numeric_agent.engine import ForecastSession

ROOT = Path(__file__).resolve().parents[1]


def session(tmp_path):
    unit = ROOT / 'track2-forecasting-public/units/t2-F3-conundrum-joint-2005'
    card = tomllib.loads((unit / 'card.toml').read_text())
    return ForecastSession(unit, unit / 'card.toml', card['provenance']['data_cutoff'],
                           tmp_path / 'forecast.parquet')


def test_random_walk_tool_registered_and_exportable(tmp_path, monkeypatch):
    import numeric_agent.agent as module
    monkeypatch.setattr(module, 'ToolCallingAgent', lambda **kwargs: SimpleNamespace(**kwargs))
    s = session(tmp_path)
    agent = module.build_agent(s, model=None)
    tools = {tool.name: tool for tool in agent.tools}
    result = json.loads(tools['evaluate_random_walk']())
    assert result['method'] == 'random_walk'
    assert result['status'] == 'ok'
    assert 0 <= result['normalized_composite'] <= 8
    assert 'proxy_composite' not in result
    assert not s.out.exists()  # evaluation does not silently export
    tools['diagnose_series']()
    exported = json.loads(tools['export_forecast']('random_walk', 'Validated numerical baseline.'))
    assert exported['fallback_export'] is True
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        'forecast.parquet', 'forecast_meta.json', 'forecast_rationale.md']


def test_alternative_first_has_same_proxy_as_reference_first(tmp_path):
    a = session(tmp_path / 'a')
    first = a.evaluate('gbm')
    assert 'random_walk' not in a.cache  # M0 scales require no candidate execution
    assert first['normalized_composite'] is not None
    b = session(tmp_path / 'b')
    b.evaluate('random_walk')
    second = b.evaluate('gbm')
    assert first['normalized_composite'] == pytest.approx(second['normalized_composite'])


def test_compare_tool_returns_reference_first(monkeypatch):
    import numeric_agent.agent as module
    monkeypatch.setattr(module, 'ToolCallingAgent', lambda **kwargs: SimpleNamespace(**kwargs))
    calls = []
    def evaluate(method):
        calls.append(method)
        return {'method': method}
    s = SimpleNamespace(evaluate=evaluate)
    agent = module.build_agent(s, model=None)
    tool = next(t for t in agent.tools if t.name == 'compare_models')
    result = json.loads(tool(['gbm', 'gbm']))
    assert calls == ['random_walk', 'gbm']
    assert result[0]['method'] == 'random_walk'
