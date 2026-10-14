"""ML candidates: gbm / ridge_factors / var produce finite joint samples; var needs >=2 assets."""
import tomllib
from pathlib import Path

import numpy as np
import pytest

from numeric_agent.engine import ForecastSession, simulate

ROOT = Path(__file__).resolve().parents[1]


def session_for(uid, tmp_path):
    u = ROOT / 'track2-forecasting-public/units' / uid
    c = tomllib.loads((u / 'card.toml').read_text())
    return ForecastSession(u, u / 'card.toml', c['provenance']['data_cutoff'], tmp_path / f'{uid}.parquet')


@pytest.mark.parametrize('uid', ['t2-F3-conundrum-joint-2005', 't2-F1-ai-mom-2024', 't2-F1-cpi-glidepath-2023', 't2-F2-whatever-it-takes-2012'])
@pytest.mark.parametrize('method', ['gbm', 'ridge_factors'])
def test_ml_methods_finite_and_shaped(tmp_path, uid, method):
    s = session_for(uid, tmp_path)
    if method == 'gbm' and s.kind == 'log_return':
        with pytest.raises(ValueError, match='complete historical target'):
            simulate(s.raw, s.kind, method, s.steps, draws=50, seed=42, window=s.window, panel=s.panel)
        return
    out = simulate(s.raw, s.kind, method, s.steps, draws=50, seed=42, window=s.window, panel=s.panel)
    assert out.shape == (50, len(s.assets), s.steps.shape[1])
    assert np.isfinite(out).all()
    # FX and returns stay in-domain.
    if s.kind == 'fx':
        assert (out > 0).all()


def test_var_requires_multiple_assets(tmp_path):
    s = session_for('t2-F3-conundrum-joint-2005', tmp_path)
    out = simulate(s.raw, s.kind, 'var', s.steps, draws=50, seed=42, window=s.window, panel=s.panel)
    assert out.shape == (50, 4, 2) and np.isfinite(out).all()
    single = session_for('t2-F1-ai-mom-2024', tmp_path)
    with pytest.raises(ValueError, match='two target assets'):
        simulate(single.raw, single.kind, 'var', single.steps, draws=50, seed=42, window=single.window, panel=single.panel)


def test_ml_context_family_filter(tmp_path):
    # Restricting context families must not change the sample shape.
    s = session_for('t2-F3-rates-into-factors-2022', tmp_path)
    from numeric_agent.features import feature_table
    full = feature_table(s.raw, s.kind, s.panel, ['factors','rates'])
    restricted = feature_table(s.raw, s.kind, s.panel, ['factors'])
    assert any(c.startswith('rates:') for c in full)
    assert not any(c.startswith('rates:') for c in restricted)
    assert any(c.startswith('own:') for c in restricted)


def test_ml_candidates_run_through_evaluate(tmp_path):
    s = session_for('t2-F3-conundrum-joint-2005', tmp_path)
    s.diagnose()
    for m in ('random_walk', 'gbm'):
        r = s.evaluate(m)
        assert r['status'] == 'ok', (m, r)
