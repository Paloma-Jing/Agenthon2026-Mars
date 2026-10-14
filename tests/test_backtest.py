"""Phase C backtest acceptance: no future leak, official component consistency, fold layout."""
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qfbench2_common.scoring import crps
from qfbench2_track_forecasting.tail import tail_pinball

from tools.run_backtest import _component_losses, _origins, _truth

ROOT = Path(__file__).resolve().parents[1]


def _session(uid, tmp_path):
    from numeric_agent.engine import ForecastSession
    u = ROOT / 'track2-forecasting-public/units' / uid
    c = tomllib.loads((u / 'card.toml').read_text())
    return ForecastSession(u, u / 'card.toml', c['provenance']['data_cutoff'],
                           tmp_path / f'{uid}.parquet')


def test_backtest_panel_truncation_no_leak(tmp_path):
    s = _session('t2-F3-conundrum-joint-2005', tmp_path)
    origin = 900
    train = s.raw.iloc[:origin]
    panel_cut = s.panel.loc[s.panel.index <= train.index[-1]]
    assert panel_cut.index.max() == train.index[-1]
    # Tampering with values AFTER the origin must not change the truncated panel.
    tampered = s.panel.copy()
    tampered.loc[tampered.index > train.index[-1]] += 1e9
    panel_cut2 = tampered.loc[tampered.index <= train.index[-1]]
    pd.testing.assert_frame_equal(panel_cut, panel_cut2)
    # And the forecast from the truncated panel must be identical.
    from numeric_agent.engine import simulate
    s1 = simulate(train, s.kind, 'ridge_factors', s.steps, draws=50, seed=0,
                  window=s.window, panel=panel_cut, context_families=s.context_families)
    s2 = simulate(train, s.kind, 'ridge_factors', s.steps, draws=50, seed=0,
                  window=s.window, panel=panel_cut2, context_families=s.context_families)
    assert np.array_equal(s1, s2)


def test_official_components_consistency():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, (200, 6))
    truth = rng.normal(0, 1, 6)
    marg, jnt, tail = _component_losses(samples, truth)
    assert marg == pytest.approx(crps.crps_marginal(samples, truth))
    assert jnt == pytest.approx(crps.variogram_score(samples, truth, p=0.5))
    assert tail == pytest.approx(tail_pinball(samples, truth, levels=(0.01, 0.05, 0.95, 0.99)))


def test_origins_non_overlapping_and_ordered():
    o = _origins(1340, 126)
    assert o == sorted(o)
    # each consecutive pair is at least H apart, and every origin leaves a full H of future
    assert all(b - a >= 126 for a, b in zip(o, o[1:]))
    assert all(x + 126 <= 1340 for x in o)


def test_truth_semantics_match_kind(tmp_path):
    s = _session('t2-F1-ai-mom-2024', tmp_path)
    origin = 3000
    future = s.raw.iloc[origin:origin + int(s.steps.max())]
    with pytest.raises(ValueError, match='missing'):
        _truth(future, s.kind, s.steps)
    # Complete synthetic labels verify the cumulative target without filling real missing data.
    clean = pd.DataFrame({'MOM': np.full(int(s.steps.max()), .01)})
    t = _truth(clean, s.kind, s.steps)
    assert t.shape == s.steps.shape
    assert t[0, 0] == pytest.approx(int(s.steps[0, 0]) * np.log1p(.01))
