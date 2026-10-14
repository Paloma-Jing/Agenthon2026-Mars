"""Factor library: family dispatch, economics semantics, and no future leakage."""
import tomllib
from pathlib import Path

import numpy as np
import pytest

from numeric_agent.engine import ForecastSession
from numeric_agent.factors import compute_factors, data_quality, family_of

ROOT = Path(__file__).resolve().parents[1]


def session_for(uid, tmp_path):
    u = ROOT / 'track2-forecasting-public/units' / uid
    c = tomllib.loads((u / 'card.toml').read_text())
    return ForecastSession(u, u / 'card.toml', c['provenance']['data_cutoff'], tmp_path / f'{uid}.parquet')


def test_family_of_dispatch():
    assert family_of(pd := __import__('pandas').DataFrame(), 'UST_10Y') == 'rates'
    assert family_of(pd, 'EUR') == 'fx'
    assert family_of(pd, 'MOM') == 'factors'
    assert family_of(pd, 'CPI_ALL') == 'macro'
    assert family_of(pd, 'UNKNOWN') is None


def test_rates_factors(tmp_path):
    s = session_for('t2-F3-conundrum-joint-2005', tmp_path)
    r = compute_factors(s.panel, 'UST_10Y', 'rates')
    f = r['factors']
    assert f['family'] == 'rates'
    assert set(f['tenors_available']) >= {2, 5, 7, 10, 20, 30}
    assert f['spread_2s10s'] == pytest.approx(f['target_level'] - s.panel['UST_2Y'].dropna().iloc[-1], abs=1e-3)
    assert np.isfinite(f['nelson_siegel_level']) and np.isfinite(f['nelson_siegel_slope'])


def test_fx_factors(tmp_path):
    s = session_for('t2-F2-whatever-it-takes-2012', tmp_path)
    f = compute_factors(s.panel, 'EUR', 'fx')['factors']
    assert f['family'] == 'fx'
    assert f['level'] == pytest.approx(1.237, abs=1e-3)
    assert 'carry_proxy_rank' in f and 'no rate panel' in f.get('carry_proxy_note', '')
    assert 0.0 <= f['carry_proxy_rank'] <= 1.0


def test_factors_factors(tmp_path):
    s = session_for('t2-F1-ai-mom-2024', tmp_path)
    f = compute_factors(s.panel, 'MOM', 'factors')['factors']
    assert f['family'] == 'factors'
    for k in ('cum_return_1m', 'vol_1m', 'cs_momentum_z'):
        assert k in f


def test_macro_factors(tmp_path):
    s = session_for('t2-F1-cpi-glidepath-2023', tmp_path)
    f = compute_factors(s.panel, 'CPI_ALL', 'macro')['factors']
    assert f['family'] == 'macro'
    assert f['yoy'] == pytest.approx(0.0404, abs=1e-3)
    assert f['mom'] == pytest.approx(0.00124, abs=1e-3)
    assert 'core_headline_yoy_spread' in f and 'sahm_change_proxy' in f


def test_data_quality_summary(tmp_path):
    s = session_for('t2-F3-rates-into-factors-2022', tmp_path)
    q = data_quality(s.panel)
    assert q['n_assets'] == 12
    assert 0.0 <= q['missing_rate'] <= 1.0
    assert 'mean_abs_cross_corr' in q and 'pca_var_ratios' in q


def test_no_future_in_factor_panel(tmp_path):
    # The panel is already cut at as-of by the session; assert the last index equals the target's
    # last observation so no factor can peek beyond it.
    s = session_for('t2-F1-cpi-glidepath-2023', tmp_path)
    assert s.panel.index.max() == s.raw.index.max() == s.panel.index[-1]
