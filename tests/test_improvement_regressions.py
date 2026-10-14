"""Regression tests for the deterministic defects found in the dev2 code review.

D1: ML recurrence lag order — training/initial state use newest-first order, but the
    per-step update appended the new increment to the END of the history, so the second
    and later steps read a stale lag.
D2: VAR anchor/rationale — `anchor_increment` indexed `fit.params[j]` (a length-n vector
    for multi-asset VAR) and passed it to float(), raising TypeError during export.
"""
import numpy as np
import pandas as pd
import pytest

import numeric_agent.engine as E
from numeric_agent.engine import simulate


def test_d1_ml_recurrence_uses_newest_lag(monkeypatch):
    # D1 is the invariant that training, initial state, and recurrence all use newest-first lag
    # order. Test the two building blocks directly: the feature rows and the history shift.
    from numeric_agent.engine import _lag_features, _push_lag

    # Training feature rows: newest first.
    s = np.array([1.0, 2, 3, 4, 5, 6, 7])
    xx = _lag_features(s, 5)
    assert xx[0].tolist() == [5, 4, 3, 2, 1]
    assert xx[1].tolist() == [6, 5, 4, 3, 2]

    # Recurrence: the new increment becomes index 0, the oldest drops.
    hist = np.array([[[5, 4, 3, 2, 1]]])  # (1, 1, 5) newest-first
    inc = np.array([[6.0]])               # (1, 1)
    out = _push_lag(hist, inc)
    assert out[0, 0].tolist() == [6, 5, 4, 3, 2]

    # End-to-end: a deterministic linear model predicting y = newest-lag on centered arithmetic
    # diffs gives step-2 state = 2*newest only when recurrence is newest-first.
    class FakeRidge:
        def __init__(self, *a, **k):
            self.coef_ = None
            self.intercept_ = None

        def fit(self, X, y):
            self.coef_ = np.zeros(X.shape[1])
            self.coef_[0] = 1.0  # predict = first lag column (newest)
            self.intercept_ = 0.0
            return self

        def predict(self, X):
            return X[:, 0]

    class IdentityScaler:
        def fit(self, X):
            return self

        def transform(self, X):
            return X

    monkeypatch.setattr(E, 'Ridge', FakeRidge)
    monkeypatch.setattr(E, 'StandardScaler', IdentityScaler)

    n = 60
    diffs = np.arange(-(n - 1) / 2, n / 2, 1.0)
    raw = pd.DataFrame({'A': np.cumsum(diffs)}, index=pd.bdate_range('2020-01-01', periods=n))
    steps = np.array([[2]])
    out = simulate(raw, 'level', 'ridge_factors', steps, draws=1, seed=0, window=n, panel=None)
    newest = diffs[-1]
    assert out[0, 0, 0] == pytest.approx(float(raw.iloc[-1, 0]) + 2 * newest)


def test_d2_var_export_full_chain(tmp_path):
    from pathlib import Path
    import tomllib
    from numeric_agent.engine import ForecastSession
    root = Path(__file__).resolve().parents[1]
    u = root / 'track2-forecasting-public/units/t2-F3-conundrum-joint-2005'
    c = tomllib.loads((u / 'card.toml').read_text())
    s = ForecastSession(u, u / 'card.toml', c['provenance']['data_cutoff'],
                        tmp_path / 'forecast.parquet')
    s.allowed_methods = ('random_walk', 'var')  # explicit legacy regression only
    s.diagnose()
    assert s.evaluate('random_walk')['status'] == 'ok'
    r = s.evaluate('var')
    assert r['status'] == 'ok', r
    s.export('var', 'Regression test: VAR rationale must derive its anchor from the fitted model.')
    meta_rationale = tmp_path / 'forecast_rationale.md'
    assert meta_rationale.exists()
    text = meta_rationale.read_text()
    assert 'drift increment' in text
    # Every asset's drift line must have a numeric value (no TypeError leak).
    for a in s.assets:
        assert f'{a}: drift increment = ' in text


def test_d9_features_filter_keeps_target_accessible(tmp_path):
    # A multi-family unit (rates + factors) whose targets are factors: selecting only the
    # 'rates' context family must not drop the target columns from the context panel and then
    # KeyError inside _context_features.
    from pathlib import Path
    import tomllib
    from numeric_agent.engine import ForecastSession
    root = Path(__file__).resolve().parents[1]
    u = root / 'track2-forecasting-public/units/t2-F3-rates-into-factors-2022'
    c = tomllib.loads((u / 'card.toml').read_text())
    s = ForecastSession(u, u / 'card.toml', c['provenance']['data_cutoff'],
                        tmp_path / 'forecast.parquet')
    assert set(s.families) == {'factors', 'rates'}
    out = simulate(s.raw, s.kind, 'ridge_factors', s.steps, draws=20, seed=0,
                   window=s.window, panel=s.panel, context_families=['rates'])
    assert out.shape == (20, len(s.assets), s.steps.shape[1])
    assert np.isfinite(out).all()


def test_d5_time_varying_ctx_no_lookahead_and_variance():
    from numeric_agent.engine import _time_varying_ctx

    # No lookahead: perturbing future increments must not change past rows.
    z = np.arange(1.0, 101)
    ctx = _time_varying_ctx(z)
    z2 = z.copy()
    z2[80:] *= 100.0
    ctx2 = _time_varying_ctx(z2)
    # Row t uses increments up to t-1, so rows 0..80 (which look back to index 79) are unchanged.
    assert np.allclose(ctx[:81], ctx2[:81], equal_nan=True)
    assert not np.allclose(ctx[82:], ctx2[82:], equal_nan=True)

    # Variance: on a real (non-constant) series the time-varying columns carry signal, so they
    # are not dropped as near-constant by the ML branch.
    rng = np.random.default_rng(0)
    zv = rng.normal(0, 1, 200)
    cv = _time_varying_ctx(zv)
    assert np.nanstd(cv[63:], axis=0).min() > 1e-6


def test_d5_time_varying_feature_affects_forecast(tmp_path):
    # A change confined to increments beyond the 5-step lag window (inside the 63-step volatility
    # window) must change the forecast, proving the time-varying context is actually consumed.
    from pathlib import Path
    import tomllib
    from numeric_agent.engine import ForecastSession
    root = Path(__file__).resolve().parents[1]
    u = root / 'track2-forecasting-public/units/t2-F1-ai-mom-2024'
    c = tomllib.loads((u / 'card.toml').read_text())
    s = ForecastSession(u, u / 'card.toml', c['provenance']['data_cutoff'],
                        tmp_path / 'forecast.parquet')
    raw = s.raw.iloc[-s.window:].copy()
    raw2 = raw.copy()
    # Perturb a level far enough back to be outside the 5-lag / 21-day window but inside 63.
    raw2.iloc[-30, 0] = raw2.iloc[-30, 0] + 50 * raw.iloc[:, 0].diff().std()
    out1 = simulate(raw, s.kind, 'ridge_factors', s.steps, draws=50, seed=0,
                    window=s.window, panel=s.panel, context_families=s.context_families)
    out2 = simulate(raw2, s.kind, 'ridge_factors', s.steps, draws=50, seed=0,
                    window=s.window, panel=s.panel, context_families=s.context_families)
    assert not np.allclose(out1, out2)


def test_evaluate_reports_proxy_composite(tmp_path):
    # evaluate must report joint+tail-aware proxy composite relative to random_walk, where the
    # reference itself has no proxy (its ratio is 1.0 by definition).
    from pathlib import Path
    import tomllib
    from numeric_agent.engine import ForecastSession
    root = Path(__file__).resolve().parents[1]
    u = root / 'track2-forecasting-public/units/t2-F3-conundrum-joint-2005'
    c = tomllib.loads((u / 'card.toml').read_text())
    s = ForecastSession(u, u / 'card.toml', c['provenance']['data_cutoff'],
                        tmp_path / 'forecast.parquet')
    s.diagnose()
    rw = s.evaluate('random_walk')
    assert rw['status'] == 'ok'
    assert 0 <= rw['normalized_composite'] <= 8
    ar = s.evaluate('gbm')
    assert ar['status'] == 'ok'
    assert ar.get('normalized_composite') is not None
    assert [r['key'] for r in rw['fold_records']] == [r['key'] for r in ar['fold_records']]
    # The proxy is the weighted geometric-mean ratio, not the old normalized marginal.
    assert ar['normalized_composite'] == pytest.approx(np.mean([r['normalized_composite'] for r in ar['fold_records']]))
