import numpy as np
import pandas as pd
import pytest
from numeric_agent.volatility import filter_innovations
from numeric_agent.engine import simulate
from numeric_agent.evidence import review_candidates


def test_filtered_shock_variance_uses_only_previous_observations():
    z = np.ones((100, 1)); z[-1] = 10
    pool, final = filter_innovations(z)
    assert final[0] == pytest.approx(.97 + .03 * 100)
    # Prior constant observations have variance one, including immediately before shock.
    assert pool[-1, 0] > 7


def test_recent_shocks_raise_forecast_variance():
    rng = np.random.default_rng(12)
    z = rng.normal(size=(200, 2))
    _, low = filter_innovations(z)
    z[-20:] *= 6
    _, high = filter_innovations(z)
    assert np.all(high > 10 * low)


def test_joint_paths_are_reproducible_positive_and_shared():
    x = np.exp(np.cumsum(np.random.default_rng(7).normal(0, .01, 250)))
    raw = pd.DataFrame({'a': x, 'b': 2*x})
    steps = np.array([[1, 10], [1, 10]])
    a = simulate(raw, 'fx', 'filtered_volatility', steps, draws=2000)
    b = simulate(raw, 'fx', 'filtered_volatility', steps, draws=2000)
    assert np.array_equal(a, b)
    assert (a > 0).all()
    assert np.allclose(a[:, 1], 2*a[:, 0])
    assert a[:, 0, 1].std() > a[:, 0, 0].std()
    from numeric_agent.engine import ForecastSession
    session = object.__new__(ForecastSession)
    assert session.anchor_increment('filtered_volatility', 'a') == 0.0


def test_gaps_do_not_become_synthetic_shocks():
    z = np.random.default_rng(0).normal(size=(100, 2))
    z[-10:, 0] = np.nan
    _, var = filter_innovations(z)
    _, before = filter_innovations(z[:-10])
    assert var[0] == before[0]
    with pytest.raises(ValueError, match='60 valid'):
        filter_innovations(z[:50])


def test_selection_instability_is_visible_without_changing_winner():
    def result(losses):
        return {'status': 'ok', 'fold_records': [dict(key=str(i), status='ok', normalized_composite=v)
                                                for i, v in enumerate(losses)]}
    r = review_candidates({'random_walk': result([2, 2, 2]), 'gbm': result([.1, 2.5, 2.5])})
    assert r['recommended_method'] == 'gbm'
    assert r['selection_stability']['winner_changes'] == 1


def test_nested_selector_cannot_see_outer_outcomes(monkeypatch):
    from types import SimpleNamespace
    from tools.evaluate_solution_policy import select_prefix
    seen = []
    class Fake(SimpleNamespace):
        def evaluate(self, method):
            seen.append((self.raw.copy(), self.panel.copy()))
            self.cache[method] = {'status': 'ok', 'normalized_composite': 1}
    data = pd.DataFrame({'a': np.arange(100.)})
    session = Fake(raw=data, panel=data, cache={})
    a, choices = select_prefix(session, 80)
    session.raw.iloc[80:] = 999999
    b, other = select_prefix(session, 80)
    assert choices == other
    assert all(len(raw)==80 and panel.index.max()==79 for raw, panel in seen)
    assert a.raw.equals(b.raw)
    assert session.cache == {}
