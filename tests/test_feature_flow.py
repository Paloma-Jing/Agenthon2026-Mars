"""Causal feature consumption, direct horizon labels and reduced tool menu."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from numeric_agent.features import feature_table
from numeric_agent.engine import METHODS, simulate
from numeric_agent.commander import validate_plan


def data():
    rng=np.random.default_rng(31);n=360
    idx=pd.bdate_range('2020-01-01',periods=n)
    signal=rng.normal(size=n)
    inc=np.r_[0,signal[:-1]]+.05*rng.normal(size=n)
    raw=pd.DataFrame({'UST_10Y':100+inc.cumsum()},index=idx)
    panel=raw.assign(UST_2Y=80+signal.cumsum())
    return raw,panel


def test_causal_feature_rows_and_future_panel_invariance():
    raw,panel=data();original=feature_table(raw,'level',panel,['rates'])
    changed=panel.copy();changed.iloc[251:,1]+=9999
    perturbed=feature_table(raw,'level',changed,['rates'])
    pd.testing.assert_frame_equal(original.iloc[:251],perturbed.iloc[:251])
    assert not original.iloc[252:].equals(perturbed.iloc[252:])
    a=simulate(raw.iloc[:250],'level','gbm',np.array([[1,3]]),draws=30,window=250,panel=panel)
    b=simulate(raw.iloc[:250],'level','gbm',np.array([[1,3]]),draws=30,window=250,panel=changed)
    np.testing.assert_array_equal(a,b)


def test_non_target_context_reaches_tree_predictions():
    raw,panel=data()
    altered=panel.copy();altered.iloc[-1,1]+=100
    a=simulate(raw,'level','gbm',np.array([[1,3]]),draws=60,seed=8,window=360,panel=panel)
    b=simulate(raw,'level','gbm',np.array([[1,3]]),draws=60,seed=8,window=360,panel=altered)
    assert not np.allclose(a,b)
    x=feature_table(raw,'level',panel,[])
    assert all(c.startswith('own:') for c in x)


def test_direct_target_and_embargo(monkeypatch):
    import numeric_agent.tree_forecast as module
    raw,panel=data();seen=[]
    class Spy:
        def __init__(self,**kwargs):pass
        def fit(self,x,y):
            seen.append((x.index.copy(),y.copy()));return self
        def predict(self,x):
            return np.zeros(len(x))
    monkeypatch.setattr(module,'HistGradientBoostingRegressor',Spy)
    out=simulate(raw,'level','gbm',np.array([[3]]),draws=20,window=360,panel=panel)
    idx,y=seen[0]
    expected=raw.UST_10Y.shift(-3)-raw.UST_10Y
    np.testing.assert_allclose(y,expected.loc[idx])
    # Complete origins 0..356; final 60 held out start at 297. Fit labels end before 297.
    assert raw.index.get_indexer(idx).max()+3<297
    assert out.shape==(20,1,1)


def test_bounded_three_candidate_menu(monkeypatch):
    import numeric_agent.agent as module
    monkeypatch.setattr(module,'ToolCallingAgent',lambda **kwargs:SimpleNamespace(**kwargs))
    assert METHODS==('random_walk','gbm','solution_ensemble')
    agent=module.build_agent(SimpleNamespace(),None)
    tools={t.name:t for t in agent.tools}
    assert {'build_features','evaluate_random_walk','compare_models'}<=tools.keys()
    with pytest.raises(ValueError,match='random_walk'):
        tools['compare_models'](['var'])
    with pytest.raises(ValueError):
        validate_plan({'decision':'modify','methods':['random_walk','ar'],'window':756,
                       'reason':'legacy','evidence':[]},[],756,False)
