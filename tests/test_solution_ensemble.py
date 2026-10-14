import numpy as np
import pandas as pd
import pytest
from numeric_agent.features import feature_table
from numeric_agent.solution_models import (joint_kernel_draws, simulate_trend,
    evidence_weights, weighted_joint_mixture, simulate_solution_ensemble)


def fixture(monthly=False):
    rng = np.random.default_rng(21); n = 250
    index = pd.period_range('2000-01', periods=n, freq='M') if monthly else pd.bdate_range('2000-01-03', periods=n)
    x = 100 + np.cumsum(.1 + rng.normal(0,.2,n))
    raw = pd.DataFrame({'UST_10Y': x}, index=index)
    return raw, raw.assign(UST_2Y=80+np.cumsum(rng.normal(0,.2,n)))


def test_joint_kernel_preserves_rank_and_covariance():
    x = np.random.default_rng(5).normal(size=50)
    pool = np.column_stack([x, 2*x, -x])
    draws = joint_kernel_draws(pool, 30000, 5)
    np.testing.assert_allclose(draws[:,1], 2*draws[:,0], atol=1e-12)
    np.testing.assert_allclose(draws[:,2], -draws[:,0], atol=1e-12)
    assert draws[:,0].var() == pytest.approx(x.var(), rel=.04)
    assert len(np.unique(draws[:,0])) > 1000


def test_mixture_selects_whole_vectors_and_rejects_bad_weights():
    a = np.tile(np.array([[[1., 2.], [3., 4.]]]), (100,1,1))
    b = 10*a
    out = weighted_joint_mixture({'a':a,'b':b}, {'a':.7,'b':.3}, 4)
    assert sum(np.all(out==a[0], axis=(1,2))) == 70
    assert sum(np.all(out==b[0], axis=(1,2))) == 30
    with pytest.raises(ValueError): weighted_joint_mixture({'a':a}, {'a':2.}, 4)


def test_weights_need_two_complete_origins_without_baseline_floor():
    losses = {'random_walk':[1.,1.], 'damped_trend':[.1,.1], 'relative_tree':[.01]}
    w = evidence_weights(losses, losses)
    assert 'relative_tree' not in w
    assert 0 < w['random_walk'] < .5
    assert sum(w.values()) == pytest.approx(1.)
    assert evidence_weights({}, losses) == {'random_walk':1.}


def test_relative_features_are_causal_and_respect_families():
    raw, panel = fixture(); x = feature_table(raw,'level',panel,['rates'],profile='relative')
    changed = panel.copy(); changed.iloc[200:] += 999
    y = feature_table(raw,'level',changed,['rates'],profile='relative')
    pd.testing.assert_frame_equal(x.iloc[:200], y.iloc[:200])
    assert 'own:UST_10Y:relative_move21' in x
    blocked = feature_table(raw,'level',panel,[],profile='relative')
    assert not any('relative_move' in c for c in blocked)


def test_trend_seasonality_and_joint_fx_paths_are_finite():
    raw, _ = fixture(monthly=True)
    raw.iloc[:,0] += .5*np.sin(2*np.pi*np.arange(len(raw))/12)
    out = simulate_trend(raw,'level',np.array([[1,12]]),500,4,180)
    assert np.isfinite(out).all() and out.shape == (500,1,2)
    fx = pd.DataFrame({'a':raw.iloc[:,0]/100,'b':raw.iloc[:,0]/50})
    out = simulate_trend(fx,'fx',np.array([[1,12],[1,12]]),500,4,180)
    assert (out>0).all()
    np.testing.assert_allclose(out[:,1],2*out[:,0],atol=1e-10)


def test_ensemble_internal_labels_and_context_end_before_forecast(monkeypatch):
    import numeric_agent.tree_forecast as tree
    raw,panel=fixture(); seen=[]
    def fake(history,kind,steps,draws,seed,window,context,families,**kwargs):
        seen.append((len(history),context.index.max(),history.index[-1],kwargs))
        return np.broadcast_to(history.iloc[-1].to_numpy()[None,:,None],(draws,*steps.shape)).copy()
    monkeypatch.setattr(tree,'simulate_tree',fake)
    audit={};steps=np.array([[2,5]])
    out=simulate_solution_ensemble(raw,'level',steps,50,4,250,panel,['rates'],audit)
    assert out.shape==(50,1,2) and np.isfinite(out).all()
    assert all(last<=end for _,last,end,_ in seen)
    assert seen[0][0]==240 and seen[1][0]==245
    assert audit['origins'][-1]['validation_end']==str(raw.index[-1])
    assert sum(audit['weights'].values()) == pytest.approx(1.)
    assert all(kwargs['profile']=='relative' and kwargs['calibration']=='joint_kde' for *_,kwargs in seen)
    changed = panel.copy(); changed.loc[pd.Timestamp('2040-01-01')] = 1e9
    other=simulate_solution_ensemble(raw,'level',steps,50,4,250,changed,['rates'])
    np.testing.assert_array_equal(out,other)


def test_relative_tree_calibration_and_fit_selection_are_isolated():
    from numeric_agent.tree_forecast import simulate_tree
    raw,panel=fixture();audit={}
    out=simulate_tree(raw,'level',np.array([[2,5]]),60,4,250,panel,['rates'],
                      profile='relative',calibration='joint_kde',audit=audit)
    assert out.shape==(60,1,2)
    assert audit['calibration']=='joint_kde'
    assert audit['train_last']+2*audit['horizon']<audit['calibration_first']
    assert all(len(cell['selected'])<=16 for cell in audit['cells'])


def test_commander_can_exclude_random_walk_and_weights_reach_output(monkeypatch):
    import numeric_agent.solution_models as module
    raw,panel=fixture()
    calls=[]
    def trend(history,kind,steps,draws,seed,window):
        calls.append('trend');return np.full((draws,*steps.shape),123.)
    monkeypatch.setattr(module,'simulate_trend',trend)
    audit={}
    output=module.simulate_solution_ensemble(raw,'level',np.array([[2,5]]),30,4,250,panel,['rates'],audit,
                                             {'random_walk':0.,'damped_trend':1.})
    assert calls==['trend'] and np.all(output==123.)
    assert audit['weights']=={'damped_trend':1.} and audit['weight_source']=='commander'
    assert audit['origins']==[]


@pytest.mark.parametrize('weights',[{'random_walk':.2},{'damped_trend':-1,'random_walk':2}, {'unknown':1}, {'random_walk':True}, {'random_walk':float('nan')}])
def test_bad_commander_weights_rejected(weights):
    from numeric_agent.solution_models import validate_ensemble_weights
    with pytest.raises(ValueError):validate_ensemble_weights(weights)
