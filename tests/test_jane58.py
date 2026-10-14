"""Compact features, offline history replay and joint distribution mixing."""
import numpy as np
import pandas as pd
from numeric_agent.features import feature_table
from numeric_agent.tree_forecast import replay_training_rows,mix_joint_samples,simulate_tree


def fixture():
    idx=pd.bdate_range('2020-01-01',periods=360);rng=np.random.default_rng(91)
    raw=pd.DataFrame({'UST_10Y':50+np.cumsum(rng.normal(size=360))},index=idx)
    return raw,raw.assign(UST_2Y=20+np.cumsum(rng.normal(size=360)))


def test_compact_last_lag_and_rolling_max_are_causal():
    raw,panel=fixture();x=feature_table(raw,'level',panel,['rates'],profile='compact')
    assert 'own:UST_10Y:lag1' in x and 'own:UST_10Y:lag2' not in x
    np.testing.assert_allclose(x['own:UST_10Y:max21'],raw.UST_10Y.diff().rolling(21,min_periods=10).max(),equal_nan=True)
    altered=raw.copy();altered.iloc[250:]+=1e5
    other=feature_table(altered,'level',panel,['rates'],profile='compact')
    pd.testing.assert_frame_equal(x.iloc[:250],other.iloc[:250])


def test_replay_uses_only_fit_rows_retains_recent_and_reproducible():
    train=np.arange(200);a=replay_training_rows(train,8);b=replay_training_rows(train,8)
    np.testing.assert_array_equal(a,b)
    assert set(a)<=set(train) and len(a)==132 and len(set(a))==132
    assert set(train[-66:])<=set(a)
    assert (np.diff(a)>0).all()


def test_mixture_keeps_whole_joint_vectors_and_between_model_variance():
    a=np.tile(np.array([[[0.,1.],[2.,3.]]]),(100,1,1));b=a+100
    result=mix_joint_samples(a,b,7)
    assert np.sum(result[:,0,0]==0)==50 and np.sum(result[:,0,0]==100)==50
    np.testing.assert_array_equal(result[:,1,1]-result[:,0,0],np.full(100,3.))
    assert result[:,0,0].var()==2500  # matched-draw averaging would erase all variance
    np.testing.assert_array_equal(result,mix_joint_samples(a,b,7))


def test_replay_keeps_labels_outside_calibration_and_reports_indices():
    raw,panel=fixture();audit={}
    out=simulate_tree(raw,'level',np.array([[3]]),30,5,360,panel,['rates'],profile='compact',replay=True,audit=audit)
    assert audit['replay'] and audit['train_rows']<audit['eligible_train_rows']
    assert max(audit['training_indices'])+audit['horizon']+audit['validation_gap']<audit['calibration_first']
    assert np.isfinite(out).all()
