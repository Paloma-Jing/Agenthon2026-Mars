"""Market feature causality, units, partition isolation and delayed calibration."""
import numpy as np
import pandas as pd
from numeric_agent.features import feature_table
from numeric_agent.tree_forecast import prepare_features,simulate_tree


def fixture():
    rng=np.random.default_rng(5);idx=pd.bdate_range('2019-01-01',periods=360)
    panel=pd.DataFrame({'UST_2Y':10+np.cumsum(rng.normal(size=360)),
                        'UST_10Y':20+np.cumsum(rng.normal(size=360))},index=idx)
    return panel[['UST_10Y']],panel


def test_market_rows_do_not_read_future_and_respect_family_filter():
    raw,panel=fixture();x=feature_table(raw,'level',panel,['rates'],profile='market')
    changed=panel.copy();changed.iloc[250:]+=1e6
    y=feature_table(raw,'level',changed,['rates'],profile='market')
    pd.testing.assert_frame_equal(x.iloc[:250],y.iloc[:250])
    assert 'market:rates:mean' in x
    blocked=feature_table(raw,'level',panel,[],profile='market')
    assert all(c.startswith('own:') for c in blocked)


def test_market_average_invariant_to_asset_units():
    raw,panel=fixture();x=feature_table(raw,'level',panel,['rates'],profile='market')
    changed=panel.copy();changed.UST_2Y*=100
    y=feature_table(raw,'level',changed,['rates'],profile='market')
    np.testing.assert_allclose(x['market:rates:mean'],y['market:rates:mean'],equal_nan=True,atol=1e-11)


def test_selection_scaling_and_imputation_fit_training_only():
    rng=np.random.default_rng(10);x=pd.DataFrame(rng.normal(size=(200,30)),columns=[f'f{i}' for i in range(30)])
    x.loc[:9,'f0']=np.nan;train=np.arange(100);target=x.f1.iloc[train].to_numpy()
    a,fit=prepare_features(x,train,target)
    changed=x.copy();changed.iloc[100:]=1e9
    b,other=prepare_features(changed,train,target)
    assert fit==other and len(fit['selected'])==16
    assert fit['selected'][0]=='f1'
    pd.testing.assert_frame_equal(a.iloc[train],b.iloc[train])
    assert np.isfinite(a.to_numpy()).all()


def test_extra_gap_and_train_only_selection_are_auditable():
    raw,panel=fixture();details={}
    out=simulate_tree(raw,'level',np.array([[3]]),20,5,360,panel,['rates'],profile='market',validation_gap=15,audit=details)
    assert details['train_last']+3+15<details['calibration_first']
    assert len(details['cells'][0]['selected'])<=16
    assert out.shape==(20,1,1) and np.isfinite(out).all()
