from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from numeric_agent.m0 import from_card, distribution, Distribution
from numeric_agent.evaluation import aggregate, score
ROOT=Path(__file__).resolve().parents[1]/'track2-forecasting-public/units'


def test_published_exemplar_and_seed():
    d,_,_=from_card(ROOT/'t2-EXAMPLE-ust-curve-1m/card.toml')
    assert d.history_rows==128
    assert d.scales()==pytest.approx({'marginal':.05260,'joint':.1772,'tail':.006051},rel=.0004)
    assert np.array_equal(d.draws('test'),d.draws('test'))
    assert np.isfinite(d.draws('test')).all()


@pytest.mark.parametrize('uid,expected',[('t2-F1-cpi-glidepath-2023',[8,9]),('t2-F1-sahm-watch-2024',[7,8]),('t2-F4-covid-nfp-2020',[2]),('t2-F4-cpi-vintage-2022',[2])])
# Sahm card now names months; current last observation is June 2024, giving 7/8.
# The narrative table still says 8/9: retain this documented discrepancy.
def test_published_monthly_step_table(uid,expected):
    d,_,_=from_card(ROOT/uid/'card.toml');assert d.steps[0].tolist()==expected


def test_gap_not_counted_as_single_jump_and_joint_horizon_covariance():
    idx=pd.date_range('2020-01-01',periods=20).append(pd.DatetimeIndex(['2025-01-01']))
    s=pd.Series(np.r_[np.arange(20.),10000.],index=idx)
    d=distribution({'a':s},'level',np.array([[1,3]]))
    assert d.history_rows==19
    assert d.mean.tolist()==[10001.,10003.]
    assert d.covariance[0,1]==pytest.approx(d.covariance[0,0]-1.1e-9)


def test_joint_scale_against_independent_quadrature():
    mean=np.array([.4,0.]);cov=np.array([[.8,.1],[.1,.5]])
    d=Distribution(mean,cov,np.linalg.cholesky(cov),['a','b'],np.ones((2,1)),100,False)
    delta=.4;sd=np.sqrt(1.1);cusp=-delta/sd
    fn=lambda z:abs(delta+sd*z)**.5*norm.pdf(z)
    half=quad(fn,-np.inf,cusp)[0]+quad(fn,cusp,np.inf)[0]
    first=quad(lambda z:abs(delta+sd*z)*norm.pdf(z),-np.inf,cusp)[0]+quad(lambda z:abs(delta+sd*z)*norm.pdf(z),cusp,np.inf)[0]
    assert d.scales()['joint']==pytest.approx(2*(first-half**2),rel=1e-8)


def test_equal_mean_failure_clip_and_duplicate_rejection():
    rows=[{'key':'a','status':'ok','normalized_composite':2.},{'key':'b','status':'failed'}, {'key':'c','status':'ok','normalized_composite':20.}]
    assert aggregate(rows)['mean_normalized_composite']==6.
    with pytest.raises(ValueError,match='duplicate'):aggregate(rows+rows[:1])


def test_single_cell_official_weight_redistribution():
    result=score(np.arange(200).reshape(-1,1)/100,[.7],{'marginal':2.,'joint':1.,'tail':3.},{})
    expected=(5/7)*result['marginal']/2+(2/7)*result['tail']/3
    assert result['normalized_composite']==pytest.approx(expected)


def test_controller_failure_exports_m0_and_records_reason(tmp_path,monkeypatch):
    import json,sys
    import numeric_agent.__main__ as cli
    unit=ROOT/'t2-F2-brl-transfer-2013'
    import tomllib
    card=tomllib.loads((unit/'card.toml').read_text())
    out=tmp_path/'output/forecast.parquet'
    monkeypatch.setenv('QFBENCH_AUDIT_DIR',str(tmp_path/'audit'))
    monkeypatch.setattr(sys,'argv',['forecast','--panels',str(unit),'--asof',card['provenance']['data_cutoff'],'--out',str(out)])
    def fail(*_):raise RuntimeError('controller unavailable')
    monkeypatch.setattr(cli,'build_model',fail)
    assert cli.main()==0
    meta=json.loads((out.parent/'forecast_meta.json').read_text())
    assert meta['selected_method']=='m0_reconstruction'
    assert len(list(out.parent.iterdir()))==3
    state=json.loads((tmp_path/'audit/run_status.json').read_text())
    assert state['fallback_trigger']=='RuntimeError'


def test_comparison_keeps_unattempted_method_in_denominator():
    from tools.compare_backtests import compare
    r={'key':'a','method':'m0','status':'ok','normalized_composite':1.,'scales':{'marginal':1.,'joint':1.,'tail':1.}}
    result=compare([r],['m0','gbm'])
    assert result['methods']['gbm']['mean_normalized_composite']==8.
    with pytest.raises(ValueError,match='duplicate'):compare([r,r],['m0'])
