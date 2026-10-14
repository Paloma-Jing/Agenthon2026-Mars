"""Check time boundaries, return semantics, joint paths and tool guardrails."""
import numpy as np
import pandas as pd
import pytest
from numeric_agent.engine import simulate,ForecastSession
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_cumulative_returns_and_shared_paths():
    rng=np.random.default_rng(3);r=rng.normal(0,.01,500)
    raw=pd.DataFrame({'A':r,'B':r},index=pd.bdate_range('2000-01-03',periods=500))
    out=simulate(raw,'log_return','random_walk',np.array([[10,20],[10,20]]),draws=3000)
    assert np.allclose(out[:,0,:],out[:,1,:])
    assert out[:,0,1].std()>out[:,0,0].std()
    assert np.corrcoef(out[:,0,0],out[:,0,1])[0,1]>.6
    assert abs(out[:,0,0].mean()-10*np.log1p(r[-500:]).mean())<.005


def test_fx_positive_and_deterministic():
    raw=pd.DataFrame({'FX':np.exp(np.cumsum(np.random.default_rng(0).normal(0,.01,200)))})
    a=simulate(raw,'fx','ewma',np.array([[10,30]]));b=simulate(raw,'fx','ewma',np.array([[10,30]]))
    assert (a>0).all();assert np.array_equal(a,b)


def test_no_future_and_monthly_horizon(tmp_path):
    u=ROOT/'track2-forecasting-public/units/t2-F1-cpi-glidepath-2023'
    s=ForecastSession(u,u/'card.toml','2023-07-12',tmp_path/'forecast.parquet')
    assert s.steps.tolist()==[[8,9]]
    assert str(s.raw.index[-1])=='2023-05'
    with pytest.raises(ValueError,match='diagnose'):s.export('random_walk','test')
    assert 'title' not in s.describe() and 'asof' not in s.describe()


def test_future_rows_filtered(tmp_path):
    import tomllib
    u=ROOT/'track2-forecasting-public/units/t2-F1-ai-mom-2024'
    p=tmp_path/'input';p.mkdir();(p/'card.toml').write_bytes((u/'card.toml').read_bytes())
    d=pd.read_parquet(u/'factors_daily.parquet');row=d.iloc[-1].copy();row['date']=pd.Timestamp('2030-01-01').date();row['value']=999;row['asset']='MOM'
    pd.concat([d,pd.DataFrame([row])]).to_parquet(p/'factors_daily.parquet',index=False)
    s=ForecastSession(p,p/'card.toml','2024-05-31',tmp_path/'out/forecast.parquet')
    assert s.raw.index.max()==pd.Timestamp('2024-05-31')
    assert s.raw.max().max()<1


def test_gap_not_treated_as_daily_jump():
    raw=pd.DataFrame({'X':np.r_[np.full(80,100.),np.full(20,np.nan),np.full(80,200.)]})
    out=simulate(raw,'level','random_walk',np.array([[1,10]]),draws=200)
    assert np.all(out==200.)


def test_missing_target_explicitly_rejected(tmp_path):
    u=ROOT/'track2-forecasting-public/units/t2-F1-ai-mom-2024'
    p=tmp_path/'input';p.mkdir();(p/'card.toml').write_bytes((u/'card.toml').read_bytes())
    d=pd.read_parquet(u/'factors_daily.parquet');d[d.asset!='MOM'].to_parquet(p/'factors_daily.parquet',index=False)
    with pytest.raises(ValueError,match='no permitted history'):
        ForecastSession(p,p/'card.toml','2024-05-31',tmp_path/'out/forecast.parquet')
