"""Participant reconstruction of published M0 §§3–5, not organizer reference files.

No realized outcome is read when constructing the distribution or expected-error scales.
Source: track2-forecasting-public/docs/M0-BASELINE.md (verified 2026-10-12).
"""
from dataclasses import dataclass
from pathlib import Path
import argparse, json, tomllib, zlib
import numpy as np
import pandas as pd
from scipy.special import gamma, hyp1f1
from scipy.stats import norm
from qfbench2_track_forecasting.cli import _read_panels, _series

TAIL_LEVELS=(.01,.05,.95,.99)

@dataclass
class Distribution:
    mean: np.ndarray
    covariance: np.ndarray
    factor: np.ndarray
    assets: list
    steps: np.ndarray
    history_rows: int
    diagonal_fallback: bool

    def draws(self, unit_id, n=500):
        seed=zlib.crc32(unit_id.encode()) & 0x7fffffff
        z=np.random.default_rng(seed).standard_normal((n,len(self.mean)))
        samples=self.mean+np.einsum('ij,kj->ik',z,self.factor)
        if not np.isfinite(samples).all():raise ValueError('nonfinite M0 draws')
        return samples

    def scales(self, tail_levels=TAIL_LEVELS):
        sd=np.sqrt(np.diag(self.covariance));joint=0.
        for i in range(len(sd)):
            for j in range(i):
                delta=float(self.mean[i]-self.mean[j])
                s2=max(0.,float(self.covariance[i,i]+self.covariance[j,j]-2*self.covariance[i,j]))
                if s2==0:continue
                s=np.sqrt(s2);z=delta/s
                first=s*np.sqrt(2/np.pi)*np.exp(-z*z/2)+delta*(1-2*norm.cdf(-z))
                half=np.sqrt(s)*2**.25*gamma(.75)/np.sqrt(np.pi)*hyp1f1(-.25,.5,-z*z/2)
                # Cancellation is possible for almost deterministic differences.
                variance=max(0.,float(first-half*half))
                joint+=2*variance
        result={'marginal':float(sd.mean()/np.sqrt(np.pi)),
                'joint':float(joint), 'tail':float(norm.pdf(norm.ppf(tail_levels)).mean()*sd.mean())}
        if not all(np.isfinite(v) and v>=0 for v in result.values()):raise ValueError('nonfinite M0 expected-error scale')
        return {k:v if v>0 else 1. for k,v in result.items()}


def distribution(series, target_type, steps, assets=None):
    """Series contain observed rows only; synthetic grid NaNs must not become observations."""
    assets=list(assets or series.keys()); changes={};anchors={}
    for a in sorted(assets):
        s=series[a].dropna().iloc[-300:].copy()
        if isinstance(s.index,pd.PeriodIndex):s.index=s.index.to_timestamp()
        s.index=pd.to_datetime(s.index)
        if len(s)<3 or s.index.has_duplicates or not s.index.is_monotonic_increasing or not np.isfinite(s).all():
            raise ValueError(f'invalid M0 history: {a}')
        gaps=s.index.to_series().diff().dt.total_seconds()/86400
        limit=max(10*float(gaps.median()),5.)
        if target_type=='level':d=s.diff();anchors[a]=float(s.iloc[-1])
        elif target_type=='log_return':
            if np.median(np.abs(s))>=.2:raise ValueError('M0 return-unit tripwire')
            d=s.copy();anchors[a]=0.
        else:raise ValueError('unsupported M0 target type')
        changes[a]=d.where(gaps<=limit)
    frame=pd.DataFrame(changes).dropna()
    if len(frame)<2:raise ValueError('not enough date-aligned M0 steps')
    mu=frame.mean();sigma=np.atleast_2d(np.cov(frame.to_numpy(),rowvar=False))
    order=list(frame.columns); steps=np.asarray(steps,dtype=int)
    if steps.ndim!=2 or steps.shape[0]!=len(assets) or np.any(steps<=0):raise ValueError('invalid M0 steps')
    ai=np.repeat([order.index(a) for a in assets],steps.shape[1]);flat=steps.ravel()
    means=np.repeat([anchors[a] for a in assets],steps.shape[1])+flat*mu.to_numpy()[ai]
    cov=np.minimum.outer(flat,flat)*sigma[np.ix_(ai,ai)]+np.eye(len(flat))*1.1e-9
    fallback=False
    try:factor=np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov=np.diag(np.diag(cov));factor=np.linalg.cholesky(cov);fallback=True
    return Distribution(means,cov,factor,assets,steps,len(frame),fallback)


def from_card(card_path, panels_path=None):
    path=Path(card_path);card=tomllib.loads(path.read_text());t=card['targets']
    assets=list(t['asset_ids']); horizons=list(t['horizons']);cutoff=str(card['provenance']['data_cutoff'])
    panels=_read_panels(Path(panels_path) if panels_path else path.parent)
    series={a:_series(panels,a,cutoff) for a in assets}
    for s in series.values():s.index=pd.to_datetime(s.index)
    specpath=path.parent/'forecast_spec.json'
    spec=json.loads(specpath.read_text()) if specpath.exists() else {}
    periods=spec.get('targets',{}).get('observation_periods')
    questions=spec.get('questions',[])
    steps=np.tile(horizons,(len(assets),1));notes=[]
    for i,a in enumerate(assets):
        s=series[a].iloc[-300:];last=s.index[-1];gaps=s.index.to_series().diff().dt.total_seconds()/86400
        threshold=max(10*float(gaps.median()),5.)
        spacing=float(gaps[gaps<=threshold].mean())
        for j,h in enumerate(horizons):
            period=periods[j] if periods else next((q.get('observation_period') for q in questions if q.get('asset',q.get('asset_id'))==a and q.get('horizon')==h),None)
            if period:
                end=pd.Period(period,freq='M');k=(end.year-last.year)*12+end.month-last.month
                if k<=0:raise ValueError('nonfuture observation period')
                steps[i,j]=k;notes.append('named observation period')
                if card['task']['id']=='t2-F1-sahm-watch-2024':
                    notes.append('Public table says 8/9 but current explicit periods imply 7/8; explicit period rule takes precedence; organizer equality unverified')
            elif card['task']['id']=='t2-F1-sahm-watch-2024':
                # Explicit public M0 §3.7 table; no inferred sealed target dates.
                steps[i,j]={145:8,165:9}[h];notes.append('published M0 table')
            else:
                dates=t.get('target_dates',[])
                date=dates[j] if j<len(dates) else None
                if date and len(s)>=3:
                    end=pd.to_datetime(date,errors='coerce')
                    if not pd.isna(end):
                        k=(end.year-last.year)*12+end.month-last.month if spacing>20 else round((end-last).days/spacing)
                        if k>0 and max(k,h)/max(min(k,h),1)>=2:steps[i,j]=k
                notes.append('declared horizon / published target date rule')
    return distribution(series,t.get('target_type','level'),steps,assets),card,notes


def export(card_path, out, panels_path=None):
    d,card,notes=from_card(card_path,panels_path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    uid=card['task']['id'];horizons=card['targets']['horizons'];samples=d.draws(uid)
    rows=[(n,a,h,float(samples[n,i*len(horizons)+j])) for n in range(500) for i,a in enumerate(d.assets) for j,h in enumerate(horizons)]
    frame=pd.DataFrame(rows,columns=['draw','asset','horizon','value']).astype({'draw':'int32','horizon':'int32'})
    frame.to_parquet(out/'forecast.parquet',index=False)
    meta={'unit_id':uid,'asof':str(card['provenance']['data_cutoff']),'asset_ids':d.assets,'horizons':horizons,'representation':'samples','n_draws':500,'selected_method':'m0_reconstruction','target':card['targets'].get('target_type','level')}
    (out/'forecast_meta.json').write_text(json.dumps(meta,indent=2))
    (out/'forecast_rationale.md').write_text('# Public M0 reconstruction\n\nText-blind Gaussian random walk; trailing 300 observations, gap-filtered date-aligned steps, joint asset/horizon covariance. Reconstructed by participant from public specification, not organizer-certified. No outcomes read.\n')
    params=card.get('scoring',{}).get('params',{})
    if params.get('joint','variogram')!='variogram' or params.get('tail_metric','pinball')!='pinball':raise ValueError('M0 scales only specified for variogram/pinball')
    return {'mean':d.mean.tolist(),'covariance':d.covariance.tolist(),'steps':d.steps.tolist(),'scales':d.scales(params.get('tail_levels',TAIL_LEVELS)), 'history_rows':d.history_rows,'step_sources':notes,'diagonal_fallback':d.diagonal_fallback}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--card',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=export(a.card,a.out)
    (a.out.parent/'m0_distribution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
