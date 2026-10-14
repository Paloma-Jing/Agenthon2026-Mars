"""Shared M0 expected-error normalization and equal-weight arithmetic aggregation.

Participant-side reconstructed scales are never passed off as organizer-certified scales.
"""
import numpy as np
from qfbench2_track_forecasting.scoring import _composite
from .m0 import distribution, TAIL_LEVELS


def parameters(card, cells):
    p=card.get('scoring',{}).get('params',{})
    joint=p.get('joint','variogram');tail=p.get('tail_metric','pinball')
    if joint!='variogram' or tail!='pinball':
        raise ValueError('published M0 expected-error formulas require variogram/pinball')
    w=p.get('weights',{'marginal':.5,'joint':.3,'tail':.2})
    weights=tuple(float(w[k]) for k in ('marginal','joint','tail'))
    if cells==1:
        total=weights[0]+weights[2];weights=(weights[0]/total,0.,weights[2]/total)
    return dict(weights=weights,tail_levels=tuple(p.get('tail_levels',TAIL_LEVELS)),joint=joint,tail_metric=tail)


def score(samples, truth, scales, card):
    y=np.asarray(truth,dtype=float).ravel();s=np.asarray(samples,dtype=float).reshape(len(samples),-1)
    if not np.isfinite(s).all() or not np.isfinite(y).all():raise ValueError('nonfinite forecast/truth')
    result=_composite(s,y,ref_scale=scales,**parameters(card,len(y)))
    result.update(normalized_composite=min(8.,float(result['composite'])),unclipped_composite=float(result['composite']))
    return result


def aggregate(records):
    """One row per committed task; duplicates or missing loss on success are errors."""
    keys=[r['key'] for r in records]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate evaluation key')
    if not records:raise ValueError('empty evaluation roster')
    losses=[min(8.,float(r['normalized_composite'])) if r['status']=='ok' else 8. for r in records]
    if not np.isfinite(losses).all() or min(losses)<0:raise ValueError('invalid losses')
    return {'count':len(records),'failed':sum(r['status']!='ok' for r in records),'mean_normalized_composite':float(np.mean(losses))}


def fold_reference(train, target, steps, card):
    d=distribution({a:train[a] for a in train.columns},target,steps,list(train.columns))
    return d,d.scales(parameters(card,steps.size)['tail_levels'])


def strict_truth(future, kind, steps):
    """No invented endpoint or skipped missing cumulative return. Shared exclusion for all models."""
    result=np.empty(steps.shape)
    for a,h in np.ndindex(steps.shape):
        k=int(steps[a,h]);v=future.iloc[:k,a]
        if len(v)!=k:raise ValueError('incomplete future window')
        if kind=='log_return':
            if v.isna().any() or (v<=-1).any():raise ValueError('missing/invalid cumulative-return observation')
            result[a,h]=np.log1p(v).sum()
        else:
            if not np.isfinite(v.iloc[-1]):raise ValueError('missing level endpoint')
            result[a,h]=v.iloc[-1]
    return result
