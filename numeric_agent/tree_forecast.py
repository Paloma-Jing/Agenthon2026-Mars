"""Direct-horizon GBM with causal features and chronological calibration residuals.

Each draw resamples a whole asset/horizon residual vector. These are joint endpoint samples,
not a claim to simulate every intermediate daily path. Short calibration history fails explicitly.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .features import feature_table


def prepare_features(features, train, target, top_k=16):
    """Fit selection, centering and imputation on the training partition only."""
    fit=features.iloc[train]
    keep=(fit.notna().sum()>=30)&(fit.nunique(dropna=True)>1)
    names=fit.columns[keep]
    if len(names)==0:raise ValueError('no usable features')
    correlations={}
    for name in names:
        v=fit[name].to_numpy();good=np.isfinite(v)&np.isfinite(target)
        c=np.corrcoef(v[good],target[good])[0,1] if good.sum()>=30 and np.std(target[good])>1e-12 else 0.
        correlations[name]=abs(float(c)) if np.isfinite(c) else 0.
    selected=sorted(names,key=lambda n:(-correlations[n],n))[:top_k]
    mean=fit[selected].mean();std=fit[selected].std().replace(0.,1.)
    values=(features[selected]-mean)/std
    # Standardized zero means the training mean, not an invented target value.
    transformed=values.fillna(0.)
    for name in selected:
        if fit[name].isna().any():transformed[name+':missing']=features[name].isna().astype(float)
    return transformed,{'selected':selected,'training_mean':mean.to_dict(),
                        'training_std':std.to_dict(),'correlations':{n:correlations[n] for n in selected}}


def replay_training_rows(train, seed):
    """Offline adaptation: retain recent third and seeded sample from earlier fit rows.

    This does not perform online learning or read newly revealed outcomes. Calibration
    rows are not eligible for replay, and original chronological indices are preserved.
    """
    train=np.asarray(train,dtype=int)
    recent_count=max(20,len(train)//3)
    if len(train)<=recent_count:return train.copy()
    recent=train[-recent_count:];older=train[:-recent_count]
    selected=np.random.default_rng(seed).choice(older,size=min(recent_count,len(older)),replace=False)
    return np.sort(np.r_[selected,recent])


def mix_joint_samples(first, second, seed=42):
    """Equal-weight distribution mixture: preserve complete asset/horizon draw vectors.

    Do not average matched draws: that would shrink uncertainty and alter the joint law.
    """
    if first.shape!=second.shape or first.ndim!=3:raise ValueError('joint sample shapes differ')
    if not np.isfinite(first).all() or not np.isfinite(second).all():raise ValueError('nonfinite mixture input')
    rng=np.random.default_rng(seed);n=len(first);nfirst=n//2
    joined=np.concatenate([first[rng.choice(n,nfirst,replace=False)],second[rng.choice(n,n-nfirst,replace=False)]])
    return joined[rng.permutation(n)]


def simulate_tree(raw, kind, steps, draws, seed, window, panel=None, families=None, profile='core', validation_gap=None, audit=None, replay=False, calibration='empirical'):
    from .engine import _increments
    xraw=raw.iloc[-window:]
    x=feature_table(xraw,kind,panel,families,profile=profile)
    inc=_increments(xraw,kind)
    labels=[]
    for a,h in np.ndindex(steps.shape):
        k=int(steps[a,h]);s=inc.iloc[:,a]
        # t+1..t+k must all be observed. Missing targets are never imputed.
        if kind == 'log_return':
            y=s.shift(-1).rolling(k,min_periods=k).sum().shift(-(k-1))
        elif kind == 'fx':
            level=xraw.iloc[:,a]
            y=np.log(level.shift(-k).where(level.shift(-k)>0)/level.where(level>0))
        else:
            level=xraw.iloc[:,a]
            y=level.shift(-k)-level
        labels.append(y.to_numpy())
    y=np.column_stack(labels)
    valid=np.flatnonzero(np.isfinite(y).all(axis=1))
    # Hold out a chronological calibration block; purge training labels reaching that block.
    count=max(15,min(60,len(valid)//4))
    if len(valid)<70:raise ValueError('tree needs 70 complete historical target windows')
    cal=valid[-count:];cut=int(cal[0]);H=int(np.max(steps))
    # Conservative default: an additional forecast horizon separates training labels
    # from calibration origins. Explicit zero is retained for reproducible ablation.
    validation_gap=H if validation_gap is None else int(validation_gap)
    if validation_gap<0:raise ValueError('negative validation gap')
    train=valid[valid+H+validation_gap<cut]
    if len(train)<40:raise ValueError('tree needs 40 training rows after horizon embargo')
    eligible_train=train.copy()
    if replay:train=replay_training_rows(train,seed)
    xt=x.iloc[train]
    keep=(xt.notna().sum()>=30)&(xt.nunique(dropna=True)>1)
    if not keep.any():raise ValueError('tree has no usable training features')
    features=x.loc[:,keep]
    if validation_gap<0:raise ValueError('negative validation gap')
    if audit is not None:audit.update(profile=profile,replay=bool(replay),eligible_train_rows=len(eligible_train),training_indices=train.tolist(),train_rows=len(train),calibration_rows=len(cal),train_last=int(train[-1]),calibration_first=cut,horizon=H,validation_gap=int(validation_gap),cells=[])
    pool=[];means=[]
    for cell in range(y.shape[1]):
        cell_features=features
        selection={'selected':features.columns.tolist()}
        if profile in ('market', 'relative'):cell_features,selection=prepare_features(x,train,y[train,cell])
        if audit is not None:audit['cells'].append(selection)
        model=HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=7,
                 min_samples_leaf=15,l2_regularization=5.0,learning_rate=.05,
                 early_stopping=False,random_state=int(seed)+cell)
        model.fit(cell_features.iloc[train],y[train,cell])
        pool.append(y[cal,cell]-model.predict(cell_features.iloc[cal]))
        means.append(float(model.predict(cell_features.iloc[[-1]])[0]))
    # Do not refit on calibration observations: residuals remain out of fitted sample.
    pool=np.column_stack(pool)
    rng=np.random.default_rng(seed)
    if calibration == 'joint_kde':
        from .solution_models import joint_kernel_draws
        residual_draws = joint_kernel_draws(pool, draws, seed)
    elif calibration == 'empirical':
        residual_draws = pool[rng.integers(0,len(pool),draws)]
    else: raise ValueError('unknown calibration method')
    if audit is not None: audit['calibration'] = calibration
    state=np.asarray(means)[None,:]+residual_draws
    out=state.reshape(draws,*steps.shape)
    anchor=raw.iloc[-1].to_numpy()
    if kind=='fx':out=anchor[None,:,None]*np.exp(out)
    elif kind!='log_return':out=anchor[None,:,None]+out
    if not np.isfinite(out).all():raise ValueError('non-finite tree forecast')
    return out
