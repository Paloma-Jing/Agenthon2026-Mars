"""Numeric-only diagnostics, rolling validation and coherent sampled forecast paths.

No text files, external market data or other units are read. Model selection scores
are historical diagnostics, never official hidden-task scores.
"""
from pathlib import Path
import json
import re
import time
import tomllib
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
from arch import arch_model
from qfbench2_track_forecasting.cli import _read_panels, _series, _monthly_inputs
from qfbench2_common.scoring import crps as _official_crps
from qfbench2_common.scoring.crps import crps_ensemble
from qfbench2_track_forecasting.tail import tail_pinball

METHODS = ('random_walk', 'gbm', 'solution_ensemble')
LEGACY_METHODS = ('ewma', 'ar', 'ridge', 'garch', 'ridge_factors', 'var')
EXPERIMENTAL_METHODS = ('filtered_volatility',)

#: Methods that use a learned conditional mean over lag/context features (ML branch).
ML_METHODS = ('gbm', 'ridge_factors', 'var')

#: Panel filename stem -> data family. Drives which factor library a context asset feeds.
FAMILY_BY_STEM = {
    'rates_daily': 'rates',
    'g10_fx_daily': 'fx',
    'factors_daily': 'factors',
    'macro_monthly': 'macro',
    'em_transfer_early': 'fx',  # transferred EM FX series, same log-level semantics as G10
}


def _asset_col(df):
    for c in ('asset', 'asset_id'):
        if c in df.columns:
            return c
    return None


def write_audit(path, text, mode='w'):
    """Best-effort audit write: the trail is diagnostic, the forecast is the deliverable.

    The platform gives the image a read-only root, a 64 MiB /tmp and a 1 GiB writable layer, so
    an audit write can legitimately fail while the run itself is fine. Losing a log line must not
    turn a scorable unit into a zero.
    """
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open(mode) as handle:handle.write(text)
    except OSError as exc:
        print(f'WARN audit write skipped ({path.name}: {exc})',flush=True)


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    return value


def ridge_predict(model,x):
    # Explicit dot avoids spurious Accelerate matmul warnings on this ARM runtime.
    scaled=model[0].transform(x)
    return np.einsum('ij,j->i',scaled,model[1].coef_)+model[1].intercept_


def _ridge_predict_plain(model, x):
    # Same einsum path for a bare Ridge (no scaler): sklearn's predict() emits spurious
    # divide-by-zero/overflow matmul warnings under Accelerate on this ARM runtime.
    return np.einsum('ij,j->i', x, model.coef_) + model.intercept_


def _push_lag(hist, inc):
    """Shift a newest-first lag history one step: `inc` becomes the newest, the oldest drops.

    `hist` has shape (..., lag) with index 0 the most recent value; this matches the feature
    rows built by `lag_features` (``s[t-lag:t][::-1]``) and the initial history. Keeping this
    single helper is what makes the training, initial and recurrence orders identical.
    """
    return np.concatenate([inc[..., None], hist[..., :-1]], axis=-1)


def _increments(raw, kind):
    """Per-asset increments in model units, from target levels/returns as supplied."""
    x = raw.copy()
    if kind == 'log_return':
        if (x <= -1).any().any(): raise ValueError('simple returns must exceed -1')
        return np.log1p(x)
    if kind == 'fx':
        if (x <= 0).any().any(): raise ValueError('FX levels must be positive')
        return np.log(x).diff()
    return x.diff()


def _context_features(panel, asset, kind):
    """A compact, point-in-time context vector for one target asset (no future rows).

    Fixed small feature set so the ML branch stays bounded and interpretable. Every value uses
    only panel history up to the last observation; family-specific signals are optional extras.
    The target asset's own history may be absent when the commander restricted the context to
    other families, so those entries fall back to 0 rather than raising.
    """
    name = str(asset)
    if name in panel.columns:
        s = panel[name].dropna()
        vals = [
            _horizon_return(s, 21),
            _horizon_return(s, 63),
            float(s.diff().std(ddof=1)) if len(s) >= 2 else np.nan,
        ]
    else:
        vals = [np.nan, np.nan, np.nan]
    # cross-sectional z of 3m momentum and vol across all panel assets
    moms, vols = {}, {}
    for c in panel.columns:
        sc = panel[c].dropna()
        if len(sc) >= 64:
            moms[c] = _horizon_return(sc, 63)
            vols[c] = sc.diff().dropna().iloc[-63:].std(ddof=1)
    if name in moms and len(moms) > 1:
        m = np.array(list(moms.values())); vals.append((moms[name] - m.mean()) / (m.std() + 1e-12))
    if name in vols and len(vols) > 1:
        v = np.array(list(vols.values())); vals.append((vols[name] - v.mean()) / (v.std() + 1e-12))
    # family-specific economics
    if name.startswith('UST_'):
        tens = sorted(_tenor_from_name(c) for c in panel.columns if _tenor_from_name(c))
        if len(tens) >= 2 and 'UST_2Y' in panel.columns and 'UST_10Y' in panel.columns:
            s2 = panel['UST_2Y'].dropna(); s10 = panel['UST_10Y'].dropna()
            if len(s2) and len(s10):
                vals.append(float(s10.iloc[-1] - s2.iloc[-1]))
    elif name in ('CPI_ALL', 'CPI_CORE', 'PCE_ALL', 'PCE_CORE'):
        if 'CPI_ALL' in panel.columns and 'CPI_CORE' in panel.columns:
            ca = panel['CPI_ALL'].dropna(); cc = panel['CPI_CORE'].dropna()
            if len(ca) > 12 and len(cc) > 12:
                vals.append(float(np.log(cc.iloc[-1] / cc.iloc[-13]) - np.log(ca.iloc[-1] / ca.iloc[-13])))
    return np.clip(np.nan_to_num(np.array(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), -1e3, 1e3)


def _horizon_return(s, periods):
    if len(s) < periods + 1: return np.nan
    x = s.iloc[-1 - periods:].to_numpy(float)
    return float(np.log(x[-1] / x[0])) if (x > 0).all() else float(x[-1] - x[0])


def _lag_features(s, lag):
    """Newest-first lag rows: row t = [s[t-1], s[t-2], ..., s[t-lag]]. Matches _push_lag order."""
    rows = []
    for t in range(lag, len(s)):
        rows.append(s[t - lag:t][::-1])
    return np.array(rows)


def _time_varying_ctx(z_j, windows=(21, 63)):
    """Per-timestep momentum and volatility of increments, no lookahead.

    Each row t uses only increments up to t-1 (shift(1)), so it can predict the increment at t
    without seeing it. Returns an (n, len(windows)+1) float array [cum_21, cum_63, vol_63].
    """
    s = pd.Series(z_j).shift(1)  # exclude the current increment (predicting it)
    cols = []
    for w in windows:
        cols.append(s.rolling(w, min_periods=max(2, w // 3)).sum().to_numpy())
    cols.append(s.rolling(63, min_periods=10).std().to_numpy())
    return np.column_stack(cols)


def _ml_feats(scaler, lags_raw, ctx_time_now, ctx_vec):
    """Build ML feature rows: standardized [lags + time-varying context] + static context vector."""
    lc = np.concatenate([lags_raw,
                         np.nan_to_num(np.asarray(ctx_time_now, dtype=float).reshape(lags_raw.shape[0], -1), nan=0.0)],
                        axis=1)
    X = scaler.transform(lc)
    if ctx_vec.size:
        X = np.concatenate([X, np.broadcast_to(ctx_vec, (lags_raw.shape[0], ctx_vec.size))], axis=1)
    return X


def _tenor_from_name(name):
    m = re.search(r'(\d+)Y\b', str(name))
    return int(m.group(1)) if m else None


def _simulate_ml(raw, kind, method, steps, draws, seed, window, panel, context_families=None, time_varying=True):
    """ML conditional-mean candidates with joint residual bootstrap.

    gbm / ridge_factors: per-asset model on 5 lag increments + context features (constant at the
    last observation). var: VAR(1) joint linear dynamics. All three resample complete residual
    rows so the estimated cross-asset dependence of innovations is preserved. Context features
    come only from panel history up to the last observation (the caller already cut at as-of).
    """
    x = raw.iloc[-window:].copy()
    delta = _increments(x, kind)
    z = delta.to_numpy(float)
    n_assets = z.shape[1]
    lag = 5
    rng = np.random.default_rng(seed)
    max_h = int(np.max(steps))
    anchor = raw.iloc[-1].to_numpy(float)

    # context features per asset (static point-in-time vector), restricted to selected families
    ctx = None
    if panel is not None and len(panel):
        pwin = panel.iloc[-window:]
        if context_families:
            from .factors import family_of
            keep = [c for c in pwin.columns if family_of(pwin, str(c)) in set(context_families)]
            pwin = pwin[keep] if keep else pwin
        ctx = np.stack([_context_features(pwin, str(c), kind) for c in raw.columns], axis=0)

    if method == 'var':
        if n_assets < 2:
            raise ValueError('VAR needs at least two target assets; use ar/ridge for single-asset')
        # Joint VAR(1) on valid overlapping increments.
        valid = np.isfinite(z).all(axis=1)
        y = z[valid]
        if len(y) < lag + 5:
            raise ValueError('insufficient overlapping increments for VAR')
        model = VAR(y).fit(1)
        resid = np.asarray(model.resid)
        pool = resid[np.isfinite(resid).all(axis=1)]
        if len(pool) < 25:
            raise ValueError('insufficient overlapping VAR residuals')
        pool = pool - pool.mean(0)
        hist = np.broadcast_to(np.nan_to_num(y[-1], nan=0.0)[None, :], (draws, n_assets)).copy()
        coef = np.asarray(model.coefs[0])  # (n_assets, n_assets)
        intercept = np.asarray(model.params[0, :])  # constant term per equation
        state = np.zeros((draws, n_assets))
        out = np.empty((draws, n_assets, steps.shape[1]))
        for t in range(1, max_h + 1):
            # einsum avoids spurious Accelerate matmul warnings on ARM (same as _ridge_predict_plain).
            mean = intercept[None, :] + np.einsum('dj,kj->dk', hist, coef)
            innovation = pool[rng.integers(0, len(pool), draws)]
            inc = mean + innovation
            state += inc
            hist = inc
            for a, h in np.argwhere(steps == t):
                out[:, a, h] = state[:, a] if kind == 'log_return' else anchor[a] * np.exp(state[:, a]) if kind == 'fx' else anchor[a] + state[:, a]
        if not np.isfinite(out).all():
            raise ValueError('non-finite forecast')
        return out

    # gbm / ridge_factors: per-asset feature model. Features = standardized lag increments
    # + TIME-VARYING momentum/volatility of increments (no lookahead, per-timestep) + the static
    # point-in-time context vector (cross-sectional z / family economics). The time-varying part
    # is what lets the ML branch learn regime signals beyond the target lags.
    models = []
    z_bufs = []  # initial 63-step increment buffers per asset, for recurrence
    for j in range(n_assets):
        s = z[:, j]
        good = np.isfinite(s)
        train = s[good]
        if len(train) < 35:
            raise ValueError('need at least 35 valid historical increments')
        xx = _lag_features(s, lag)
        yy = s[lag:]
        valid = np.isfinite(xx).all(1) & np.isfinite(yy)
        if valid.sum() < 30:
            raise ValueError('insufficient contiguous lag history')
        ctx_time = (np.nan_to_num(_time_varying_ctx(s), nan=0.0) if time_varying
                    else np.zeros((len(s), 3)))  # (n, 3), row t uses z[:t] when enabled
        ctx_time_train = ctx_time[lag:]  # align to lag_features rows
        # Standardize lag + time-varying context together so their scales are comparable and the
        # Ridge solver stays well-conditioned (the static point vector is left unscaled).
        scaler = StandardScaler().fit(np.concatenate([xx[valid], ctx_time_train[valid]], axis=1))
        Y = yy[valid]
        ctx_vec = np.asarray(ctx[j], dtype=float) if ctx is not None else np.zeros(0)
        X = _ml_feats(scaler, xx[valid], ctx_time_train[valid], ctx_vec)
        # Drop near-constant columns so the linear solver stays well-conditioned.
        keep = X.std(axis=0) > 1e-9
        X = X[:, keep]
        if method == 'gbm':
            model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05,
                                                  max_depth=3, random_state=int(seed) + j)
        else:
            model = Ridge(alpha=10.0, solver='svd')
        with warnings.catch_warnings():
            # Ridge's SVD emits spurious divide-by-zero/overflow matmul warnings under Accelerate
            # on this ARM runtime; the fit is finite (checked downstream), so suppress the noise.
            warnings.simplefilter('ignore', RuntimeWarning)
            model.fit(X, Y)
        models.append((scaler, model, ctx_vec, keep))
        buf = np.zeros(63)
        nz = min(63, len(z))
        buf[-nz:] = np.nan_to_num(z[-nz:, j], nan=0.0)
        z_bufs.append(buf if time_varying else np.zeros(63))

    # Joint residual pool: only dates where every asset has a finite residual.
    residuals = np.full_like(z, np.nan)
    for j in range(n_assets):
        s = z[:, j]
        xx = _lag_features(s, lag)
        yy = s[lag:]
        valid = np.isfinite(xx).all(1) & np.isfinite(yy)
        if valid.sum() < 30:
            raise ValueError('insufficient contiguous lag history')
        scaler, model, ctx_vec, keep = models[j]
        if time_varying:
            ctx_time_train = np.nan_to_num(_time_varying_ctx(s)[lag:], nan=0.0)[valid]
        else:
            ctx_time_train = np.zeros((valid.sum(), 3))
        X = _ml_feats(scaler, xx[valid], ctx_time_train, ctx_vec)[:, keep]
        pred = _ridge_predict_plain(model, X) if method == 'ridge_factors' else model.predict(X)
        residuals[np.flatnonzero(valid) + lag, j] = yy[valid] - pred
    pool = residuals[np.isfinite(residuals).all(axis=1)]
    if len(pool) < 25:
        raise ValueError('insufficient overlapping residual dates')
    pool = pool - pool.mean(0)

    # Initial lag history, most-recent-first: [z[t-1], z[t-2], ..., z[t-lag]].
    histories = np.array([[np.nan_to_num(z[-1 - k, j], nan=np.nanmean(z[:, j]))
                           for k in range(lag)] for j in range(n_assets)])
    hist = np.broadcast_to(histories[None, :, :], (draws, n_assets, lag)).copy()
    # Per-asset rolling increment buffers for the time-varying context (63-step lookback).
    zbuf = np.broadcast_to(np.stack(z_bufs)[None, :, :], (draws, n_assets, 63)).copy()
    state = np.zeros((draws, n_assets))
    out = np.empty((draws, n_assets, steps.shape[1]))

    def predict_mean(j, h, zbuf_j):
        scaler, model, ctx_vec, keep = models[j]
        lags = h[:, j, :]  # most-recent-first lags
        cum21 = zbuf_j[:, -21:].sum(axis=1)
        cum63 = zbuf_j.sum(axis=1)
        vol63 = zbuf_j.std(axis=1)
        ctx_time = np.column_stack([cum21, cum63, vol63])
        return _ml_feats(scaler, lags, ctx_time, ctx_vec)[:, keep]

    def model_predict(model, X):
        return _ridge_predict_plain(model, X) if method == 'ridge_factors' else model.predict(X)

    for t in range(1, max_h + 1):
        innovation = pool[rng.integers(0, len(pool), draws)]
        inc = np.empty_like(state)
        for j in range(n_assets):
            X = predict_mean(j, hist, zbuf[:, j, :])
            mean = model_predict(models[j][1], X)
            scale = max(float(np.nanstd(z[:, j])), 1e-10)
            mean = np.clip(mean, -3 * scale, 3 * scale)
            inc[:, j] = mean + innovation[:, j]
        state += inc
        hist = _push_lag(hist, inc)
        zbuf = np.concatenate([zbuf[:, :, 1:], inc[:, :, None]], axis=2)
        for a, h in np.argwhere(steps == t):
            out[:, a, h] = state[:, a] if kind == 'log_return' else anchor[a] * np.exp(state[:, a]) if kind == 'fx' else anchor[a] + state[:, a]
    if not np.isfinite(out).all():
        raise ValueError('non-finite forecast')
    return out


def simulate(raw, kind, method, steps, draws=500, seed=0, window=756, panel=None, context_families=None, time_varying=True, audit=None, ensemble_weights=None):
    """Fit only raw history supplied; resample joint residual rows along common paths.

    The five baseline methods model increments with hand-built structure. The ML branch
    (gbm / ridge_factors / var) learns a conditional mean over lag + context features and
    bootstraps the same joint residual rows, so cross-asset dependence is preserved by design.
    `time_varying=False` restores the pre-D5 point-in-time context (no rolling momentum/vol),
    kept as an ablation switch for the locked holdout comparison.
    """
    if method not in METHODS + LEGACY_METHODS + EXPERIMENTAL_METHODS: raise ValueError('unknown method')
    if method == 'solution_ensemble':
        from .solution_models import simulate_solution_ensemble
        return simulate_solution_ensemble(raw, kind, steps, draws, seed, window, panel, context_families, audit, ensemble_weights)
    if method == 'filtered_volatility':
        from .volatility import simulate_volatility
        return simulate_volatility(raw, kind, steps, draws, seed, window)
    if method == 'gbm':
        from .tree_forecast import simulate_tree
        if not time_varying:raise ValueError('gbm now uses causal direct features; legacy time_varying ablation is unsupported')
        return simulate_tree(raw, kind, steps, draws, seed, window, panel, context_families)
    if method in ML_METHODS:
        return _simulate_ml(raw, kind, method, steps, draws, seed, window, panel, context_families, time_varying)
    x = raw.iloc[-window:].copy()
    delta = _increments(x, kind)
    # Preserve missing calendar positions while making lag features.
    z = delta.to_numpy(float)
    n_assets = z.shape[1]; lag=5
    models=[]; residuals=np.full_like(z,np.nan); histories=[]; variances=[]
    params=[]
    for j in range(n_assets):
        s=z[:,j]; good=np.isfinite(s); train=s[good]
        if len(train)<35: raise ValueError('need at least 35 valid historical increments')
        scale=max(float(np.std(train,ddof=1)),1e-10)
        mu=float(train.mean()) if kind=='log_return' else 0.0
        model=None; g=None
        if method in ('ar','ridge'):
            xx=np.array([s[t-lag:t][::-1] for t in range(lag,len(s))])
            yy=s[lag:]; valid=np.isfinite(xx).all(1)&np.isfinite(yy)
            if valid.sum()<30 or not np.isfinite(s[-lag:]).all():
                raise ValueError('insufficient contiguous lag history')
            if method=='ridge':
                model=make_pipeline(StandardScaler(),Ridge(alpha=10.0,solver='svd'))
                model.fit(xx[valid], yy[valid]); pred=ridge_predict(model,xx[valid])
            else:
                # OLS AR(5) on valid calendar-lag rows, never across missing observations.
                fit=OLS(yy[valid],np.column_stack([np.ones(valid.sum()),xx[valid]])).fit()
                model=np.asarray(fit.params); pred=model[0]+np.einsum('ij,j->i',xx[valid],model[1:])
            residuals[np.flatnonzero(valid)+lag,j]=yy[valid]-pred
        elif method=='garch':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit=arch_model(train/scale,mean='Constant' if kind=='log_return' else 'Zero',vol='GARCH',p=1,q=1,dist='t',rescale=False).fit(disp='off',options={'maxiter':150})
            if fit.convergence_flag: raise ValueError('GARCH did not converge')
            p=fit.params; mu=float(p.get('mu',0))*scale
            g=(float(p['omega'])*scale**2,float(p['alpha[1]']),float(p['beta[1]']))
            residuals[good,j]=np.asarray(fit.std_resid)
            variance=float(np.asarray(fit.conditional_volatility)[-1]*scale)**2
        else:
            residuals[good,j]=train-mu
        if method=='ewma':
            weights=.97**np.arange(len(train)-1,-1,-1);weights/=weights.sum()
            variance=max(float(weights@((train-mu)**2)),1e-20)
            residuals[good,j]=(train-mu)/scale
        elif method!='garch': variance=scale**2
        models.append((model,mu,scale));params.append(g);variances.append(variance)
        histories.append(np.nan_to_num(s[-lag:],nan=mu))
    pool=residuals[np.isfinite(residuals).all(1)]
    if len(pool)<25: raise ValueError('insufficient overlapping residual dates')
    # Center residuals. Standardize innovations for conditional variance models.
    pool=pool-pool.mean(0)
    if method in ('ewma','garch'):
        pool=pool/np.maximum(pool.std(0,ddof=1),1e-10)
    rng=np.random.default_rng(seed); max_h=int(np.max(steps))
    state=np.zeros((draws,n_assets)); hist=np.broadcast_to(np.array(histories)[None,:,:],(draws,n_assets,lag)).copy()
    var=np.broadcast_to(variances,(draws,n_assets)).copy()
    out=np.empty((draws,n_assets,steps.shape[1]))
    anchor=raw.iloc[-1].to_numpy(float)
    for t in range(1,max_h+1):
        innovation=pool[rng.integers(0,len(pool),draws)].copy(); inc=np.empty_like(state)
        for j,(model,mu,scale) in enumerate(models):
            mean=np.full(draws,mu)
            if method=='ridge': mean=ridge_predict(model,hist[:,j,::-1])
            if method=='ar': mean=model[0]+np.einsum('ij,j->i',hist[:,j,::-1],model[1:])
            # Bound explosive extrapolation in fitted lag models, in training units.
            mean=np.clip(mean,-3*scale,3*scale)
            error=innovation[:,j]*np.sqrt(var[:,j]) if method in ('garch','ewma') else innovation[:,j]
            inc[:,j]=mean+error
            if method=='garch':
                omega,alpha,beta=params[j];var[:,j]=np.maximum(omega+alpha*error**2+beta*var[:,j],1e-20)
            elif method=='ewma': var[:,j]=.97*var[:,j]+.03*error**2
        state+=inc;hist=np.concatenate([hist[:,:,1:],inc[:,:,None]],axis=2)
        for a,h in np.argwhere(steps==t):
            out[:,a,h]=state[:,a] if kind=='log_return' else anchor[a]*np.exp(state[:,a]) if kind=='fx' else anchor[a]+state[:,a]
    if not np.isfinite(out).all(): raise ValueError('non-finite forecast')
    return out


class ForecastSession:
    def __init__(self,panels,card_path,asof,out,draws=1000,seed=42,controller_status=None,audit_root='unset'):
        self.start=time.monotonic();self.out=Path(out);self.out.parent.mkdir(parents=True,exist_ok=True)
        # `audit_root=None` means the caller already established that no writable audit location
        # exists (the platform's read-only root), so the trail is disabled rather than invented.
        # The sentinel keeps "no trail" distinct from "caller did not say".
        if audit_root == 'unset':
            self.audit=self.out.parent.with_name(self.out.parent.name+'_audit')
        elif audit_root is None:
            self.audit=None
        else:
            self.audit=Path(audit_root)
        if self.audit is not None:self.audit.mkdir(parents=True,exist_ok=True)
        self.events=[];self.cache={};self._raw_losses={};self.diagnosed=False;self.exported=False;self.seed=seed
        self.controller_status=controller_status if controller_status is not None else {}
        self.model_calls=0;self.total_tokens=0;self.last_diagnostics=None
        self.allowed_methods=METHODS;self.commander=None;self.ensemble_weights=None
        self.card_path=Path(card_path);self.card=tomllib.loads(self.card_path.read_text())
        self.asof=asof; t=self.card['targets'];self.assets=list(t['asset_ids']);self.horizons=list(map(int,t['horizons']));self.target=t.get('target_type','level')
        if self.target not in ('level','log_return'):raise ValueError('unsupported target type')
        cutoff=str(self.card['provenance']['data_cutoff'])
        if asof!=cutoff:raise ValueError('asof must match task cutoff')
        self.draws=max(200,draws,int(self.card.get('scoring',{}).get('params',{}).get('n_draws_min',0)))
        from qfbench2_track_forecasting.limits import ParseLimits
        if self.draws>ParseLimits().max_draws:raise ValueError('draw count exceeds contract maximum')
        panels=_read_panels(Path(panels)); series={}
        for a in self.assets:
            try: s=_series(panels,a,asof)
            except SystemExit: raise ValueError(f'target {a} has no permitted history; transfer inference is unsupported, no invented anchor') from None
            s.index=pd.to_datetime(s.index)
            if s.index.has_duplicates or not np.isfinite(s.to_numpy()).all():raise ValueError('invalid/duplicate historical rows')
            series[a]=s
        self.month_steps=_monthly_inputs(panels,self.card,self.card_path,asof)
        self.monthly=self.month_steps is not None
        if len({s.index[-1] for s in series.values()})!=1:raise ValueError('unequal final observation dates unsupported')
        raw=pd.DataFrame(series).sort_index()
        if self.monthly:
            raw.index=raw.index.to_period('M');raw=raw.reindex(pd.period_range(raw.index.min(),raw.index.max(),freq='M'))
        else:raw=raw.reindex(pd.bdate_range(raw.index.min(),raw.index.max()))
        self.raw=raw
        self.steps=np.asarray(self.month_steps,dtype=int) if self.monthly else np.tile(self.horizons,(len(self.assets),1))
        self.kind='log_return' if self.target=='log_return' else 'fx' if any('fx' in name.lower() and any(a in df.get('asset',df.get('asset_id',pd.Series(dtype=str))).values for a in self.assets) for name,df in panels.items()) else 'level'
        self.window=180 if self.monthly else 756
        self.panel=self._load_panel_wide(panels,asof)
        self.families=sorted({FAMILY_BY_STEM[name] for name in panels if name in FAMILY_BY_STEM})
        self.context_families=list(self.families)
        self.log('loaded',assets=self.assets,rows=len(raw),monthly=self.monthly,
                 context_assets=sorted(self.panel.columns),families=self.families)

    def apply_workflow(self, record):
        """Freeze the validated commander plan before any numerical tool runs."""
        if self.diagnosed or self.cache or self.exported or self.commander is not None:
            raise ValueError('workflow must be set once before numerical work')
        self.commander=json.loads(json.dumps(record))
        self.allowed_methods=tuple(record['plan']['methods'])
        self.window=record['plan']['window']
        self.ensemble_weights=record['plan'].get('ensemble_weights')
        feats=record['plan'].get('features') if isinstance(record.get('plan'), dict) else None
        self.context_families=list(feats) if feats else list(self.families)

    def trail(self,name,text,mode='w'):
        """Write one audit file, or nothing at all when the platform left us no writable path."""
        if self.audit is None:return
        write_audit(self.audit/name,text,mode)

    def log(self,event,**data):
        row=clean({'event':event,'elapsed_seconds':round(time.monotonic()-self.start,2),**data});self.events.append(row)
        self.trail('progress.jsonl',json.dumps(row)+'\n','a')
        print(json.dumps(row),flush=True)

    def describe(self):
        # Do not expose titles, family labels, event narratives or calendar dates to LLM.
        return clean({'assets':self.assets,'target_type':self.target,'numeric_transform':self.kind,'frequency':'monthly' if self.monthly else 'business_daily','horizon_steps':self.steps,'observations':len(self.raw),'missing_cells':int(self.raw.isna().sum().sum()),'methods':self.allowed_methods,'ensemble_weights':self.ensemble_weights,'training_window':self.window,'text_access':False})

    def feature_report(self):
        from .features import feature_report
        return clean(feature_report(self.raw.iloc[-self.window:], self.kind, self.panel, self.context_families))

    def factor_report(self, asset=None):
        """Classic factor / economic-metric construction for target assets (read-only).

        Each entry documents the data family, the panel-wide data-quality summary, and the
        family-specific factors. Some metrics are labelled proxies where the textbook input
        (e.g. interest rates for FX carry) is not present in the panel.
        """
        from .factors import compute_factors
        targets = [asset] if asset else list(self.assets)
        return clean({'families': self.families,
                      'factors': {t: compute_factors(self.panel, t) for t in targets}})

    def data_quality_report(self):
        """Missing rate, cross-asset correlation and PCA structure of the whole panel."""
        from .factors import data_quality
        return clean(data_quality(self.panel))


    def _load_panel_wide(self, panels, asof):
        """All panel assets (targets and context) as one date x asset wide table, cut at asof.

        Only panel values up to the as-of are retained; this is the same leak boundary the
        official reader enforces for targets. Context assets are read-only predictors for
        factor/feature construction, never graded targets.
        """
        frames = []
        for name, df in panels.items():
            col = _asset_col(df)
            if col is None:
                continue
            sub = df.copy()
            sub['date'] = sub['date'].astype(str).str.slice(0, 10)
            sub = sub[sub['date'] <= asof]
            wide = sub.pivot_table(index='date', columns=col, values='value', aggfunc='last')
            frames.append(wide)
        if not frames:
            return pd.DataFrame(index=self.raw.index)
        wide = pd.concat(frames, axis=1).sort_index()
        # Keep each asset column from its first panel only (a duplicated asset across panels
        # would otherwise yield collinear duplicate columns).
        wide = wide.loc[:, ~wide.columns.duplicated()].astype(float)
        # Same calendar grid as the target panel so lag features align.
        if self.monthly:
            idx = pd.to_datetime(wide.index).to_period('M')
            wide.index = idx
            wide = wide[~wide.index.duplicated()].reindex(pd.period_range(idx.min(), idx.max(), freq='M'))
        else:
            wide.index = pd.to_datetime(wide.index)
            wide = wide[~wide.index.duplicated()].reindex(pd.bdate_range(wide.index.min(), wide.index.max()))
        return wide


    def trace_step(self,step):
        """Record one controller step: model text, tool calls with arguments, token usage."""
        raw_output=getattr(step,'model_output','') or ''
        text=' '.join(b.get('text','') for b in raw_output if isinstance(b,dict)) if isinstance(raw_output,list) else str(raw_output)
        usage=getattr(step,'token_usage',None)
        if usage is not None:
            self.model_calls+=1
            self.total_tokens+=int((getattr(usage,'input_tokens',0) or 0)+(getattr(usage,'output_tokens',0) or 0))
        row={'step':getattr(step,'step_number',None),'model_text':text.strip(),
             'tool_calls':[{'name':c.name,'arguments':clean(getattr(c,'arguments',None))} for c in (getattr(step,'tool_calls',None) or [])],
             'error_type':type(step.error).__name__ if getattr(step,'error',None) else None,
             'is_final_answer':bool(getattr(step,'is_final_answer',False)),
             'model_calls_so_far':self.model_calls,'total_tokens_so_far':self.total_tokens}
        self.log('agent_step',**row)
        if self.controller_status:
            self.controller_status['model_calls']=self.model_calls;self.controller_status['total_tokens']=self.total_tokens

    def diagnose(self):
        delta=np.log1p(self.raw) if self.kind=='log_return' else np.log(self.raw).diff() if self.kind=='fx' else self.raw.diff()
        result={}
        for a in self.assets:
            v=delta[a].dropna().iloc[-self.window:];n=len(v)
            if n<35:raise ValueError('too little increment history')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p=float(adfuller(v,maxlag=min(10,n//5),autolag='AIC')[1]) if v.std()>0 else None
            result[a]={'count':n,'mean':v.mean(),'std':v.std(),'recent_std':v.tail(30).std(),'skew':v.skew(),'excess_kurtosis':v.kurt(),'acf1':acf(v,nlags=1,fft=True)[1],'squared_acf1':acf(v*v,nlags=1,fft=True)[1],'adf_pvalue_increments':p}
        joint=delta.dropna().iloc[-self.window:];pca=None
        if len(joint)>10 and len(self.assets)>1:
            pca=PCA().fit(StandardScaler().fit_transform(joint)).explained_variance_ratio_.tolist()
        self.diagnosed=True
        answer=clean({'series':result,'increment_correlation':joint.corr().to_dict(),'pca_variance_ratios':pca,'caveat':'ADF/ACF diagnostics drop missing increments; gaps may affect lag interpretation. No causal interpretation.'})
        self.trail('diagnostics.json',json.dumps(answer,indent=2));self.last_diagnostics=answer;self.log('diagnosed');return answer

    def evaluate(self,method):
        from .evaluation import strict_truth, fold_reference, score, aggregate
        if method not in self.allowed_methods:raise ValueError('method excluded by workflow')
        if method in self.cache:return self.cache[method]
        H=int(self.steps.max());n=len(self.raw)
        origins=list(range(n-H,n-4*H,-H))[::-1]
        records=[];excluded=[]
        if method=='gbm':
            self.trail('features.json',json.dumps(self.feature_report(),indent=2))
        for origin in origins:
            if origin<70:
                excluded.append({'origin':origin,'reason':'insufficient training history'});continue
            train=self.raw.iloc[:origin];future=self.raw.iloc[origin:origin+H]
            try:
                truth=strict_truth(future,self.kind,self.steps)
                reference,scales=fold_reference(train,self.target,self.steps,self.card)
            except ValueError as e:
                excluded.append({'origin':origin,'reason':str(e)});continue
            row={'key':str(origin),'origin':origin,'status':'failed','normalized_composite':8.,'scales':scales}
            try:
                details = {}
                samples=simulate(train,self.kind,method,self.steps,draws=200,seed=self.seed,
                    window=self.window,panel=self.panel.loc[self.panel.index<=train.index[-1]],
                    context_families=self.context_families,audit=details,ensemble_weights=self.ensemble_weights)
                if details: row['model_audit'] = details
                row.update(score(samples,truth,scales,self.card),status='ok')
            except Exception as e:row['error']=str(e)
            records.append(row);self.log('validation_fold',method=method,origin=origin,status=row['status'])
        feasible=True;error=None
        try:
            simulate(self.raw,self.kind,method,self.steps,draws=20,seed=self.seed,window=self.window,
                     panel=self.panel,context_families=self.context_families,ensemble_weights=self.ensemble_weights)
        except Exception as e:feasible=False;error=str(e)
        metric=aggregate(records)['mean_normalized_composite'] if records else None
        result={'method':method,'status':'ok' if records and feasible else 'failed' if not feasible else 'insufficient_history',
                'normalized_composite':metric,'folds':len(records),'failed_folds':sum(r['status']!='ok' for r in records),
                'fold_records':records,'excluded_origins':excluded,'error':error,
                'normalization':'public_M0_expected_error_reconstruction',
                'aggregation':'arithmetic_mean_with_failure_8_and_clip_8',
                'warning':'Historical selection diagnostics, not independent holdout or official hidden score.'}
        self.cache[method]=clean(result);self.trail('backtests.json',json.dumps(self.cache,indent=2))
        return self.cache[method]

    def review_evidence(self):
        from .evidence import review_candidates
        report = clean(review_candidates(self.cache))
        self.trail('evidence_review.json', json.dumps(report, indent=2))
        return report

    def anchor_increment(self,method,asset):
        """The method's own mean increment at the last observation, in forecast units.

        This is the anchor adjustment the controller did not choose: it is what the
        selected method adds on top of a flat continuation, reported here so the
        derivation can be checked term by term.
        """
        if method == 'filtered_volatility':
            return 0.0
        j=self.assets.index(asset);x=self.raw.iloc[-self.window:].copy()
        delta=_increments(x,self.kind)
        s=delta.to_numpy(float)[:,j];good=np.isfinite(s);train=s[good];lag=5
        mu=float(train.mean()) if self.kind=='log_return' else 0.0
        if method in ('ar','ridge'):
            hist=np.nan_to_num(s[-lag:],nan=mu)[::-1]
            if method=='ridge':
                model=make_pipeline(StandardScaler(),Ridge(alpha=10.0,solver='svd'))
                xx=np.array([s[t-lag:t][::-1] for t in range(lag,len(s))]);yy=s[lag:]
                valid=np.isfinite(xx).all(1)&np.isfinite(yy)
                model.fit(xx[valid],yy[valid]);mu=float(ridge_predict(model,hist[None,:])[0])
            else:
                xx=np.array([s[t-lag:t][::-1] for t in range(lag,len(s))]);yy=s[lag:]
                valid=np.isfinite(xx).all(1)&np.isfinite(yy)
                fit=OLS(yy[valid],np.column_stack([np.ones(valid.sum()),xx[valid]])).fit()
                mu=float(fit.params[0]+np.einsum('j,j->',hist,fit.params[1:]))
        elif method in ('gbm','ridge_factors'):
            # Conditional mean at the last observation, same features as _simulate_ml
            # (standardized lags + time-varying momentum/vol + static context).
            hist=np.nan_to_num(s[-lag:],nan=mu)[::-1]
            xx=_lag_features(s,lag);yy=s[lag:]
            valid=np.isfinite(xx).all(1)&np.isfinite(yy)
            ctx_time=np.nan_to_num(_time_varying_ctx(s),nan=0.0)[lag:]
            scaler=StandardScaler().fit(np.concatenate([xx[valid],ctx_time[valid]],axis=1))
            Y=yy[valid]
            ctx=_context_features(self.panel.iloc[-self.window:],str(asset),self.kind)
            X=_ml_feats(scaler,xx[valid],ctx_time[valid],ctx)
            keep=X.std(axis=0)>1e-9
            X=X[:,keep]
            if method=='gbm':
                model=HistGradientBoostingRegressor(max_iter=100,learning_rate=0.05,max_depth=3,random_state=int(self.seed)+j)
            else:
                model=Ridge(alpha=10.0,solver='svd')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                model.fit(X,Y)
            buf=np.zeros(63);nz=min(63,len(s));buf[-nz:]=np.nan_to_num(s[-nz:],nan=0.0)
            ctx_time_last=np.array([[buf[-21:].sum(),buf.sum(),buf.std()]])
            feat=_ml_feats(scaler,hist[None,:],ctx_time_last,ctx)[:,keep]
            mu=float(_ridge_predict_plain(model,feat)[0] if method=='ridge_factors' else model.predict(feat)[0])
        elif method=='var':
            z=delta.to_numpy(float);valid=np.isfinite(z).all(axis=1);y=z[valid]
            if len(y)>=lag+5:
                fit=VAR(y).fit(1)
                mu=float((np.asarray(fit.coefs[0])[j]@y[-1])+np.asarray(fit.params)[0, j])
        return mu

    def rationale_markdown(self,method,reason):
        rows=self.cache.get(method,{})
        lines=['# Numerical forecast', '',reason,'',
               f'Method: {method}; training window: {self.window}; joint residual paths.',
               'Historical evaluation uses reconstructed public M0 expected-error scales.',
               'Composite aggregation: equal arithmetic mean, failed folds = 8, clipping = 8.',
               'Random walk is a candidate; its normalized loss is not fixed to 1.',
               'Missing historical truth is excluded for all candidates and reported explicitly.',
               'These are selection diagnostics, not independent holdout or official hidden scores.',
               '', '```json',json.dumps(rows,indent=2),'```']
        for asset in self.assets:
            lines.append(f'{asset}: drift increment = {self.anchor_increment(method,asset):.8g}')
        if self.commander is not None:
            lines+=['','## Commander plan','',json.dumps(self.commander,indent=2),
                    'Historical folds do not replay historical text vintages.']
        return '\n'.join(lines)+'\n'

    def export(self,method,reason):
        if not self.diagnosed:raise ValueError('run diagnose_series first')
        eligible=[v for v in self.cache.values() if v['status']=='ok']
        if not eligible:raise ValueError('no successful candidates to export')
        # One successfully validated candidate is enough to export when it is the only one left;
        # flag it so the record does not pretend a two-candidate comparison happened.
        fallback_export=len(eligible)==1
        if method not in {v['method'] for v in eligible}:raise ValueError('select a successfully validated method')
        evidence_review = self.review_evidence()
        if len(eligible)>1 and not evidence_review['comparable']:
            raise ValueError('candidate evidence is incomparable; align fold roster and scoring contract')
        panel_t = self.panel.loc[self.panel.index <= self.raw.index[-1]]
        model_audit = {}
        samples=simulate(self.raw,self.kind,method,self.steps,self.draws,self.seed,self.window,panel=panel_t,context_families=self.context_families,audit=model_audit,ensemble_weights=self.ensemble_weights)
        idx=pd.MultiIndex.from_product([range(self.draws),self.assets,self.horizons],names=['draw','asset','horizon'])
        frame=idx.to_frame(index=False);frame['value']=samples.reshape(-1);frame.to_parquet(self.out,index=False)
        meta={'unit_id':self.card['task']['id'],'asof':self.asof,'representation':'samples','asset_ids':self.assets,'horizons':self.horizons,'n_draws':self.draws,'target':self.target,'text_used':bool(self.commander and self.commander['text_used']),'commander':self.commander,'training_window':self.window,'selected_method':method,'feature_pipeline':'causal_direct_horizon_v2_gap' if method=='gbm' else 'none','fallback_export':fallback_export,'rationale':{'file':'forecast_rationale.md','method':method+' joint residual path bootstrap; bounded workflow plan'}}
        meta['evidence_review'] = evidence_review
        if model_audit:
            meta['model_audit'] = model_audit
            meta['feature_pipeline'] = 'relative_top16_joint_kde_evidence_mixture_v1'
        (self.out.parent/'forecast_meta.json').write_text(json.dumps(meta,indent=2))
        if method == 'solution_ensemble':
            rationale = '# Evidence-weighted joint mixture\n\n' + reason + '\n\nRandom walk plus eligible damped trend/relative-feature tree. Weights come from the commander analysis or its delegated historical-error rule; random walk has no minimum weight. Whole asset/horizon draw vectors are mixed. Tree residuals use covariance-matched joint KDE; no independent marginal quantile assembly. Not FFORMA or ES-RNN reproduction, no coverage guarantee.\n\n```json\n' + json.dumps(model_audit, indent=2) + '\n```\n'
        elif method == 'gbm':
            rationale = '# Direct-horizon tree forecast\n\nCausal lag, momentum, volatility and selected cross-asset features.\nChronological residual calibration with horizon embargo plus one extra maximum-horizon gap; full asset/horizon residual vectors sampled jointly.\nNo future context is required. Endpoint joint samples are not daily simulated paths.\n\n' + reason + '\n\nFeature report:\n```json\n' + json.dumps(self.feature_report(), indent=2) + '\n```\n'
        else:
            rationale = self.rationale_markdown(method,reason)
        rationale += '\n## Candidate evidence review\n\n```json\n' + json.dumps(evidence_review, indent=2) + '\n```\n'
        (self.out.parent/'forecast_rationale.md').write_text(rationale)
        self.exported=True;self.log('exported',method=method,draws=self.draws,fallback_export=fallback_export);return {'output':str(self.out),'method':method,'rows':len(frame),'fallback_export':fallback_export}
