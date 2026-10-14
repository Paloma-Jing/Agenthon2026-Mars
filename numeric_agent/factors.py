"""Classic factor / economic-metric construction and data-quality summaries.

Read-only feature library over the as-of-truncated panel. Every function is a pure map from a
wide date x asset DataFrame (already cut at the task as-of by the caller) to JSON-serializable
values. No future rows, no external data, no file or network access: the caller enforces the
as-of boundary, and these functions only use columns present in the supplied frame.

Factors are grouped by data family. A "proxy" label marks metrics whose textbook definition
needs an input this panel does not carry (e.g. FX carry needs interest rates), so the value is a
documented stand-in, not the canonical quantity.
"""
import json
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#: Business-day lookback for momentum/volatility features. Monthly series use calendar periods.
MOMENTUM_WINDOWS = (21, 63, 126, 252)


def _clean(value):
    if isinstance(value, (dict, list, tuple)):
        return json.loads(json.dumps(value, default=lambda o: None if isinstance(o, (np.integer, np.floating, float)) and not np.isfinite(o) else o))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    return value


def _series(wide, name):
    if name not in wide.columns:
        return None
    return pd.to_numeric(wide[name], errors='coerce')


def _roll(series, window, func):
    """Rolling statistic with at least min_periods; safe on all-NaN slices."""
    out = series.rolling(window, min_periods=max(2, window // 3)).apply(func, raw=True)
    return out


def _last(series):
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else None


def _ret(series, periods):
    """period-over-period log return if levels look multiplicative/positive, else simple diff."""
    s = series.dropna()
    if len(s) < periods + 1:
        return None
    if (s > 0).all():
        return float(np.log(s.iloc[-1] / s.iloc[-1 - periods]))
    return float(s.iloc[-1] - s.iloc[-1 - periods])


def _mom(series, periods):
    """Cumulative return over `periods`, log where positive else simple."""
    s = series.dropna()
    if len(s) < periods + 1:
        return None
    x = s.iloc[-1 - periods:].to_numpy(float)
    if (x > 0).all():
        return float(np.log(x[-1] / x[0]))
    return float(x[-1] - x[0])


def _realized_vol(series, periods):
    s = series.dropna()
    if len(s) < periods + 1:
        return None
    d = s.diff().dropna().iloc[-periods:]
    return float(d.std(ddof=1)) if len(d) > 1 else None


def _reversal(series, short, long):
    """Short-horizon vs long-horizon drift; positive means recent momentum dominates."""
    sm = _mom(series, short)
    lm = _mom(series, long)
    if sm is None or lm is None:
        return None
    return float(sm - lm)


def _zscore(series, periods):
    s = series.dropna()
    if len(s) < periods + 1:
        return None
    m = s.iloc[-periods:].mean()
    sd = s.iloc[-periods:].std(ddof=1)
    return float((s.iloc[-1] - m) / sd) if sd and np.isfinite(sd) else None


# --------------------------------------------------------------------------- rates

_TENOR_RE = re.compile(r'(?:_|\b)(\d+)Y\b')


def _tenor(name):
    m = _TENOR_RE.search(str(name))
    return int(m.group(1)) if m else None


def rates_factors(wide, target):
    """Term-structure factors for a UST target: level, slope, curvature, spreads, NS beta.

    Uses the cross-section of tenors present in the panel at the last observation. A two-factor
    Nelson-Siegel fit (level + slope, fixed decay) is a compact summary; curvature is reported as
    the butterfly 2*mid - short - long where a mid tenor exists.
    """
    tenors = {}
    for c in wide.columns:
        t = _tenor(c)
        if t is not None:
            tenors[c] = t
    out = {'tenors_available': sorted(set(tenors.values()))}
    if not tenors:
        return {'family': 'rates', 'note': 'no tenor-labelled columns', **out}

    last = {c: wide[c].dropna().iloc[-1] for c in tenors if wide[c].notna().any()}
    if len(last) < 2:
        return {'family': 'rates', 'note': 'insufficient tenor overlap', **out}

    ordered = sorted(last.items(), key=lambda kv: tenors[kv[0]])  # by tenor in years
    ys = np.array([v for _, v in ordered], dtype=float)
    ts = np.array([tenors[c] for c, _ in ordered], dtype=float)
    short, long = ordered[0], ordered[-1]
    out['level'] = float(np.mean(ys))
    out['slope'] = float(long[1] - short[1])
    out['spread_short_long'] = float(long[1] - short[1])
    # butterfly with nearest mid tenor
    mid_cands = [t for t in ts if short[1] < t < long[1]]
    if mid_cands:
        mid_t = min(mid_cands, key=lambda t: abs(t - (short[1] + long[1]) / 2))
        mid_c = [c for c, t in tenors.items() if t == mid_t][0]
        out['curvature'] = float(2 * last[mid_c] - short[1] - long[1])
    # canonical spreads if the tenors exist
    def spread(a, b):
        if a in last and b in last:
            return float(last[b] - last[a])
        return None
    out['spread_2s10s'] = spread('UST_2Y', 'UST_10Y')
    out['spread_5s30s'] = spread('UST_5Y', 'UST_30Y')
    # two-factor Nelson-Siegel level/slope on the cross-section (fixed lambda=0.0609)
    tau = 0.0609
    X = np.column_stack([np.ones_like(ts), (1 - np.exp(-ts * tau)) / (ts * tau)])
    beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
    out['nelson_siegel_level'] = float(beta[0])
    out['nelson_siegel_slope'] = float(beta[1])
    if target in wide.columns:
        out['target_level'] = _last(wide[target])
        out['target_change_1m'] = _ret(wide[target], 21)
        out['target_vol_3m'] = _realized_vol(wide[target], 63)
    out['family'] = 'rates'
    return out


# --------------------------------------------------------------------------- fx

def fx_factors(wide, target):
    """FX factors for a USD-quoted currency target.

    Carry is the textbook driver but this panel carries no interest rates, so `carry_proxy` is a
    cross-sectional trend rank (positive-trend currencies tend to be high yielders in sample) and
    is labelled a proxy. Momentum, realized vol, reversal and a rolling-mean PPP-style deviation
    are computed from levels.
    """
    out = {'family': 'fx'}
    if target in wide.columns:
        out['level'] = _last(wide[target])
        out['mom_1m'] = _mom(wide[target], 21)
        out['mom_3m'] = _mom(wide[target], 63)
        out['mom_12m'] = _mom(wide[target], 252)
        out['vol_1m'] = _realized_vol(wide[target], 21)
        out['vol_3m'] = _realized_vol(wide[target], 63)
        out['reversal_1m_12m'] = _reversal(wide[target], 21, 252)
        s = wide[target].dropna()
        if len(s) > 252:
            m = s.rolling(252, min_periods=120).mean()
            sd = s.rolling(252, min_periods=120).std(ddof=1)
            out['ppp_deviation_proxy'] = float((s.iloc[-1] - m.iloc[-1]) / sd.iloc[-1]) if sd.iloc[-1] else None
    # carry proxy: rank of the target's 3m trend among all panel currencies
    trends = {}
    for c in wide.columns:
        t = _mom(wide[c], 63)
        if t is not None and np.isfinite(t):
            trends[c] = t
    if target in trends and len(trends) > 1:
        ranked = sorted(trends, key=trends.get)
        out['carry_proxy_rank'] = ranked.index(target) / max(1, len(ranked) - 1)
        out['carry_proxy_note'] = 'no rate panel; cross-sectional trend-rank stand-in for carry'
    return out


# --------------------------------------------------------------------------- factors

def factors_factors(wide, target):
    """Cross-sectional equity-style factor features for a target return series.

    Momentum, reversal, realized vol, and the target's cross-sectional z-score of trailing
    cumulative return and volatility (value/quality-like ranking) among panel factor columns.
    """
    out = {'family': 'factors'}
    if target in wide.columns:
        out['cum_return_1m'] = _mom(wide[target], 21)
        out['cum_return_3m'] = _mom(wide[target], 63)
        out['cum_return_12m'] = _mom(wide[target], 252)
        out['vol_1m'] = _realized_vol(wide[target], 21)
        out['vol_3m'] = _realized_vol(wide[target], 63)
        out['reversal_1m_12m'] = _reversal(wide[target], 21, 252)
    rets, vols = {}, {}
    for c in wide.columns:
        r = _mom(wide[c], 63)
        v = _realized_vol(wide[c], 63)
        if r is not None and np.isfinite(r):
            rets[c] = r
        if v is not None and np.isfinite(v):
            vols[c] = v
    if target in rets and len(rets) > 1:
        vals = np.array(list(rets.values()))
        out['cs_momentum_z'] = float((rets[target] - vals.mean()) / (vals.std() + 1e-12))
    if target in vols and len(vols) > 1:
        vals = np.array(list(vols.values()))
        out['cs_vol_z'] = float((vols[target] - vals.mean()) / (vals.std() + 1e-12))
    return out


# --------------------------------------------------------------------------- macro

def macro_factors(wide, target):
    """Macro indicators for an index/rate target (CPI/PCE index, NFP level, unemployment rate).

    YoY/MoM use log changes for positive indices and simple differences for rates/counts.
    core-vs-headline spread, unemployment gap (vs trailing mean) and a Sahm-rule-style change in
    the unemployment rate are reported when the companion series exist in the panel.
    """
    out = {'family': 'macro'}

    def growth(periods):
        s = wide[target].dropna()
        if len(s) < periods + 1:
            return None
        a, b = s.iloc[-1 - periods], s.iloc[-1]
        if a > 0 and b > 0:
            return float(np.log(b / a))
        return float(b - a)

    if target in wide.columns:
        out['yoy'] = growth(12)
        out['mom'] = growth(1)
        out['mom_3m_ann'] = growth(3) * 4
        out['level'] = _last(wide[target])
        out['zscore_12m'] = _zscore(wide[target], 12)
    if 'CPI_ALL' in wide.columns and 'CPI_CORE' in wide.columns:
        a = _ret(wide['CPI_ALL'], 12)
        b = _ret(wide['CPI_CORE'], 12)
        if a is not None and b is not None:
            out['core_headline_yoy_spread'] = float(b - a)
    if 'PCE_ALL' in wide.columns and 'PCE_CORE' in wide.columns:
        a = _ret(wide['PCE_ALL'], 12)
        b = _ret(wide['PCE_CORE'], 12)
        if a is not None and b is not None:
            out['pce_core_headline_yoy_spread'] = float(b - a)
    if 'UNRATE' in wide.columns:
        s = wide['UNRATE'].dropna()
        if len(s) > 12:
            out['unemployment_gap'] = float(s.iloc[-1] - s.rolling(12, min_periods=6).mean().iloc[-1])
            out['sahm_change_proxy'] = float(s.rolling(3, min_periods=2).mean().iloc[-1] - s.rolling(12, min_periods=6).mean().iloc[-1])
    if 'NFP' in wide.columns:
        out['nfp_change_3m'] = _ret(wide['NFP'], 3)
    return out


# --------------------------------------------------------------------------- dispatch + quality

FACTOR_BUILDERS = {
    'rates': rates_factors,
    'fx': fx_factors,
    'factors': factors_factors,
    'macro': macro_factors,
}


def family_of(wide, target):
    """Classify an asset id by name. `wide` is accepted for a uniform signature; the mapping is
    name-based because panel asset ids are the fixed catalog names (UST_*, G10 codes, factor and
    macro labels)."""
    name = str(target)
    if 'UST_' in name:
        return 'rates'
    if name in {'AUD', 'CAD', 'CHF', 'DKK', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK', 'BRL', 'INR', 'CNY'}:
        return 'fx'
    if name in {'MKT', 'SMB', 'HML', 'MOM', 'BAB', 'QMJ'}:
        return 'factors'
    if name in {'CPI_ALL', 'CPI_CORE', 'PCE_ALL', 'PCE_CORE', 'NFP', 'UNRATE'}:
        return 'macro'
    return None


def data_quality(wide):
    """Missing rate, overlap, cross-sectional correlation and PCA share for the whole panel."""
    n = len(wide)
    out = {'rows': int(n), 'n_assets': int(wide.shape[1])}
    out['missing_rate'] = float(wide.isna().mean().mean()) if n else None
    out['per_asset_missing'] = {str(c): float(wide[c].isna().mean()) for c in wide.columns}
    num = wide.dropna(axis=1, how='all')
    if num.shape[1] >= 2 and n > 10:
        corr = num.corr().abs().where(np.triu(np.ones(num.corr().shape), 1).astype(bool)).stack()
        out['mean_abs_cross_corr'] = float(corr.mean())
        scaled = StandardScaler().fit_transform(num.fillna(num.mean()))
        pca = PCA().fit(scaled)
        out['pca_var_ratios'] = [float(v) for v in pca.explained_variance_ratio_[:5]]
    return out


def compute_factors(wide, target, family=None):
    """All factors for one target, plus panel data quality. Returns a JSON-safe dict."""
    fam = family or family_of(wide, target)
    out = {'target': str(target), 'family': fam}
    out['quality'] = data_quality(wide)
    builder = FACTOR_BUILDERS.get(fam)
    out['factors'] = builder(wide, target) if builder else {'note': 'no factor builder for family'}
    return _clean(out)


def compute_all(wide, targets):
    """Factors for every target, for inspection/diagnosis (no fitting)."""
    return {str(t): compute_factors(wide, t) for t in targets}
