"""Filtered historical simulation, inspired by adaptive volatility forecasting.

An original fixed EWMA recipe, NOT an implementation of AdaVol/AdaGaussMC.
Hyperparameters are frozen before evaluation; no validation-fitted weights.
"""
import numpy as np


def filter_innovations(z, decay=0.97, warmup=35):
    """Scale each shock by variance estimated strictly before that shock.

    Missing calendar rows stay missing. Each asset updates on its observed dates;
    only simultaneous finite innovation rows enter the joint sampling pool.
    """
    z = np.asarray(z, dtype=float)
    shocks = np.full_like(z, np.nan)
    final = np.empty(z.shape[1])
    for j in range(z.shape[1]):
        valid = np.flatnonzero(np.isfinite(z[:, j]))
        if len(valid) < warmup + 25:
            raise ValueError('need 60 valid increments for filtered volatility')
        initial = z[valid[:warmup], j]
        variance = max(float(np.mean(initial ** 2)), 1e-20)
        for t in range(valid[warmup - 1] + 1, len(z)):
            if not np.isfinite(z[t, j]):
                continue
            shocks[t, j] = z[t, j] / np.sqrt(variance)
            variance = max(decay * variance + (1 - decay) * z[t, j] ** 2, 1e-20)
        final[j] = variance
    pool = shocks[np.isfinite(shocks).all(axis=1)]
    if len(pool) < 25:
        raise ValueError('need 25 overlapping filtered shock dates')
    pool = pool - pool.mean(axis=0)
    pool /= np.maximum(pool.std(axis=0, ddof=1), 1e-10)
    return pool, final


def simulate_volatility(raw, kind, steps, draws, seed, window):
    from .engine import _increments
    # Zero drift avoids fitting noisy means; all output horizons share each path.
    z = _increments(raw.iloc[-window:], kind).to_numpy(float)
    pool, variance = filter_innovations(z)
    rng = np.random.default_rng(seed)
    state = np.zeros((draws, z.shape[1]))
    var = np.broadcast_to(variance, state.shape).copy()
    out = np.empty((draws, *steps.shape))
    anchor = raw.iloc[-1].to_numpy(float)
    for t in range(1, int(steps.max()) + 1):
        shock = pool[rng.integers(len(pool), size=draws)] * np.sqrt(var)
        state += shock
        var = np.maximum(.97 * var + .03 * shock ** 2, 1e-20)
        for a, h in np.argwhere(steps == t):
            out[:, a, h] = (state[:, a] if kind == 'log_return' else
                           anchor[a] * np.exp(state[:, a]) if kind == 'fx' else
                           anchor[a] + state[:, a])
    if not np.isfinite(out).all():
        raise ValueError('non-finite volatility forecast')
    return out
