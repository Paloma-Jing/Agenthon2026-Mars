"""Small bounded adaptations of forecasting competition ideas, not reproductions.

Exponentially smoothed trends, joint residual KDE and historical-evidence mixtures.
Only two internal weight-estimation origins; no search over recipes or parameters.
"""
import numpy as np
import pandas as pd
from qfbench2_common.scoring.crps import crps_ensemble


def joint_kernel_draws(pool, draws, seed):
    """Smooth whole residual vectors with covariance-matched noise.

    Bandwidth is fixed by sample size. Variance correction avoids mechanically
    inflating the residual covariance. Rank-deficient dependencies stay intact.
    This is smoothed empirical calibration, not a coverage guarantee.
    """
    pool = np.asarray(pool, dtype=float)
    if pool.ndim != 2 or len(pool) < 15 or not np.isfinite(pool).all():
        raise ValueError('need 15 finite joint calibration residuals')
    rng = np.random.default_rng(seed)
    mean = pool.mean(axis=0); centered = pool - mean
    # Population covariance matches uniform empirical sampling exactly.
    _, singular, vectors = np.linalg.svd(centered / np.sqrt(len(pool)), full_matrices=False)
    factor = singular[:, None] * vectors
    bandwidth = min(.35, len(pool) ** (-1 / (pool.shape[1] + 4)))
    sampled = centered[rng.integers(len(pool), size=draws)]
    noise = np.einsum('ij,jk->ik', rng.standard_normal((draws, len(singular))), factor)
    return mean + (sampled + bandwidth * noise) / np.sqrt(1 + bandwidth**2)


def simulate_trend(raw, kind, steps, draws, seed, window):
    """Fixed-coefficient damped local trend with shrunk monthly seasonality.

    Smooth increments causally; missing observations never become residuals.
    Monthly seasonal effects need at least three prior observations of that month.
    Return targets keep zero drift; level and log-FX targets use damped trend.
    """
    from .engine import _increments
    x = raw.iloc[-window:]; z = _increments(x, kind).to_numpy(float)
    monthly = isinstance(x.index, pd.PeriodIndex)
    residuals = np.full_like(z, np.nan); trends = []; seasonal = []
    for a in range(z.shape[1]):
        mean = 0.; counts = np.zeros(12); season = np.zeros(12); observed = 0
        for t, value in enumerate(z[:, a]):
            if not np.isfinite(value):
                continue
            month = x.index[t].month - 1 if monthly else 0
            effect = season[month] * counts[month] / (counts[month] + 6) if monthly and counts[month] >= 3 else 0.
            prediction = mean + effect if kind != 'log_return' else 0.
            if observed >= 35: residuals[t, a] = value - prediction
            if monthly:
                season[month] = .8*season[month] + .2*(value-mean)
                counts[month] += 1
            mean = .85*mean + .15*(value-effect)
            observed += 1
        if observed < 60: raise ValueError('trend needs 60 valid increments')
        trends.append(mean if kind != 'log_return' else 0.)
        seasonal.append(np.where(counts >= 3, season*counts/(counts+6), 0.))
    pool = residuals[np.isfinite(residuals).all(axis=1)]
    if len(pool) < 25: raise ValueError('trend needs 25 simultaneous residual dates')
    pool = pool - pool.mean(axis=0)
    rng = np.random.default_rng(seed); state = np.zeros((draws, z.shape[1]))
    out = np.empty((draws, *steps.shape)); anchor = raw.iloc[-1].to_numpy(float)
    for t in range(1, int(steps.max())+1):
        drift = np.asarray(trends) * .95**t
        if monthly and kind != 'log_return':
            month = (x.index[-1] + t).month - 1
            drift += np.asarray(seasonal)[:, month]
        state += drift + pool[rng.integers(len(pool), size=draws)]
        for a, h in np.argwhere(steps == t):
            out[:, a, h] = (state[:, a] if kind == 'log_return' else
                           anchor[a]*np.exp(state[:, a]) if kind == 'fx' else
                           anchor[a]+state[:, a])
    if not np.isfinite(out).all(): raise ValueError('non-finite trend forecast')
    return out


def evidence_weights(losses, available):
    """Shrink inverse validation CRPS weights with no privileged component floor.

    Two usable origins are required to give a component weight. Candidate failure
    gives zero weight, never a good score. No weights are fitted on outer folds.
    """
    if 'random_walk' not in available:
        raise ValueError('ensemble requires random_walk reference')
    eligible = [m for m in available if len(losses.get(m, [])) == 2 and
                np.isfinite(losses[m]).all() and min(losses[m]) >= 0]
    if 'random_walk' not in eligible or len(eligible) < 2:
        return {'random_walk': 1.}
    inverse = np.asarray([1/max(float(np.mean(losses[m])), 1e-8) for m in eligible])
    inverse /= inverse.sum()
    shrunk = .5*inverse + .5/len(eligible)
    return {m: float(w) for m, w in zip(eligible, shrunk)}


def weighted_joint_mixture(samples, weights, seed):
    """Select complete draw vectors, never average draws or mix cells separately."""
    names = list(weights); first = samples[names[0]]; n = len(first)
    if first.ndim != 3 or any(samples[m].shape != first.shape or not np.isfinite(samples[m]).all() for m in names):
        raise ValueError('invalid joint mixture shapes/values')
    p = np.array([weights[m] for m in names], dtype=float)
    if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError('invalid mixture weights')
    counts = np.floor(n*p).astype(int)
    for i in np.argsort(-(n*p-counts), kind='stable')[:n-counts.sum()]: counts[i] += 1
    rng = np.random.default_rng(seed)
    parts = [samples[m][rng.choice(n, size=k, replace=False)] for m, k in zip(names, counts) if k]
    joined = np.concatenate(parts)
    return joined[rng.permutation(n)]


def simulate_solution_ensemble(raw, kind, steps, draws, seed, window, panel, families, audit=None, commander_weights=None):
    from .engine import simulate
    from .tree_forecast import simulate_tree
    from .evaluation import strict_truth
    components = ('random_walk', 'damped_trend', 'relative_tree')
    details = {'recipe': 'solutions_v1', 'weight_objective': 'mean marginal CRPS / historical increment std / sqrt(horizon)',
               'origins': [], 'failures': [], 'weights': {}, 'tree': {}}
    def predict(name, history, n, info=None):
        context = panel.loc[panel.index <= history.index[-1]] if panel is not None else None
        if name == 'random_walk': return simulate(history, kind, name, steps, n, seed, window)
        if name == 'damped_trend': return simulate_trend(history, kind, steps, n, seed, window)
        return simulate_tree(history, kind, steps, n, seed, window, context, families,
                             profile='relative', calibration='joint_kde', audit=info)
    if commander_weights is not None:
        weights = validate_ensemble_weights(commander_weights)
        selected = {name: predict(name, raw, draws, details['tree'] if name == 'relative_tree' else None)
                    for name, weight in weights.items() if weight > 0}
        weights = {name: weight for name, weight in weights.items() if weight > 0}
        details.update(weights=weights, weight_source='commander', requested_weights=commander_weights,
                       note='Commander selected weights from supplied numerical analysis. Component failure rejects this candidate; no silent reweighting.')
        if audit is not None: audit.update(details)
        return weighted_joint_mixture(selected, weights, seed)
    H = int(steps.max()); losses = {m: [] for m in components}
    from .engine import _increments
    # Only two H-separated inner checks. All labels end before final forecast origin.
    for origin in (len(raw)-2*H, len(raw)-H):
        if origin < 80: continue
        history = raw.iloc[:origin]
        try: truth = strict_truth(raw.iloc[origin:origin+H], kind, steps)
        except ValueError as error:
            details['failures'].append({'origin': origin, 'component': 'truth', 'reason': str(error)}); continue
        # Score in output units (FX levels, not log increments).
        inc = history.iloc[-window:].diff() if kind == 'fx' else _increments(history.iloc[-window:], kind)
        scale = np.maximum(inc.std().to_numpy()[:, None] * np.sqrt(steps), 1e-10)
        details['origins'].append({'origin': origin, 'train_end': str(history.index[-1]),
                                   'validation_end': str(raw.index[origin+H-1])})
        for name in components:
            try:
                sample = predict(name, history, 128)
                loss = np.mean([crps_ensemble(sample[:,a,h], truth[a,h], fair=False) / scale[a,h]
                                for a,h in np.ndindex(steps.shape)])
                if not np.isfinite(loss): raise ValueError('nonfinite component loss')
                losses[name].append(float(loss))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                details['failures'].append({'origin': origin, 'component': name, 'reason': str(error)})
    preliminary = evidence_weights(losses, components)
    samples = {}
    for name in preliminary:
        try: samples[name] = predict(name, raw, draws, details['tree'] if name == 'relative_tree' else None)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            details['failures'].append({'origin': len(raw), 'component': name, 'reason': str(error)})
    weights = evidence_weights(losses, samples)
    details.update(weights=weights, weight_source='historical_error_no_floor', validation_losses=losses,
                   note='Internal weight diagnostics, not independent holdout. Fixed shrinkage, no parameter search.')
    if audit is not None: audit.update(details)
    if weights == {'random_walk': 1.}: return samples['random_walk']
    return weighted_joint_mixture(samples, weights, seed)


def validate_ensemble_weights(weights):
    allowed = {'random_walk', 'damped_trend', 'relative_tree'}
    if not isinstance(weights, dict) or not weights or not set(weights) <= allowed:
        raise ValueError('invalid ensemble component names')
    if any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('ensemble weights must be finite and nonnegative')
    if not np.isclose(sum(weights.values()), 1., atol=1e-8, rtol=0):
        raise ValueError('ensemble weights must sum to one')
    return dict(weights)
