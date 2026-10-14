"""Read-only paired review of cached chronological experiments; never fit on validation."""
import numpy as np

from .evaluation import aggregate


def review_candidates(cache):
    ledger = {}
    for method, result in cache.items():
        records = result.get('fold_records', [])
        metrics = aggregate(records) if records else None
        ledger[method] = {'status': result['status'], 'metrics': metrics,
                          'excluded_origins': result.get('excluded_origins', [])}
    eligible = [m for m in ledger if ledger[m]['status'] == 'ok' and ledger[m]['metrics']]
    report = {'experiments': ledger, 'paired': {}, 'comparable': True,
              'recommended_method': None,
              'warning': 'Selection diagnostics only; three folds do not establish independent gain. '
                         'Paired loss differences are not prediction diversity or statistical significance.'}
    baseline = cache.get('random_walk')
    if not baseline or 'random_walk' not in eligible:
        report.update(comparable=False, reason='No successful evaluated random_walk reference.')
        return report
    reference = {r['key']: r for r in baseline['fold_records']}
    for method in eligible:
        if method == 'random_walk':
            continue
        candidate = cache[method]
        rows = {r['key']: r for r in candidate['fold_records']}
        if set(rows) != set(reference) or any(
                candidate.get(k) != baseline.get(k) for k in ('normalization', 'aggregation')) or any(
                rows[k].get('scales') != reference[k].get('scales') for k in set(rows) & set(reference)):
            report['comparable'] = False
            report['paired'][method] = {'status': 'incomparable',
                                        'reason': 'Fold roster or scoring contract differs.'}
            continue
        differences = []
        for key, ref in reference.items():
            row = rows[key]
            losses = [aggregate([r])['mean_normalized_composite'] for r in (row, ref)]
            differences.append({'key': key, 'loss_difference': losses[0] - losses[1],
                                'candidate_status': row['status'], 'reference_status': ref['status']})
        deltas = [r['loss_difference'] for r in differences]
        report['paired'][method] = {'status': 'comparable', 'folds': differences,
                                    'wins': sum(d < 0 for d in deltas),
                                    'ties': sum(d == 0 for d in deltas),
                                    'losses': sum(d > 0 for d in deltas),
                                    'mean_loss_difference': float(np.mean(deltas)),
                                    'worst_loss_increase': max(0., max(deltas)),
                                    'new_failure_keys': [r['key'] for r in differences
                                                         if r['candidate_status'] != 'ok' and r['reference_status'] == 'ok']}
    if report['comparable']:
        # Preserve the existing mean-loss selection criterion; baseline wins exact ties.
        report['recommended_method'] = min(['random_walk'] + [m for m in eligible if m != 'random_walk'],
                                          key=lambda m: ledger[m]['metrics']['mean_normalized_composite'])
        # Sensitivity to a single time block, not a confidence interval: adjacent
        # blocks and common market regimes can remain statistically dependent.
        ordered = ['random_walk'] + [m for m in eligible if m != 'random_walk']
        omitted = []
        if len(reference) > 1:
            for key in reference:
                losses = {m: aggregate([r for r in cache[m]['fold_records'] if r['key'] != key])[
                    'mean_normalized_composite'] for m in ordered}
                omitted.append({'omitted_key': key, 'recommended_method': min(ordered, key=losses.get)})
        report['selection_stability'] = {
            'leave_one_block_out': omitted,
            'winner_changes': sum(r['recommended_method'] != report['recommended_method'] for r in omitted),
            'interpretation': 'Sensitivity diagnostic only; not independent evidence or statistical significance.'}
    return report
