"""Causal, named features consumed by direct-horizon trees and exposed to agents.

Feature rows are indexed by the last observed time t. Labels begin at t+1. No backward
fill, no future context, and no random validation split. Monthly snapshot vintages remain
an explicit limitation: truncating a snapshot does not recreate historical releases.
"""
import numpy as np
import pandas as pd
from .factors import family_of


def catalog(monthly=False):
    return {'models': ['random_walk', 'gbm', 'solution_ensemble'],
            'feature_profiles': ['core', 'market', 'compact', 'relative'],
            'active_profile': 'core',
            'calibration_gap': 'one additional maximum forecast horizon after training labels',
            'market_status': 'experimental; failed aggregate development ablation; not active by default',
            'compact_status': 'experimental; compact/replay did not improve development; blend failed per-task confirmation guard; active core unchanged',
            'market_profile': 'within-family standardized market mean/dispersion, rolling statistics and target-relative market; supervised top-16 selection uses fit partition only',
            'groups': {'own': 'target increment lags, momentum and volatility',
                       'context': 'time-varying increments and momentum of selected panel families',
                       'rates': 'observed 10Y minus 2Y curve spread'},
            'windows': [3, 12] if monthly else [21, 63],
            'alignment': 'row t contains information through t; predicts t+1 onward',
            'missing_policy': 'no backfill; missing features remain NaN; trees handle missingness',
            'vintage_limit': 'snapshot history is not a historical release-vintage replay'}


def feature_table(raw, kind, panel=None, families=None, profile='core'):
    """Build the same feature matrix used for training, preview and final prediction."""
    from .engine import _increments
    monthly = isinstance(raw.index, pd.PeriodIndex)
    short, long = (3, 12) if monthly else (21, 63)
    inc = _increments(raw, kind)
    columns = {}
    for asset in raw:
        s = inc[asset]
        for lag in ((0,) if profile=='compact' else (0, 1, 2, 4)):
            columns[f'own:{asset}:lag{lag+1}'] = s.shift(lag)
        for w in (short, long):
            columns[f'own:{asset}:momentum{w}'] = s.rolling(w, min_periods=max(2,w//2)).sum()
            columns[f'own:{asset}:vol{w}'] = s.rolling(w, min_periods=max(2,w//2)).std()
            if profile=='compact':
                columns.pop(f'own:{asset}:momentum{w}')
                columns[f'own:{asset}:mean{w}']=s.rolling(w,min_periods=max(2,w//2)).mean()
                columns[f'own:{asset}:max{w}']=s.rolling(w,min_periods=max(2,w//2)).max()
    if panel is not None:
        # Cut before deriving any context, even when the caller supplies a longer panel.
        p = panel.loc[panel.index <= raw.index[-1]].reindex(raw.index)
        for asset in sorted(p.columns):
            family = family_of(p, str(asset))
            if asset in raw or family is None or (families is not None and family not in families):
                continue
            s = p[asset]
            if family == 'fx':
                s = np.log(s.where(s > 0)).diff()
            elif family == 'factors':
                s = np.log1p(s.where(s > -1))
            else:
                s = s.diff()
            columns[f'{family}:{asset}:increment'] = s
            columns[f'{family}:{asset}:momentum{short}'] = s.rolling(short, min_periods=max(2,short//2)).sum()
        if (families is None or 'rates' in families) and {'UST_10Y','UST_2Y'} <= set(p.columns):
            columns['rates:curve:10Y_minus_2Y'] = p['UST_10Y'] - p['UST_2Y']
    if profile not in ('core','market','compact','relative'):
        raise ValueError('unknown feature profile')
    if profile == 'relative':
        # Price-only adaptation: no order-book, volume or future market aggregates.
        windows = (3, 6, 12) if monthly else (5, 21, 63)
        for asset in raw:
            s = inc[asset]
            for w in windows:
                vol = s.shift(1).rolling(w, min_periods=max(3,w//2)).std()
                move = s.rolling(w, min_periods=w).sum()
                columns[f'own:{asset}:scaled_move{w}'] = move/(vol.where(vol>1e-12)*np.sqrt(w))
            short_vol = s.rolling(windows[0], min_periods=windows[0]).std()
            long_vol = s.rolling(windows[-1], min_periods=windows[-1]).std()
            columns[f'own:{asset}:vol_ratio'] = short_vol/long_vol.where(long_vol>1e-12)
            if panel is not None:
                family = family_of(p, str(asset))
                if family is None or (families is not None and family not in families): continue
                peers = [a for a in p if a != asset and family_of(p,str(a)) == family]
                if not peers: continue
                values = p[peers]
                changes = (np.log(values.where(values>0)).diff() if family=='fx' else
                           np.log1p(values.where(values>-1)) if family=='factors' else values.diff())
                for w in windows:
                    sd = changes.shift(1).rolling(w,min_periods=max(3,w//2)).std()
                    moves = changes.rolling(w,min_periods=w).sum()/(sd.where(sd>1e-12)*np.sqrt(w))
                    peer = moves.mean(axis=1)
                    columns[f'own:{asset}:relative_move{w}'] = columns[f'own:{asset}:scaled_move{w}']-peer
        if monthly:
            columns['calendar:month_sin'] = np.sin(2*np.pi*raw.index.month/12)
            columns['calendar:month_cos'] = np.cos(2*np.pi*raw.index.month/12)
    if profile == 'market':
        # Adapt the writeup's per-symbol rolling statistics and same-time market averages.
        for asset in raw:
            s=inc[asset]
            for w in (short,long):
                mean=s.rolling(w,min_periods=max(2,w//2)).mean()
                std=s.rolling(w,min_periods=max(2,w//2)).std()
                columns[f'own:{asset}:mean{w}']=mean
                columns[f'own:{asset}:z{w}']=(s-mean)/std.where(std>1e-12)
        if panel is not None:
            grouped={}
            for asset in sorted(p.columns):
                family=family_of(p,str(asset))
                if family is None or (families is not None and family not in families):continue
                v=p[asset]
                if family=='fx':v=np.log(v.where(v>0)).diff()
                elif family=='factors':v=np.log1p(v.where(v>-1))
                else:v=v.diff()
                # Heterogeneous macro levels cannot be averaged in their original units.
                # Normalize each increment by past-only moments before combining peers.
                past=v.shift(1).rolling(long,min_periods=max(3,long//2))
                sd=past.std()
                z=(v-past.mean())/sd.where(sd>1e-12)
                grouped.setdefault(family,{})[asset]=z
            for family,items in grouped.items():
                z=pd.DataFrame(items,index=raw.index)
                enough=z.notna().sum(axis=1)>=2
                market=z.mean(axis=1).where(enough)
                columns[f'market:{family}:mean']=market
                columns[f'market:{family}:dispersion']=z.std(axis=1).where(enough)
                columns[f'market:{family}:rolling_mean{short}']=market.rolling(short,min_periods=max(2,short//2)).mean()
                columns[f'market:{family}:rolling_std{short}']=market.rolling(short,min_periods=max(2,short//2)).std()
                for asset in raw:
                    if asset in z:
                        peers=z.drop(columns=asset)
                        peer_mean=peers.mean(axis=1).where(peers.notna().sum(axis=1)>=1)
                        columns[f'own:{asset}:relative_market']=z[asset]-peer_mean
    return pd.DataFrame(columns, index=raw.index).replace([np.inf,-np.inf],np.nan)


def feature_report(raw, kind, panel=None, families=None):
    x = feature_table(raw, kind, panel, families)
    market = feature_table(raw, kind, panel, families, profile='market')
    compact = feature_table(raw, kind, panel, families, profile='compact')
    relative = feature_table(raw, kind, panel, families, profile='relative')
    valid = x.notna().sum()
    usable = (valid >= 30) & (x.nunique(dropna=True) > 1)
    return {'catalog': catalog(isinstance(raw.index,pd.PeriodIndex)),
            'rows':len(x), 'feature_count':len(x.columns),
            'compact_profile_features':compact.columns.tolist(),
            'market_profile_added_features':[c for c in market if c not in x],
            'ensemble_tree_profile':'relative; top16 fit-only selection and joint residual KDE',
            'relative_profile_added_features':[c for c in relative if c not in x],
            'usable_features':x.columns[usable].tolist(),
            'dropped_features':x.columns[~usable].tolist(),
            'missing_fraction':x.isna().mean().to_dict(),
            'latest_missing':x.columns[x.iloc[-1].isna()].tolist(),
            'selected_context_families':families,
            'note':'Preview over supplied history; each fit chooses columns from its training partition only.'}
