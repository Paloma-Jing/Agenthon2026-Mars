"""Frozen prefix replay + version-held-out empirical leaderboard calibration.

Offline source replay is a proxy, not House reproduction. No scorer edits, hidden
outcomes, API calls, component ablations, or fitting sample choice to online labels.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time, tomllib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UNITS = ['t2-F3-conundrum-joint-2005', 't2-F1-conflicting-texts-2024',
         't2-F2-whatever-it-takes-2012', 't2-F3-funding-carry-fx-2024',
         't2-F4-factor-joint-credit-2007', 't2-F4-hml-covid-2020',
         't2-F1-cpi-glidepath-2023']
# Supply private chronological scores through runs/.../online_scores.json.
LABELS = {}
M0_SCORE = None


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


def hashes(paths):
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def prepare(out):
    from numeric_agent.engine import ForecastSession
    from numeric_agent.evaluation import strict_truth, fold_reference, score
    manifest = out/'roster.json'
    if manifest.exists():
        roster = json.loads(manifest.read_text())
        for path, digest in roster['hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Frozen input changed: {path}')
        return roster
    population = {}
    for p in (ROOT/'track2-forecasting-public/units').glob('t2-*/card.toml'):
        if p.parent.name.startswith('t2-EXAMPLE'): continue
        family = tomllib.loads(p.read_text())['metadata']['asset_panel']
        population[family] = population.get(family, 0)+1
    cases = []; excluded = []; inputs = []
    for uid in UNITS:
        unit = ROOT/'track2-forecasting-public/units'/uid
        card = tomllib.loads((unit/'card.toml').read_text())
        s = ForecastSession(unit, unit/'card.toml', card['provenance']['data_cutoff'], out/'unused', draws=500, audit_root=None)
        H = int(s.steps.max()); chosen = None
        # Data-only endpoint search, never inspect model performance to choose a fold.
        for offset in range(64 if not s.monthly else 12):
            origin = len(s.raw)-H-offset
            if origin < 70: break
            try:
                truth = strict_truth(s.raw.iloc[origin:origin+H], s.kind, s.steps)
                reference, scales = fold_reference(s.raw.iloc[:origin], s.target, s.steps, card)
                chosen = origin, truth, reference, scales; break
            except ValueError: pass
        if chosen is None:
            excluded.append({'unit':uid, 'reason':'no complete truth within fixed endpoint search'}); continue
        origin, truth, reference, scales = chosen
        last = s.raw.index[origin-1]
        cutoff = str(last.to_timestamp(how='end').date() if s.monthly else last.date())
        stage = out/'inputs'/uid; stage.mkdir(parents=True)
        # The release process gets physically truncated panels, never full panels/truth.
        for p in unit.glob('*.parquet'):
            frame = pd.read_parquet(p)
            frame.loc[pd.to_datetime(frame.date) <= pd.Timestamp(cutoff)].to_parquet(stage/p.name,index=False)
            inputs.append(p)
        original = (unit/'card.toml').read_text()
        (stage/'card.toml').write_text(original.replace(str(card['provenance']['data_cutoff']),cutoff))
        inputs.append(unit/'card.toml')
        if s.monthly:
            spec = json.loads((unit/'forecast_spec.json').read_text())
            spec['targets']['observation_periods'] = [str(last+int(k)) for k in s.steps[0]]
            save(stage/'forecast_spec.json',spec); inputs.append(unit/'forecast_spec.json')
        draws = reference.draws(f'{uid}:{origin}',500)
        cases.append({'unit':uid,'family':card['metadata']['asset_panel'],'origin':origin,
                      'cutoff':cutoff,'steps':s.steps.tolist(),'assets':s.assets,'horizons':s.horizons,
                      'truth':truth.tolist(),'scales':scales,'m0_loss':score(draws,truth,scales,card)['normalized_composite'],
                      'test_start':str(s.raw.index[origin]),'test_end':str(s.raw.index[origin+H-1]),
                      'endpoint_search_offset':offset})
    # All families must be represented; do not silently renormalize missing coverage.
    if set(population) != {c['family'] for c in cases}: raise ValueError('sample misses a population family')
    frozen = inputs+list((out/'inputs').rglob('*.*'))+[ROOT/'numeric_agent/evaluation.py', ROOT/'numeric_agent/m0.py', Path(__file__)]
    roster = {'cases':cases,'excluded':excluded,'population':population,'hashes':hashes(frozen),
              'protocol':'7 predeclared cards, one complete historical window/card; family population weights; 500 draws; source offline replay',
              'limitations':['small purposive sample','no historical macro vintages','no House/text policy replay','shared current dependency runtime','online version IDs inferred from chronology']}
    save(manifest,roster); return roster


def aggregate(cases, rows, population):
    by_unit = {r['unit']:r for r in rows}
    if len(by_unit)!=len(rows): raise ValueError('duplicate unit')
    if set(by_unit)-{c['unit'] for c in cases}: raise ValueError('unknown unit')
    family = {}
    for f in population:
        values = [8. if by_unit.get(c['unit'],{}).get('status')=='failed' else float(by_unit.get(c['unit'],{}).get('loss',8.)) for c in cases if c['family']==f]
        if not values or not np.isfinite(values).all() or min(values)<0: raise ValueError('invalid roster/loss')
        family[f] = float(np.mean(np.minimum(values,8.)))
    return sum(family[f]*population[f] for f in population)/sum(population.values()), family


def replay(out, source, name, roster):
    fingerprint = hashes(list((source/'numeric_agent').glob('*.py')))
    if not fingerprint: raise ValueError('source must contain numeric_agent/*.py')
    config = out/'replays'/name/'source.json'
    if config.exists() and json.loads(config.read_text())!=fingerprint: raise ValueError('source changed; choose a new run name')
    save(config,fingerprint); rows=[]
    for n,c in enumerate(roster['cases'],1):
        case = config.parent/c['unit']; result = case/'result.json'
        if result.exists(): rows.append(json.loads(result.read_text())); continue
        case.mkdir(parents=True,exist_ok=True)
        stage = out/'inputs'/c['unit']; forecast = case/'output/forecast.parquet'
        cmd = [sys.executable,'-m','numeric_agent','--panels',str(stage),'--card',str(stage/'card.toml'),
               '--asof',c['cutoff'],'--out',str(forecast),'--mode','offline','--draws','500','--timeout','180']
        env = os.environ.copy();env['PYTHONPATH']=str(source);env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
        started=time.monotonic();print(f'START {name} {n}/{len(roster["cases"])} {c["unit"]}',flush=True)
        with (case/'run.log').open('w') as log:
            proc = subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: print(f'HEARTBEAT {name} {n} elapsed={time.monotonic()-started:.0f}s',flush=True)
                if time.monotonic()-started>200: proc.kill();proc.wait();break
        row={'unit':c['unit'],'status':'failed','loss':8.,'returncode':proc.returncode}
        try:
            if proc.returncode: raise RuntimeError('release process failed')
            frame=pd.read_parquet(forecast)
            # Align explicitly, never rely on incidental parquet row ordering.
            id_col='draw'; asset_col='asset'
            grid=pd.MultiIndex.from_product([sorted(frame[id_col].unique()),c['assets'],c['horizons']],names=[id_col,asset_col,'horizon'])
            series=frame.set_index([id_col,asset_col,'horizon']).value
            if series.index.has_duplicates or len(series)!=len(grid): raise ValueError('invalid forecast grid')
            samples=series.reindex(grid).to_numpy().reshape(-1,len(c['assets'])*len(c['horizons']))
            from numeric_agent.evaluation import score
            card=tomllib.loads((stage/'card.toml').read_text())
            scored=score(samples,c['truth'],c['scales'],card)
            meta=json.loads((forecast.parent/'forecast_meta.json').read_text())
            row.update(status='ok',loss=scored['normalized_composite'],selected=meta.get('selected_method'),components=scored)
        except Exception as exc: row.update(error=f'{type(exc).__name__}: {str(exc)[:180]}')
        row['seconds']=time.monotonic()-started;save(result,row);rows.append(row)
        print(f'DONE {name} {n} {row["status"]} loss={row["loss"]:.4f}',flush=True)
    loss,family=aggregate(roster['cases'],rows,roster['population'])
    summary={'name':name,'local_loss':loss,'family_losses':family,'failed':sum(r['status']!='ok' for r in rows),'rows':rows}
    save(config.parent/'summary.json',summary);return summary


def fit(xs,ys):
    # A single monotone slope; no version IDs/features to memorize five labels.
    x=np.asarray(xs,float);y=np.asarray(ys,float)
    variance=float(np.sum((x-x.mean())**2))
    slope=min(0.,float(np.sum((x-x.mean())*(y-y.mean()))/variance)) if variance>1e-12 else 0.
    return {'intercept':float(y.mean()-slope*x.mean()),'slope':slope}


def predict(model,x): return model['intercept']+model['slope']*x


def calibrate(out,roster,labels=None):
    labels = LABELS if labels is None else labels
    if len(labels)<3 or not np.isfinite(list(labels.values())).all(): raise ValueError('need at least three finite chronological version labels')
    records=[]
    for name,y in labels.items():
        p=out/'replays'/name/'summary.json'
        if not p.exists(): raise ValueError(f'missing historical replay {name}')
        r=json.loads(p.read_text());records.append({'version':name,'x':r['local_loss'],'y':y,'failed':r['failed']})
    folds=[]
    for held in records:
        train=[r for r in records if r['version']!=held['version']]
        model=fit([r['x'] for r in train],[r['y'] for r in train])
        prediction=predict(model,held['x']); baseline=float(np.mean([r['y'] for r in train]))
        folds.append(dict(held,prediction=prediction,error=prediction-held['y'],mean_baseline=baseline,baseline_error=baseline-held['y']))
    mae=float(np.mean([abs(r['error']) for r in folds]));base=float(np.mean([abs(r['baseline_error']) for r in folds]))
    final=fit([r['x'] for r in records],[r['y'] for r in records])
    latest=folds[-1]
    # Predeclared operational bar: beats intercept-only and latest error <=0.5 display points.
    trusted=mae<base and abs(latest['error'])<=.5
    report={'historical':records,'leave_one_version_out':folds,'heldout_mae':mae,'mean_baseline_mae':base,
            'latest_version_holdout':latest,'latest_score_baseline':{'prediction':records[-2]['y'],'absolute_error':abs(records[-2]['y']-records[-1]['y'])},'calibration':final,'usable_for_close_prediction':trusted,
            'acceptance_rule':'LOVO MAE < mean-only baseline MAE AND latest chronological version absolute error <= 0.5',
            'empirical_error_radius':max(abs(r['error']) for r in folds),
            'interval_note':'Observed held-out worst error, not a confidence interval or future coverage guarantee.',
            'm0_local_loss':aggregate(roster['cases'],[{'unit':c['unit'],'loss':c['m0_loss']} for c in roster['cases']],roster['population'])[0],
            'm0_online_score':M0_SCORE,'m0_note':'Reported separately; not inserted as a sixth entrant training version.',
            'limitations':roster['limitations']}
    save(out/'calibration.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'runs/score_predictor_20261014')
    p.add_argument('--source',type=Path,help='New candidate root containing numeric_agent; same frozen protocol')
    p.add_argument('--name',default='candidate');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);roster=prepare(out)
    if a.prepare_only:print(f'Frozen {len(roster["cases"])} cases');return
    label_path=out/'online_scores.json'
    if not label_path.exists():
        raise SystemExit('Provide online_scores.json in --out: chronological version names mapped to scores (at least three).')
    labels=json.loads(label_path.read_text())
    if not a.source:
        for name in labels:replay(out,out/'releases'/name,name,roster)
    report=calibrate(out,roster,labels)
    if a.source:
        if a.name in labels:raise ValueError('candidate name cannot overwrite historical release')
        r=replay(out,a.source.resolve(),a.name,roster);estimate=predict(report['calibration'],r['local_loss'])
        radius=report['empirical_error_radius']
        result={'name':a.name,'local_loss':r['local_loss'],'estimated_display_score':estimate if report['usable_for_close_prediction'] else None,
                'diagnostic_calibrated_score':estimate,'latest_known_online_score':list(labels.values())[-1],
                'empirical_range':[estimate-radius,estimate+radius],'validated_for_close_prediction':report['usable_for_close_prediction'],
                'extrapolation':not min(v['x'] for v in report['historical'])<=r['local_loss']<=max(v['x'] for v in report['historical'])}
        save(out/'replays'/a.name/'prediction.json',result);print(json.dumps(result,indent=2))
    print(json.dumps({k:report[k] for k in ('heldout_mae','mean_baseline_mae','latest_version_holdout','usable_for_close_prediction')},indent=2))

if __name__=='__main__':main()
