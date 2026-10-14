"""Inventory public numeric inputs only, reporting progress every twenty units."""
from pathlib import Path
import json,time,tomllib,collections
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];start=time.monotonic();rows=[];counts=collections.Counter()
for i,p in enumerate(sorted((ROOT/'track2-forecasting-public/units').glob('*/card.toml')),1):
    if 'EXAMPLE' in p.parent.name:continue
    c=tomllib.loads(p.read_text());assets=set();panels=[]
    for f in p.parent.glob('*.parquet'):
        d=pd.read_parquet(f);ac='asset' if 'asset' in d else 'asset_id'
        d=d[d.date.astype(str).str[:10]<=str(c['provenance']['data_cutoff'])];assets.update(d[ac].unique())
        panels.append({'file':f.name,'rows':len(d),'assets':sorted(d[ac].unique().tolist()),'columns':list(d.columns),'start':str(d.date.min()),'end':str(d.date.max())})
    kind=c['metadata']['asset_panel'];counts[kind]+=1
    rows.append({'unit':p.parent.name,'panel_type':kind,'target':c['targets'],'missing_targets':sorted(set(c['targets']['asset_ids'])-assets),'panels':panels})
    if i%20==0:print(f'Inventory {i} units; elapsed={time.monotonic()-start:.1f}s',flush=True)
out=ROOT/'runs/numeric_agent/data_inventory.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'counts':dict(counts),'units':rows},indent=2,default=str))
print(json.dumps({'counts':dict(counts),'units_missing_targets':[(x['unit'],x['missing_targets']) for x in rows if x['missing_targets']],'elapsed_seconds':time.monotonic()-start}),flush=True)
