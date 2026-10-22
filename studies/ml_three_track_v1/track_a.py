"""Isolated interval-sweep histories. No Spark session, model fit or feature publication."""
from bisect import bisect_right
from collections import defaultdict, Counter
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import resource
import time

HOUR = 3_600_000_000
WINDOWS = (1,3)
DATES = tuple(date.fromisoformat(x) for x in (
    '2023-01-15','2023-01-16','2023-07-15','2023-07-16','2024-05-15','2024-05-16'))


def index(sources, role, hours):
    events = defaultdict(lambda:defaultdict(lambda:[0,0]))
    for source in sources:
        dep,available,label = source['prediction'],source['available'],source['label']
        if dep is None or available is None:
            continue
        start,end = max(dep,available)+1,dep+hours*HOUR+1
        if start>=end:
            continue
        for moment,sign in ((start,1),(end,-1)):
            pair=events[source[role]][moment]
            pair[0]+=sign;pair[1]+=sign*label
    result={}
    for key,changes in events.items():
        times=[];counts=[];delays=[];n=y=0
        for moment,(dn,dy) in sorted(changes.items()):
            n+=dn;y+=dy
            assert 0<=y<=n
            times.append(moment);counts.append(n);delays.append(y)
        result[key]=(times,counts,delays)
    return result


def query(lookup,key,prediction):
    n=y=0
    if prediction is not None and key in lookup:
        times,counts,delays=lookup[key]
        i=bisect_right(times,prediction)-1
        if i>=0:n,y=counts[i],delays[i]
    return {'count':n,'rate':y/n if n else None,'missing':int(n==0)}


def enrich(targets,sources):
    indexes={(role,h):index(sources,role,h) for role in ('origin','destination') for h in WINDOWS}
    return [(row,{f'{role}_{h}h':query(lookup,row[role],row['prediction'])
                  for (role,h),lookup in indexes.items()}) for row in targets]


def key(row):
    return (str(row['flight_date']),row['reporting_airline'],int(row['flight_number_reporting_airline']),
            int(row['origin_airport_id']),int(row['dest_airport_id']),int(row['crs_dep_time']))


def main():
    import numpy as np
    import pyarrow.dataset as ds
    from airline.dws.flight_time import scheduled_converter, airport_zones, time_provenance
    from airline.dws.ml_feature_contract import TARGET_COLUMN
    out=Path('/output/a');out.mkdir(parents=True,exist_ok=False)
    tick=time.monotonic();convert=scheduled_converter(airport_zones())
    columns=['flight_date','reporting_airline','flight_number_reporting_airline','origin_airport_id',
             'dest_airport_id','crs_dep_time','origin','dest','dep_delay','actual_elapsed_time',
             'is_cancelled','is_diverted',TARGET_COLUMN]
    sources=[];targets=[];files=[];read_rows=0;excluded=Counter()
    for year,month in ((2023,1),(2023,7),(2024,5)):
        parts=sorted(Path(f'/dwd/year={year}/month={month}').glob('*.parquet'))
        assert parts
        files.extend({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in parts)
        lo,hi=date(year,month,13),date(year,month,18)
        table=ds.dataset([str(p) for p in parts],format='parquet').to_table(columns=columns,
            filter=(ds.field('flight_date')>=lo)&(ds.field('flight_date')<=hi))
        for ordinal,row in enumerate(table.to_pylist()):
            read_rows+=1
            if row['is_cancelled']!=0 or row['is_diverted']!=0 or row[TARGET_COLUMN] not in (0,1):
                excluded['ineligible']+=1;continue
            dep=convert(row['flight_date'],row['crs_dep_time'],row['origin_airport_id'])
            delay,elapsed=row['dep_delay'],row['actual_elapsed_time']
            available=None
            if dep is not None and delay is not None and elapsed is not None and math.isfinite(delay) and math.isfinite(elapsed) and elapsed>=0:
                offset=(delay+elapsed)*60_000_000
                if math.isfinite(offset) and abs(offset)<=8e18:
                    # DWD source is minute precision; half-up matches Spark round.
                    available=dep+int(math.copysign(math.floor(abs(offset)+0.5),offset))
            source={'key':key(row),'occurrence':(year,month,ordinal),'origin':row['origin'],
                    'destination':row['dest'],'prediction':dep,'available':available,'label':row[TARGET_COLUMN]}
            sources.append(source)
            if row['flight_date'] in DATES:targets.append(source)
        del table
    baseline_cols=['flight_date','reporting_airline','flight_number_reporting_airline','origin_airport_id',
                   'dest_airport_id','crs_dep_time']
    existing=[f'{role}_{kind}_{days}d' for role in ('origin','destination')
              for kind in ('flight_count','delay_rate') for days in (7,30,90)]
    table=ds.dataset('/selection',format='parquet',partitioning='hive',exclude_invalid_files=True).to_table(
        columns=baseline_cols+existing,filter=ds.field('flight_date').isin(DATES))
    baseline=defaultdict(list)
    for row in table.to_pylist():baseline[key(row)].append(row)
    assert Counter(x['key'] for x in targets)==Counter({k:len(v) for k,v in baseline.items()})
    result=enrich(targets,sources)
    assert len(result)==len(targets) and all(a is b for a,(b,_) in zip(targets,result))
    identity=lambda rows:hashlib.sha256(json.dumps([r['occurrence'] for r in rows]).encode()).hexdigest()
    summaries=[]
    for name in ('origin_1h','origin_3h','destination_1h','destination_3h'):
        role,h=name.split('_');counts=np.array([x[name]['count'] for _,x in result])
        rates=np.array([np.nan if x[name]['rate'] is None else x[name]['rate'] for _,x in result])
        comparisons={}
        for days in (7,30,90):
            for kind,values in [('flight_count',counts),('delay_rate',rates)]:
                old=np.array([baseline[row['key']][0][f'{role}_{kind}_{days}d'] for row,_ in result],dtype=float)
                good=np.isfinite(old)&np.isfinite(values)
                comparisons[f'{kind}_{days}d']={'paired_rows':int(good.sum()),
                    'pearson':float(np.corrcoef(old[good],values[good])[0,1]),
                    'unequal_fraction':float(np.mean(np.abs(old[good]-values[good])>1e-12))}
        per_date={str(d):{'rows':sum(row['key'][0]==str(d) for row,_ in result),
            'covered':sum(row['key'][0]==str(d) and x[name]['count']>0 for row,x in result)} for d in DATES}
        summaries.append({'feature':name,'rows':len(targets),'nonmissing':int((counts>0).sum()),
            'missing_fraction':float(np.mean(counts==0)),'count_quantiles':dict(zip(['min','p50','p90','p99','max'],map(float,np.quantile(counts,[0,.5,.9,.99,1])))),
            'coverage_by_date':per_date,'comparison_to_existing':comparisons})
    # Independent brute-force oracle on first 8 target rows per role/window.
    oracle_checks=0
    for row,features in result[:8]:
        for role in ('origin','destination'):
            for h in WINDOWS:
                p=row['prediction']
                eligible=[s for s in sources if p is not None and s[role]==row[role] and s['prediction'] is not None
                    and p-h*HOUR<=s['prediction']<p and s['available'] is not None and s['available']<p]
                value=features[f'{role}_{h}h']
                assert value['count']==len(eligible)
                assert value['rate']==(sum(s['label'] for s in eligible)/len(eligible) if eligible else None)
                oracle_checks+=1
    report={'status':'PASS','time_policy':time_provenance(),'target_dates':[str(d) for d in DATES],
        'source_local_date_ranges':['2023-01-13/18','2023-07-13/18','2024-05-13/18'],
        'dwd_rows_read':read_rows,'eligible_source_rows':len(sources),'target_rows':len(targets),
        'excluded':dict(excluded),'null_target_prediction_times':sum(x['prediction'] is None for x in targets),
        'duplicate_target_occurrences':len(targets)-len({x['key'] for x in targets}),
        'existing_dataset_population_exact':True,'input_occurrence_sha256':identity(targets),
        'output_occurrence_sha256':identity([r for r,_ in result]),'oracle_checks':oracle_checks,
        'features':summaries,'source_files':files,'seconds':time.monotonic()-tick,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'limitations':['Arrival-event availability assumption, not observed receipt.','Descriptive feature comparisons, no predictive gain test.','Purposive six dates, not nationwide annual coverage.','One-hour departure lookback disproportionately excludes long or delayed flights still in progress.']}
    (out/'results.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('features','source_files')},indent=2))


if __name__=='__main__':main()
