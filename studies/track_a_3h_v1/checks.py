"""Spark parity checks against the reviewed Python oracle."""
from collections import Counter
from pyspark.sql import functions as F
from features import append_role, ROLES, HOUR
from studies.ml_three_track_v1.track_a import enrich, DATES
from airline.dws.ml_feature_contract import TARGET_COLUMN


def parity(spark, rows, targets):
    schema = 'origin string, destination string, _prediction_us long, _available_us long, ' + TARGET_COLUMN + ' int'
    def frame(values):
        return spark.createDataFrame([(r['origin'],r['destination'],r['prediction'],r['available'],r['label']) for r in values], schema)
    source, target = frame(rows), frame(targets)
    result = target
    for role in ROLES:
        result = append_role(result, source, role)
    def key(row):
        return tuple(row[c] for c in ('origin','destination','_prediction_us','_available_us',TARGET_COLUMN))
    expected = Counter()
    for row, values in enrich(targets, rows):
        expected[(row['origin'],row['destination'],row['prediction'],row['available'],row['label'],
                  *[values[f'{r}_3h'][k] for r in ROLES for k in ('rate','count','missing')])] += 1
    actual = Counter((*key(r), *[r[f'{role}_{kind}_3h'] for role in ROLES for kind in ('delay_rate','flight_count','history_missing')]) for r in result.collect())
    assert expected == actual
    return sum(actual.values())


def generated(spark):
    def row(d,a,y=1,airport='A'):
        return dict(origin=airport,destination=airport,prediction=d,available=a,label=y)
    t=10*HOUR
    sources=[row(t-3*HOUR,t-1),row(t-3*HOUR,t-1),row(t-3*HOUR-1,t-1),
             row(t-1,t),row(t-1,t+1),row(t,t-1),row(t+1,0),row(t-1,None),
             row(t-2,t-1,0),row(None,None),row(t-2,t-1,0,None)]
    targets=[row(t,t+1),row(t,t+1),row(None,None),row(t,t+1,0,None),row(t,t+1,0,'EMPTY')]
    count=parity(spark,sources,targets)
    changed=[dict(s,label=1-s['label']) if s['prediction'] is None or s['prediction']>=t or s['available'] is None or s['available']>=t else s for s in sources]
    # Compare oracle and Spark after forbidden-outcome perturbations, and oracle invariance.
    assert [v for _,v in enrich(targets,sources)] == [v for _,v in enrich(targets,changed)]
    parity(spark,changed,targets)
    parity(spark,list(reversed(sources)),list(reversed(targets)))
    return {'status':'PASS','cases':['exact microsecond boundaries','duplicate sources/targets','null entity/time','future/current/inflight exclusion','forbidden outcome perturbation','order invariance'], 'target_occurrences_per_check':count}


def sample(spark, source):
    bounded=source.filter(((F.year('flight_date')==2023)&F.month('flight_date').isin(1,7) | (F.year('flight_date')==2024)&(F.month('flight_date')==5)) & F.dayofmonth('flight_date').between(13,18))
    rows=[];targets=[]
    for r in bounded.select('flight_date','origin','destination','_prediction_us','_available_us',TARGET_COLUMN).toLocalIterator():
        value=dict(origin=r.origin,destination=r.destination,prediction=r._prediction_us,available=r._available_us,label=r[TARGET_COLUMN])
        rows.append(value)
        if r.flight_date in DATES: targets.append(value)
    assert len(rows)==331355 and len(targets)==109423
    count=parity(spark,rows,targets)
    return {'status':'PASS','eligible_sources':len(rows),'targets':count,'comparison':'all six values, every target occurrence, exact multiset parity with track_a.enrich'}
