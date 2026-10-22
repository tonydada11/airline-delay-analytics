"""One gated train/validation build. Exclusive outputs, never a model fit."""
import json
import time
from pathlib import Path
from datetime import date
from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F
from airline.dws.build_ml_flight_features import eligibility_condition
from airline.dws.flight_time import with_event_times, scheduled_converter, airport_zones, time_provenance
from airline.dws.ml_feature_contract import TARGET_COLUMN
from features import append_role, FEATURES, ROLES, HOUR
from checks import generated, sample

OUT=Path('/output')
def save(name,value):
    (OUT/(name+'.json')).write_text(json.dumps(value,indent=2,default=str)+'\n')

def group(spark,name):
    spark.sparkContext.setJobGroup(name,name)
    save('phase',{'name':name,'epoch':time.time()})


def main():
    spark=(SparkSession.builder.master('spark://a-master:7077').appName('isolated-track-a-3h')
        .config('spark.driver.host','a-driver').config('spark.driver.bindAddress','0.0.0.0')
        .config('spark.cores.max','2').config('spark.executor.cores','1').config('spark.executor.memory','640m')
        .config('spark.task.cpus','1').config('spark.dynamicAllocation.enabled','false')
        .config('spark.sql.shuffle.partitions','4').config('spark.sql.files.maxPartitionBytes','67108864')
        .config('spark.sql.inMemoryColumnarStorage.batchSize','1000')
        .config('spark.driver.maxResultSize','128m').config('spark.sql.session.timeZone','UTC')
        .config('spark.task.maxFailures','1').config('spark.stage.maxConsecutiveAttempts','1')
        .config('spark.eventLog.enabled','true').config('spark.eventLog.dir','file:///output/events').getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    started=time.monotonic()
    try:
        save('runtime',dict(spark.sparkContext.getConf().getAll()))
        group(spark,'generated')
        save('generated',generated(spark))
        # Explicit partition paths prevent any test-period parquet from being read.
        paths=[f'/dwd/year={y}/month={m}' for y in (2023,2024) for m in range(1,13 if y==2023 else 9)]
        raw=spark.read.option('basePath','/dwd').parquet(*paths).filter(F.col('flight_date').between('2023-01-01','2024-08-31')).filter(eligibility_condition())
        def timed(frame):
            return with_event_times(frame).select('flight_date','origin',F.col('dest').alias('destination'),'_prediction_us','_available_us',TARGET_COLUMN)
        group(spark,'sample')
        bounded=raw.filter(((F.year('flight_date')==2023)&F.month('flight_date').isin(1,7) | (F.year('flight_date')==2024)&(F.month('flight_date')==5)) & F.dayofmonth('flight_date').between(13,18))
        save('sample',sample(spark,timed(bounded)))
        # No full build may start until both Spark parity gates pass.
        (OUT/'FULL_BUILD_STARTED').write_text(str(time.time()))
        group(spark,'source_preparation')
        source=timed(raw).persist(StorageLevel.DISK_ONLY)
        n=source.count()
        assert n==8946370+2418739, n
        save('source',{'rows':n,'time_policy':time_provenance(),'paths':paths,
             'nulls':source.agg(F.sum(F.col('_prediction_us').isNull().cast('long')).alias('prediction'),F.sum(F.col('_available_us').isNull().cast('long')).alias('availability')).first().asDict()})
        baseline=spark.read.parquet('/selection')
        old=baseline.columns
        assert not set(FEATURES)&set(old)
        convert=F.udf(scheduled_converter(airport_zones()),'long')
        months=[];totals={'train':0,'validation':0}
        for y in (2023,2024):
            for m in range(1,13 if y==2023 else 9):
                split='train' if (y,m)<(2024,5) else 'validation'
                name=f'{split}-{y}-{m:02d}'
                group(spark,name);tick=time.monotonic()
                original=baseline.filter((F.col('split')==split)&(F.year('flight_date')==y)&(F.month('flight_date')==m))
                featured=original.withColumn('_prediction_us',convert('flight_date','crs_dep_time','origin_airport_id')).persist(StorageLevel.DISK_ONLY)
                bounds=featured.agg(F.min('_prediction_us').alias('lo'),F.max('_prediction_us').alias('hi')).first()
                # Filter by exact UTC bounds, not guessed local-calendar halos.
                relevant=source.filter(F.col('_prediction_us').between(bounds.lo-3*HOUR,bounds.hi-1)).persist(StorageLevel.DISK_ONLY)
                source_rows=relevant.count()
                for role in ROLES:
                    previous=featured
                    featured=append_role(previous,relevant,role).persist(StorageLevel.DISK_ONLY)
                    featured.count()
                    previous.unpersist()
                path=str(OUT/'features'/name)
                featured.select(*old,*FEATURES).write.mode('errorifexists').parquet(path)
                featured.unpersist();relevant.unpersist()
                output=spark.read.parquet(path)
                # Exact all-column multiset equality subsumes business-key/label multiplicity.
                assert original.exceptAll(output.select(*old)).limit(1).count()==0
                assert output.select(*old).exceptAll(original).limit(1).count()==0
                invalid=F.lit(False);metrics=[F.count('*').alias('rows')]
                for role in ROLES:
                    c=F.col(f'{role}_flight_count_3h');r=F.col(f'{role}_delay_rate_3h');missing=F.col(f'{role}_history_missing_3h')
                    invalid=invalid|c.isNull()|(c<0)|missing.isNull()|(missing!=(c==0).cast('double'))|((c==0)&r.isNotNull())|((c>0)&(r.isNull()|F.isnan(r)|~r.between(0,1)))
                    metrics += [F.sum(missing).alias(role+'_missing'),F.avg(missing).alias(role+'_missing_fraction'),F.max(c).alias(role+'_max_count')]
                metrics += [F.sum(invalid.cast('long')).alias('invalid')]
                stats=output.agg(*metrics).first().asDict()
                assert stats['invalid']==0
                totals[split]+=stats['rows']
                record=dict(month=name,elapsed_seconds=time.monotonic()-tick,source_rows=source_rows,utc_bounds=bounds.asDict(),exact_original_multiset=True,**stats)
                months.append(record);save('months',months)
                save('schema',{'preserved_columns':old,'added':{c:dict(output.dtypes)[c] for c in FEATURES},'original_width':784,'proposed_width':790,'preprocessing_fit':False})
                print(json.dumps(record),flush=True)
        assert totals=={'train':8946370,'validation':2418739}, totals
        source.unpersist()
        save('result',{'status':'PASS','totals':totals,'seconds':time.monotonic()-started,'no_model_fit':True})
    finally:
        spark.stop()

if __name__=='__main__':main()
