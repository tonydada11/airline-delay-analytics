"""One 790-coordinate full-population LR; no baseline scoring or publication."""
import json, math, time
from pathlib import Path
from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import LogisticRegressionModel
from airline.ml.model_contract import load_bundle, build_logistic_regression, PREDICTION_COLUMNS, MODEL_SETTINGS
from airline.ml.experiments import continuous_score, threshold_counts, best_threshold
from airline.ml.evaluation_artifacts import validate_predictions
from airline.ml.metrics import calculate_metrics
from airline.common.lineage import BUSINESS_KEY
from studies.lr_v1.diagnose import budget_metrics
from studies.lr_v1.metrics import MONTHS, monthly_comparison

OUT=Path('/output');TARGET='is_arrival_delayed_15m'
SCHEMA=json.loads(Path('/prior/evidence/schema.json').read_text())
FEATURES=list(SCHEMA['added']);RATES=[FEATURES[0],FEATURES[3]]
def save(name,value):
    (OUT/(name+'.json')).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')
def phase(spark,name):
    spark.sparkContext.setJobGroup(name,name);save('phase',{'name':name,'epoch':time.time()});print(name,flush=True)

def append(frame,means):
    old=frame.withColumnRenamed('features','original_features')
    for rate in RATES:old=old.withColumn(rate+'_imputed',F.coalesce(F.col(rate),F.lit(means[rate])))
    coordinates=[c+'_imputed' if c in RATES else c for c in FEATURES]
    return VectorAssembler(inputCols=['original_features',*coordinates],outputCol='features',handleInvalid='error').transform(old)

def session():
    return (SparkSession.builder.master('spark://a-master:7077').appName('isolated-lr-3h-v2')
        .config('spark.driver.host','a-driver').config('spark.driver.bindAddress','0.0.0.0')
        .config('spark.cores.max','2').config('spark.executor.cores','1').config('spark.executor.memory','640m')
        .config('spark.task.cpus','1').config('spark.dynamicAllocation.enabled','false')
        .config('spark.sql.shuffle.partitions','4').config('spark.sql.files.maxPartitionBytes','67108864')
        .config('spark.sql.inMemoryColumnarStorage.batchSize','1000')
        .config('spark.driver.maxResultSize','128m').config('spark.sql.session.timeZone','UTC')
        .config('spark.task.maxFailures','1').config('spark.stage.maxConsecutiveAttempts','1')
        .config('spark.eventLog.enabled','true').config('spark.eventLog.dir','file:///output/events').getOrCreate())

def main():
    spark=session()
    spark.sparkContext.setLogLevel('WARN');cached=[];started=time.monotonic()
    try:
        save('runtime',dict(spark.sparkContext.getConf().getAll()))
        phase(spark,'verify_population_schema_labels_multiplicity')
        paths=[str(Path('/prior')/r['path']) for r in json.loads(Path('/prior/evidence/output-files.json').read_text())['files']]
        data=spark.read.parquet(*paths)
        assert data.columns==SCHEMA['preserved_columns']+FEATURES
        assert {c:dict(data.dtypes)[c] for c in FEATURES}==SCHEMA['added']
        original=spark.read.parquet('/selection');assert set(original.columns)==set(SCHEMA['preserved_columns'])
        assert not data.filter(F.col('split').isNull()|~F.col('split').isin('train','validation')|F.col('flight_date').isNull()|~F.col('flight_date').between('2023-01-01','2024-08-31')|F.col(TARGET).isNull()|~F.col(TARGET).isin(0,1)).limit(1).count()
        profiles=[]
        for record in json.loads(Path('/prior/evidence/months.json').read_text()):
            name=record['month'];split,year,month=name.split('-')
            predicate=(F.col('split')==split)&(F.year('flight_date')==int(year))&(F.month('flight_date')==int(month))
            current=data.filter(predicate);old=original.filter(predicate)
            assert current.count()==record['rows']
            assert not old.exceptAll(current.select(*original.columns)).limit(1).count()
            assert not current.select(*original.columns).exceptAll(old).limit(1).count()
            invalid=F.lit(False)
            for role in ('origin','destination'):
                c=F.col(role+'_flight_count_3h');r=F.col(role+'_delay_rate_3h');m=F.col(role+'_history_missing_3h')
                invalid=invalid|c.isNull()|(c<0)|m.isNull()|(m!=(c==0).cast('double'))|((c==0)&r.isNotNull())|((c>0)&(r.isNull()|F.isnan(r)|~r.between(0,1)))
            assert not current.filter(invalid).limit(1).count()
            profiles.append({'month':name,'rows':record['rows'],'exact_original_multiset':True});save('population',profiles)
        counts={r['split']:r['count'] for r in data.groupBy('split').count().collect()}
        assert counts=={'train':8946370,'validation':2418739}
        preprocessing,accepted,manifest=load_bundle('/accepted')
        assert manifest['bundle_id']=='d0098a9572834c53bf76b04d732a94a6' and manifest['threshold_step']==39
        assert accepted.numFeatures==784
        estimator=build_logistic_regression()
        for param in estimator.params:
            if estimator.isDefined(param) and accepted.hasParam(param.name):
                assert estimator.getOrDefault(param)==accepted.getOrDefault(param.name),param.name
        phase(spark,'training_only_new_rate_means')
        training=data.filter("split='train'");validation=data.filter("split='validation'")
        means=training.agg(*[F.avg(c).alias(c) for c in RATES]).first().asDict()
        assert all(v is not None and math.isfinite(v) for v in means.values())
        save('append-contract',{'original_width':784,'coordinates':FEATURES,'width':790,'training_only_rate_means':means,'settings':MODEL_SETTINGS,'positive_weight':2.0,'negative_weight':1.0,'baseline_bundle':manifest['bundle_id']})
        def prepare(frame):
            result=append(preprocessing.transform(frame),means)
            # Verify every retained coordinate before fit, without rounding/casting.
            mismatch=(F.size(vector_to_array('original_features'))!=784)|(F.size(vector_to_array('features'))!=790)|(F.slice(vector_to_array('features'),1,784)!=vector_to_array('original_features'))
            assert not result.filter(mismatch).limit(1).count()
            return result.drop('original_features')
        phase(spark,'frozen_preprocessing_and_prefix_verification')
        train=prepare(training).withColumn('class_weight',F.when(F.col(TARGET)==1,2.0).otherwise(1.0)).select('features',TARGET,'class_weight').repartition(36).persist(StorageLevel.DISK_ONLY);cached.append(train)
        assert train.count()==8946370
        valid=prepare(validation).select('flight_id',*BUSINESS_KEY,'split',TARGET,'features').persist(StorageLevel.DISK_ONLY);cached.append(valid)
        assert valid.count()==2418739
        phase(spark,'single_classifier_fit');save('fit-started',{'classifier_fits_started':1})
        tick=time.monotonic();model=estimator.fit(train);fit_seconds=time.monotonic()-tick
        save('optimizer',{'fit_seconds':fit_seconds,'iterations':model.summary.totalIterations,'objective_history':model.summary.objectiveHistory})
        model.write().save('/output/classifier');loaded=LogisticRegressionModel.load('/output/classifier');assert loaded.numFeatures==790
        for key,value in MODEL_SETTINGS.items():assert loaded.getOrDefault(key)==value
        train.unpersist()
        phase(spark,'validation_score_save_reload_and_threshold_selection')
        def scores(m):return continuous_score(m,valid).select('flight_id',*BUSINESS_KEY,'split',TARGET,'probability')
        scores(loaded).write.parquet('/output/validation-scores');saved=spark.read.parquet('/output/validation-scores')
        assert not scores(model).exceptAll(saved).limit(1).count()
        assert not saved.exceptAll(scores(model)).limit(1).count()
        thresholds=threshold_counts(saved);best=best_threshold(thresholds);threshold=best['step']/100
        save('threshold-selection',{'thresholds':thresholds,'selected':best,'threshold':threshold})
        predicted=saved.withColumn('prediction',(F.col('probability')>=threshold).cast('int')).select(*PREDICTION_COLUMNS)
        predicted.write.parquet('/output/validation-predictions');saved=spark.read.parquet('/output/validation-predictions')
        assert validate_predictions(saved,validation,threshold)==2418739
        valid.unpersist()
        baseline=json.loads(Path('/baseline/results.json').read_text())['validation_months'];months={}
        for month in MONTHS:
            phase(spark,'evaluate-'+month)
            frame=saved.filter(F.date_format('flight_date','yyyy-MM')==month).persist(StorageLevel.DISK_ONLY)
            try:
                metrics=calculate_metrics(frame,threshold)
                positives=metrics['confusion_matrix']['true_positive']+metrics['confusion_matrix']['false_negative']
                assert metrics['rows']==baseline[month]['rows']
                assert positives==baseline[month]['confusion_matrix']['true_positive']+baseline[month]['confusion_matrix']['false_negative']
                budgets,duplicates=budget_metrics(frame,metrics['rows'],positives)
                months[month]={**metrics,'budgets':budgets,'duplicate_score_id_check':duplicates}
                save('validation-'+month,months[month])
            finally:frame.unpersist()
        comparison=monthly_comparison(baseline,months)
        save('results',{'status':'complete','counts':counts,'months':months,'comparison':comparison,'threshold':threshold,'mean_monthly_PR_AUC':sum(v['pr_auc'] for v in months.values())/4,'fit_seconds':fit_seconds,'classifier_fits_started':1,'baseline_scored':False,'preprocessing_refit':False,'seconds':time.monotonic()-started})
    finally:
        for frame in cached:frame.unpersist()
        spark.stop()

if __name__=='__main__':main()
