"""Self-contained offline LR-3h bundle and opt-in precomputed-row scorer. Never fits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

VERSION='airline-offline-lr3h-v1'
MODEL_ID='offline-lr3h-ca23fe0-v1'
POLICY='research-scheduled-utc-zero-reporting-latency-v1'
KEYS=['flight_date','reporting_airline','flight_number_reporting_airline','origin_airport_id','dest_airport_id','crs_dep_time']
TARGET='is_arrival_delayed_15m'
EXTRA=[f'{r}_{k}_3h' for r in ('origin','destination') for k in ('delay_rate','flight_count','history_missing')]
MEANS={EXTRA[0]:0.08079999358628864,EXTRA[3]:0.0796816299126744}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def inventory(root):
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and p.name not in ('manifest.json','_draft.json')}

def validate_files(root,manifest):
    if manifest.get('offline_only') is not True or manifest.get('operationally_validated') is not False or manifest['version']!=VERSION or manifest['model_id']!=MODEL_ID or manifest['policy']!=POLICY or manifest['threshold']!=0.39 or manifest['feature_order']!=EXTRA or manifest['width']!=790:
        raise ValueError('Incompatible offline contract')
    if inventory(root)!=manifest['files']:raise ValueError('Offline bundle file hashes differ')
    means=json.loads((Path(root)/'means.json').read_text())
    if means!=MEANS or any(not math.isfinite(v) for v in means.values()):raise ValueError('Training means differ')
    return means

def stage(root,source_root,qualified,accepted):
    root=Path(root);source_root=Path(source_root);qualified=Path(qualified);accepted=Path(accepted)
    # Verify required artifacts before creating any candidate bundle.
    for record in json.loads((source_root/'evidence/artifact-hashes.json').read_text()):
        p=source_root/record['path']
        if not p.is_file() or p.stat().st_size!=record['bytes'] or digest(p)!=record['sha256']:raise ValueError(f'Missing/changed candidate artifact: {p}')
    for name,sha in json.loads((qualified/'evidence/preprocessing-sha256.json').read_text()).items():
        relative=Path(name).relative_to('artifacts/flighttime-v3/models/preprocessing')
        if digest(accepted/'preprocessing'/relative)!=sha:raise ValueError('Changed frozen preprocessing')
    contract=json.loads((source_root/'evidence/append-contract.json').read_text())
    if contract['coordinates']!=EXTRA or contract['training_only_rate_means']!=MEANS:raise ValueError('Append contract differs')
    root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source_root/'runtime/output/classifier',root/'classifier')
    shutil.copytree(accepted/'preprocessing',root/'preprocessing')
    shutil.copyfile(__file__,root/'offline.py')
    write(root/'means.json',MEANS)
    evidence=root/'evidence';evidence.mkdir()
    for filename in ('results.json','threshold-selection.json','append-contract.json','artifact-hashes.json','optimizer.json'):
        shutil.copyfile(source_root/'evidence'/filename,evidence/filename)
    for filename in ('schema.json','output-files.json','preprocessing-sha256.json','source.json'):
        shutil.copyfile(qualified/'evidence'/filename,evidence/('qualified-'+filename))
    shutil.copyfile(accepted/'manifest.json',evidence/'parent-manifest.json')
    parent_data=json.loads((accepted/'manifest.json').read_text())
    if parent_data['bundle_id']!='d0098a9572834c53bf76b04d732a94a6' or parent_data['threshold']!=0.39:raise ValueError('Parent identity differs')
    # The frozen assembler's named inputs come from the original dataset schema.
    saved=json.loads((qualified/'evidence/schema.json').read_text())
    metadata=['flight_id','flight_date','flight_number_reporting_airline','origin_airport_id','dest_airport_id','crs_dep_time',TARGET,'split']
    predictors=[c for c in saved['preserved_columns'] if c not in metadata]
    if len(predictors)!=50:raise ValueError('Original predictors differ')
    manifest={'version':VERSION,'model_id':MODEL_ID,'policy':POLICY,'offline_only':True,'original_width':784,'width':790,'threshold':0.39,'feature_order':EXTRA,'required_predictors':predictors,'business_keys':KEYS,'candidate_commit':'ca23fe0','parent_bundle_id':parent_data['bundle_id'],'classifier_settings':contract['settings'],'training_rows':8946370,'training_period':['2023-01-01','2024-04-30'],'availability':'t-3h <= source scheduled departure UTC < t AND reconstructed arrival < t; zero reporting latency assumed, not measured receipt','operationally_validated':False}
    manifest['files']=inventory(root);write(root/'_draft.json',manifest)
    return manifest

def components(root,manifest):
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import LogisticRegressionModel
    root=Path(root);means=validate_files(root,manifest)
    pre=PipelineModel.load(str(root/'preprocessing'));model=LogisticRegressionModel.load(str(root/'classifier'))
    encoder=pre.stages[5]
    width=sum(n+int(encoder.getHandleInvalid()=='keep')-int(encoder.getDropLast()) for n in encoder.categorySizes)+len(pre.stages[-1].getInputCols())-5
    if width!=784 or model.numFeatures!=790 or model.numClasses!=2:raise ValueError('Offline model dimensions differ')
    for k,v in manifest['classifier_settings'].items():
        if model.getOrDefault(k)!=v:raise ValueError(f'Classifier setting differs: {k}')
    return pre,model,means

def seal(root):
    root=Path(root);manifest=json.loads((root/'_draft.json').read_text())
    components(root,manifest)
    write(root/'manifest.json',manifest)  # Completion only after artifact/load validation.
    (root/'_draft.json').unlink()

def load(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    return (*components(root,manifest),manifest)

def prepare(frame,pre,means,manifest):
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import functions as F
    required=set(manifest['required_predictors']+EXTRA+KEYS)
    if not required<=set(frame.columns):raise ValueError('Missing required offline predictors/business keys')
    invalid=F.lit(False)
    for role in ('origin','destination'):
        c=F.col(role+'_flight_count_3h');r=F.col(role+'_delay_rate_3h');m=F.col(role+'_history_missing_3h')
        invalid=invalid|c.isNull()|(c<0)|(c!=F.floor(c))|m.isNull()|(m!=(c==0).cast('double'))|((c==0)&r.isNotNull())|((c>0)&(r.isNull()|F.isnan(r)|~r.between(0,1)))
    if frame.filter(invalid).limit(1).count():raise ValueError('Invalid 3h feature domain')
    old=pre.transform(frame).withColumnRenamed('features','original_features')
    for rate in MEANS:old=old.withColumn(rate+'_imputed',F.coalesce(F.col(rate),F.lit(means[rate])))
    return VectorAssembler(inputCols=['original_features',*[c+'_imputed' if c in MEANS else c for c in EXTRA]],outputCol='features',handleInvalid='error').transform(old)

def decide(frame,threshold):
    from pyspark.sql import functions as F
    return frame.withColumn('prediction',(F.col('probability')>=F.lit(threshold)).cast('int'))

def scores(prepared,model,manifest):
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    metadata=[c for c in ['flight_id',*KEYS,'split',TARGET] if c in prepared.columns]
    output=model.transform(prepared).select(*metadata,vector_to_array('probability')[1].alias('probability'))
    return decide(output,manifest['threshold']).withColumn('model_id',F.lit(manifest['model_id'])).withColumn('feature_policy',F.lit(manifest['policy']))

def session():
    Path('/output/events').mkdir(parents=True,exist_ok=True)
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master('spark://a-master:7077').appName('offline-lr3h-v1')
        .config('spark.driver.host','a-driver').config('spark.driver.bindAddress','0.0.0.0')
        .config('spark.cores.max','2').config('spark.executor.cores','1').config('spark.executor.memory','640m')
        .config('spark.task.cpus','1').config('spark.dynamicAllocation.enabled','false')
        .config('spark.sql.shuffle.partitions','4').config('spark.sql.files.maxPartitionBytes','67108864')
        .config('spark.sql.inMemoryColumnarStorage.batchSize','1000').config('spark.driver.maxResultSize','128m')
        .config('spark.sql.session.timeZone','UTC').config('spark.task.maxFailures','1').config('spark.stage.maxConsecutiveAttempts','1')
        .config('spark.eventLog.enabled','true').config('spark.eventLog.dir','file:///output/events').getOrCreate())

def batch(bundle,inputs,output):
    from pyspark import StorageLevel
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    output=Path(output)
    if output.exists():raise ValueError('Scoring output must be new')
    spark=session();prepared=None
    try:
        spark.sparkContext.setLogLevel('WARN');spark.sparkContext.setJobGroup('fresh-offline-batch','fresh-offline-batch')
        pre,model,means,manifest=load(bundle);frame=spark.read.parquet(*inputs)
        prepared=prepare(frame,pre,means,manifest).persist(StorageLevel.DISK_ONLY)
        if prepared.filter(F.size(vector_to_array('features'))!=790).limit(1).count():raise ValueError('Wrong vector width')
        result=scores(prepared,model,manifest);result.write.mode('errorifexists').parquet(str(output))
        saved=spark.read.parquet(str(output));keys=[c for c in ['flight_id',*KEYS,'split',TARGET] if c in frame.columns]
        if frame.select(*keys).exceptAll(saved.select(*keys)).limit(1).count() or saved.select(*keys).exceptAll(frame.select(*keys)).limit(1).count():raise ValueError('Batch multiplicity changed')
        rows=saved.count()
        invalid=F.col('probability').isNull()|F.isnan('probability')|~F.col('probability').between(0,1)|(F.col('prediction')!=(F.col('probability')>=0.39).cast('int'))
        if saved.filter(invalid).limit(1).count():raise ValueError('Invalid offline predictions')
        write(output/'_completion.json',{'version':VERSION,'model_id':MODEL_ID,'bundle_manifest_sha256':digest(Path(bundle)/'manifest.json'),'rows':rows,'threshold':0.39,'files':inventory(output),'input_files':{str(p):digest(p) for directory in inputs for p in sorted(Path(directory).rglob('*.parquet'))},'offline_only':True})
    finally:
        if prepared is not None:prepared.unpersist()
        spark.stop()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',required=True);parser.add_argument('--input',nargs='+',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();batch(args.bundle,args.input,args.output)
