"""Generated fixtures only. Accepted preprocessing is loaded read-only, never fitted."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import time

import numpy as np
import psutil
from pyspark import BarrierTaskContext
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector
from pyspark.sql import SparkSession, functions as F, types as T
import xgboost as xgb
from xgboost.spark import SparkXGBClassifier, SparkXGBClassifierModel

from airline.dws.ml_feature_contract import (
    CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, HISTORY_COUNTS, HISTORY_MISSING,
    FEATURE_COLUMNS, TARGET_COLUMN,
)

OUT = Path('/output')
WIDTH = 784
MAX_ROWS = 32768
RESULT = {'status':'running', 'generated_data_only':True, 'fits':[], 'checks':{}}


def write(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def check(name, condition, evidence):
    RESULT['checks'][name] = {'passed':bool(condition), 'evidence':evidence}
    write(OUT/'results.json',RESULT)
    assert condition, name


class MemoryCallback(xgb.callback.TrainingCallback):
    def __init__(self, case):
        self.case = case

    def after_iteration(self, model, epoch, evals_log):
        rank = xgb.collective.get_rank()
        record = {'case':self.case,'rank':rank,'world_size':xgb.collective.get_world_size(),
                  'host':socket.gethostname(),'pid':os.getpid(),'iteration':epoch,
                  'rss_bytes':psutil.Process().memory_info().rss,
                  'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        write(OUT/f'memory-{self.case}-{rank}.json',record)
        return False


def estimator(case, **overrides):
    params = dict(features_col='features', label_col=TARGET_COLUMN, weight_col='class_weight',
                  num_workers=2, device='cpu', tree_method='hist', n_estimators=3,
                  max_depth=2, max_bin=32, learning_rate=0.3, subsample=1.0,
                  colsample_bytree=1.0, reg_lambda=1.0, min_child_weight=1.0,
                  random_state=42, missing=float('nan'), enable_sparse_data_optim=False,
                  force_repartition=True, repartition_random_shuffle=False,
                  base_score=0.5, callbacks=[MemoryCallback(case)])
    params.update(overrides)
    return SparkXGBClassifier(**params)


def fit(case, frame, **overrides):
    rows = frame.count()
    assert 0 < rows <= MAX_ROWS
    started = time.monotonic()
    model = estimator(case,**overrides).fit(frame)
    workers = [json.loads((OUT/f'memory-{case}-{rank}.json').read_text()) for rank in (0,1)]
    check(case+'_distributed', {x['rank'] for x in workers}=={0,1}
          and len({x['host'] for x in workers})==2
          and all(x['world_size']==2 for x in workers), workers)
    RESULT['fits'].append({'case':case,'rows':rows,'seconds':time.monotonic()-started,
                          'features':model.get_booster().num_features(),'workers':workers})
    write(OUT/'results.json',RESULT)
    return model


def scores(model, frame):
    return model.transform(frame).select('id',TARGET_COLUMN,'prediction',
        vector_to_array('probability').alias('p'),vector_to_array('rawPrediction').alias('margin')).orderBy('id').collect()


def generated_prepared(spark, preprocessing):
    fields = [T.StructField('id',T.LongType(),False)]
    fields += [T.StructField(c,T.StringType(),True) for c in CATEGORICAL_COLUMNS]
    fields += [T.StructField(c,T.DoubleType(),True) for c in NUMERIC_COLUMNS]
    fields += [T.StructField(c,T.LongType(),False) for c in (*HISTORY_COUNTS,*HISTORY_MISSING)]
    fields += [T.StructField(TARGET_COLUMN,T.DoubleType(),False),T.StructField('class_weight',T.DoubleType(),False)]
    rows = []
    for i in range(512):
        categories = [stage.labels[i%len(stage.labels)] if i%17 else '__UNSEEN_FIXTURE__'
                      for stage in preprocessing.stages[:5]]
        numeric = [None if i%13==0 else float(((i+3*j)%101)-50)/10 for j in range(len(NUMERIC_COLUMNS))]
        label = float(i%2)
        rows.append(tuple([i,*categories,*numeric,*[i%7 for _ in HISTORY_COUNTS],
                           *[int(i%13==0) for _ in HISTORY_MISSING],label,1.0+label]))
    rows += [rows[7],rows[7]]  # Deliberate duplicate IDs and identical payloads.
    raw = spark.createDataFrame(rows,T.StructType(fields))
    return preprocessing.transform(raw).select('id','features',TARGET_COLUMN,'class_weight')


def vector_frame(spark, rows):
    schema = T.StructType([T.StructField('id',T.LongType(),False),
        T.StructField('features',VectorUDT(),False),T.StructField(TARGET_COLUMN,T.DoubleType(),False),
        T.StructField('class_weight',T.DoubleType(),False)])
    return spark.createDataFrame(rows,schema)


def main():
    (OUT/'events').mkdir(exist_ok=True)
    spark = (SparkSession.builder.appName('isolated-xgb-generated-feasibility')
        .master('spark://xgb-master:7077').config('spark.driver.host','xgb-driver')
        .config('spark.driver.bindAddress','0.0.0.0').config('spark.cores.max','2')
        .config('spark.executor.cores','1').config('spark.executor.memory','640m')
        .config('spark.task.cpus','1').config('spark.dynamicAllocation.enabled','false')
        .config('spark.sql.shuffle.partitions','4').config('spark.default.parallelism','2')
        .config('spark.sql.execution.arrow.maxRecordsPerBatch','1000')
        .config('spark.driver.maxResultSize','128m').config('spark.eventLog.enabled','true')
        .config('spark.eventLog.dir','file:///output/events')
        .config('spark.scheduler.barrier.maxConcurrentTasksCheck.interval','5s')
        .config('spark.scheduler.barrier.maxConcurrentTasksCheck.maxFailures','3')
        .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    try:
        packages = {x:importlib.metadata.version(x) for x in
                    ['xgboost-cpu','numpy','pandas','scipy','scikit-learn','pyarrow','joblib','threadpoolctl']}
        RESULT['runtime'] = {'spark':spark.version,'python':platform.python_version(),
            'architecture':platform.machine(),'packages':packages,'spark_conf':dict(spark.sparkContext.getConf().getAll())}
        check('exact_runtime',spark.version=='3.5.9' and platform.python_version()=='3.10.12'
              and xgb.__version__=='3.0.5',RESULT['runtime'])
        preprocessing = PipelineModel.load('/accepted-preprocessing')
        prepared = generated_prepared(spark,preprocessing).cache()
        vectors = [r.features for r in prepared.select('features').collect()]
        sizes = list(preprocessing.stages[5].categorySizes)
        check('accepted_preprocessing_784',preprocessing.uid=='PipelineModel_196025fd5a4b'
              and len(FEATURE_COLUMNS)==50 and all(v.size==WIDTH and np.isfinite(v.toArray()).all() for v in vectors),
              {'uid':preprocessing.uid,'category_sizes':sizes,'dimensions':sorted({v.size for v in vectors}),
               'sparse_rows':sum(isinstance(v,SparseVector) for v in vectors),'rows':len(vectors),
               'mean_nonzero':float(np.mean([v.numNonzeros() for v in vectors]))})
        sparse_model = fit('accepted_sparse',prepared)
        dense = vector_frame(spark,[(r.id,Vectors.dense(r.features.toArray()),r[TARGET_COLUMN],r.class_weight)
                                   for r in prepared.collect()]).cache()
        dense_model = fit('accepted_dense',dense)
        a,b = scores(sparse_model,prepared),scores(dense_model,dense)
        check('dense_sparse_training_parity',len(a)==len(b)==514
              and max(abs(x.p[1]-y.p[1]) for x,y in zip(a,b))<1e-6,{'rows':len(a),'max_probability_delta':max(abs(x.p[1]-y.p[1]) for x,y in zip(a,b))})
        check('labels_predictions_multiplicity',sum(r.id==7 for r in a)==3 and
              all(r[TARGET_COLUMN] in (0.,1.) and r.prediction in (0.,1.) and len(r.p)==2
                  and abs(sum(r.p)-1)<1e-6 and all(0<=p<=1 for p in r.p)
                  and r.prediction==float(r.p[1]>0.5) for r in a),{'rows':len(a),'duplicate_id_7_count':sum(r.id==7 for r in a)})
        path = str(OUT/'saved-model')
        sparse_model.write().save(path)
        loaded = SparkXGBClassifierModel.load(path)
        reloaded = scores(loaded,prepared)
        check('save_reload_exact',a==reloaded and loaded.get_booster().num_features()==WIDTH
              and loaded.getOrDefault(loaded.weightCol)=='class_weight'
              and loaded.getOrDefault(loaded.labelCol)==TARGET_COLUMN,{'prediction_rows':len(a),'model_features':loaded.get_booster().num_features()})
        prepared.unpersist(); dense.unpersist()

        # In Spark, inactive sparse entries mean zero, not an unavailable value.
        rows = [(i,Vectors.sparse(WIDTH,{} if i%3==0 else {0:1.0 if i%3==1 else float('nan')}),
                 float(i%3!=0),1.) for i in range(192)]
        semantics = vector_frame(spark,rows)
        normal = fit('zero_nan',semantics,max_depth=1)
        normal_scores = scores(normal,semantics.limit(3))
        p = [x.p[1] for x in normal_scores]
        check('zero_distinct_from_nan',p[2]-p[0]>0.2,{'zero_one_nan_probabilities':p})
        explicit = vector_frame(spark,[(i,Vectors.sparse(WIDTH,{0:0.0}) if i==0 else Vectors.dense(np.zeros(WIDTH)),0.,1.) for i in range(2)])
        explicit_scores = scores(normal,explicit)
        check('implicit_explicit_dense_zero_parity',all(abs(x.p[1]-p[0])<1e-7 for x in explicit_scores),{'probabilities':[x.p[1] for x in explicit_scores]})
        error = None
        try:
            estimator('invalid_sparse',enable_sparse_data_optim=True).fit(semantics)
        except ValueError as exc:
            error = str(exc)
        check('sparse_optimization_rejects_nan_missing',error is not None and 'missing param != 0' in error,{'error':error})
        unsafe = fit('sparse_missing_zero',semantics,max_depth=1,enable_sparse_data_optim=True,missing=0.0)
        u = [x.p[1] for x in scores(unsafe,semantics.limit(3))]
        check('sparse_optimization_changes_contract',abs(u[0]-u[2])<1e-7 and abs(u[0]-p[0])>0.1,{'zero_one_nan_probabilities':u})

        constant = vector_frame(spark,[(i,Vectors.sparse(WIDTH,{}),float(i%4==0),2. if i%4==0 else 1.) for i in range(256)])
        weighted = fit('weighted',constant,n_estimators=1,learning_rate=1.,reg_lambda=0.,min_child_weight=0.,max_depth=1)
        unweighted = fit('unweighted',constant.withColumn('class_weight',F.lit(1.)),n_estimators=1,learning_rate=1.,reg_lambda=0.,min_child_weight=0.,max_depth=1)
        wp = scores(weighted,constant.limit(1))[0].p[1]
        up = scores(unweighted,constant.limit(1))[0].p[1]
        check('class_weight_newton_step',abs(wp-1/(1+math.exp(0.4)))<1e-6
              and abs(up-1/(1+math.exp(1)))<1e-6,{'weighted_p':wp,'unweighted_p':up,'expected_weighted_margin':-0.4,'expected_unweighted_margin':-1.0})

        # Fixed sizes chosen before execution; these are scaling fixtures, not tuning.
        for size in (4096,32768):
            def generate(indices):
                for i in indices:
                    values = {j:float(((i+7*j)%97)+1)/97 for j in range(0,WIDTH,17)}
                    yield (i,Vectors.sparse(WIDTH,values),float(i%2),float(1+i%2))
            generated = spark.createDataFrame(spark.sparkContext.parallelize(range(size),2).mapPartitions(generate),constant.schema)
            fit('scaling_'+str(size),generated,n_estimators=2)
        RESULT['status']='PASS_FIXTURES_ONLY'
        RESULT['fit_count']=len(RESULT['fits'])
        write(OUT/'results.json',RESULT)
    except BaseException as exc:
        RESULT['status']='FAILED'
        RESULT['error']=repr(exc)
        write(OUT/'results.json',RESULT)
        raise
    finally:
        spark.stop()


if __name__=='__main__':
    main()
