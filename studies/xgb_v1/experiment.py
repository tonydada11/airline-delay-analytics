"""One fixed XGBoost candidate, with a pre-fit resource gate; no publication."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

OUT = Path('/output')
TARGET = 'is_arrival_delayed_15m'
PARAMS = dict(n_estimators=100, max_depth=4, learning_rate=0.05,
              tree_method='hist', device='cpu', seed=42, num_workers=2,
              features_col='features', label_col=TARGET, weight_col='class_weight')
WIDTH = 784
TRAIN_ROWS = 8946370
VALIDATION_ROWS = 2418739
BASELINE = Path(__file__).resolve().parents[1] / 'lr_v1/evidence/diagnostics/results.json'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False, default=str) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_bound(rows, width, physical, swap):
    dense = rows * width * 4
    return dict(dense_bytes=dense, dense_bytes_per_worker=math.ceil(rows / 2) * width * 4,
                host_physical_bytes=physical, host_swap_bytes=swap,
                blocked=dense > physical + swap)


def session(name):
    from pyspark.sql import SparkSession
    return (SparkSession.builder.appName('xgb-v1-' + name)
        .master('spark://xgb-master:7077').config('spark.driver.host', 'xgb-driver')
        .config('spark.driver.bindAddress', '0.0.0.0').config('spark.cores.max', '2')
        .config('spark.executor.cores', '1').config('spark.executor.memory', '640m')
        .config('spark.task.cpus', '1').config('spark.task.maxFailures', '1')
        .config('spark.stage.maxConsecutiveAttempts', '1')
        .config('spark.dynamicAllocation.enabled', 'false')
        .config('spark.sql.files.maxPartitionBytes', '67108864')
        .config('spark.sql.shuffle.partitions', '4')
        .config('spark.default.parallelism', '2')
        .config('spark.sql.inMemoryColumnarStorage.batchSize', '1000')
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', '1000')
        .config('spark.driver.maxResultSize', '128m').getOrCreate())


def estimator():
    from xgboost.spark import SparkXGBClassifier
    # This API forbids setting objective; it derives binary:logistic from labels.
    return SparkXGBClassifier(**PARAMS)


def environment_probe(_):
    import importlib.metadata as metadata
    import platform
    import socket
    from pyspark import BarrierTaskContext
    context = BarrierTaskContext.get()
    context.barrier()
    yield dict(host=socket.gethostname(), partition=context.partitionId(),
               python=platform.python_version(), packages={p: metadata.version(p) for p in
               ('xgboost-cpu', 'pyarrow', 'numpy', 'pandas', 'scipy', 'scikit-learn',
                'joblib', 'threadpoolctl')})


def load_inputs(spark):
    from pyspark.sql import functions as F
    from airline.ml.model_contract import load_bundle
    from airline.dws.dataset_manifest import validate_dataset_manifest
    from airline.dws.build_ml_flight_features import validate_feature_frame
    from airline.dws.ml_feature_contract import FEATURE_COLUMNS
    from studies.lr_v1.metrics import MONTHS
    baseline_path = BASELINE
    baseline = json.loads(baseline_path.read_text())
    pre, lr, manifest = load_bundle('/accepted-models', sha('/raw-manifest.csv'))
    assert manifest['bundle_id'] == baseline['baseline_bundle_id'] == 'd0098a9572834c53bf76b04d732a94a6'
    assert manifest['threshold'] == 0.39
    assert len(FEATURE_COLUMNS) == 50 and lr.numFeatures == WIDTH
    assert pre.uid == 'PipelineModel_196025fd5a4b'
    features = spark.read.parquet('/selection')
    validate_dataset_manifest('/selection', features, sha('/raw-manifest.csv'), phase='selection')
    assert sha('/selection/_dataset_manifest.json') == manifest['selection']['dataset_manifest_sha256']
    assert validate_feature_frame(features, TRAIN_ROWS + VALIDATION_ROWS) == TRAIN_ROWS + VALIDATION_ROWS
    print('Full selection feature contract passed: 11,365,109 rows', flush=True)
    profiles = [r.asDict() for r in features.groupBy('split').agg(
        F.count('*').alias('rows'), F.sum(TARGET).alias('positives'),
        F.min('flight_date').alias('start'), F.max('flight_date').alias('end')).collect()]
    by_split = {p['split']: p for p in profiles}
    assert set(by_split) == {'train', 'validation'}
    for split, count, start, end in [('train', TRAIN_ROWS, '2023-01-01', '2024-04-30'),
                                   ('validation', VALIDATION_ROWS, '2024-05-01', '2024-08-31')]:
        p = by_split[split]
        assert p['rows'] == count and str(p['start']) == start and str(p['end']) == end
    validation = features.filter("split = 'validation'")
    monthly = {r.month: r.asDict() for r in validation.withColumn(
        'month', F.date_format('flight_date', 'yyyy-MM')).groupBy('month').agg(
            F.count('*').alias('rows'), F.sum(TARGET).alias('positives')).collect()}
    assert set(monthly) == set(MONTHS)
    for month, p in monthly.items():
        b = baseline['validation_months'][month]
        assert p['rows'] == b['rows']
        assert p['positives'] == b['confusion_matrix']['true_positive'] + b['confusion_matrix']['false_negative']
    assert sum(baseline['validation_months'][m]['pr_auc'] for m in MONTHS) / 4 == 0.4280983226531411
    identity = dict(preprocessing_uid=pre.uid, bundle_id=manifest['bundle_id'],
                    baseline_sha256=sha(baseline_path), feature_count=len(FEATURE_COLUMNS),
                    feature_order=list(FEATURE_COLUMNS), dimensions=WIDTH,
                    dataset_manifest_sha256=sha('/selection/_dataset_manifest.json'),
                    populations=profiles, validation_months=monthly)
    write(OUT / 'population.json', identity)
    print('Exact training/validation populations and monthly positive counts passed', flush=True)
    return pre, features.filter("split = 'train'"), validation, baseline, manifest


def metric_checks(spark):
    from studies.lr_v1.diagnose import budget_metrics
    from studies.lr_v1.metrics import expected_budget
    from airline.ml.metrics import calculate_metrics
    from airline.ml.experiments import best_threshold, threshold_counts
    fixture = [(0.9, 'same', 1), (0.9, 'same', 0), *[(0.1, str(i), 0) for i in range(8)]]
    frame = spark.createDataFrame(fixture, ['probability', 'flight_id', TARGET])
    actual, _ = budget_metrics(frame, 10, 1)
    for p in (10, 20):
        for k, value in expected_budget(fixture, p).items():
            assert abs(actual[str(p)][k] - value) < 1e-12
    metrics = calculate_metrics(frame, .9)
    assert metrics['precision'] == .5 and metrics['recall'] == 1
    assert abs(metrics['roc_auc'] - 17 / 18) < 1e-12
    assert abs(metrics['pr_auc'] - .5) < 1e-12
    grid = threshold_counts(frame)
    assert len(grid) == 101 and best_threshold(grid)['step'] == 11
    return dict(duplicate_boundary='passed', unweighted_auc='passed', inclusive_threshold_grid='passed')


def smoke(spark, pre):
    import numpy as np
    from pyspark.sql import functions as F
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.functions import vector_to_array
    from xgboost.spark import SparkXGBClassifierModel
    from xgboost.spark.data import alias
    from airline.dws.ml_feature_contract import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, HISTORY_COUNTS, HISTORY_MISSING
    raw = spark.range(256).withColumn(TARGET, (F.col('id') % 4 == 0).cast('int'))
    for i, name in enumerate(CATEGORICAL_COLUMNS):
        raw = raw.withColumn(name, F.lit(pre.stages[i].labels[0]))
    for name in NUMERIC_COLUMNS:
        raw = raw.withColumn(name, F.lit(None).cast('double'))
    for name in HISTORY_COUNTS:
        raw = raw.withColumn(name, F.lit(0.0))
    for name in HISTORY_MISSING:
        raw = raw.withColumn(name, F.lit(1.0))
    prepared = pre.transform(raw).withColumn('class_weight', 1.0 + F.col(TARGET)).select(
        'id', 'features', TARGET, 'class_weight').repartition(2).cache()
    vectors = prepared.select('features').collect()
    assert len(vectors) == 256 and all(r.features.size == WIDTH and np.isfinite(r.features.toArray()).all() for r in vectors)
    est = estimator()
    assert math.isnan(est.getOrDefault('missing')) and not est.getOrDefault('enable_sparse_data_optim')
    # Inspect the actual adapter's conversion: implicit zeros stay real zeros;
    # NaN remains missing, labels and sample weights retain their alignment.
    probes = spark.createDataFrame([
        (Vectors.sparse(WIDTH, {}), 0, 1.),
        (Vectors.dense([0.] * WIDTH), 0, 1.),
        (Vectors.sparse(WIDTH, {0: float('nan')}), 1, 2.)], ['features', TARGET, 'class_weight'])
    cols, _ = est._prepare_input_columns_and_feature_prop(probes)
    converted = probes.select(*cols).collect()
    assert converted[0][alias.data] == converted[1][alias.data] == [0.] * WIDTH
    assert math.isnan(converted[2][alias.data][0])
    assert [(r[alias.label], r[alias.weight]) for r in converted] == [(0, 1.), (0, 1.), (1, 2.)]
    reused = (OUT / 'smoke-fit-started').exists()
    if reused:
        # Continue this study without repeating its already successful smoke fit.
        prior = OUT / 'prior/original-output'
        recorded = json.loads((prior / 'environment.json').read_text())
        current = json.loads((OUT / 'environment.json').read_text())
        assert all(recorded[k] == current[k] for k in ('packages', 'python', 'spark'))
        for name in ('installed-core.py.txt', 'installed-data.py.txt'):
            assert sha(prior / name) == sha(OUT / name)
        model = SparkXGBClassifierModel.load(str(OUT / 'smoke-model'))
        config = json.loads((prior / 'smoke-booster-config.json').read_text())
        fit_seconds = json.loads((prior / 'preflight.json').read_text())['smoke']['fit_seconds']
    else:
        with (OUT / 'smoke-fit-started').open('x') as f:
            f.write('Exactly one generated-data fit; no retries.\n')
        started = time.monotonic()
        model = est.fit(prepared)
        fit_seconds = time.monotonic() - started
        config = json.loads(model.get_booster().save_config())
    learner = config['learner']
    assert learner['objective']['name'] == 'binary:logistic'
    assert learner['generic_param']['seed'] == '42'
    assert model.get_booster().num_boosted_rounds() == 100
    saved_path = OUT / ('smoke-reload-check' if reused else 'smoke-model')
    model.write().save(str(saved_path))
    loaded = SparkXGBClassifierModel.load(str(saved_path))
    for key in ('device', 'learning_rate', 'max_depth', 'n_estimators', 'num_workers', 'tree_method'):
        assert loaded.getOrDefault(key) == PARAMS[key]
    assert loaded.getOrDefault('arbitrary_params_dict') == {'seed': 42}
    assert loaded.get_booster().num_features() == WIDTH
    assert loaded.getOrDefault('labelCol') == TARGET and loaded.getOrDefault('weightCol') == 'class_weight'
    def scores(m, frame):
        return m.transform(frame).select('id', TARGET, vector_to_array('probability').alias('p')).orderBy('id').collect()
    before = scores(model, prepared)
    after = scores(loaded, prepared)
    assert before == after and len(after) == 256
    # Constant features: weighted positive fraction = (64*2)/(64*2+192) = .4.
    assert all(abs(r.p[1] - .4) < 1e-5 for r in after)
    dense = spark.createDataFrame([(r.id, Vectors.dense(r.features.toArray()), r[TARGET], r.class_weight)
                                  for r in prepared.collect()], prepared.schema)
    assert scores(loaded, dense) == after
    prepared.unpersist()
    write(OUT / 'smoke-booster-config.json', config)
    (OUT / 'estimator-parameters.txt').write_text(est.explainParams() + '\n')
    return dict(rows=256, fit_seconds=fit_seconds, reused_existing_fit=reused, trees=100, dimensions=WIDTH,
                weight_expected_probability=.4, observed_probability=after[0].p[1],
                exact_reload_and_dense_sparse_inference='passed', adapter_zero_nan_label_weights='passed')


def preflight(spark):
    import importlib.metadata as metadata
    import platform
    import xgboost.spark.core as core
    import xgboost.spark.data as data
    workers = spark.sparkContext.parallelize([0, 1], 2).barrier().mapPartitions(environment_probe).collect()
    assert len({r['host'] for r in workers}) == 2
    packages = {p: metadata.version(p) for p in workers[0]['packages']}
    assert all(r['packages'] == packages and r['python'] == platform.python_version() for r in workers)
    assert spark.version == '3.5.9' and platform.python_version() == '3.10.12'
    write(OUT / 'environment.json', dict(spark=spark.version, python=platform.python_version(),
          packages=packages, workers=workers, spark_conf=dict(spark.sparkContext.getConf().getAll())))
    for name, module in [('core', core), ('data', data)]:
        (OUT / ('installed-' + name + '.py.txt')).write_text(inspect.getsource(module))
    pre, _, _, _, _ = load_inputs(spark)
    checks = metric_checks(spark)
    print('Synthetic metric checks passed; verifying smoke model (reuse if already fitted)', flush=True)
    smoke_result = smoke(spark, pre)
    mem = {k: int(v.strip().split()[0]) * 1024 for k, v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    bound = memory_bound(TRAIN_ROWS, WIDTH, mem['MemTotal'], mem['SwapTotal'])
    bound.update(host_available_bytes=mem['MemAvailable'], swap_free_bytes=mem['SwapFree'],
                 explanation='Installed adapter densifies float32 vectors and caches all batches before QuantileDMatrix; bound excludes JVM, Arrow, labels, weights and XGBoost allocations.')
    result = dict(status='blocked' if bound['blocked'] else 'ready', memory=bound,
                  metrics_checks=checks, smoke=smoke_result, full_classifier_fits_started=0)
    write(OUT / 'preflight.json', result)
    return result


def full(spark):
    from pyspark import StorageLevel
    from pyspark.sql import functions as F
    from xgboost.spark import SparkXGBClassifierModel
    from airline.ml.experiments import continuous_score, threshold_counts, best_threshold
    from airline.ml.metrics import calculate_metrics
    from airline.ml.model_contract import PREDICTION_COLUMNS
    from airline.ml.evaluation_artifacts import validate_predictions
    from airline.common.lineage import BUSINESS_KEY
    from studies.lr_v1.diagnose import budget_metrics
    from studies.lr_v1.metrics import MONTHS, monthly_comparison
    assert json.loads((OUT / 'preflight.json').read_text())['status'] == 'ready', 'Resource gate blocks full fit'
    destination = OUT / 'full'
    destination.mkdir(exist_ok=False)
    pre, training, validation, baseline, manifest = load_inputs(spark)
    prepared = pre.transform(training).withColumn('class_weight', 1.0 + F.col(TARGET)).select('features', TARGET, 'class_weight')
    (destination / 'fit-started').write_text('One full fit; never retry.\n')
    started = time.monotonic()
    model = estimator().fit(prepared)
    fit_seconds = time.monotonic() - started
    model.write().save(str(destination / 'classifier'))
    reloaded = SparkXGBClassifierModel.load(str(destination / 'classifier'))
    assert reloaded.get_booster().num_features() == WIDTH
    assert reloaded.get_booster().num_boosted_rounds() == 100
    for key in ('device', 'learning_rate', 'max_depth', 'n_estimators', 'num_workers', 'tree_method'):
        assert reloaded.getOrDefault(key) == PARAMS[key]
    assert reloaded.getOrDefault('arbitrary_params_dict') == {'seed': 42}
    assert json.loads(model.get_booster().save_config())['learner']['objective']['name'] == 'binary:logistic'
    write(destination / 'booster-config.json', json.loads(model.get_booster().save_config()))
    vp = pre.transform(validation).select('flight_id', *BUSINESS_KEY, 'split', TARGET, 'features')
    columns = [c for c in PREDICTION_COLUMNS if c != 'prediction']
    saved_path = str(destination / 'validation-scores')
    continuous_score(reloaded, vp).select(*columns).write.parquet(saved_path)
    saved = spark.read.parquet(saved_path)
    original = continuous_score(model, vp).select(*columns)
    assert not original.exceptAll(saved).limit(1).count() and not saved.exceptAll(original).limit(1).count()
    grid = threshold_counts(saved)
    selected = best_threshold(grid)
    threshold = selected['step'] / 100
    scored = saved.withColumn('prediction', (F.col('probability') >= threshold).cast('int'))
    assert validate_predictions(scored, validation, threshold) == VALIDATION_ROWS
    scored.write.parquet(str(destination / 'validation-predictions'))
    months = {}
    for month in MONTHS:
        frame = scored.filter(F.date_format('flight_date', 'yyyy-MM') == month).persist(StorageLevel.DISK_ONLY)
        try:
            m = calculate_metrics(frame, threshold)
            positives = m['confusion_matrix']['true_positive'] + m['confusion_matrix']['false_negative']
            b = baseline['validation_months'][month]
            assert m['rows'] == b['rows'] and positives == b['confusion_matrix']['true_positive'] + b['confusion_matrix']['false_negative']
            m['budgets'], m['duplicates'] = budget_metrics(frame, m['rows'], positives)
            months[month] = m
        finally:
            frame.unpersist()
    result = dict(status='complete', months=months, pooled=calculate_metrics(scored, threshold),
                  threshold=threshold, threshold_grid=grid, fit_seconds=fit_seconds,
                  evaluation_seconds=time.monotonic() - started - fit_seconds,
                  comparison=monthly_comparison(baseline['validation_months'], months),
                  baseline_months=baseline['validation_months'],
                  baseline_pooled=manifest['selection']['validation_results']['lr']['validation'])
    write(destination / 'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['preflight', 'full'])
    phase = parser.parse_args().phase
    spark = session(phase)
    spark.sparkContext.setLogLevel('WARN')
    try:
        print(json.dumps((preflight if phase == 'preflight' else full)(spark), default=str), flush=True)
    finally:
        spark.stop()
