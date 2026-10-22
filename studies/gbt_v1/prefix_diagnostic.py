"""Read-only GBT30 prefixes. Use launch_prefix.py --execute for the bounded run."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

from studies.gbt_v1 import run_candidate as study

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('data/studies/gbt-improvement-v1/prefix-diagnostic-001')
REFERENCE = Path('studies/gbt_v1/evidence_30iter')
MODEL = Path('data/studies/gbt-improvement-v1/attempt-002/gbt-all-30iter/classifier')
PREFIXES = (10, 20, 25, 30)
LIMIT, GRACE = 3600, 60
SCORE_ATOL, AUC_ATOL = 1e-12, 1e-10
RESULTS_SHA = '6c02d060a8b974e89d59ac210410bb048c74720f5212af76918191162675718e'
CHECKS = {'native_scores', 'saved_monthly_auc', 'population', 'inputs_unchanged'}


def prefix_members(trees, weights, count):
    """Ordered selection only: retain the original, unnormalized fitted weights."""
    study.require(type(count) is int and 0 < count <= len(trees) == len(weights), 'Invalid prefix')
    # PySpark exposes NumPy float64 weights; Python float preserves their bits
    # and is accepted by both Py4J's double-array protocol and compact JSON.
    return list(trees[:count]), [float(w) for w in weights[:count]]


def prefix_model(model, count, spark):
    """Construct a native fitted ensemble; maxIter is NOT a truncation mechanism."""
    from pyspark.ml.classification import GBTClassificationModel
    trees, weights = prefix_members(model.trees, model.treeWeights, count)
    gateway, jvm = spark.sparkContext._gateway, spark.sparkContext._jvm
    jtrees = gateway.new_array(jvm.org.apache.spark.ml.regression.DecisionTreeRegressionModel, count)
    jweights = gateway.new_array(jvm.double, count)
    for index, (tree, weight) in enumerate(zip(trees, weights)):
        jtrees[index], jweights[index] = tree._java_obj, weight
    native = jvm.org.apache.spark.ml.classification.GBTClassificationModel(
        model.uid + '_prefix_' + str(count), jtrees, jweights, model.numFeatures, model.numClasses)
    prefix = model._copyValues(GBTClassificationModel(native))
    study.require(prefix.getNumTrees == count and prefix.numFeatures == model.numFeatures
                  and prefix.numClasses == model.numClasses and list(prefix.treeWeights) == weights
                  and [t.uid for t in prefix.trees] == [t.uid for t in trees], 'Prefix construction differs')
    return prefix


def exclusive_output(root):
    path = Path(root).absolute()
    for component in [path, *path.parents]:
        study.require(not component.is_symlink(), 'Symlink root forbidden')
    for part in OUTPUT.parts:
        path /= part
        study.require(not path.is_symlink(), 'Symlink output forbidden')
    if path.exists():
        raise FileExistsError('Preserve existing diagnostic; no overwrite, resume or retry')
    return path


def inspect(root):
    """Reuse frozen input checks; reconcile local GBT30 records and classifier bytes."""
    from airline.ml.evaluation_artifacts import inventory
    paths, manifest, lr, defaults, identities = study.inspect_inputs(root, ROOT, candidate='gbt-all-30iter')
    attempt = Path(root) / MODEL.parent.parent
    records = {}
    for name in ('completion.json', 'results.json', 'inputs.json', 'worker-status.json'):
        local = attempt / ('completion.json' if name == 'completion.json' else
                           'gbt-all-30iter/' + ('status.json' if name == 'worker-status.json' else name))
        published = ROOT / REFERENCE / name
        study.require(local.is_file() and published.is_file() and study.digest(local) == study.digest(published),
                      'Missing/changed GBT30 record: ' + name)
        records[name] = {'sha256': study.digest(local), 'path': str(local)}
    completion = study.read_json(attempt / 'completion.json')
    result = study.read_json(ROOT / REFERENCE / 'results.json')
    inputs = study.read_json(ROOT / REFERENCE / 'inputs.json')
    state = study.read_json(ROOT / REFERENCE / 'worker-status.json')
    study.require(records['results.json']['sha256'] == RESULTS_SHA == completion['results_sha256']
                  and completion['status'] == 'complete' and completion['exit_code'] == 0
                  and completion['classifier_fits_started'] == 1 and completion['shutdown_successful']
                  and state['stage'] == 'evaluated' and state['shutdown_successful']
                  and all(result['checks'].values()) and set(result['months']) == set(lr)
                  and inputs['effective_parameters'] == defaults, 'GBT30 reference incomplete/incompatible')
    model_inventory = inventory(Path(root) / MODEL)
    study.require(model_inventory == result['artifacts']['classifier'], 'Saved GBT30 classifier inventory differs')
    selection = paths.resolve(paths.dws_ml_flight_features)
    parquet = {p.relative_to(selection).as_posix(): p for p in selection.rglob('*')
               if p.is_file() and (p.suffix == '.parquet' or p.name == '_SUCCESS')}
    study.require(set(parquet) == set(identities['expected_selection_parquet']), 'Selection file set differs')
    # Hash only the validation partition, never the raw warehouse or training data.
    validation_hashes = {name: study.digest(p) for name, p in parquet.items() if name.startswith('split=validation/')}
    study.require(bool(validation_hashes) and all(value == identities['expected_selection_parquet'][name]
                  for name, value in validation_hashes.items()), 'Validation Parquet differs')
    metadata = {p.relative_to(selection).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
                for p in selection.rglob('*') if p.is_file()}
    return paths, manifest, lr, result['months'], inputs, {
        'accepted': identities, 'gbt30_records': records, 'classifier': model_inventory,
        'validation_parquet_sha256': validation_hashes, 'selection_file_metadata': metadata}


def reconcile(months, saved):
    from studies.lr_v1.metrics import MONTHS
    study.require(set(months) == set(saved) == set(MONTHS), 'Missing validation month')
    errors = {m: abs(months[m]['pr_auc']['30'] - saved[m]['pr_auc']) for m in MONTHS}
    study.require(all(math.isfinite(e) and e <= AUC_ATOL for e in errors.values()),
                  '30-tree PR-AUC differs from saved results: ' + repr(errors))
    return errors


def summarize(months, lr):
    from studies.lr_v1.metrics import MONTHS
    study.require(set(months) == set(lr) == set(MONTHS), 'Missing validation month')
    means = {str(n): sum(months[m]['pr_auc'][str(n)] for m in MONTHS) / 4 for n in PREFIXES}
    changes = {}
    for a, b in zip(PREFIXES, PREFIXES[1:]):
        paired = {m: months[m]['pr_auc'][str(b)] - months[m]['pr_auc'][str(a)] for m in MONTHS}
        changes[f'{a}->{b}'] = {'monthly': paired, 'mean': sum(paired.values()) / 4,
                              'worst': min(paired.values())}
    return {'equal_month_means': means, 'changes': changes,
            'lr_monthly_pr_auc': {m: lr[m]['pr_auc'] for m in MONTHS},
            'lr_equal_month_mean': sum(lr[m]['pr_auc'] for m in MONTHS) / 4}


def evaluate(spark, validation, preprocessing, classifier, lr, saved, output):
    from pyspark import StorageLevel
    from pyspark.sql import functions as F
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from airline.common.lineage import BUSINESS_KEY
    from airline.dws.ml_feature_contract import TARGET_COLUMN
    from airline.dws.build_ml_flight_features import validate_feature_frame
    from airline.ml.experiments import continuous_score
    from studies.lr_v1.metrics import MONTHS
    lineage = ['flight_id', *BUSINESS_KEY, 'split', TARGET_COLUMN]
    validate_feature_frame(validation, sum(lr[m]['rows'] for m in MONTHS))
    population = validation.groupBy(F.date_format('flight_date', 'yyyy-MM').alias('month')).agg(
        F.count('*').alias('rows'), F.sum(TARGET_COLUMN).alias('positives')).collect()
    study.require({r.month for r in population} == set(MONTHS), 'Validation dates differ')
    for row in population:
        cm = lr[row.month]['confusion_matrix']
        study.require(row.rows == lr[row.month]['rows'] and row.positives == cm['true_positive'] + cm['false_negative'],
                      'Validation population differs: ' + row.month)
    prepared = scored = None
    try:
        prepared = preprocessing.transform(validation).select(*lineage, 'features').persist(StorageLevel.DISK_ONLY)
        study.same_multiset(validation.select(*lineage), prepared.select(*lineage))
        frame = prepared
        for n in PREFIXES:
            frame = continuous_score(prefix_model(classifier, n, spark), frame).withColumnRenamed(
                'probability', f'probability_{n}').drop('rawPrediction', 'prediction')
        scored = continuous_score(classifier, frame).withColumnRenamed('probability', 'native_probability').select(
            *lineage, *[f'probability_{n}' for n in PREFIXES], 'native_probability').persist(StorageLevel.DISK_ONLY)
        invalid = F.lit(False)
        for name in [*[f'probability_{n}' for n in PREFIXES], 'native_probability']:
            invalid = invalid | F.col(name).isNull() | F.isnan(name) | ~F.col(name).between(0, 1)
        score_check = scored.agg(F.count('*').alias('rows'), F.sum(F.when(invalid, 1).otherwise(0)).alias('invalid'),
            F.max(F.abs(F.col('probability_30') - F.col('native_probability'))).alias('max_absolute_error')).first().asDict()
        study.require(not score_check['invalid'] and score_check['max_absolute_error'] <= SCORE_ATOL,
                      '30-tree/native scores differ or invalid scores: ' + repr(score_check))
        study.same_multiset(validation.select(*lineage), scored.select(*lineage))
        prepared.unpersist()
        prepared = None
        months = {}
        for month in MONTHS:
            frame = scored.filter(F.date_format('flight_date', 'yyyy-MM') == month).persist(StorageLevel.DISK_ONLY)
            try:
                # Exact same unweighted evaluator/settings as airline.ml.metrics.calculate_metrics.
                evaluator = BinaryClassificationEvaluator(labelCol=TARGET_COLUMN, metricName='areaUnderPR', numBins=0)
                months[month] = {'rows': lr[month]['rows'], 'positives': next(r.positives for r in population if r.month == month),
                                 'pr_auc': {str(n): evaluator.setRawPredictionCol(f'probability_{n}').evaluate(frame)
                                            for n in PREFIXES}}
            finally:
                frame.unpersist()
        study.write_json(output / 'monthly-pr-auc.json', months)
        errors = reconcile(months, saved)  # Gate interpretation on the original GBT30 result.
        return {'status': 'pending_termination', 'months': months, 'summary': summarize(months, lr),
                'native_score_check': score_check, 'saved_auc_absolute_errors': errors,
                'score_absolute_tolerance': SCORE_ATOL, 'auc_absolute_tolerance': AUC_ATOL,
                'prefix_tree_weights': {str(n): [float(w) for w in classifier.treeWeights[:n]] for n in PREFIXES},
                'prefix_tree_uids': {str(n): [t.uid for t in classifier.trees[:n]] for n in PREFIXES}}
    finally:
        for frame in (scored, prepared):
            if frame is not None:
                frame.unpersist()


def worker(root, token):
    from pyspark.ml.base import Estimator
    from pyspark.ml.classification import GBTClassificationModel
    from airline.common.spark_session import create_spark_session
    from airline.ml.model_contract import load_bundle
    from pyspark import StorageLevel
    def forbidden_fit(*args, **kwargs):
        raise RuntimeError('Fitting is forbidden in the prefix diagnostic')
    Estimator.fit = forbidden_fit
    output = Path(root) / OUTPUT
    study.require(os.environ.get('AIRLINE_PREFIX_TOKEN') == token == study.read_json(output / 'attempt.json')['token'],
                  'Diagnostic worker requires its supervisor')
    state = {'stage': 'checking_inputs', 'classifier_fits_started': 0, 'shutdown_successful': False}
    spark = validation = None
    def save():
        study.write_json(output / 'worker-status.json', state)
    def interrupted(signum, frame):
        raise RuntimeError('Diagnostic interrupted: ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    save()
    try:
        paths, manifest, lr, saved, inputs, identities = inspect(root)
        study.write_json(output / 'inputs.json', {'identities': identities, 'gbt30_effective_parameters': inputs['effective_parameters'],
            'baseline_bundle_id': manifest['bundle_id'], 'preprocessing_uid': 'PipelineModel_196025fd5a4b',
            'selection_rows_manifest_claim': 11365109})
        spark = create_spark_session('gbt30-prefix-diagnostic')
        state['spark_application_id'] = spark.sparkContext.applicationId
        actual_runtime = {key: spark.sparkContext.getConf().get(key, None) if not key.startswith('spark.sql.')
                          else spark.conf.get(key) for key in study.RUNTIME}
        study.require(spark.version == '3.5.9' and actual_runtime == study.RUNTIME, 'Pinned Spark runtime/resources differ')
        state['runtime'] = actual_runtime
        save()
        preprocessing, accepted_lr, _ = load_bundle(paths.resolve(paths.artifact_root) / 'models', manifest['raw_manifest_sha256'])
        study.require(preprocessing.uid == 'PipelineModel_196025fd5a4b', 'Accepted preprocessing identity differs')
        classifier = GBTClassificationModel.load(str(Path(root) / MODEL))
        study.validate_candidate(classifier, inputs['effective_parameters'], accepted_lr.numFeatures)
        dataset = spark.read.parquet(str(paths.resolve(paths.dws_ml_flight_features)))
        from airline.dws.ml_feature_contract import OUTPUT_COLUMNS
        dataset_manifest = study.read_json(paths.resolve(paths.dws_ml_flight_features) / '_dataset_manifest.json')
        study.require(dataset_manifest['phase'] == 'selection' and dataset_manifest['splits'] == ['train', 'validation']
                      and dataset_manifest['columns'] == [{'name': n, 'type': t} for n, t in dataset.select(*OUTPUT_COLUMNS).dtypes],
                      'Selection manifest/schema differs')
        # Partition pruning: no training or test rows are evaluated or materialized.
        validation = dataset.filter("split = 'validation'").select(*OUTPUT_COLUMNS).persist(StorageLevel.DISK_ONLY)
        state['stage'] = 'scoring_prefixes'
        save()
        result = evaluate(spark, validation, preprocessing, classifier, lr, saved, output)
        study.write_json(output / 'results.json', result)
        study.require(inspect(root)[-1] == identities, 'Inputs changed during diagnostic')
        result['checks'] = {check: True for check in CHECKS}
        study.write_json(output / 'results.json', result)
        state['stage'] = 'evaluated'
    except BaseException as exc:
        state.update(stage='incomplete', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if validation is not None:
                validation.unpersist()
            if spark is not None:
                spark.stop()
            state['shutdown_successful'] = True
        finally:
            save()


def supervise(root, command=None, *, revision, preparation_seconds=0.0, limit=LIMIT, grace=GRACE, cleanup=None):
    """The Spark worker's process group is bounded inside the container."""
    started = time.monotonic()
    study.require(math.isfinite(preparation_seconds) and preparation_seconds >= 0
                  and preparation_seconds < limit - grace, 'Preparation exhausted diagnostic allowance')
    output = exclusive_output(root)
    output.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    record = {'status': 'running', 'token': token, 'source_revision': revision, 'limit_seconds': limit,
              'termination_allowance_seconds': grace, 'preparation_seconds': preparation_seconds,
              'started_at': study.utc(), 'classifier_fits_started': 0, 'automatic_retries': 0,
              'diagnostic_sha256': study.digest(Path(__file__))}
    study.write_json(output / 'attempt.json', record)
    command = command or [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', token,
                          '--project-root', str(root)]
    process = None
    handlers = {}
    def interrupted(signum, frame):
        raise RuntimeError('Diagnostic supervisor interrupted: ' + str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupted)
        with (output / 'worker.log').open('x') as log:
            process = subprocess.Popen(command, env=dict(os.environ, AIRLINE_PREFIX_TOKEN=token,
                PYTHONDONTWRITEBYTECODE='1'), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process.wait(timeout=max(0, limit - grace - preparation_seconds - (time.monotonic() - started)))
        study.require(process.returncode == 0, f'Diagnostic worker exited {process.returncode}')
        state, result = study.read_json(output / 'worker-status.json'), study.read_json(output / 'results.json')
        study.require(state['stage'] == 'evaluated' and state['shutdown_successful']
                      and state['classifier_fits_started'] == 0 and result['status'] == 'pending_termination'
                      and set(result['checks']) == CHECKS and all(result['checks'].values()), 'Diagnostic incomplete')
        reconcile(result['months'], study.read_json(ROOT / REFERENCE / 'results.json')['months'])
        record.update(status='complete', results_sha256=study.digest(output / 'results.json'))
    except BaseException as exc:
        record.update(status='incomplete', error=repr(exc), traceback=traceback.format_exc())
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        if process is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                if sig == signal.SIGTERM:
                    try:
                        process.wait(timeout=max(0, limit - preparation_seconds - (time.monotonic() - started) - 1))
                    except subprocess.TimeoutExpired:
                        record.update(status='incomplete', error='Worker termination timed out')
            process.wait()
            record['exit_code'] = process.returncode
        try:
            if cleanup:
                cleanup()
        except BaseException as exc:
            record.update(status='incomplete', cleanup_error=repr(exc))
        record.update(total_seconds=preparation_seconds + time.monotonic() - started, finished_at=study.utc())
        if record['total_seconds'] >= limit:
            record.update(status='incomplete', error='Diagnostic wall-clock ceiling exceeded')
        study.write_json(output / 'attempt.json', record)
        if record['status'] == 'complete':
            study.write_json(output / 'completion.json', record)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 0 if record['status'] == 'complete' else 1


def main(argv=None, *, cleanup=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--execute', action='store_true')
    action.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--source-revision')
    parser.add_argument('--preparation-seconds', type=float, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.project_root, args.worker)
        return 0
    study.require(args.source_revision is not None, 'Use the bounded launcher')
    return supervise(args.project_root, revision=args.source_revision, cleanup=cleanup,
                     preparation_seconds=args.preparation_seconds + time.monotonic() - started)


if __name__ == '__main__':
    sys.exit(main())
