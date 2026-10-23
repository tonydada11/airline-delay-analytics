"""Saved-result reproduction only. No fit, threshold selection or test-period reads."""
import argparse,json,math,time
from pathlib import Path
from pyspark import StorageLevel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
import offline as api

OUT=Path('/output')
def phase(spark,name):
    spark.sparkContext.setJobGroup(name,name);api.write(OUT/'phase.json',{'name':name,'epoch':time.time()});print(name,flush=True)
def equal(a,b):
    assert not a.exceptAll(b).limit(1).count()
    assert not b.exceptAll(a).limit(1).count()
def features(spark):
    records=json.loads(Path('/prior/evidence/output-files.json').read_text())['files']
    return spark.read.parquet(*[str(Path('/prior')/r['path']) for r in records])
def reconcile():
    from metrics import monthly_comparison,MONTHS
    candidate=json.loads(Path('/candidate/evidence/results.json').read_text())
    baseline=json.loads(Path('/baseline/results.json').read_text())['validation_months']
    comparison=monthly_comparison(baseline,candidate['months'])
    assert comparison==candidate['comparison']
    assert math.isclose(sum(v['pr_auc'] for v in baseline.values())/4,0.4280983226531411,abs_tol=1e-15)
    assert math.isclose(candidate['mean_monthly_PR_AUC'],0.4637348613945793,abs_tol=1e-15)
    assert math.isclose(comparison['mean_monthly_PR_AUC_gain'],0.0356365387414382,abs_tol=1e-15)
    grid=json.loads(Path('/candidate/evidence/threshold-selection.json').read_text())
    assert len(grid['thresholds'])==101 and grid['threshold']==0.39 and grid['selected']['step']==39
    return {'saved_results_reconciled':True,'new_independent_evaluation':False,'threshold_reselected':False,'mean_monthly_PR_AUC':candidate['mean_monthly_PR_AUC'],'baseline_mean_monthly_PR_AUC':sum(v['pr_auc'] for v in baseline.values())/4,'comparison':comparison}

def main(mode):
    spark=api.session();cached=[]
    try:
        spark.sparkContext.setLogLevel('WARN')
        if mode=='final':
            phase(spark,'fresh-process-full-parity')
            final=spark.read.parquet('/output/scored-offline-v1')
            original=spark.read.parquet('/candidate/runtime/output/validation-predictions')
            equal(final.select(*original.columns),original)
            assert final.count()==2418739
            completion=json.loads(Path('/output/scored-offline-v1/_completion.json').read_text());assert completion['threshold']==0.39
            assert final.filter((F.col('model_id')!=api.MODEL_ID)|(F.col('feature_policy')!=api.POLICY)).count()==0
            # Load sealed bundle again, independently of the accepted loader.
            api.load('/output/staged-bundle')
            api.write(OUT/'final-parity.json',{'status':'PASS','rows':2418739,'full_score_prediction_label_multiset_exact':True,'model_id':api.MODEL_ID,'fresh_process':True,'threshold':0.39,'fits':0})
            return
        if mode=='focused':
            phase(spark,'focused-offline-tests')
            root=OUT/'test-bundle';manifest=api.stage(root,'/candidate','/prior','/accepted')
            pre,model,means=api.components(root,manifest)
            frame=spark.read.parquet('/prior/runtime/output/features/validation-2024-05').orderBy('flight_id').limit(16)
            for role in ('origin','destination'):
                frame=frame.withColumn(role+'_delay_rate_3h',F.lit(None).cast('double')).withColumn(role+'_flight_count_3h',F.lit(0).cast('long')).withColumn(role+'_history_missing_3h',F.lit(1.0))
            frame=frame.cache();cached.append(frame);assert frame.count()==16
            prepared=api.prepare(frame,pre,means,manifest)
            for row in prepared.select(vector_to_array('original_features').alias('old'),vector_to_array('features').alias('new'),*api.EXTRA).collect():
                assert len(row.old)==784 and len(row.new)==790 and row.new[:784]==row.old
                assert row.new[784:]==[means[c] if c in api.MEANS else float(row[c]) for c in api.EXTRA]
            no_labels=api.prepare(frame.drop(api.TARGET,'split'),pre,means,manifest)
            keys=['flight_id',*api.KEYS,'probability','prediction']
            equal(api.scores(prepared,model,manifest).select(*keys),api.scores(no_labels,model,manifest).select(*keys))
            duplicated=frame.unionByName(frame.limit(1));assert api.scores(api.prepare(duplicated,pre,means,manifest),model,manifest).count()==17
            decisions=api.decide(spark.createDataFrame([(0.389999,),(0.39,),(0.49999,),(0.5,)],'probability double'),0.39).orderBy('probability').collect()
            assert [r.prediction for r in decisions]==[0,1,1,1]
            from pyspark.ml.classification import LogisticRegressionModel
            model.write().save(str(OUT/'classifier-roundtrip'))
            roundtrip=LogisticRegressionModel.load(str(OUT/'classifier-roundtrip'))
            equal(api.scores(prepared,model,manifest),api.scores(prepared,roundtrip,manifest))
            api.seal(root);pre2,model2,means2,manifest2=api.load(root);assert means2==api.MEANS and manifest2['threshold']==0.39
            equal(api.scores(prepared,model,manifest),api.scores(api.prepare(frame,pre2,means2,manifest2),model2,manifest2))
            # Focused integrity tests never modify source artifacts.
            original=(root/'means.json').read_bytes();api.write(root/'means.json',{**means,api.EXTRA[0]:0.1})
            try:
                api.load(root)
            except ValueError:pass
            else:raise AssertionError('Corrupted means accepted')
            (root/'means.json').write_bytes(original)
            bad=dict(manifest2,threshold=0.5)
            try:api.validate_files(root,bad)
            except ValueError:pass
            else:raise AssertionError('Default threshold accepted')
            bad=dict(manifest2,feature_order=list(reversed(api.EXTRA)))
            try:api.validate_files(root,bad)
            except ValueError:pass
            else:raise AssertionError('Wrong coordinate order accepted')
            api.write(OUT/'focused-tests.json',{'status':'PASS','rows':16,'unlabeled_scoring':True,'duplicate_occurrences':17,'widths':[784,790],'prefix_exact':True,'means_exact':True,'threshold_boundary':True,'reload_scores_exact':True,'classifier_save_reload':True,'tamper_and_order_rejection':True,'fits':0})
            return
        phase(spark,'package-and-validate')
        root=OUT/'staged-bundle';manifest=api.stage(root,'/candidate','/prior','/accepted');pre,model,means=api.components(root,manifest)
        data=features(spark)
        counts={r['split']:r['count'] for r in data.groupBy('split').count().collect()};assert counts=={'train':8946370,'validation':2418739}
        validation=data.filter("split='validation'")
        phase(spark,'full-validation-feature-assembly')
        prepared=api.prepare(validation,pre,means,manifest).persist(StorageLevel.DISK_ONLY);cached.append(prepared)
        old=vector_to_array('original_features');new=vector_to_array('features')
        wrong=(F.size(old)!=784)|(F.size(new)!=790)|(F.slice(new,1,784)!=old)
        # Independent array-concatenation reference reproduces the approved append expression.
        expected=F.concat(old,F.array(*[F.coalesce(F.col(c),F.lit(means[c])) if c in api.MEANS else F.col(c).cast('double') for c in api.EXTRA]))
        assert not prepared.filter(wrong|(new!=expected)).limit(1).count()
        assert prepared.count()==2418739
        phase(spark,'full-saved-prediction-parity')
        api.scores(prepared,model,manifest).write.parquet('/output/verified-scores')
        output=spark.read.parquet('/output/verified-scores');original=spark.read.parquet('/candidate/runtime/output/validation-predictions')
        equal(output.select(*original.columns),original)
        original_scores=spark.read.parquet('/candidate/runtime/output/validation-scores')
        equal(output.select(*original_scores.columns),original_scores)
        keys=[c for c in original.columns if c not in ('probability','prediction')]
        equal(validation.select(*keys),output.select(*keys))
        phase(spark,'legacy-load-and-evidence-reconciliation')
        from airline.ml.model_contract import load_bundle
        oldpre,oldmodel,parent=load_bundle('/accepted');assert oldmodel.numFeatures==784 and parent['threshold']==0.39
        reconciled=reconcile()
        api.seal(root)
        api.write(OUT/'reconciliation.json',reconciled)
        api.write(OUT/'verification.json',{'status':'PASS','counts':counts,'all_validation_vectors':[784,790],'prefix_and_six_values_exact':True,'score_prediction_and_label_multisets_exact':True,'legacy_schema3_loadable':True,'legacy_scored':False,'threshold':0.39,'fits':0,'preprocessing_refits':0})
    finally:
        for frame in cached:frame.unpersist()
        spark.stop()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['focused','stage','final'],required=True);main(p.parse_args().mode)
