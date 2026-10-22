"""Actual runtime imports and frozen-transform checks; no classifier fit/score."""
import json
from pathlib import Path
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from experiment import append, session, FEATURES, RATES
from studies.lr_v1.diagnose import budget_metrics
from studies.lr_v1.metrics import expected_budget

spark=session()
try:
    assert spark.version=='3.5.9'
    import sys
    assert sys.version_info[:3]==(3,10,12),sys.version
    model=PipelineModel.load('/accepted/preprocessing')
    # Only sixteen saved feature rows, transformed twice with train-only
    # placeholder means to exercise missingness and fixed append order.
    source=spark.read.parquet('/prior/runtime/output/features/train-2023-01').orderBy('flight_id').limit(16).cache()
    for rate in RATES:source=source.withColumn(rate,F.lit(None).cast('double'))
    source=source.cache()
    assert source.count()==16
    means={RATES[0]:0.123,RATES[1]:0.456}
    old=model.transform(source)
    new=append(old,means)
    rows=new.select(vector_to_array('original_features').alias('old'),vector_to_array('features').alias('new'),*FEATURES).collect()
    assert len(rows)==16
    for row in rows:
        assert len(row.old)==784 and len(row.new)==790
        assert row.new[:784]==row.old
        expected=[means[c] if c in RATES else float(row[c]) for c in FEATURES]
        assert row.new[784:]==expected
    model.write().save('/output/preprocessing-roundtrip')
    loaded=PipelineModel.load('/output/preprocessing-roundtrip')
    first=old.select(vector_to_array('features').alias('v'))
    second=loaded.transform(source).select(vector_to_array('features').alias('v'))
    assert first.exceptAll(second).count()==second.exceptAll(first).count()==0
    fixture=[(.9,'same',1),(.9,'same',0)]+[(.1,str(i),0) for i in range(8)]
    frame=spark.createDataFrame(fixture,'probability double, flight_id string, is_arrival_delayed_15m int')
    budgets,duplicates=budget_metrics(frame,10,1)
    for p in (10,20):
        oracle=expected_budget(fixture,p)
        for k in ('selected_mass','true_positive_mass','precision','recall'):assert budgets[str(p)][k]==oracle[k]
    Path('/output/preflight.json').write_text(json.dumps({'status':'PASS','spark':spark.version,'python':sys.version,'runtime_imports':True,'frozen_preprocessor_width':784,'appended_width':790,'rows':16,'missing_rate_imputation_and_feature_order':True,'preprocessing_save_reload':True,'distributed_duplicate_budget_fixture':True,'classifier_fits':0},indent=2)+'\n')
finally:spark.stop()
