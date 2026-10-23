"""One 16-row smoke test of the unchanged offline scoring helpers; never fits."""
from collections import Counter
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
sys.path.insert(0,'/bundle')
import offline as api
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F


def main():
    started=time.monotonic()
    spark=api.session()
    try:
        spark.sparkContext.setLogLevel('WARN')
        spark.sparkContext.setJobGroup('offline-final-location-smoke','16 precomputed rows only')
        pre,model,means,manifest=api.load('/bundle')
        assert manifest['model_id']=='offline-lr3h-ca23fe0-v1'
        assert model.numFeatures==790 and manifest['threshold']==0.39
        # A single pinned Parquet file, limited to sixteen original feature rows.
        source=next(iter(sorted(Path('/input').glob('*.parquet'))))
        frame=spark.read.parquet(str(source)).limit(16).cache()
        assert frame.count()==16 and frame.select('flight_id').distinct().count()==16
        # The target is deliberately omitted from the scoring input.
        prepared=api.prepare(frame.drop(api.TARGET,'split'),pre,means,manifest)
        widths={r.width for r in prepared.select(F.size(vector_to_array('features')).alias('width')).collect()}
        assert widths=={790}
        saved_reference=spark.read.parquet('/expected').join(frame.select('flight_id'),'flight_id','left_semi')
        output=Path('/output/scores')
        result=api.scores(prepared,model,manifest)
        result.write.mode('errorifexists').parquet(str(output))
        saved=spark.read.parquet(str(output))
        columns=saved.columns
        observed=Counter(tuple(r) for r in saved.select(*columns).collect())
        expected=Counter(tuple(r) for r in saved_reference.select(*columns).collect())
        assert sum(observed.values())==16 and observed==expected
        assert saved.filter((F.col('model_id')!=api.MODEL_ID)|(F.col('feature_policy')!=api.POLICY)|(F.col('prediction')!=(F.col('probability')>=0.39).cast('int'))).count()==0
        # Reload exactly the same immutable manifest and means, without rounding.
        assert json.loads(Path('/bundle/means.json').read_text())==means==api.MEANS
        assert api.digest('/bundle/manifest.json')=='9157123a82926767a0ac69b6c1f9ab04899d0ad86ec32c95b42ffe46809a564f'
        api.write('/output/smoke.json',{'status':'PASS','model_id':api.MODEL_ID,'bundle_manifest_sha256':api.digest('/bundle/manifest.json'),'threshold':manifest['threshold'],'feature_width':790,'rows':16,'unlabeled_input':True,'saved_score_prediction_multiset_exact':True,'means_exact':True,'classifier_fits':0,'preprocessing_refits':0,'input_file':source.name,'input_sha256':api.digest(source),'scored_files':api.inventory(output)})
        frame.unpersist()
    finally:
        spark.stop()
        print('Spark stopped; elapsed seconds:',time.monotonic()-started,flush=True)

if __name__=='__main__':main()
