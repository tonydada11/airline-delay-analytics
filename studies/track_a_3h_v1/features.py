"""Study-only 3h interval sweep; no preprocessing, fitting or publication."""
from pyspark.sql import Window, functions as F
from airline.dws.ml_feature_contract import TARGET_COLUMN

HOUR = 3_600_000_000
ROLES = ('origin', 'destination')
FEATURES = tuple(f'{r}_{k}_3h' for r in ROLES for k in
                 ('delay_rate', 'flight_count', 'history_missing'))


def history(targets, sources, role):
    query = targets.select(role, F.col('_prediction_us').alias('_time')).distinct()
    source = sources.filter(F.col('_prediction_us').isNotNull() & F.col('_available_us').isNotNull())
    start = F.greatest('_prediction_us', '_available_us') + 1
    end = F.col('_prediction_us') + 3 * HOUR + 1
    source = source.filter(start < end)
    timeline = query.select('*', F.lit(1).alias('_query'),
                            F.lit(0).cast('long').alias('_n'), F.lit(0).cast('long').alias('_y'))
    for point, sign in ((start, 1), (end, -1)):
        timeline = timeline.unionByName(source.select(role, point.alias('_time'),
            F.lit(0).alias('_query'), F.lit(sign).cast('long').alias('_n'),
            (F.col(TARGET_COLUMN) * sign).cast('long').alias('_y')))
    timeline = timeline.groupBy(role, '_time').agg(F.max('_query').alias('_query'),
        F.sum('_n').alias('_n'), F.sum('_y').alias('_y'))
    window = Window.partitionBy(role).orderBy('_time').rowsBetween(Window.unboundedPreceding, 0)
    timeline = timeline.withColumn('_n', F.sum('_n').over(window)).withColumn('_y', F.sum('_y').over(window))
    return timeline.filter('_query = 1').select(role, F.col('_time').alias('_prediction_us'),
        F.when(F.col('_n') > 0, F.col('_y').cast('double') / F.col('_n')).alias(f'{role}_delay_rate_3h'),
        F.col('_n').alias(f'{role}_flight_count_3h'),
        (F.col('_n') == 0).cast('double').alias(f'{role}_history_missing_3h'))


def append_role(targets, sources, role):
    h = history(targets, sources, role).alias('h')
    t = targets.alias('t')
    condition = F.col('t._prediction_us').eqNullSafe(F.col('h._prediction_us')) & F.col(f't.{role}').eqNullSafe(F.col(f'h.{role}'))
    return t.join(h, condition, 'left').select('t.*', *[F.col('h.' + c).alias(c) for c in FEATURES if c.startswith(role+'_')])
