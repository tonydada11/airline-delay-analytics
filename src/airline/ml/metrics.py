"""Natural-prevalence confusion and continuous-score ranking metrics."""
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
from airline.dws.ml_feature_contract import TARGET_COLUMN
from airline.ml.model_contract import CLASSIFICATION_THRESHOLD


def confusion_metrics(tp, fp, fn, tn):
    rows = tp + fp + fn + tn
    if rows == 0:
        raise ValueError("Cannot evaluate an empty split")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"rows": rows, "prevalence": (tp + fn) / rows,
            "confusion_matrix": {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn},
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "predicted_positive_rate": (tp + fp) / rows}


def calculate_metrics(scored):
    positive = F.col("probability") >= CLASSIFICATION_THRESHOLD
    label = F.col(TARGET_COLUMN) == 1
    row = scored.agg(*[F.sum(F.when(c, 1).otherwise(0)).alias(name) for name, c in
                      (("tp", positive & label), ("fp", positive & ~label), ("fn", ~positive & label), ("tn", ~positive & ~label))]).first()
    result = confusion_metrics(*(int(row[k] or 0) for k in ("tp", "fp", "fn", "tn")))
    if result["prevalence"] in (0.0, 1.0):
        result.update(roc_auc=None, pr_auc=None, ranking_status="undefined: single-class split")
    else:
        evaluator = BinaryClassificationEvaluator(labelCol=TARGET_COLUMN, rawPredictionCol="probability", numBins=0)
        result.update(roc_auc=evaluator.setMetricName("areaUnderROC").evaluate(scored),
                      pr_auc=evaluator.setMetricName("areaUnderPR").evaluate(scored), ranking_status="defined")
    return result
