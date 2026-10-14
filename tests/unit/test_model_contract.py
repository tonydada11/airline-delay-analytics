# AI-generated test code: generated with OpenAI Codex.
import ast
from pathlib import Path
import pytest
from airline.dws import ml_feature_contract as features
from airline.ml.model_contract import MODEL_SETTINGS, CLASSIFICATION_THRESHOLD
from airline.ml.metrics import confusion_metrics


def test_pure_feature_contract_and_frozen_settings():
    tree = ast.parse(Path(features.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert imports == ["datetime"]
    assert len(features.FEATURE_COLUMNS) == 50
    assert len(set(features.FEATURE_COLUMNS)) == 50
    assert not set(features.FEATURE_COLUMNS) & features.LEAKAGE_BLACKLIST
    assert MODEL_SETTINGS["maxIter"] == 100
    assert MODEL_SETTINGS["regParam"] == 0.1
    assert MODEL_SETTINGS["elasticNetParam"] == 0
    assert MODEL_SETTINGS["tol"] == 1e-6
    assert CLASSIFICATION_THRESHOLD == 0.34
    assert "seed" not in MODEL_SETTINGS


def test_evaluation_has_no_fit_calls():
    from airline.ml import evaluate
    tree = ast.parse(Path(evaluate.__file__).read_text())
    assert not any(isinstance(n, ast.Attribute) and n.attr == "fit" for n in ast.walk(tree))


def test_confusion_metrics_and_zero_denominators():
    result = confusion_metrics(3, 2, 1, 4)
    assert result["rows"] == 10
    assert result["precision"] == 0.6
    assert result["recall"] == 0.75
    assert result["prevalence"] == 0.4
    assert result["predicted_positive_rate"] == 0.5
    assert confusion_metrics(0, 0, 0, 3)["f1"] == 0
    with pytest.raises(ValueError, match="empty"): confusion_metrics(0, 0, 0, 0)


def test_feature_contract_import_does_not_load_spark_or_ml():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c", "import sys; import airline.dws.ml_feature_contract; assert not any(n.startswith(('pyspark', 'airline.ml')) for n in sys.modules)"], check=True)
