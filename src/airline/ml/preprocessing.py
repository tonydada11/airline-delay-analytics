"""Deterministic preprocessing; fit exclusively on the training split."""
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from airline.dws.ml_feature_contract import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, HISTORY_COUNTS, HISTORY_MISSING


def build_preprocessing_pipeline() -> Pipeline:
    indexers = [StringIndexer(inputCol=c, outputCol=f"_{c}_index", handleInvalid="keep",
                              stringOrderType="alphabetAsc") for c in CATEGORICAL_COLUMNS]
    encoder = OneHotEncoder(inputCols=[f"_{c}_index" for c in CATEGORICAL_COLUMNS],
                            outputCols=[f"_{c}_ohe" for c in CATEGORICAL_COLUMNS], handleInvalid="keep")
    imputer = Imputer(inputCols=list(NUMERIC_COLUMNS), outputCols=[f"_{c}_imputed" for c in NUMERIC_COLUMNS], strategy="mean")
    assembler = VectorAssembler(inputCols=[*[f"_{c}_ohe" for c in CATEGORICAL_COLUMNS],
                                           *[f"_{c}_imputed" for c in NUMERIC_COLUMNS], *HISTORY_COUNTS, *HISTORY_MISSING], outputCol="features")
    return Pipeline(stages=[*indexers, encoder, imputer, assembler])
