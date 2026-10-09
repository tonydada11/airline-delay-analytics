"""Small, deterministic experiment policy; no Spark or dataset access."""

from dataclasses import asdict, dataclass
from datetime import date
from statistics import mean
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Backtest:
    name: str
    train_start: date
    train_end: date
    evaluation_start: date
    evaluation_end: date


BACKTESTS = (
    Backtest("autumn_2023", date(2023, 1, 1), date(2023, 8, 31),
             date(2023, 9, 1), date(2023, 12, 31)),
    Backtest("spring_2024", date(2023, 1, 1), date(2023, 12, 31),
             date(2024, 1, 1), date(2024, 4, 30)),
)
STAGES = ("baseline", "schedule", "history", "tuning", "weighting", "selection", "validation")


def training_partition_count(rows: int) -> int:
    """Target <=250k rows per partition before caching wide feature vectors."""
    return max(2, (rows + 249_999) // 250_000)


@dataclass(frozen=True)
class Candidate:
    bundle: str
    family: str
    iterations: int
    regularization: float = 0.1
    positive_weight: float = 1.0

    @property
    def key(self) -> str:
        return (f"{self.bundle}-{self.family}-i{self.iterations}"
                f"-r{self.regularization:g}-w{self.positive_weight:g}")

    def document(self) -> dict:
        return asdict(self)


def baseline_candidates(bundle: str) -> tuple[Candidate, Candidate]:
    return (Candidate(bundle, "logistic_regression", 20), Candidate(bundle, "gbt", 10))


def tuning_candidates(bundle: str, family: str) -> tuple[Candidate, ...]:
    if family == "logistic_regression":
        return (Candidate(bundle, family, 100, 0.1), Candidate(bundle, family, 100, 0.01))
    if family == "gbt":
        # Keep tree tuning bounded for the project's two-worker local cluster.
        # The 50/100-tree grid repeatedly exceeded an hour per fold and caused
        # executor heartbeat loss under the documented 512 MiB profile.
        return (Candidate(bundle, family, 20),)
    raise ValueError(f"Unknown model family: {family}")


def feature_gate(previous: Mapping[str, Sequence[float]], proposed: Mapping[str, Sequence[float]]) -> dict:
    """Compare the fixed two-model panel on the same two backtests.

    Each fold's PR-AUC is averaged across LR and GBT, so a feature bundle
    cannot win simply by choosing a different model on each fold.
    """
    families = ("logistic_regression", "gbt")
    if set(previous) != set(families) or set(proposed) != set(families):
        raise ValueError("Feature gates require both model families.")
    if any(len(panel[family]) != len(BACKTESTS) for panel in (previous, proposed) for family in families):
        raise ValueError("Feature gates require both chronological backtests.")
    before = [mean(previous[family][i] for family in families) for i in range(len(BACKTESTS))]
    after = [mean(proposed[family][i] for family in families) for i in range(len(BACKTESTS))]
    return {
        "accepted": mean(after) > mean(before) and all(b >= a for a, b in zip(before, after)),
        "metric": "mean_model_pr_auc_per_backtest",
        "previous": before,
        "proposed": after,
        "mean_gain": mean(after) - mean(before),
    }


def choose_ranking_candidate(results: Sequence[tuple[Candidate, Sequence[float]]]) -> Candidate:
    """Prefer mean PR-AUC; keep existing / lower-iteration models on ties."""
    return max(results, key=lambda item: (mean(item[1]), -item[0].iterations,
                                         item[0].regularization))[0]


def choose_f1_candidate(results: Sequence[tuple[Candidate, Mapping]]) -> Candidate:
    """Formal final selection: F1, LR, unweighted, then lower complexity."""
    return max(results, key=lambda item: (
        item[1]["f1"], item[0].family == "logistic_regression",
        -item[0].positive_weight, -item[0].iterations, item[0].regularization,
    ))[0]
