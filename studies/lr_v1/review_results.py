"""Verify the saved LR comparison offline using only Python's standard library."""
import hashlib
import json
import math
from pathlib import Path

from studies.lr_v1.candidate_spec import validate_approval
from studies.lr_v1.metrics import MONTHS, monthly_comparison, rank_complete_candidates


CODE = Path(__file__).resolve().parent
EVIDENCE = CODE / "evidence"


def load(path):
    return json.loads(path.read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    approval = load(CODE / "approved-candidates.json")
    specs = validate_approval(approval)
    for path, expected in [(EVIDENCE / "diagnostics/results.json", approval["baseline_results_sha256"]),
                           (CODE / "protocol.json", approval["protocol_sha256"])]:
        check(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
              "Frozen input hash differs: " + str(path))
    baseline = load(EVIDENCE / "diagnostics/results.json")["validation_months"]
    raw = load(EVIDENCE / "candidate-results.json")
    summary = load(CODE / "candidate-results-summary.json")
    check([c["name"] for c in raw["candidates"]] == [s["name"] for s in specs],
          "Candidate list differs from approval")
    for candidate in raw["candidates"]:
        check(candidate["status"] == "complete", "Candidate did not complete")
        saved = next(c for c in summary["candidates"] if c["name"] == candidate["name"])
        comparison = monthly_comparison(baseline, candidate["months"])
        check(comparison == candidate["comparison"] == saved["comparison"],
              "Paired comparison differs: " + candidate["name"])
        check(saved["months"] == candidate["months"], "Monthly values differ")
        check(math.isclose(saved["mean_monthly_PR_AUC"],
                           sum(candidate["months"][m]["pr_auc"] for m in MONTHS) / 4,
                           rel_tol=0, abs_tol=1e-15),
              "Equal-month mean differs")
        check(all(candidate["months"][m]["rows"] == baseline[m]["rows"] for m in MONTHS),
              "Monthly evaluation populations differ")
    ranking = rank_complete_candidates(baseline, raw["candidates"])
    check(ranking == raw["qualified_ranking"] == summary["qualified_ranking"] == [],
          "Qualification result differs")
    check(raw["study_recommendation"] == summary["study_recommendation"] == "retain baseline",
          "Study recommendation differs")
    check(not raw["historical_test_used_for_selection"] and not summary["historical_test_used_for_selection"],
          "Historical test was used for selection")
    production = CODE.parents[1] / "results/production"
    model = load(production / "model-manifest.json")
    check(load(production / "selection/selection.json") == model["selection"], "Model selection differs")
    check(model["bundle_id"] == approval["baseline_bundle_id"] and model["threshold"] == 0.39,
          "Production baseline differs")
    for split in ("validation", "test"):
        report = load(production / (split + ".json"))
        check(report["bundle_id"] == model["bundle_id"], "Evaluation bundle differs")
        check(sum(report["confusion_matrix"].values()) == report["rows"], "Confusion counts differ")
    check(sum(baseline[m]["rows"] for m in MONTHS) == load(production / "validation.json")["rows"],
          "Study and production validation populations differ")
    print("Verified: frozen inputs, two complete candidates, paired monthly comparisons, no qualifying candidate, and unchanged production baseline.")


if __name__ == "__main__":
    main()
