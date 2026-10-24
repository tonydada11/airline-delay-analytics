# Three-hour delay history research

## Question

Can recent airport delay history improve arrival-delay predictions?

## Result

The saved equal-month validation PR-AUC was 0.4637, compared with 0.4281 for the accepted LR. The experiment added three-hour airport delay features.

## Decision

The candidate was not promoted. Recent arrival outcomes are not proven to be available by prediction time; offline scoring with precomputed inputs does not establish an online feature path. These are historical results, not new fits.

[Full research, availability review, and evidence](studies/RECENT_DELAY_FEATURES.md).
