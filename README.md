# Model comparison research

## Question

Did LR tuning, GBT, or XGBoost improve on the accepted logistic regression model?

## Results

Historical equal-month validation PR-AUC was 0.4281 for the accepted LR, 0.4277 for LR with positive-class weight 1, 0.4237 for LR with regularization 0.3, and 0.4274 for a 60-iteration GBT. The full-data XGBoost run did not fit within available memory.

## Decision

Keep the accepted LR. No alternate candidate qualified for promotion. These are saved historical results, not new fits.

[Full comparison and evidence](studies/MODEL_COMPARISON.md).
