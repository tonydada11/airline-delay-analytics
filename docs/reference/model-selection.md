# Authoritative M7 selection

Production calls the same selection implementation as `airline.ml.experiments select`. It fits preprocessing on train, then exactly one LR and one GBT sequentially. Both use all training rows and positive-class weight 2.0. No sampling, feature ablations, parameter search, or train-plus-validation refit occurs.

LR: maxIter=100, regParam=0.1, elasticNetParam=0.0, tol=1e-6, family=binomial. GBT: maxIter=10, maxDepth=4, maxBins=32, stepSize=0.1, subsamplingRate=0.5, featureSubsetStrategy=sqrt, maxMemoryInMB=128, seed=42. Estimator settings are separate from the operating threshold.

For each model, evaluate inclusive probability thresholds at integer steps 0..100. Across both classifiers, maximize exact rational validation F1; on equal F1 choose the lower integer threshold step, and choose LR only when both F1 and threshold step tie. Neither ranking metrics nor the training-prevalence baseline selects the winner. Baseline score is training prevalence with threshold 0.5; a prevalence tie predicts positive.

`airline.ml.train --run-id NAME` performs selection and publishes the exact selected model. Omitting the ID generates a unique one. Production manifest schema 3 embeds the validation evidence and authoritative classifier/threshold step; loading rejects contradictory contracts. Test materialization and evaluation occur only after publication. Neither can refit or retune. See the [pipeline report](../guides/project-guide.html#individual-ml-stages) for the ML stages and their responsibilities.

`airline.ml.experiments select --run-id NAME` remains a selection-only entrypoint. It creates the same frozen artifacts but does not publish a production bundle. Its historical evaluation command can evaluate test features bound to that same run, such as a run created by `airline.ml.train --run-id NAME`. Evaluation loads only the selected classifier and baseline. Reusing an incomplete selection ID fails; retry with a new ID.

The September–December 2024 period was previously inspected. Causal rolling test history uses only earlier available arrivals and cannot influence selection. The zero-latency arrival availability assumption is explained in [pipeline rules](pipeline-rules.md).

Saved accepted metrics are under `results/production/`. Research is confined to the two independent experiment branches. M9 is a separate execution study and never selects a classifier.
