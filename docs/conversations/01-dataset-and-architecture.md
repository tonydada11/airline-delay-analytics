# Project Brainstorming and Architecture


## Choosing a dataset

**User:**

I'm planning a personal Spark project. I want to practice data engineering, especially ODS, DWD, DIM, DWS, and ADS. Maybe some ML too, but I don't want the whole thing to become a model-training exercise. Any dataset ideas?

**ChatGPT:**

A few worth comparing are Olist e-commerce, NYC taxi trips, and BTS airline records. They give you different things: Olist has related business tables, while taxi and airline data give you much more volume.

You could also consider the common credit-card fraud dataset if you're interested in fintech, although it's a weaker fit for those warehouse layers.

For the shortlist, I'd evaluate four things separately:

| Criterion | What it lets you demonstrate |
|---|---|
| Real historical volume | Multi-file scans, partitioning, and distributed writes |
| Stable entities | Meaningful dimensions and joins |
| Several useful grains | A reason to separate facts, summaries, and reports |
| Nontrivial correctness rules | Tests that establish more than successful execution |

The strongest choice is one where those requirements arise naturally from the data. A large flat table can still make a weak warehouse project if every downstream output is essentially the same count with a different grouping key.

**User:**

Fintech was actually my first idea. Why weaker?

**ChatGPT:**

That fraud dataset has about 285,000 rows, but most predictors are anonymized numerical components. You don't get useful customer, account, or merchant relationships to model.

It works for a classification exercise. For your project, we'd struggle to justify much of the DIM layer.

In a transaction warehouse, you'd normally expect a transaction fact referencing customer, merchant, account, and date entities. Those relationships support reusable dimensions, customer histories, merchant-level summaries, and risk features. An anonymized vector doesn't give us that structure merely because it originated from financial transactions.

We could create a date dimension or bucket numerical values, but that would leave most of the warehouse design artificial. It would also make the interesting work primarily about class imbalance and classifier evaluation, which reverses your stated priorities. I'd only keep fintech on the shortlist if we identify a richer source with meaningful entity identifiers.

**User:**

Then let's drop that one. I don't want to invent dimensions just to use the names. What about e-commerce?

**ChatGPT:**

Olist is much better structurally. It has orders, items, customers, sellers, products, payments, and reviews. You could practice joins and different fact grains with real relationships.

The catch is size: roughly 100,000 orders. You can use Spark, but many runtime comparisons will mostly measure its startup and scheduling overhead.

The grain is particularly instructive. An order can have several items and several payment records, so joining everything on order ID without thinking can multiply rows and inflate revenue. A useful design would distinguish an order-level fact from an order-item fact and aggregate each at its appropriate grain.

That makes Olist valuable for warehouse correctness even at modest scale. The limitation is experimental: broadcasting a small table or changing shuffle partitions may show different plans without producing an informative runtime improvement. We should distinguish “this demonstrates the API and semantics” from “this demonstrates an optimization under a substantial workload.”

**User:**

Is there a larger version of that kind of dataset? I don't want to rule out e-commerce just because one dataset is small.

**ChatGPT:**

We'd need to inspect a specific source. Some larger retail datasets are mostly click events rather than orders and payments. More rows wouldn't necessarily give us the relationships you want.

If distributed processing is the priority, I'd compare taxi and airline data next. Both have repeated historical releases and enough real records to avoid manufacturing volume.

Before choosing a larger source, I'd check its schema, identifier stability, time coverage, and whether its tables genuinely belong together. Ten million anonymous clicks could support sessionization but still lack the customer/order/payment relationships you originally wanted.

I'd also avoid combining unrelated public retail datasets just because they share column names like `product_id`. Those identifiers usually refer to different namespaces. The integration would require invented mappings rather than real relationships. A somewhat simpler domain with internally consistent historical records would give us a better foundation for repeatable engineering experiments.

**User:**

Yes, no duplicating rows to make it look big. And I'd prefer one coherent source rather than joining unrelated datasets.

**ChatGPT:**

Then a fixed range of monthly files is a good starting point. It gives you schema validation, source-file tracking, partitioning, and reruns before you even get into the downstream analysis.

Keeping the monthly releases separate also creates useful failure cases: a month can be missing, a file can be duplicated, or a release can arrive with unexpected columns. We can check those conditions before building downstream tables.

Duplicating rows would increase bytes processed, but it would preserve the original distributions and repeat the same keys. That can distort join cardinality, skew, and compression in ways that aren't representative of additional real history. A fixed genuine snapshot gives us a clearer interpretation of both correctness checks and performance measurements.

**User:**

What would we actually do with taxi data beyond counting trips?

**ChatGPT:**

Origin–destination patterns, demand by hour, trip duration, and fare distributions. The zone lookup gives you a useful join, and trip-duration prediction could be a downstream ML task.

You'd need to define what is known at pickup. Actual drop-off time, for example, can't be an input to a prediction made before the trip finishes.

For example, you could build a trip fact partitioned by pickup date, join the zone lookup in both pickup and drop-off roles, and produce hourly origin–destination summaries. Those operations exercise filtering, projections, joins, aggregation shuffles, and partitioned output.

Duration prediction adds a different information boundary. A report may group trips by actual duration after completion, while a model predicting at pickup must use only information available then. We'd need to keep those consumers separate. The potential scope expansion is spatial processing and external context: useful topics, but they could consume time you intended to spend on Spark execution.

**User:**

That sounds okay, but I can see myself adding weather and maps just to make it interesting. I'd rather spend that time on Spark. What does airline data give us?

**ChatGPT:**

BTS Reporting Carrier On-Time Performance has scheduled flight records with carriers, origin and destination airports, schedules, cancellations, diversions, and delay outcomes.

That gives you several useful grains: individual flights, daily carrier operations, airport activity, and directed routes. It also raises some worthwhile correctness questions. Does a cancelled flight count toward a delay rate? When is a previous flight's outcome available to a prediction? Those questions affect the pipeline itself.

A possible reporting workload would include scheduled volume and cancellation rates by carrier, delay patterns by origin airport, and directed-route comparisons over time. Those outputs reuse common facts but have different grouping keys and denominators.

The feature workload introduces another kind of computation: joining each target flight to an eligible portion of historical activity. That is a stronger reason to discuss event time, bounded windows, shuffles, and independent validation than simply adding more `groupBy` calls. It also gives you something substantive to explain when a transformation produces valid Parquet but incorrect business results.

**User:**

These are individual flights, right? Not monthly airline totals?

**ChatGPT:**

Right, flight-level records. You'd build your own aggregates from them.

The distinction is between the source release frequency and the row grain. A monthly file is a delivery unit; it doesn't mean each row is a monthly summary. Individual flight records retain dates, routes, carriers, operational status, and available outcomes.

That lets us derive daily or monthly tables without losing the ability to return to the underlying occurrences. It also means we should define identity carefully: a convenient flight business key isn't automatically proof that every physical input row is unique. Ingestion should preserve source occurrences unless we have a justified rule for removing them.

**User:**

Would 2023 and 2024 be enough data?

**ChatGPT:**

That gives you 24 monthly files and roughly fourteen million flight records. It should be a substantial local Spark workload without making the project open-ended.

Once the files are downloaded, record the exact inventory and row count. That becomes the snapshot you reproduce against.

Whether that volume produces useful performance differences also depends on the workload and resources. A narrow count, a wide historical join, and a model fit can have very different memory and shuffle costs on the same rows.

I'd use small fixtures to establish correctness, then exercise the full snapshot for reconciliation and selected experiments. That avoids making every development iteration a full-data rebuild. Record the file identities as well as the total count: two different snapshots could happen to contain the same number of rows, so count equality alone doesn't establish reproducibility.

**User:**

Okay. Keep the monthly files separate and have Spark read them. I don't want a manually combined CSV.

**ChatGPT:**

Yes. Keep the originals unchanged, read them with an explicit schema, and retain the source filename in the ingested data. Then you can trace an unexpected record back to its month.

There are two separate boundaries here. The raw directory preserves the downloaded source, while ODS records our typed interpretation of it. If we later discover a parsing mistake, we can rebuild ODS from the unchanged files instead of trying to reverse a preprocessing edit.

An explicit schema also makes type expectations visible in source control. We should still verify representative values across the wide record, since a positional schema can be consistently wrong if fields shift. File lineage then helps locate the affected release when validation finds an unexpected value or missing column.

**User:**

What would the ML problem be? Arrival delay?

**ChatGPT:**

A reasonable target is whether a flight arrives at least 15 minutes late. We'd first define when the prediction is made—scheduled departure is one possible cutoff.

Schedule, route, carrier, calendar, and distance information would be candidates. Actual departure and arrival outcomes from the flight being predicted would be excluded.

I'd describe the initial modeling contract in three parts:

1. **Prediction time:** the scheduled departure instant, with airport-local time interpreted consistently.
2. **Eligible population:** flights for which arrival delay can actually be assessed.
3. **Target:** whether the observed arrival delay reaches the 15-minute boundary.

Those decisions come before choosing an estimator. Otherwise it's easy to report a good score for a task that accidentally uses post-departure information or treats unavailable outcomes as negatives. A simple classifier with a defensible information boundary would be more useful here than a complex model trained on an ambiguous label.

**User:**

Could an airport's previous delays help? I don't want a random train/test split mixing everything together.

**ChatGPT:**

They could, but we'd need more than an “earlier flight date” filter. A flight that departed earlier might still be airborne at the prediction cutoff, so its arrival delay wouldn't be available yet.

I'd start with scheduled features and a chronological split. Historical rates can be a separate step once we've worked out timestamps and availability assumptions.

There are two different leakage risks. A random split can mix later periods into training, while a historical feature can use future outcomes even under a chronological split. Fixing the split doesn't automatically fix the feature computation.

For each target, we need an explicit cutoff and an eligible source set. The feature should be calculated from that set alone, with tests showing that changing later outcomes cannot alter earlier feature values. We can develop that logic on tiny timestamped examples before asking Spark to compute it across the complete flight history.

**User:**

And cancelled flights? We can't label them as on time just because arrival delay is missing.

**ChatGPT:**

Exactly. Keep them in the warehouse because operations reporting needs them. Exclude them from this classifier's eligible population, along with diverted flights and records without a usable arrival-delay outcome.

Consider a day with 100 scheduled flights, 10 cancellations, and 90 flights with assessable arrival outcomes. The cancellation rate uses 100 as its denominator. The arrival-delay rate uses the eligible arrival population, which is a different denominator.

If DWD drops the cancellations, the first rate becomes impossible to reconstruct correctly from that table. If it instead labels them as on time, the second rate becomes artificially low. Preserving the full fact population and applying task-specific eligibility downstream avoids both errors while keeping the source-to-warehouse reconciliation meaningful.

**User:**

Airline seems like the better fit. What's the main downside?

**ChatGPT:**

Time handling. Local airport schedules, overnight flights, and missing outcomes will need care. Also, historical BTS files don't establish when each outcome was actually published or received. We can build an offline model, but that alone won't establish that the same inputs are available in a live system.

The wide source schema is another practical issue. Explicit parsing and reconciliation will matter.

Another downside is that exact temporal reconstruction can become more involved than the classifier itself. Local times need airport timezone context; midnight conventions and overnight arrivals require explicit handling; missing fields need a defined outcome rather than an accidental conversion.

I'd keep those problems bounded by the chosen snapshot and prediction task. We don't need to add a live serving system to demonstrate that the offline computations are internally consistent. We do need to distinguish the assumptions we can test in code from operational facts that the historical source never recorded.

**User:**

That's fine. Those are useful problems to work through. Let's choose BTS for 2023–2024, with Spark and data engineering first. We can decide the architecture separately.

**ChatGPT:**

Then the dataset decision is settled. Next I'd work through what each layer contributes and where the reporting and ML paths diverge.

The requirements we have now are enough to constrain the next discussion: one historical source, unmodified monthly inputs, enough volume for meaningful Spark work, and a warehouse that remains useful without ML.

When we discuss layers, I'd ask the same question at each boundary: what new grain, reusable contract, or independently testable responsibility does this output introduce? If the answer is only “the diagram includes this acronym,” we should simplify it. That will keep the architecture connected to the learning objective rather than to a predetermined collection of components.

## Working out the architecture

**User:**

Before we pick files and functions, can you explain the overall flow?

**ChatGPT:**

Start with the data flow. You have monthly source files, a trusted flight dataset, and two kinds of consumers: business reporting and model training.

Spark does the processing. Docker can host the local Spark services. Storage holds the inputs and outputs. We can decide whether anything else is needed after those responsibilities are clear.

The tools sit at different levels of that flow:

- **Spark** executes transformations and model workloads.
- **Docker Compose** gives the local services a repeatable environment.
- **Parquet** represents persisted tabular outputs.
- **Storage paths and mounts** determine what the driver and executors can access.
- **A runner** orders jobs and stops downstream work when a prerequisite fails.

HDFS, Kafka, and orchestration platforms are possible additions, but none is a mandatory dependency of Spark batch processing. We'd add one only if its particular responsibility is needed by the source or the execution requirements.

**User:**

Do we actually need ODS? The raw files are already there. Why not clean them straight into DWD?

**ChatGPT:**

You could. Keeping ODS gives parsing its own checkpoint: read the wide CSV schema, retain rows and source lineage, write typed Parquet, and reconcile it with the inputs.

Then DWD handles business rules. If a count changes, that separation helps you tell whether it happened during parsing or transformation. The cost is an extra write and read.

For example, suppose ODS matches the source count but DWD loses several thousand records. We can investigate business transformations without reopening CSV decoding first. Conversely, if ODS already has shifted fields, debugging a downstream label formula would be premature.

The persisted boundary also lets you rebuild business logic without repeatedly parsing CSV. That benefit has to be balanced against storage and I/O cost. In this learning project, independently verifying ingestion is a substantial objective, so the extra materialization has a clear purpose. In a smaller single-purpose script, I might combine the stages.

**User:**

For this project, that seems worth it. What should DWD do?

**ChatGPT:**

One trusted record per source flight occurrence, with standardized names and types, usable schedule fields, consistent status flags, and a nullable delay label.

It should retain cancelled and diverted flights. The ML branch can filter those later; removing them here would make cancellation rates and scheduled-flight totals wrong.

I would make the guarantees explicit enough to test: required columns exist with the intended types, schedule values follow the chosen convention, status flags use valid domains, and unassessable labels remain null. The output count should reconcile with ODS under the row-preserving transformation contract.

Standardization should also preserve distinctions downstream users need. A missing arrival outcome, a cancellation, and an observed on-time arrival are different states. If cleaning collapses them into one value, later consumers cannot recover the original semantics simply by choosing a different aggregation formula.

**User:**

So cleaning doesn't mean dropping everything the model can't use?

**ChatGPT:**

Right. DWD serves more than the model. Its population should still represent the scheduled flights in the source.

A useful way to design the boundary is to separate data validity from task eligibility. A cancelled flight can be a valid operational fact while being ineligible for an arrival-delay classifier. Likewise, an incomplete outcome can still contribute to scheduled volume or missing-data reporting.

That allows each consumer to express its own denominator without rereading raw files or reversing an earlier filter. It also makes reconciliation easier: the warehouse preserves the source population, and the ML branch accounts explicitly for the records it excludes and why.

**User:**

What about dimensions? The airline and airport IDs are already on each flight.

**ChatGPT:**

Date, carrier, airport, and route dimensions can define those entities once and provide reusable attributes. But a daily count doesn't need to join every dimension just to prove they're there.

One airport dimension should cover both origins and destinations. Routes should be directional: ORD to LAX and LAX to ORD are different keys.

Dimensions need their own grain checks. Carrier keys should resolve consistently, airport identifiers should not appear twice merely because the airport serves both roles, and route keys should preserve endpoint order. If attributes vary across records, the rule for choosing an attribute value needs to be deterministic.

A dimension join is useful when a consumer needs those attributes. If the fact already contains the required grouping key, another join may add no information and can introduce duplication if the dimension grain is wrong. Reference validation and query-time enrichment are related responsibilities, but they don't require identical execution paths.

**User:**

Explain DWS and ADS. Aren't both just aggregates?

**ChatGPT:**

They can overlap if we don't give them distinct jobs. I'd use analytical DWS for reusable daily metrics by carrier, origin airport, and route. Keep counts and denominators there so later reports can aggregate them correctly.

ADS would produce specific outputs: monthly operations, scorecards, and rankings. For a monthly delay rate, sum delayed and eligible flights first, then divide. Averaging daily percentages would give quiet days too much weight.

For example, imagine one day has 10 delayed flights out of 20 and another has 10 out of 100. The average of the daily rates is 30%, but the combined rate is 20/120, or about 16.7%.

Storing counts alongside daily rates preserves the information needed to aggregate correctly. The same principle applies to average delay: carry the relevant sum and count, or derive the final average from trusted facts. A persisted rate alone is often insufficient for a later report at a different grain.

**User:**

Where does the classifier go in that diagram? After ADS?

**ChatGPT:**

It branches from DWD. It needs eligible individual flights, not monthly scorecards.

```text
BTS CSVs → ODS → DWD
                   ├→ DIM
                   ├→ analytical DWS → ADS
                   └→ flight-level features → training and evaluation
```

Historical features may summarize earlier flights, but they need a prediction-time cutoff. Reporting aggregates don't have that restriction.

The difference is also temporal. A monthly scorecard can legitimately use every observed outcome in that month because it is retrospective. Giving that scorecard to a model predicting a flight near the start of the month would expose outcomes from flights that had not happened yet.

A feature aggregate therefore needs both a grain and an availability rule. “Average delay by airport” is not a complete feature definition until we specify which historical flights qualify for each target. Keeping the branches separate makes that distinction visible in both the code and the architecture diagram.

**User:**

Can the feature table live under DWS? I know it's not really a daily summary.

**ChatGPT:**

That's a workable project convention if we name it clearly and document its grain as one row per eligible flight occurrence. A separate features package would also work. The distinction matters more than the directory name.

I'd document each materialized table with its key, population, time meaning, and intended consumers. For the feature output, “one eligible flight occurrence” should be visible wherever daily DWS tables are also described.

That matters more than whether the directory follows a particular warehouse naming convention. A misleading grain can lead someone to aggregate again or join on an incomplete key, causing row multiplication. A clear name and contract prevent that ambiguity without requiring a broad package rename solely to align with terminology from another organization.

**User:**

Would you save that table or build it inside training?

**ChatGPT:**

I'd save it. Historical calculations could be expensive, and we'll want to inspect and test feature values independently of fitting a model. Training can then reuse the same inputs.

We should validate a completed write before treating it as ready. A directory containing some Parquet files doesn't necessarily mean the job finished.

Materialization also creates a reproducible boundary for model comparison. If two estimators read the same validated feature snapshot, a metric difference is less likely to come from accidentally changing the data preparation between runs.

The dataset should have a manifest describing its schema, split, source identity, and completion state. Publish completion only after the written data has been reloaded and checked. That doesn't make the features correct by itself, but it prevents a partial or incompatible feature directory from silently becoming a model input.

**User:**

Let's work through the history rule. Landing doesn't necessarily mean the result was published immediately.

**ChatGPT:**

That's the limitation. Without publication or receipt timestamps, we'd be assuming zero reporting latency when using reconstructed arrival time as availability. We should state that assumption and describe the result as offline historical modeling.

A timestamp added when we load the monthly files won't solve it; that's bulk ingestion time.

There are at least three timestamps to distinguish: when the operational event occurred, when the provider published the record, and when our system received it. A historical reconstruction may give us the first while providing no evidence for the other two.

Assuming immediate availability is an explicit offline approximation. It permits a well-defined experiment, but a live implementation would need an actual availability contract and appropriately timestamped inputs. We should preserve that distinction in the feature documentation rather than implying that accurate arithmetic can compensate for missing provenance.

**User:**

What lookback windows would you start with?

**ChatGPT:**

Seven, thirty, and ninety days for carrier, origin, destination, and directed route would give us short and longer histories. I'd introduce them after the basic schedule-only feature dataset works, with separate boundary tests.

For each entity and window, rates need context. A 50% delay rate based on two flights should not be interpreted the same way as 50% based on thousands, so retain counts alongside rates. An absence of eligible history should also be distinguishable from an observed zero delay rate.

The different windows trade responsiveness against stability. A short window reacts more quickly but may be sparse for a route; a long one provides more observations but can smooth over recent changes. Those are hypotheses to evaluate later, not guarantees that every additional history improves the classifier.

**User:**

And the split?

**ChatGPT:**

A workable split is January 2023 through April 2024 for training, May through August 2024 for validation, and September through December 2024 for test.

Fit preprocessing on train. Use validation to choose the model and threshold, then freeze them before test evaluation. Earlier test-period outcomes can contribute to a later test flight's history only if they pass the same availability rule; they must never feed selection.

There are two kinds of state to keep separate: learned model state and historical observations. The preprocessing parameters and model coefficients are fitted from train and then frozen. Historical features for a later prediction may incorporate newly available past observations without retraining that model.

For validation, that means a later validation target can use an earlier qualifying outcome under the declared temporal rule. It does not mean validation labels can be used to fit an imputer or estimator. Separating those operations makes a chronological evaluation more faithful to the intended prediction sequence.

**User:**

Do we need a feature store or MLflow for that?

**ChatGPT:**

Not for this local batch workflow. Persisted features, saved Spark ML artifacts, and manifests describing their inputs and settings should be enough. If we later need online features or multiple deployed model versions, we can revisit those tools.

A feature store becomes useful when multiple consumers need consistent feature definitions and online/offline retrieval. A model registry helps manage versions, approvals, and deployment state. Those are concrete organizational problems, but this project has one batch workflow and a small set of saved candidates.

We can preserve the essential information locally: feature schema, fitted preprocessing, estimator configuration, selected threshold, source provenance, and compatible evaluation outputs. That gives us a repeatable experiment without operating extra services whose lifecycle would become a separate project.

**User:**

I'd skip Airflow too. One script is fine. And Kafka doesn't seem useful for historical CSVs.

**ChatGPT:**

Agreed. A sequential shell runner can execute the stages and stop when a quality check fails. Each stage should also run independently while we're developing it.

Kafka would add a streaming interface to data we're already processing as a fixed batch. There's plenty of Spark work here without it.

The runner should encode dependency order and observable failure behavior. If a stage's validation fails, the next consumer must not execute, and the run must exit nonzero with enough context to identify the failed step. A success message at the end should mean every required gate actually ran successfully.

It should also reuse the same module entry points you run during development. Otherwise the interactive path and full-pipeline path can drift apart. A small orchestrator that invokes established jobs is easier to inspect than a second implementation of the pipeline inside shell logic.

**User:**

Storage-wise, can we just use Parquet in mounted directories? Or do we need HDFS to call this distributed processing?

**ChatGPT:**

Shared mounted directories are enough for this setup. Spark workers can execute tasks across separate processes while reading the same accessible files.

That demonstrates distributed computation on one host. It doesn't give you replicated storage or protection against losing that host. HDFS would be a separate storage exercise.

Parquet and HDFS answer different questions: Parquet specifies how tabular data is encoded, while HDFS provides distributed storage. Choosing Parquet doesn't determine whether the underlying bytes live on a shared local disk, HDFS, or object storage.

For Docker workers, the practical requirement is that every process can resolve the configured paths to the same intended data. We should verify that access and document the shared-host limitation. The resulting experiment can demonstrate task scheduling and shuffle behavior while remaining explicit that its storage is a single point of failure.

**User:**

One master and two workers in Docker, then? I'd like notebooks for EDA, but I don't want the pipeline depending on running cells in the right order.

**ChatGPT:**

That setup should let you inspect executor placement, stages, shuffles, joins, and caching. Put repeatable jobs in Python modules and keep notebooks for exploration and explanation.

Later, we can compare execution strategies on real warehouse workloads. A separate worker-loss experiment could show task retry, provided the worker is killed while a task is running and we verify the final result.

I would keep notebooks as consumers of the same modules and outputs used by the scripted pipeline. They can show an execution plan, inspect a distribution, or explain a stage without becoming the only copy of its logic.

For experiments, record the resource configuration along with results. Two one-core workers on one host won't behave like a large multi-machine cluster, but they can still reveal task boundaries, shuffle exchanges, and recomputation. Performance claims should identify the workload and tested conditions so readers can distinguish an observation from a universal tuning rule.