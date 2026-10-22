# Isolated Track A three-hour qualification

Reference: `16e841e`. Only the six approved three-hour columns are appended to
saved train/validation rows. No preprocessing or classifier is fit; original
784 coordinates remain supported by unchanged columns and preprocessing artifacts.
The proposed width is 790, not a fitted vector claim.

Run once with `python3 studies/track_a_3h_v1/run.py`. The exclusive `runtime/`
directory prevents retries. The pinned Spark 3.5.9 image, two one-core workers,
640m executor heaps, 768m driver heap, four shuffle partitions and 64 MiB input
partitions follow the existing production runtime. Container caps are unchanged
from the preceding isolated Spark study (2 GiB driver/worker, 512 MiB master).
No dependencies are installed. Spark task/stage retries are disabled.

Generated Spark checks and exact six-date parity against `track_a.enrich` must
pass before the full build marker is created. Previous sample source SHA-256s
are checked first. Sources use the existing eligibility and UTC-time converter,
with signed delay plus elapsed time and accepted zero reporting latency. Only
January 2023–August 2024 partition paths are read. Interval contributions apply
when `t-3h <= departure < t` and `availability < t`; ties collapse before sweeping.
Null times retain targets with empty histories. Null entities use null-safe joins.

The full build is one monthly sequence, not independent retries. Each month's
source bounds are computed from exact target UTC extrema. Full source rows are
staged on disk once; each role is staged separately. Every output month is written
exclusively, then compared in both directions with `exceptAll` over every original
column, which includes business keys and labels. Counts and missingness are
validated by month; final split counts must be 8,946,370 and 2,418,739.

The 14,400-second supervisor cap includes cleanup, with work stopped by 14,220
seconds. Event logs record shuffle/spill/task-memory metrics; 10-second cgroup
samples are phase-tagged and lifetime peaks are captured before teardown.
Per-month sampled memory is a lower bound, while cgroup lifetime peaks are exact.
Protected tracked files are SHA-256 checked; existing data/artifact/study files
are size/mtime checked. All input mounts are read-only. Incomplete outputs are
retained as evidence and are not qualified. Do not fit or promote a model here.

Completed single-attempt results: [PASS report](REPORT.md). Recompute only the
offline summaries with `python3 studies/track_a_3h_v1/summarize.py` and
`python3 studies/track_a_3h_v1/report.py`; do not rerun the Spark build.
