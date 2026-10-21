# Spark worker recovery

## Question

Can a Spark job finish if one of its workers disappears during an active task?

## Experiment and result

We stopped one worker during a task in an isolated two-worker cluster. Spark retried the task on the remaining worker and produced the expected output.

## Why it was merged

The experiment verified worker task recovery without changing production behavior.

