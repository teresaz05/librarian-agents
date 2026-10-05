# Results

Compact records used to reproduce the paper metrics.

- `browsecomp_per_corpus.jsonl` supports the 443-question reuse analysis.
- `bright_per_question.jsonl` contains four binary judgments per method-question cell.
- `hierarchy_per_question.jsonl` supports the original, flattened, and shuffled comparison.
- `training_reward.csv` contains the reward series used for the training curve.
- The corresponding JSON files contain generated summaries.
- `reported_aggregates.json` records secondary paper tables whose underlying
  response-level rows are not included in this release.

Run `scripts/reproduce_results.sh` from the repository root to recompute the
reported metrics.
