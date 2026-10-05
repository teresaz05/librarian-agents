# Analysis

Metric and uncertainty computation for the reported experiments.

- `reproduce_results.py` reproduces all bundled result tables.
- `bright_qa.py` computes pass@1 and unbiased pass@3 with per-domain and macro-average SE.
- `browsecomp.py` computes BrowseComp+ accuracy with corpus-bootstrap uncertainty.
- `hierarchy_ablation.py` builds and scores original, flattened, and shuffled memories.
- `browsecomp_break_even.py` computes amortization and break-even statistics.
- `training_curve.py` summarizes the training-reward series.
