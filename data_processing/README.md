# Data Processing

Commands for preparing BrowseComp+ and BRIGHT from their official distributions.

- `prepare_browsecomp.py` creates the training and held-out BrowseComp+ splits.
- `validate_browsecomp.py` checks a prepared BrowseComp+ split.
- `prepare_bright_qa50.py` creates the fixed 150-question BRIGHT evaluation.
- `prepare_bright_frozen_construction.py` creates question-blind BRIGHT construction tasks.
- `prepare_training_index.py` writes synthetic-filesystem training indices.

Dataset access and licenses are described in [`data/README.md`](../data/README.md).
