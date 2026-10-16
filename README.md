# Librarian Agents: Training LLMs to Curate Long-Context Workspaces that Anticipate Future Queries

![Librarian Agents](assets/overview.png)

Librarian Agents trains a builder agent to organize a document corpus into a
reusable hierarchical filesystem-based workspace before future queries are
revealed. An answering agent then navigates that workspace to answer queries.

[![arXiv](https://img.shields.io/badge/arXiv-2610.22273-b31b1b.svg)](https://arxiv.org/abs/2610.22273)
[![Project website](https://img.shields.io/badge/Project-Website-1f6feb.svg)](https://librarian-agents.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Contents

- [`environments/`](environments/README.md): synthetic-filesystem RL environment.
- [`experiments/`](experiments/README.md): training, answering, and retrieval methods.
- [`data_processing/`](data_processing/README.md): benchmark preparation.
- [`configs/`](configs/README.md): paper configurations.
- [`analysis/`](analysis/README.md): metric computation and statistical summaries.
- [`results/`](results/README.md): compact records for reported results.
- [`scripts/`](scripts/README.md): end-to-end commands.

## Quick Start

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[builder,dev]'
cp .env.example .env
```

Install `.[baselines]` on a CUDA machine to generate the dense, hybrid,
late-interaction, reranking, GraphRAG, Agentic RAG, and AutoCompaction results.

## Training

Training uses [Tinker](https://thinkingmachines.ai/tinker/) for model
post-training and [Modal](https://modal.com/) to host the Qwen3.5-35B executor
used for inference. See
[`scripts/train_browsecomp_plus.sh`](scripts/train_browsecomp_plus.sh) for the
BrowseComp+ training launch used by this project.

Note the launcher expects `TINKER_API_KEY`, `GEMINI_API_KEY`, and the
OpenAI-compatible executor variables documented in [`.env.example`](.env.example).

## Reproduce Reported Results

```bash
scripts/reproduce_results.sh
```

This command recomputes the BrowseComp+ reuse, BRIGHT, and hierarchy-ablation
metrics from the bundled compact score records. End-to-end commands and protocol
details are in [REPRODUCING.md](REPRODUCING.md). Numerical results are summarized
in [RESULTS.md](RESULTS.md), and checkpoint assignments are in
[CHECKPOINTS.md](CHECKPOINTS.md).

## Citation

```bibtex
@article{singh2026librarian,
  title         = {Librarian Agents: Training LLMs to Curate Long-Context Workspaces that Anticipate Future Queries},
  author        = {Singh, Anikait and Zhang, Teresa and Lee, Yoonho and Nair, Roshen Sanjay and Kumar, Aviral and Finn, Chelsea},
  year          = {2026},
  eprint        = {2610.22273},
  archivePrefix = {arXiv},
}
```

## Acknowledgments

We thank [Tinker](https://thinkingmachines.ai/tinker/) for generously providing
training compute and [Modal](https://modal.com/) for supporting model inference.

## License

Code is released under the [MIT License](LICENSE). Benchmark data remains under
its upstream terms; see [data/README.md](data/README.md) and
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
