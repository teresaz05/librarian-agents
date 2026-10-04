# Reproducing the experiments

## Bundled records

```bash
scripts/reproduce_results.sh
```

The command recomputes evaluations from the bundled score records and checks
the reported aggregates, training configuration, and training-reward series.

## Training

BrowseComp+ training uses 680 tasks, batch size 8, group size 4, Qwen3.5-4B
with rank-32 LoRA, Qwen3.5-35B-A3B for file creation and merging, ten epochs,
32 construction turns, 1,024 tokens per builder action, and a 60,000-token
trajectory cap. Answerability-delta rewards are evaluated every eight turns,
with at most four probes per trajectory, followed by terminal answer
correctness. Explicit structure and construction-cost reward terms are zero.

```bash
scripts/prepare_browsecomp_plus.sh
scripts/train_browsecomp_plus.sh
```

The configuration is in `configs/paper_browsecomp.json`; the executable
training command is `scripts/train_browsecomp_plus.sh`.

## BrowseComp+ reuse evaluation

```bash
export BROWSECOMP_CONSTRUCTION_INDEX=/path/to/prepared/amortization/index.jsonl
scripts/build_browsecomp_filesystems.sh

export LIBRARIAN_SNAPSHOTS_ROOT=outputs/browsecomp/snapshots/librarian
export BASE_SNAPSHOTS_ROOT=outputs/browsecomp/snapshots/base
scripts/evaluate_browsecomp_plus.sh
```

The bundled manifest fixes 50 corpora and 443 questions. Both arms use the same
construction caps, executor, answerer, judge, and question-to-corpus mapping.
Incomplete constructions score zero; answerer or judge errors invalidate a run.

The paper contains two BrowseComp+ cost protocols:

- The cross-answerer construction experiment reports 178,926 Librarian tokens.
- The 443-question reuse evaluation records 147,634 Librarian builder tokens per
  corpus and approximately 16.8K recorded builder and filesystem-read tokens per
  query at 20 queries per corpus.

## BRIGHT evaluation

```bash
scripts/prepare_bright_qa.sh
export BRIGHT_SNAPSHOTS_ROOT=outputs/bright/snapshots
scripts/build_bright_filesystems.sh
scripts/evaluate_bright_qa.sh
```

The fixed manifest contains 50 questions from each of AOPS, Biology, and Stack
Overflow. Four answers are sampled per question at temperature 0.7 and top-p
1.0. Each domain corpus contains every selected question's gold documents and
sampled distractors up to 50 documents. The scorer reports pass@1 and unbiased
pass@3, with standard errors over questions. For a question with `c` correct
responses among four samples, pass@3 is `1 - C(4-c, 3) / C(4, 3)`.

Static retrieval generation is implemented in
`experiments/inference/strong_retrieval.py`; GraphRAG, Agentic RAG, and
AutoCompaction are implemented in their correspondingly named modules. Their
shared generation settings are in `configs/bright_baselines.json`. Generate all
eight matched baselines with:

```bash
scripts/run_bright_baselines.sh
```

## Hierarchy ablation

The fixed manifest contains the 49 corpora and 435 questions used to compare
the original Librarian filesystem with content-preserving flattened and
shuffled variants. The transformation seed is `20260924`. The runner checks
that node payloads and identifiers are unchanged, flattening removes all parent
edges, and shuffling preserves the hierarchy depth profile.

```bash
export BROWSECOMP_SNAPSHOTS_ROOT=/path/to/librarian/snapshots
scripts/evaluate_hierarchy_ablation.sh
```

The command prints accuracy and mean navigation tokens for all three variants
and writes the compact score rows to
`outputs/hierarchy_ablation/per_question.jsonl`.

## Checkpoints

The BrowseComp+ reuse evaluation uses training step 55, and the BRIGHT
evaluation uses training step 89. Their assignments are listed in
`CHECKPOINTS.md`. Set `BROWSECOMP_LIBRARIAN_CHECKPOINT` and
`BRIGHT_LIBRARIAN_CHECKPOINT` in `.env` to the corresponding exported model path.

## Training curve

```bash
python -m analysis.training_curve
python -m analysis.training_curve --output outputs/training_curve.png
```

The second command requires the `plots` extra.
