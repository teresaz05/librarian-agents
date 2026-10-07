# Results

Run `scripts/reproduce_results.sh` to reproduce these metrics from the bundled records.

## BrowseComp+ reuse

| Builder | Accuracy +/- SE | Correct / questions | Incomplete corpora | Builder tokens / corpus |
|---|---:|---:|---:|---:|
| Base Qwen3.5-4B | 24.15% +/- 3.55% | 107 / 443 | 14 / 50 | 126,012 |
| Librarian | 63.66% +/- 2.95% | 282 / 443 | 0 / 50 | 147,634 |

The paired gain is 39.50 points (95% corpus-bootstrap CI 31.47 to 47.75).
Correct answers per builder token improve by 2.25x (95% CI 1.75 to 3.01).
At ten questions per corpus, Librarian uses 33.8% fewer recorded tokens per
expected correct answer (ratio 0.662, 95% CI 0.504 to 0.836). The two methods
have equal recorded tokens per expected correct answer at an estimated 31
questions per corpus (95% CI 18 to 54).

These values are recomputed from `results/browsecomp_per_corpus.jsonl`.

## Hierarchy ablation

The 49 corpora with completed filesystem snapshots, comprising 435 questions,
are evaluated with the original, flattened, and shuffled Librarian filesystems.
This experiment uses the expanded amortization questions and a fresh matched
answering run, rather than the original one-question-per-corpus evaluation in
the paper's main BrowseComp+ figure.

| Filesystem | Accuracy +/- SE | Mean navigation tokens +/- SE |
|---|---:|---:|
| Original | 55.86% +/- 2.58% | 3,634.79 +/- 216.02 |
| Flattened | 57.70% +/- 2.81% | 5,497.08 +/- 356.17 |
| Shuffled | 51.72% +/- 2.87% | 3,611.66 +/- 220.97 |

Flattening increases navigation cost by 1,862.29 tokens per question; its 95%
paired corpus-bootstrap interval is 1,474.43 to 2,274.35. Shuffling reduces
accuracy by 4.14 points; its 95% paired corpus-bootstrap interval is -7.50 to
-0.91 points. Standard errors use corpora as the independent unit. The fixed
transformation seed and structural checks are recorded
in `results/hierarchy_ablation.json`, and the 1,305 compact score rows are in
`results/hierarchy_per_question.jsonl`.

## BrowseComp+ cross-answerer result

| Answerer | Embedding | Hybrid | Late interaction | Rerank | Graph | Agentic | AutoComp. | No FT | Librarian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 42% | 24% | 38% | 44% | 26% | 44% | 2% | 25% | **75%** |
| Gemini 3.1 Flash Lite | 30% | 18% | 30% | 42% | 36% | 42% | 4% | 28% | **77%** |

The reported aggregates are stored in
`results/browsecomp_cross_answerer.json`.

## BRIGHT question answering

Each cell reports pass@1 ± SE / pass@3 ± SE. Per-domain standard errors use
questions as the independent unit. The Average row is an equal-weight macro-
average, with standard errors propagated from the three domain estimates.

| Domain | BM25 | Dense | Hybrid | Rerank | Agentic | Late interaction | Graph | AutoComp. | Librarian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AOPS | .305±.059 / .370±.068 | .315±.060 / .380±.069 | .330±.064 / .370±.068 | .320±.061 / .375±.069 | .285±.056 / .370±.068 | .285±.055 / .375±.069 | .125±.044 / .155±.051 | .060±.034 / .060±.034 | **.360±.056 / .500±.069** |
| Biology | .175±.046 / .260±.060 | .195±.048 / .290±.061 | .140±.043 / .205±.056 | .140±.041 / .230±.056 | .170±.045 / .255±.059 | .180±.049 / .245±.060 | .050±.029 / .060±.034 | .060±.034 / .060±.034 | **.345±.049 / .570±.065** |
| Stack Overflow | .090±.033 / .155±.048 | .115±.039 / .170±.052 | .070±.024 / .155±.048 | .060±.026 / .110±.043 | .065±.025 / .145±.045 | .070±.025 / .145±.048 | .015±.011 / .035±.025 | .015±.015 / .020±.020 | **.855±.039 / .935±.034** |
| Average | .190±.027 / .262±.034 | .208±.029 / .280±.035 | .180±.027 / .243±.033 | .173±.026 / .238±.033 | .173±.025 / .257±.033 | .178±.026 / .255±.034 | .063±.018 / .083±.022 | .045±.017 / .047±.017 | **.520±.028 / .668±.034** |

`results/bright_per_question.jsonl` contains all 5,400 binary judgments in
1,350 method-question rows.

## Additional results

`results/reported_aggregates.json` records the construction-cost table,
BRIGHT and BrowseComp+ filesystem statistics, and the nine-domain BRIGHT
transfer table exactly as reported. `results/training_reward.csv` contains the
54 reward observations underlying the training curve.

The construction-cost evaluation reports 178,926 Librarian builder tokens. The
443-question reuse protocol records 147,634 tokens per corpus. These values
correspond to distinct evaluation protocols and are reported separately.
