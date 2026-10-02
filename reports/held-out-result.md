# The held-out comparison

The first result in this project that is not a development diagnostic. Eighty questions
authored by agents with no access to any measured result, verified mechanically, sealed,
and run once.

Run: `experiments/runs/m13-heldout-comparison-01`. Bundle `datasets/processed/m10-heldout-01`,
split `test`, 80 questions × 4 methods × 3 repeats = 960 records. Offline verification
passed. Evaluation fingerprint:

```text
8eb8b328764a38612825c3f1bfa66c5b55a30b3e670cffd6f6cc67670fdc6e0c
```

Every repeat was identical, no question was unreachable from the full chunk collection, and
the index and graph are the same frozen artifacts the development comparison used — the
ingestion manifests are byte-identical, so nothing was rebuilt and nothing could drift.

## Headline

Complete evidence after a shared 2,000-token budget:

| Method | K=5 | K=10 |
| --- | ---: | ---: |
| BM25 | **35/80** | **39/80** |
| Vector | 26/80 | 30/80 |
| Hybrid | 21/80 | 29/80 |
| Graph | 6/80 | 6/80 |

## The result the project was built to find

Splitting by whether a question needs one document or two, 40 questions each:

| Stratum | n | BM25 | Vector | Hybrid | Graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| One document, K=5 | 40 | 0.800 | 0.650 | 0.525 | 0.150 |
| One document, K=10 | 40 | 0.850 | 0.750 | 0.725 | 0.150 |
| **Two documents, K=5** | 40 | **0.075** | **0.000** | **0.000** | **0.000** |
| **Two documents, K=10** | 40 | **0.125** | **0.000** | **0.000** | **0.000** |

Two things here, and the second is new.

**Dense retrieval, graph traversal, and rank fusion complete zero of forty
multi-document questions, at both cutoffs.** The development set showed the same thing on
8 questions, where it could fairly be dismissed as small-sample. Forty questions across
two independently authored strata — 20 hidden-bridge and 20 comparison — is not a small
sample, and the answer does not move.

**BM25 is the only method that completes any of them**: 3 at K=5 and 5 at K=10. Those are
`held-a-01`, `held-a-11`, `held-c-01`, `held-c-10`, `held-c-18`. This is the project's
first non-zero multi-document result, and it arrives from the simplest method in the set.
It is 12.5% at K=10, so the honest description is that lexical matching occasionally gets
both halves into the budget while nothing else ever does.

## A development finding that did not replicate

On the development set, hybrid beat vector by **+5.0 points at K=5** (3 wins, 36 ties, 1
loss) and lost by 7.5 at K=10. That asymmetry was explained in detail in
[m12-fusion-analysis.md](m12-fusion-analysis.md), and the explanation was mechanically
sound: reciprocal rank fusion promotes graph candidates into the band vector ranks 7th to
10th, which is free upside at a tight cutoff and a cost at a loose one.

On held-out data the K=5 advantage is **gone, and reversed**:

| Comparison | Development | Held-out |
| --- | ---: | ---: |
| hybrid − vector, K=5 | **+5.0 pp** (3W/36T/1L) | **−6.2 pp** (1W/73T/6L) |
| hybrid − vector, K=10 | −7.5 pp (0W/37T/3L) | −1.2 pp (1W/77T/2L) |

The development report called that K=5 result "a direction, not a measured effect" and
noted the counts were too small to support a significance claim. That caution was correct.
Held-out, fusion is worse than vector alone at both cutoffs, and the losing questions are
all single-document lookups where graph candidates displaced a correct vector hit —
`held-d-01`, `held-d-03`, `held-d-10`, `held-d-11`, `held-d-16` at K=5.

The underlying mechanism survives; the sign of its net effect did not. That distinction
matters: the ablation's finding that no fusion parameter can remove the penalty still
holds, and on this evidence unweighted fusion is simply not worth running.

## What survived from development to held-out

- **BM25 leads every non-zero row.** It did on development and it does here, by 9
  questions at K=5 and 9 at K=10.
- **The graph arm is near-inert.** 6 of 80 at both cutoffs, from a graph of 111 assertions.
  It never completes a multi-document question.
- **Retrieval depth is the binding constraint**, not method choice. Single-document
  questions are mostly solvable by every method except graph; two-document questions are
  solvable by almost none.

## A spot-check of ten questions, performed after the run

Ten of the 80 were reviewed by the project owner after this run was reported: the five the
drafting agents self-reported as weakest, plus the five BM25 completed. All ten are
multi-document questions, the stratum the headline rests on. **Four were approved and six
need revision.** Record: `datasets/papers/research/held-out-spot-check.json`.

Three are defects that four drafting agents and two mechanical verifiers all passed, each
the same shape — a quote that is true but does not prove the specific claim the question
asks about:

- `held-c-01` — the GraphRAG quote stops at "in a hierarchical manner", cutting off the
  clause that says what Leiden partitions; the statement's "partition its graph index" is
  unproven.
- `held-c-20` — the M3 quote sits under a "Dense retrieval" heading and is followed by the
  passage-embedding clause, but the sliced span excludes both, so as quoted it covers only
  queries and never says "dense".
- `held-a-10` — the evidence proves MEDI and 330 datasets but not the task-instruction
  input the question also asks for.

**These defects are deliberately not fixed, and the numbers above are unchanged.** Every
correction would make its question *stricter*, and `held-a-11` and `held-c-01` are two of
the five questions BM25 completed. Repairing them after seeing the results could lower the
only non-zero figure in the headline. Editing a test set once its scores are known is the
contamination the protocol exists to prevent, so the sealed run stands as reported and the
defects stand as a known limitation of it.

What this costs the result is bounded and worth stating precisely. Six of ten questions in
the hardest stratum carry a labelling defect, and **70 of the 80 have had no human read
them at all**. The three zero columns cannot be raised by stricter questions, so the
central negative finding is unaffected. BM25's 3/40 and 5/40 are the figures at risk: both
would fall if the corrected questions were used, so treat them as an upper bound rather
than a measurement.

A corrected question set would need a fresh evaluation to mean anything, and this project
does not run one.

## What this does and does not establish

**Does.** On this corpus, under this budget, with these methods: graph-based retrieval does
not outperform vector retrieval for evidence spread across documents. It underperforms it
everywhere, and both are beaten by BM25. The question the project set out to answer has an
answer, and the answer is negative.

**Does not.** This is one corpus of 30 related papers, one embedding model, one chunking
scheme, one budget, and a graph built by an extractor that recovers 2 of 6 sampled facts.
A denser graph could change the graph and hybrid columns; nothing here bounds what graph
retrieval could do with better extraction. The 80 questions are agent-authored and
results-blinded, not independently human-authored, and the
[protocol document](../docs/held-out-questions.md) lists the specific ways that is weaker.
No confidence intervals and no significance tests are reported, and three identical repeats
are not independent samples.

**One structural caveat worth stating plainly.** The held-out questions are new but the
corpus is not: the system was developed against these same 30 papers. Held-out questions
over a familiar corpus is weaker than a held-out corpus.

## Reproduction

```bash
.venv/bin/graphrag-bench prepare-papers \
  --catalog datasets/papers/research/catalog.json \
  --annotations datasets/papers/research/held-out-annotations.json \
  --raw datasets/raw/m10-expanded --config configs/papers-research.toml \
  --output datasets/processed/m10-heldout-01 --allow-test-split
.venv/bin/graphrag-bench benchmark-papers datasets/processed/m10-heldout-01 \
  --index experiments/runs/m11-vectors-05 --graph experiments/runs/m11-graph-05 \
  --config configs/benchmark-papers.toml \
  --output experiments/runs/m13-heldout-comparison-01
.venv/bin/graphrag-bench verify-paper-benchmark experiments/runs/m13-heldout-comparison-01
```

`--allow-test-split` is required and refused by default, so a held-out label cannot be
acquired by editing one field in an annotation file.

**This run has been reported once. Using it to choose a configuration, fix an annotation,
or tune a parameter would end its held-out status and require a fresh set.**
