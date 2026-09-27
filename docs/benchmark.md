# M7: measuring whether retrieval finds the evidence

M7 adds a benchmark runner, evidence metrics, a shared context budget, saved experiment records, and reports. It evaluates the four implemented retrieval methods: vector, graph, hybrid, and the BM25 keyword baseline. It reuses M1's 20 annotated questions while building all chunks, embeddings, and graph assertions from source documents.

## Start in everyday language

Imagine an open-book exam. A question needs two pieces of information from the book. A search method hands you several passages. Our first task is to check whether those passages contain the information needed to answer the question.

We already know the supporting passages because we annotated the fictional documents ourselves. Those annotations are the marking guide. The search methods receive the question and the source material. Only the evaluator sees the marking guide.

Finding one of two necessary facts is useful, but does not provide everything needed. M7 therefore reports both partial evidence coverage and whether a complete answer could be supported by the selected evidence. It does not ask a language model to answer or grade the question.

We also restrict the amount of text each method can provide. Otherwise, a method could retrieve the entire collection and appear successful. Every method uses the same maximum number of candidate chunks at each reported K and the same 2,000-token context limit.

## Run the offline lesson

```bash
.venv/bin/python scripts/study_m7.py
```

This script explains partial versus complete evidence, alternative valid sources, graph versus keyword retrieval, and the effect of a tiny context budget. Its budget arithmetic example deliberately counts characters, with an explicit teaching-only label. The actual benchmark uses the pinned model tokenizer described below.

The lesson can run with core development dependencies. It reads gold questions for evaluation, but constructs its retrievers from source documents alone. It requires no model download or external inference service.

## What counts as supporting evidence?

The existing `BenchmarkQuestion` contract identifies one or more **sufficient evidence sets**. Each set contains necessary facts; each fact contains all source spans required to support it. The rules are:

1. A source span is present only if the selected evidence covers its entire original interval in the correct document.
2. All spans belonging to a fact must be present before that fact earns credit.
3. All facts within one sufficient set must be present for complete evidence.
4. If several sufficient sets exist, completing any one is enough.

Adjacent or overlapping retrieved fragments can jointly cover a span. A missing character interval prevents full span credit. Repeated retrieval does not multiply credit. The same words in another document or at another location receive credit only if those coordinates are also annotated as a valid alternative.

This is an exact source-coverage measure. It does not judge whether the source is true, whether the annotation is complete, or whether a model could infer a missing fact. That boundary makes the evaluation deterministic and inspectable, but requires careful annotation on real papers.

The loader checks document hashes, unique IDs, exact gold quotes and offsets, question schema, consistent fact IDs within alternative sets, and question-group split consistency. Gold chunk IDs and relevant entity IDs remain annotation metadata. They are not matched to runtime graph IDs or used as retrieval hints. Document coordinates allow us to evaluate different chunking schemes without rewriting the gold evidence.

## The metrics, with a worked example

Question q09 asks which dataset evaluates the model that Alder builds on. The necessary chain is:

```text
Fact A: Alder is built on Birch.
Fact B: Birch is evaluated on Cedar.
```

| Selected facts | Evidence coverage | Complete evidence |
| --- | --- | --- |
| Neither | 0/2 = 0% | No |
| Only A or only B | 1/2 = 50% | No |
| Both | 2/2 = 100% | Yes |

For each sufficient set, coverage is `fully supported facts / required facts`. If alternatives exist, we take the maximum coverage of an individual set. We never combine incomplete parts of different sets into a fictional complete set.

M7 records these values at each cutoff:

| Output | Meaning |
| --- | --- |
| `fact_recall_at_k` | Mean fact coverage using the first K retrieved chunks, before budget selection |
| `raw_complete_rate` | Fraction of questions with a complete sufficient set in those retrieved chunks |
| `context_evidence_coverage` | Mean fact coverage after context selection and the token limit |
| `complete_evidence_rate` | Fraction of questions whose selected context contains a complete sufficient set |
| `mean_context_tokens` | Mean size of the rendered selected evidence under the recorded tokenizer |
| `retrieval_median_ms`, `retrieval_p95_ms` | Descriptive warm retrieval timings |
| `context_mean_ms` | Context selection and token counting time, separate from retrieval |

Conventional recall measures the proportion of relevant items retrieved; see the [Stanford information-retrieval textbook](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html). Our named **Fact Recall@K** adapts that idea to complete annotated facts and alternative sufficient sets. It is not conventional chunk/document Recall@K. Counting every alternative source as required would penalize a method even after it had retrieved a complete valid answer route.

We defer Precision@K because these annotations specify sufficient evidence, not exhaustive relevance judgments for every passage. MRR focuses on the first relevant result and does not measure whether a multi-fact answer is fully supported. An “any evidence hit” rate has a similar weakness. Complete evidence rate is the primary metric here.

## Context selection and the shared budget

The default comparison reports K=5 and K=10. A method retrieves up to the largest cutoff once per timed query. Each cutoff is evaluated separately using the corresponding prefix of that ranking.

For each prefix, the context assembler processes chunks in ranked order:

1. Resolve the chunk ID to validated source text.
2. Remove portions already selected from the same document coordinates.
3. Render the remaining portions with source headers, preserving exact text.
4. Count tokens for the **entire proposed context**, including headers and separators.
5. Accept all new portions of that chunk if the result fits. Otherwise skip that chunk and try the next one.

There is no clipping merely to fit the budget. A lower-ranked smaller chunk can fit after a larger chunk is skipped. Portions removed because they duplicate already selected coordinates are not counted twice. Identical text in different source documents remains separate evidence. A chunk entirely covered by previous selections is recorded under `skipped_overlap`; a chunk that would exceed the budget is recorded under `skipped_budget`.

The rendered format is:

```text
[doc-alder:0:53]
Alder is a retrieval method built on the Birch model.

[doc-birch:0:76]
Birch (also called Base) is a language model evaluated on the Cedar dataset.
```

Headers and separators consume tokens. Tokenization is not assumed to be additive across strings, so the full candidate context is recounted at each selection step. The budget applies only to this evidence string. A future answer prompt's instructions, question, special tokens, and generated answer are outside the M7 count.

The production counter uses the tokenizer associated with the pinned M4 encoder, `sentence-transformers/msmarco-MiniLM-L6-cos-v5`, revision `14ca9be4bbcf1402eac0f43a2e2ccb6e0f994ba3`. It disables truncation, prefixes, padding, and special tokens. The [Hugging Face tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer) describes these controls. Counting a 2,000-token context does not run it through the encoder: the encoder's 384-token input limit still applies separately when embedding an individual chunk or question.

These are tokens from a fixed comparison tokenizer, not an estimate of a future answer model's bill. All strategies in a run share it. If a later experiment changes embedding models, it must also preserve a common independent context tokenizer or explicitly treat the budget definition as another changed variable.

Graph paths, entity records, and assertion quotes outside the selected evidence are **audit metadata**. They never enter context construction or metric calculation. This prevents a graph method from receiving additional unbudgeted evidence through its diagnostic output.

## Run the full comparison

Use the M4 optional CPU dependency/model setup from the [README](../README.md). From the project root:

```bash
.venv/bin/graphrag-bench benchmark datasets/fixtures/tiny/corpus/documents.jsonl \
  --questions datasets/fixtures/tiny/gold/questions.jsonl \
  --split fixture \
  --config configs/benchmark.toml \
  --rules configs/extraction.toml \
  --embedding-config configs/embedding.toml \
  --output experiments/runs/m7-fixture-example

.venv/bin/graphrag-bench verify-benchmark experiments/runs/m7-fixture-example
```

Choose a new output directory for every run. Existing directories are refused before model loading. The model is cached-only unless `--allow-download` is supplied. The full benchmark uses the pinned tokenizer even for a graph/BM25-only configuration, so its CLI still requires the optional model setup. Unit tests and the default teaching script stay offline without those dependencies.

The benchmark accepts source `Document` JSONL and `BenchmarkQuestion` JSONL, not a special-purpose gold fixture object. For a future corpus, use source records with the exact document IDs and text on which the questions were annotated. Passing the two-document README ingestion example with the eight-document fixture's questions correctly fails validation.

The runner builds M2 chunks, M3 extraction/graph outputs, and M4 vectors from source documents for each run. It does not reuse M1's manually annotated chunks or graph. It embeds chunks once per run; vector and hybrid queries reuse that index. Each strategy receives only question text and requested K. Gold answers, source spans, entities, and group labels reach the evaluator after retrieval.

## Configuration and experimental discipline

[configs/benchmark.toml](../configs/benchmark.toml) records the complete comparison policy. Its top-level defaults are:

```toml
[benchmark]
strategies = ["vector", "graph", "hybrid", "bm25"]
cutoffs = [5, 10]
max_context_tokens = 2000
repeats = 3
seed = 0
```

Nested tables record sentence chunking, M5 traversal limits, M6 fusion settings, and BM25 parameters. They match the preceding milestones: 800 maximum characters, one sentence per chunk without overlap; two graph hops with mentions enabled; RRF constant 60 and 20 candidates per branch. Cutoffs must be positive, unique, and increasing. Strategy names must be known and unique. No setting is automatically tuned from the observed results.

The existing 20 questions all have `split: "fixture"`. They have been inspected during development, so relabeling them as test questions would not make them unseen. The CLI requires an explicit split and rejects empty selections. It checks that a group does not cross splits before filtering. Real development and test data must be authored separately with related templates/evidence chains grouped consistently.

This loader checks a declared split policy; it cannot discover paraphrase leakage, prevent a person from repeatedly inspecting test scores, or establish independent annotation quality. M7 does not implement automatic parameter selection. The planned RRF comparison of 10, 60, and 100 belongs on independently prepared development questions; freeze the choice before held-out evaluation.

## Repeats, timing, and aggregate results

The runner performs one excluded warm-up query per selected strategy, then evaluates every question three times by default. Strategy order is shuffled reproducibly for each question/repeat using the recorded seed. The seed controls this schedule; it is not a claim about arbitrary providers' determinism.

With 20 questions, four strategies, and three repeats, the output has **240 timed query records**. It still represents 20 questions in 10 fixture groups. Repeats supply timing samples and a stability check; they do not increase the number of independent questions.

Each question has the same repeat count, so averaging its repeated scores and then averaging questions equals the reported mean over those balanced records. The report includes separate groups by question type, reasoning hops, and minimum required document count. These categories overlap; do not add their counts together. q19 illustrates why they differ: it requires two reasoning steps but has a sufficient one-document evidence route.

Warm retrieval time measures the retrieval callback, including diagnostic construction. It excludes model loading, chunking, graph construction, vector indexing, context assembly, and scoring. Construction/model timings appear separately in the manifest; context assembly has its own timing field. Each max-K retrieval feeds both cutoff rows, so the same retrieval timing samples appear at K=5 and K=10. Do not add those rows together as distinct queries.

The stability check compares hits, scores, paths, selected contexts, and metric results across repeats. Timing values and diagnostic trace serialization are excluded. With only one repeat, `repeatable` is `null`, meaning not assessed. The evaluation fingerprint hashes these same non-timing results in canonical question/strategy/repeat order.

Paired comparisons report each question's hybrid-minus-vector difference and wins/ties/losses on complete evidence. Confidence intervals and grouped bootstrap analysis remain part of the later comparative study: a tiny, repeatedly inspected fixture does not support a general effectiveness claim. Timing p95 here is descriptive and should not be treated as a service-level guarantee.

## Saved files and verification

| File | Purpose |
| --- | --- |
| `documents.jsonl` | Validated source snapshot |
| `questions.jsonl` | Selected split's annotation snapshot, used only by evaluation |
| `chunks.jsonl` | Generated chunks in canonical ID order |
| `graph.json`, `extraction_issues.jsonl` | Source-derived graph and diagnostics, when graph/hybrid is selected |
| `vectors.npy` | Normalized vectors aligned with `chunks.jsonl`, when vector/hybrid is selected |
| `results.jsonl` | Each timed query, ranking, graph diagnostics, cutoff scores, selected context, and missing facts |
| `summary.json` | Aggregate groups, paired differences, repeatability, and evaluation fingerprint |
| `report.md` | Human-readable comparison and audit notes |
| `manifest.json` | Run ID, timestamp, raw input hashes, artifact checksums, complete settings, versions, code fingerprint, and environment |

The existing M1 `RunManifest` contract is now populated. The Git revision and dirty state are recorded when running in a source checkout. A hash of package Python sources identifies the actual local code even before a milestone commit; an installed wheel may have no Git revision. The manifest is written last as a completion marker. An interrupted write can leave a partial new directory, which is not a complete run.

`verify-benchmark` checks the supported run version, allowed filenames, artifact hashes, result schemas, unique query/repeat keys, record count, and evaluation fingerprint without loading the neural model. It detects accidental corruption. It does not retokenize context, rerun retrieval, prove semantic correctness, or authenticate deliberately rehashed data.

Source/question snapshots are canonical reserializations. Their artifact hashes may differ from the original input-file hashes despite preserving the same records. Replaying those snapshots can therefore change the input byte hashes while retaining the evaluation fingerprint. Complete run directories differ because timestamps, IDs, timing samples, and sometimes Git state differ.

Generated runs live under ignored `experiments/runs/`. A small reviewed observation report is tracked at [reports/m7-fixture.md](../reports/m7-fixture.md). Do not commit model caches or generated run directories.

## What the first real-model fixture run showed

Using the unchanged default settings and the local cached CPU model:

| Strategy | Complete evidence at K=5 | Complete evidence at K=10 |
| --- | --- | --- |
| Vector | 15/20 = 75% | 19/20 = 95% |
| Graph | 11/20 = 55% | 14/20 = 70% |
| Hybrid | 15/20 = 75% | 19/20 = 95% |
| BM25 | 15/20 = 75% | 19/20 = 95% |

At K=5, hybrid improves q10 and harms q13 relative to vector. Their aggregate complete-evidence rates tie. BM25 reaches the same aggregate rate while failing a different set of questions. Equal totals therefore do not mean identical behavior.

For q10, graph contributions bring in the Elm–Fern source needed to finish the Drift chain. For q13, fusion displaces Cedar's task-description passage, leaving the Birch question incomplete. The saved missing-fact lists and rankings make both outcomes inspectable.

The maximum selected context in this run was 216 tokens. No chunk hit the 2,000-token limit. Raw and budgeted coverage therefore match here; tests and the teaching example separately exercise binding budgets. All gold facts were coverable by the generated chunk collection. Extraction still reported the two known unsupported numeric statements from M3.

Every default query was stable across three repeats. A second process rebuilding from the saved source/question snapshots produced the same evaluation fingerprint. These are successful checks of implementation and reproducibility. They do not demonstrate that graph retrieval is generally worse, or that hybrid never helps. M8–M10 must address extraction quality and real-paper data before the later study can answer the research question.

## Code map and verification

- [dataset.py](../src/graphrag_bench/benchmark/dataset.py): source/question loading, annotation integrity, group splits.
- [context.py](../src/graphrag_bench/benchmark/context.py): source-coordinate deduplication, rendering, shared budget.
- [metrics.py](../src/graphrag_bench/benchmark/metrics.py): complete facts and alternative sufficient sets.
- [runner.py](../src/graphrag_bench/benchmark/runner.py): query-only retrieval calls, repeats, cutoff evaluation, stable fingerprints.
- [report.py](../src/graphrag_bench/benchmark/report.py): aggregates, paired differences, human-readable output.
- [pipeline.py](../src/graphrag_bench/benchmark/pipeline.py): source-only construction, run manifests, artifacts, verification.

The focused tests cover known metric values, incomplete/alternative evidence, coordinate gaps and overlaps, identical text at different sources, headers in the token budget, greedy skipping, malformed annotations, split groups, no gold-answer leakage, stable/unstable repeats, denominator checks, CLI behavior, corruption detection, and refusal to overwrite outputs. A tokenizer adapter test confirms that long context counting does not encode or truncate text.

```bash
.venv/bin/python -m pytest tests/test_benchmark.py tests/test_benchmark_metrics.py \
  tests/test_benchmark_context.py tests/test_embeddings.py -q
```

CI uses a deliberately nonsemantic embedding provider and a small test counter for integration checks. The actual local benchmark uses the pinned neural model and its tokenizer. Neither kind of check substitutes for a real-paper evaluation.

## Exercises with answers

1. **A question needs two facts, but retrieval finds only one. What are the scores?** Coverage 50%; complete evidence false.
2. **A single fact requires two quoted spans, and only one is present. Does it earn half credit?** No. Atomic facts receive credit only when all required spans are covered.
3. **One of two alternative sufficient sets is complete. Must we retrieve the other?** No. One complete set is sufficient.
4. **Both necessary facts were retrieved, but one chunk did not fit the budget. Which metric reveals the difference?** Raw complete evidence can be true while budgeted complete evidence is false.
5. **Can assertion quotes hidden in a graph trace make the answer complete?** No. Only retrieved chunk prefixes or explicitly selected context pieces are scored.
6. **What does a 75% complete-evidence rate mean here?** Fifteen of the twenty questions have a full annotated evidence route in the selected context. It is not answer accuracy.
7. **Three repetitions of twenty questions give how many independent questions?** Twenty, with only ten declared fixture groups; repetitions are not new questions.
8. **Why can vector and hybrid tie while helping different questions?** Gains and losses can cancel in an aggregate. Inspect paired outcomes and missing facts.
9. **Does a stable fingerprint imply identical runtime?** No. Timing is intentionally excluded.
10. **Can we now put “hybrid improves retrieval” on the CV?** No. We can state that we implemented a reproducible comparison and observed specific fixture outcomes. A broader claim requires the real-data study.

M7 is complete. M8 adds AI-assisted extraction, using this evaluation machinery to inspect downstream effects without supplying gold answers to extraction or retrieval.
