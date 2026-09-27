# M7 fixture observation report

This is a development diagnostic, not a held-out result or a real-paper experiment. The corpus contains eight fictional documents and 20 previously inspected questions in 10 groups. It was measured locally during M7 development using the actual cached CPU embedding model, not the test double.

## Recorded setup

- Package: `graphrag-bench` 0.7.0.
- Configuration: [configs/benchmark.toml](../configs/benchmark.toml).
- Model: `sentence-transformers/msmarco-MiniLM-L6-cos-v5`, revision `14ca9be4bbcf1402eac0f43a2e2ccb6e0f994ba3`.
- Source chunks: 25, generated from documents using one sentence per chunk.
- Graph: extracted from source text using the existing deterministic rules; two numeric statements remain unsupported as relations.
- Fusion: RRF constant 60, up to 20 candidates per branch.
- Cutoffs: 5 and 10; context limit: 2,000 tokens from the pinned encoder tokenizer, including source headers.
- Repeats: three per question and strategy; 240 timed retrievals, plus one excluded warm-up per method.
- Evaluation fingerprint: `b2c4d399f1588fe9a468498bc78affbb502824b19e5cbdfdaadea3685ff96b04`.

## Outcomes

| Method | Fact Recall@5 | Complete@5 | Fact Recall@10 | Complete@10 |
| --- | --- | --- | --- | --- |
| Vector | 0.875 | 15/20 | 0.975 | 19/20 |
| Graph | 0.775 | 11/20 | 0.850 | 14/20 |
| Hybrid | 0.875 | 15/20 | 0.975 | 19/20 |
| BM25 | 0.875 | 15/20 | 0.975 | 19/20 |

“Complete” means that all facts from at least one annotated sufficient set are present in the selected context. It does not measure generated-answer accuracy. Fact Recall@K is the best sufficient set's fraction of completely covered facts before applying the budget.

No context exceeded 216 tokens, so the budget did not bind and pre-/post-budget coverage was identical. All gold evidence was coverable by the full generated chunk collection. Binding budgets are exercised separately in tests. An isolated installation of the built wheel also ran with a 40-token limit: the actual model tokenizer confirmed every saved context stayed within that limit, and some chunks were skipped because they did not fit.

For the nine two-hop questions, vector, hybrid, and BM25 each completed four at K=5 and eight at K=10. Graph completed zero at K=5 and three at K=10. These are small fixture subgroups, not estimates for real research questions.

## Why paired analysis matters

At K=5, hybrid versus vector has one win, eighteen ties, and one loss. At K=10, all twenty outcomes tie.

- **q10 improves:** “Which dataset evaluates the model that Drift builds on?” Hybrid brings the Elm–Fern source into the top five, completing the Drift–Elm–Fern evidence chain.
- **q13 regresses:** “Which task does the dataset used to evaluate Birch support?” Hybrid removes Cedar's task-description source from the top five, breaking evidence coverage that vector had completed.

Incomplete questions at K=5:

| Method | Question IDs |
| --- | --- |
| Vector | q09, q10, q15, q16, q19 |
| Graph | q09, q10, q11, q12, q13, q14, q15, q16, q19 |
| Hybrid | q09, q13, q15, q16, q19 |
| BM25 | q11, q12, q13, q14, q19 |

Equal totals hide different failures. We retained the original defaults rather than choosing a setting from these already familiar examples.

## Timing and reproducibility

The first local run's descriptive warm retrieval p95 values were approximately 8.89 ms for vector, 10.87 ms for graph, 38.80 ms for hybrid, and 0.10 ms for BM25. They exclude model/index loading and context construction, include retrieval diagnostic construction, and depend on this machine and tiny corpus. They are not deployment guarantees or a meaningful scalability comparison.

All repeated queries produced the same rankings, selected contexts, and metric scores. A second process rebuilt the indexes from the saved source/question snapshots and produced the same evaluation fingerprint. An isolated installation of the built wheel reproduced all 240 results with that fingerprint using the cached model and network access disabled for model loading. The original raw input hashes and canonical snapshot hashes can differ because of JSON serialization; the manifest records both through separate input and artifact fields.

Full local outputs are in ignored `experiments/runs/m7-fixture-01` and `m7-fixture-02`. These directories are not shipped with the repository. Use the [M7 walkthrough](../docs/benchmark.md) to reproduce the experiment into a fresh directory. Preserve its complete manifest for exact code/environment provenance; timestamps and timings are expected to differ.

This establishes a working comparison harness. It does not establish a general benefit or disadvantage for graph retrieval. Real-paper extraction, independent annotations, a frozen development/test protocol, and the later comparative study remain necessary.
