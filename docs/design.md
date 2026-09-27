# M0 architecture and experiment contract

## Research question

When does graph traversal improve retrieval of the evidence needed to answer questions across documents?

The smallest hypothesis is that adding two-hop graph evidence to vector retrieval improves complete evidence retrieval when a required bridge entity does not appear in the question. Graph-only retrieval is a diagnostic comparison. The primary comparison is hybrid against vector.

For example, one document states that Alder builds on Birch. Another states that Birch is evaluated on Cedar. A question about the dataset evaluating Alder's underlying model requires both facts. This is a synthetic mechanism test, not a scientific result.

## Planned implementation

Use Python, Pydantic, NetworkX, a local sentence-transformers embedding provider, and exact NumPy cosine search. Exact search avoids introducing approximate-index recall as another variable at this scale. Sentence Transformers documents direct embedding similarity search for small corpora: <https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html>.

Keep documents and chunks independent from the graph. Represent semantic entities with stable IDs, explicit aliases, and configurable types. Store relations as separate attributed assertions with source spans, source chunks, extraction versions, and qualifiers. Repeated or conflicting assertions remain distinct. A future NetworkX adapter will preserve these using a directed multigraph.

Graph retrieval will link query entity names and aliases, retain ambiguity, traverse bounded neighborhoods, and collect source evidence. Initially cap traversal at two relation edges, limit hub expansion, and rank by query-seed coverage followed by path length and stable IDs. Traversal paths preserve assertion IDs so evidence selection can be inspected. Benchmark entity labels are never available to the query linker.

Hybrid retrieval will fuse chunk rankings using reciprocal rank fusion. For each ranking containing a chunk, add `1 / (c + rank)`, with ranks starting at one. Missing chunks contribute zero. Compare `c` values 10, 60, and 100 on development questions and freeze the choice before test evaluation. This avoids requiring calibrated similarity and graph scores. Reference: <https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf>.

Implementation status through M6: this fusion rule now exists with configurable `c` and separate candidate windows. The initial settings are `c=60` and 20 candidates per branch; they are starting values, not development-set selections. No strategy weights or reranker have been added. The [M6 walkthrough](hybrid-retrieval.md) records the implementation, sensitivity examples, and observed misses; development/test tuning and evidence metrics remain M7 work.

Implementation status through M7: the [benchmark runner](benchmark.md) now measures complete source evidence and fact coverage at K=5/10, selects context under a shared 2,000-token budget, records repeats/timings, and exports verified experiment records. Its fixed comparison tokenizer is the pinned M4 encoder's tokenizer, with source headers included and special tokens excluded. Precision, MRR, confidence intervals, and automatic development-set tuning remain deferred. The existing twenty questions remain fixtures; [their observed results](../reports/m7-fixture.md) are diagnostics rather than held-out evidence for the hypothesis.

The primary experiment will omit reranking. A later ablation can apply the same reranker to every strategy. Context assembly will deduplicate evidence and enforce the same token budget for every strategy. All generation runs will share the provider, prompt, and decoding configuration.

This project studies local evidence traversal. The original Microsoft GraphRAG paper instead emphasizes global question-focused summarization with community summaries, so its results do not directly validate this hypothesis: <https://arxiv.org/abs/2404.16130>.

## Experiment protocol

Start with approximately 30 papers and 120 manually verified questions: 40 development and 80 held out. Aim for 40 held-out two-hop questions and 40 single-hop controls. Keep related templates and evidence chains in the same split. A shared retrieval corpus is permitted; held-out labels must remain unavailable during tuning and extraction.

Separate question type, minimum required document count, and required reasoning hops. Comparison and relationship questions can fall into either hop category. Verify that supposed cross-document questions do not have a sufficient single-passage shortcut.

Author evidence requirements from documents independently of the extracted graph. Annotate alternative sufficient evidence sets. Build real graphs without using benchmark labels. A small, explicitly labeled oracle graph diagnostic can separate extraction failures from traversal failures.

## Measures and targets

- Primary measure: fraction of questions whose assembled context contains a complete sufficient evidence set.
- Evidence coverage: fraction of required atomic facts supported in the context, accepting alternative sufficient sets. Final metric implementation belongs to M7.
- Ranking diagnostics: evidence Recall@5 and Recall@10.
- Shared evidence budget: at most 2,000 tokens per question using a fixed tokenizer.
- Practical effect target: at least 10 percentage points improvement in two-hop complete evidence retrieval, with no more than 5 percentage points regression on single-hop controls.
- Runtime target: warm retrieval p95 no greater than two seconds on the development machine. Record indexing and extraction time separately.
- Reproducibility target: three runs over frozen artifacts produce identical rankings; timing fields may differ.
- Provenance target: every returned evidence reference resolves to an existing source span.

Report paired differences and 95% bootstrap intervals, grouping related questions. Eighty held-out questions constitute a pilot; intervals may remain wide. A negative result with reproducible evidence and useful failure analysis is a successful project. Do not tune against the held-out results to meet the effect target.

Precision@K requires sufficiently complete relevance judgments. MRR does not capture retrieval of all required facts. Defer both rather than misrepresent partial annotations. Later answer evaluation will inspect reference-fact correctness, supported-claim fraction, and citation entailment on the same subset across strategies. Record context size, latency, token usage, and applicable costs separately.

## Scope controls

The largest risks are extraction quality, entity resolution, benchmark bias, and annotation effort. Use a retrieval-trained encoder and a small BM25 sanity baseline. Prioritize zero-, one-, and two-hop ablations before adding infrastructure. Begin with a few real papers before expanding the corpus. A two- to three-week schedule requires keeping the UI optional and avoiding broad PDF-layout support in v1.

M1 establishes contracts and fixtures only. No graph, embedding, retrieval, extraction provider, or generation implementation is included in this milestone.
