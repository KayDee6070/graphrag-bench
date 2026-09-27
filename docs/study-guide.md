# Study guide: understanding the project incrementally

The project currently implements M1–M6. In everyday terms: M1 defines our record cards, M2 divides documents into traceable pieces, M3 draws connections, M4 finds pieces by comparing numerical text representations, M5 follows the connections to collect evidence, and M6 combines the two search lists. Every method keeps the source receipts. Start with the working data flow rather than reading every model at once.

## First session: what the project is testing

Read [the experiment design](design.md). Be able to state the primary hypothesis and explain why retrieval evaluation must be separate from answer evaluation. A fluent answer does not prove that the system retrieved all supporting evidence.

Then inspect [the tiny fixture](../datasets/fixtures/tiny/README.md). Read q09 in `gold/questions.jsonl`: Alder is named in the question, Birch is the bridge entity, and Cedar is the answer entity. The source assertions appear in separate documents. Contrast this with q17, which compares two direct facts without a two-edge reasoning chain.

Gold answers and evidence exist so we can evaluate retrieval. Their bridge/answer entity labels must never become retrieval query hints.

## Second session: understand M1 contracts

Read [data-contracts.md](data-contracts.md), then [models.py](../src/graphrag_bench/models.py) in this order:

| Records | Question to answer |
| --- | --- |
| `Document`, `TextSpan`, `Chunk` | What exactly does a source coordinate address? |
| `EvidenceSpan`, `RelationAssertion` | How will a graph edge retain evidence? |
| `Entity` | Why can a label/alias differ from identity? |
| `EvidenceFact`, `EvidenceSet`, `BenchmarkQuestion` | How are necessary facts and alternative sufficient evidence represented? |
| `RetrievalHit`, `RetrievalResult`, `TraversalPath` | How will a retrieved result remain inspectable? |
| `RunManifest` | What must an experiment record to be reproducible? |

Run the fixture audit:

```bash
.venv/bin/graphrag-bench validate-fixtures datasets/fixtures/tiny
```

Study one negative test in `tests/test_fixtures.py`. A quote can look plausible while its source offset or document reference is wrong. Learn what the validator checks and what it cannot establish: consistency is not entailment.

## Third session: trace M2 end to end

Read [the M2 walkthrough](ingestion.md) and run:

```bash
.venv/bin/python scripts/study_m2.py
```

Predict the chunk boundaries before looking at the output. Follow `parse_bytes`, `chunk_document`, and `write_artifacts` in that order. Pay particular attention to why the text is not rewritten after coordinates are assigned.

Read `test_sentence_configuration_reproduces_all_m1_reference_spans` in `tests/test_chunker.py`. It connects the earlier manually annotated fixtures to the new implementation without using gold answers to decide how text is split.

## Fourth session: change one parameter and inspect the effect

Run the ingestion example using the commands in the walkthrough. Then choose a new output directory and change only one setting, such as `--max-chars 40`. Compare the chunk texts, boundaries, and manifest configuration.

Changing a chunking parameter is an experimental change. It may affect evidence completeness, duplicate retrieval, and the eventual context budget. Do not compare graph and vector retrieval with different chunking settings and then attribute the difference only to graph traversal.

For a small focused test run:

```bash
.venv/bin/python -m pytest tests/test_parser.py tests/test_chunker.py -q
```

## Fifth session: understand the map and its receipts

Read [the M3 walkthrough](extraction-and-graph.md), starting with its nontechnical explanation. Run:

```bash
.venv/bin/python scripts/study_m3.py
```

Describe Alder, Birch, and Cedar as boxes connected by two labeled arrows. Locate the original quote for each arrow. Explain why Birch connects the documents, why `Base` remains ambiguous, and why two sources produce two Cedar–Accuracy assertion edges.

Then follow the walkthrough's graph CLI commands and inspect `graph.json`, `issues.jsonl`, and `manifest.json`. Complete its exercises before reading the full extractor. A source quote, a graph assertion, and a verified fact are different things.

## Sixth session: understand vector retrieval

Read [the M4 walkthrough](vector-retrieval.md), starting with its cards-and-coordinates explanation. Run `.venv/bin/python scripts/study_m4.py` to work through cosine arithmetic and a lexical comparison. After the optional local model setup, add `--semantic` to inspect real embeddings and retrieved quotes.

Explain why chunks are encoded once, why questions must use the same embedding space, why 0.6 is not a probability of correctness, and why a high-ranked result can discuss a metric when the question asks for a dataset. Compare the direct Birch question with the two-step Alder question. This example demonstrates vector mechanics; it is not a measured graph-versus-vector result.

## Seventh session: follow connections and audit the evidence

Read [the M5 walkthrough](graph-retrieval.md) and run `.venv/bin/python scripts/study_m5.py`. Trace Orion, Nova, and Harbor across two sources. Compare zero, one, and two hops with mentions disabled, then enable mentions and explain why one hop can expose both facts.

Inspect the full fixture's top-five miss. Distinguish failing to link a name, failing to reach evidence, and failing to rank reached evidence high enough. Explain why walking an edge backward does not reverse its assertion. Check a returned path against the graph and source quotes before claiming the evidence supports an answer.

## Eighth session: combine two search lists

Read [the M6 walkthrough](hybrid-retrieval.md) and run `.venv/bin/python scripts/study_m6.py`. Calculate one chunk's two contributions by hand. Explain why graph scores and cosine similarities cannot simply be averaged, why a missing candidate contributes zero, and why both methods selecting a chunk does not prove it is correct.

Distinguish the candidate window from the final top K. Predict what happens if a useful chunk appears second in both lists but each list is cut to one item before fusion. Work through the example where noisy graph evidence removes a necessary source from the final selection.

After the M4 model setup, run `.venv/bin/python scripts/study_m6.py --semantic`. Inspect the actual Birch and Alder outputs. Changing the rank constant changes the Alder fifth result; explain why choosing that setting from a single example would not establish a general improvement. M7 will provide the benchmark and metrics needed to compare retrieval quality.

## Explain the current system in an interview

A precise description at M6 is:

> I implemented vector, lexical, graph, and hybrid retrieval over shared source chunks. The graph retriever preserves alias ambiguity and returns auditable evidence paths. Hybrid retrieval combines rank positions using reciprocal rank fusion, with explicit candidate limits and per-source contributions. Tests cover the arithmetic, provenance, compatibility, and reproducibility. The study examples expose successes and top-K misses. The controlled benchmark remains future work; I have not yet demonstrated a retrieval improvement.

Avoid claiming a complete GraphRAG system or measured retrieval improvements at this stage. Later milestones will add those capabilities and their evidence.

## Terminology to keep distinct

- **Parsing:** turn a source representation into stored text and structural metadata.
- **Chunking:** choose retrievable spans within that text.
- **Entity/node:** a named thing represented in the graph, such as a model or dataset.
- **Relation assertion/edge:** a source's claim connecting two entities, with a label and direction.
- **Extraction:** identify entities and explicit relation assertions with source evidence; limited deterministic rules exist in M3.
- **Embedding/vector:** a numerical representation of text produced by a model.
- **Cosine similarity:** a comparison of vector directions; it becomes a dot product after unit normalization.
- **Seed:** a graph entity identified from the question as a traversal starting point.
- **Hop/path:** one followed edge / an ordered record of followed entities and assertions.
- **Retrieval:** select evidence for a question; vector, lexical, graph, and hybrid retrieval exist through M6.
- **Candidate window:** how many results from one method enter fusion, before the final selection.
- **Rank fusion:** combine ordered lists using positions rather than incompatible raw scores.
- **Generation:** produce a cited answer from selected evidence; planned later.
- **Provenance:** the references and coordinates connecting an output back to its source.
- **Determinism:** the same specified inputs and implementation produce the same result.
- **Reproducibility:** preserve the inputs, settings, versions, and artifacts needed to repeat and inspect the result.
