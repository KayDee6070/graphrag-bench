# Study guide: understanding the project incrementally

The project currently implements M1 and M2. Start with the working data flow rather than reading every model at once. The order below separates what exists from later research work.

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

## Explain the current system in an interview

A precise description at M2 is:

> I implemented deterministic ingestion for UTF-8 text and Markdown. Chunks preserve exact source coordinates and use configurable sentence windows with bounded overlap. Tests reproduce the annotated fixture spans and verify that exported artifacts are identical after moving the source corpus. The retrieval comparison is planned but not implemented yet.

Avoid claiming a complete GraphRAG system or measured retrieval improvements at this stage. Later milestones will add those capabilities and their evidence.

## Terminology to keep distinct

- **Parsing:** turn a source representation into stored text and structural metadata.
- **Chunking:** choose retrievable spans within that text.
- **Extraction:** identify entities and supported relation assertions from chunks; planned for M3.
- **Retrieval:** select evidence for a question; planned for M4–M6.
- **Generation:** produce a cited answer from selected evidence; planned later.
- **Provenance:** the references and coordinates connecting an output back to its source.
- **Determinism:** the same specified inputs and implementation produce the same result.
- **Reproducibility:** preserve the inputs, settings, versions, and artifacts needed to repeat and inspect the result.
