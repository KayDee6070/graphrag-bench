# M10 real-paper pilot: local observations

This report records corpus preparation and integration checks on 2026-09-28. It is not a comparative retrieval experiment. M11 has not started.

## Sources and preparation

The [tracked catalog](../datasets/papers/pilot/catalog.json) pins three arXiv PDF snapshots. The [data card](../datasets/papers/pilot/README.md) documents source licenses and annotation limitations. Raw PDFs remain local and ignored.

| Paper | Version | PDF bytes | Pages | Canonical characters | Final chunks |
| --- | --- | ---: | ---: | ---: | ---: |
| [GraphRAG](https://arxiv.org/abs/2404.16130v2) | 2404.16130v2 | 6,893,854 | 26 | 89,713 | 267 |
| [RAPTOR](https://arxiv.org/abs/2401.18059v1) | 2401.18059v1 | 2,547,113 | 23 | 77,317 | 250 |
| [LightRAG](https://arxiv.org/abs/2410.05779v3) | 2410.05779v3 | 1,123,301 | 16 | 61,171 | 176 |
| Total | | 10,564,268 | 65 | 228,201 | 693 |

Extraction used Python 3.12.3 and `pypdf==6.19.0`, plain mode, preserving extracted characters and adding a `\n\f\n` separator after each page. No OCR or layout repair was applied. Page numbers refer to physical PDF positions. The corpus contains complete papers, including references and appendices; header and figure noise is retained.

## A real input-size failure

Initial preparation in `datasets/processed/m10-pilot-01` used the M2 800-character configuration and produced 565 chunks. Neural indexing rejected input 183 because it encoded to 441 tokens against a 384-token limit. No successful index was written from that attempt.

An offline tokenizer preflight inspected smaller character limits on the same source text:

| Maximum characters | Chunks | Maximum model tokens, including special tokens | Overlong chunks |
| ---: | ---: | ---: | ---: |
| 600 | 693 | 335 | 0 |
| 512 | 799 | 287 | 0 |
| 400 | 962 | 228 | 0 |

The final [paper configuration](../configs/papers.toml) uses 600 characters, five units, and one overlapping unit. This choice followed the encoder constraint, before strategy evaluation. It is not a tuned retrieval result. The final prepared bundle is `datasets/processed/m10-pilot-02`.

## Draft questions and provenance checks

The 12 questions contain six factual, three relationship, and three comparison questions. Nine use one annotated source document and three use two. All have `reasoning_hops=1` and `split=dev`. All share one related-question group because facts are reused. There are 17 span references to 10 distinct source intervals.

The authoring agent inspected the methods passages and rendered PDF pages: GraphRAG page 5, RAPTOR page 3, and LightRAG page 4. These same-agent source checks are separate from graph extraction, but they are not independent human review. No graph proposals, retrieval rankings, or model-generated answers were used to author the labels. Questions were frozen before the retrieval smoke checks below.

Every annotation resolves to an exact unique quotation on its named page and loads through the existing M7 benchmark contract. Fifteen of the 17 span references fit wholly inside at least one final chunk. The two references to LightRAG's high-level-retrieval explanation span overlapping chunks; their union covers the required interval. With all source chunks available, all 12 questions have complete annotated evidence. This is an input-coverage sanity check, not measured retrieval recall.

The labels remain `pending-independent-review`. Their reference answers, minimum-document assumptions, alternatives, and absence of shortcuts have not been independently established. No held-out set or genuine hidden-bridge two-hop set was created. The approximately 30-paper/120-question study remains an expansion target.

## Reproducibility and integration

`verify-papers --raw` rechecked the original PDF bytes and reproduced the saved corpus. A second preparation into `datasets/processed/m10-pilot-repeated` produced byte-identical contents for all nine bundle files, including the completion manifest. Ordinary verification needs no PDF parser or neural dependencies; it reconstructs saved chunks and compiled labels and checks attribution/review outputs.

Final ingestion SHA-256 values:

```text
documents.jsonl  f24d90a6cbdb5c74423bf280ecacc33afae19eb06680e5425700b5a68ec017b9
chunks.jsonl     4c499fe10ee04ffbdc04b7be84a3986b7123f1115021395e3308b0be11f7d37c
manifest.json    64fa401475f398fa5ede3d32aff85d4463823d2210de28dffb276ad67a1cca0f
```

The existing cached `sentence-transformers/msmarco-MiniLM-L6-cos-v5`, revision `14ca9be4bbcf1402eac0f43a2e2ccb6e0f994ba3`, indexed all 693 chunks into 384-dimensional vectors. Configuration used CPU, four threads, batch size 16, and a 384-token maximum. Reported indexing time was about 17.61 seconds on this machine; this single timing is not a latency benchmark.

One integration query asked which community-detection algorithm GraphRAG applies hierarchically. Both the BM25 and vector commands returned the relevant GraphRAG page-5 methods passage first. The vector top-five pages were 5, 3, 12, 12, and 10; BM25's were 5, 4, 5, 3, and 3. Later hits include background/bibliography material, illustrating why page/document hits alone do not establish complete evidence. No strategy aggregates, statistical comparison, or answer generation were run.

The vector artifacts are in `experiments/runs/m10-pilot-vectors`, with the sample vector query saved in `experiments/runs/m10-vector-query.json`. No new embedding or language-model weights were downloaded. No paid API was used. Model graph extraction over the 693 chunks is not part of this pilot run.

## Validation and next work

The test suite checks bounded acquisition, cache integrity, source-version and license validation, PDF/Unicode coordinates, empty-page preservation, quote ambiguity, invalid references, dev-only labels, annotation/corpus separation, rechunking, deterministic regeneration, artifact tampering, page mismatches, optional-dependency boundaries, and CLI failures. All tests run offline with original tiny generated PDFs; CI does not download the papers or model weights.

Final local validation: **517 tests passed**, including 42 new paper tests; Ruff lint and formatting checks passed. Version 0.10.0 built successfully as both wheel and source distribution. The installed wheel verified the real prepared bundle from outside the repository with PDF and neural imports deliberately blocked. The source distribution includes the catalog, draft annotations, licensing notice, configuration, tests, and lesson; neither distribution includes downloaded PDFs or generated datasets. The editable local installation also reports 0.10.0.

Study documentation includes the short [Scout lesson](../scripts/study_m10.py), the [walkthrough](../docs/real-paper-corpus.md), and a generated `review.md` with page links and pending reviewer fields. Acquisition and pilot engineering are implemented. Independent annotation review and corpus expansion remain before the full M11 comparison. The current evidence supports a reproducible real-paper preparation pipeline, not a claim that graph retrieval helps or hurts on real research questions.
