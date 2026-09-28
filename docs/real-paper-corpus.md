# M10: from training reports to real research papers

## The Scout story

Until now, our Scouts trained with reports that we wrote ourselves. We knew where Eren was, which squad he belonged to, and which gate that squad guarded. Those reports made excellent **software tests**: if a Scout missed an obvious connection, we could find the bug.

Real reports are harder. A PDF has page headers, diagrams, references, abbreviations, and sentences split across lines. One sentence may discuss a competing method rather than the paper's own method. The same words can have different meanings in different sections.

M10 introduces three real papers and a way to trace their text. It prepares the material for a later experiment. It does not establish that one retrieval strategy is better.

Keep two boxes in mind:

- **Reading box: the corpus.** The full paper text that search and extraction may inspect.
- **Exam box: the benchmark.** Questions, reference answers, and the source facts required by the evaluator.

If we build the Scouts' map from the exam's answer sheet, the exam becomes misleading. Our paper corpus is built only from the pinned PDFs. A separate step compiles annotations after parsing. Tests verify that changing an expected answer cannot change any ingestion artifact.

## What exists now

The [pilot data card](../datasets/papers/pilot/README.md) identifies GraphRAG, RAPTOR, and LightRAG, their exact paper versions, authors, licenses, and annotation status. The download totals 10,564,268 bytes, about 10.6 MB. Together they contain 65 PDF pages. The final paper chunking configuration produces 693 chunks.

The 12 draft questions contain six factual questions, three relationship questions, and three comparisons. Nine use one annotated document and three use two. Every question is `dev`; independent review remains pending. The pilot contains no held-out labels and no established two-hop cases. It is a reproducible starting corpus, not the planned 30-paper comparative benchmark.

The three papers are reading material for our own retrieval system. For example, GraphRAG Bench's local graph traversal differs from the global community-summary approach described by [Edge et al.](https://arxiv.org/abs/2404.16130v2). Do not describe this project as reproducing that paper's results.

## Five steps from a PDF to an evidence receipt

1. **Pin a source.** The catalog names an exact arXiv version, authors, license, byte count, and SHA-256. The hash acts like a fingerprint of that PDF snapshot. A changed PDF is rejected even if its filename is unchanged.
2. **Extract pages.** Pinned `pypdf==6.19.0` extracts each page in plain mode. The parser preserves its output, including awkward spaces and line-break hyphens. It adds `\n\f\n` after every page, including the last one. Empty pages retain their physical page number.
3. **Store canonical text.** Concatenating those page strings creates the document. Every character offset refers to this stored Unicode text. Offsets do not refer to PDF bytes, visual bounding boxes, or a cleaned HTML rendition.
4. **Make page-bounded chunks.** The existing sentence-window algorithm runs within each page section. It assigns `Chunk.page` using physical PDF numbering from 1. Chunks cannot cross pages. Headers, references, and figure text remain present.
5. **Compile evidence labels.** Each annotation names a paper, page, and exact quotation. The quotation must occur exactly once on that page. The compiler calculates absolute character coordinates and leaves `chunk_id` unset. Rechunking therefore does not require rewriting the reference labels.

For a small example, suppose page 1 extracts as `α Scout.` and page 2 as `Gate east.`. Page 2 starts at character 11: eight characters from page 1 plus the three-character separator. The UTF-8 byte position differs because `α` occupies two bytes. This is why the project consistently distinguishes characters from bytes.

The parser follows the [pypdf text-extraction API](https://pypdf.readthedocs.io/en/6.19.0/user/extract-text.html). This is deliberately limited PDF support: no OCR, scientific table reconstruction, automatic reference removal, dehyphenation, or reliable multi-column reading-order repair. Page/content limits help bound this pilot; they are not a secure sandbox for arbitrary hostile PDFs.

## Why the chunk size changed

The first attempt used the M2 default of 800 characters and produced 565 chunks. A dense real-paper passage encoded to 441 tokens, exceeding our embedding model's 384-token limit. M4 rejected the input before writing an index. Silently losing the end of a passage would make its evidence coordinates misleading.

The paper configuration now uses 600 characters, at most five sentence/fragment units, and one overlapping unit. On these exact PDFs, all 693 chunks pass the pinned tokenizer check; the largest has 335 tokens including special tokens. A character limit is not a universal token guarantee. A different corpus must pass the same preflight again.

An annotated quotation may span two overlapping chunks. That is valid: the evidence scorer can combine overlapping source intervals. All 12 pilot questions have complete evidence when all chunks are available. This checks that the chunk representation can cover the labels; it is not retrieval accuracy.

## Run it locally

From the repository root, install the small optional PDF dependency. The tests' development lock already includes it:

```bash
.venv/bin/python -m pip install --no-build-isolation -e '.[papers]'
.venv/bin/graphrag-bench fetch-papers \
  --catalog datasets/papers/pilot/catalog.json \
  --output datasets/raw/m10-pilot
.venv/bin/graphrag-bench prepare-papers \
  --catalog datasets/papers/pilot/catalog.json \
  --annotations datasets/papers/pilot/annotations.json \
  --raw datasets/raw/m10-pilot --config configs/papers.toml \
  --output datasets/processed/my-paper-pilot
.venv/bin/graphrag-bench verify-papers datasets/processed/my-paper-pilot \
  --raw datasets/raw/m10-pilot
.venv/bin/python scripts/study_m10.py
```

Only `fetch-papers` contacts arXiv. Valid cached files are reused. Invalid files are preserved and rejected; missing files download serially with a three-second interval. Each download is bounded by its catalog length and verified before publication. The pilot catalog allows at most 10 papers, 16 MB per PDF, and 32 MB total. A larger study needs an explicitly reviewed acquisition plan and limits.

`prepare-papers` and `verify-papers` use local files only. Preparation requires a fresh output directory outside the raw directory. An incomplete output without its top-level manifest is not a valid bundle. To repeat preparation, choose another output name.

The prepared bundle contains:

| File | Purpose |
| --- | --- |
| `ingestion/documents.jsonl` | Full canonical paper text and versioned source URLs |
| `ingestion/chunks.jsonl` | Page-bounded chunks and exact document coordinates |
| `ingestion/manifest.json` | Parser/chunker versions, PDF fingerprints, page sections, chunk configuration |
| `catalog.json` | Snapshot of the curated acquisition metadata |
| `annotations.json` | Snapshot of the source-authored draft annotations |
| `questions.jsonl` | Compiled `BenchmarkQuestion` records, dev only |
| `review.md` | Review worksheet with paper/page links, quotations, answers, rationales, and pending decisions |
| `attribution.md` | Author credits, source licenses, and transformation notice |
| `manifest.json` | Bundle counts, review status, and hashes of every listed artifact |

Without `--raw`, verification checks saved artifacts, page/chunk provenance, reconstructed chunks, and compiled labels. It needs neither pypdf nor neural libraries. With `--raw`, it also checks original PDF fingerprints, repeats parsing, and requires the rebuilt corpus to equal the saved corpus. Hash checks detect drift against a manifest; they do not authenticate a malicious replacement of both data and manifest.

Existing retrieval commands consume the bundle's `ingestion/` directory:

```bash
.venv/bin/graphrag-bench query-bm25 datasets/processed/my-paper-pilot/ingestion \
  --query "Which community detection algorithm does the GraphRAG pipeline apply hierarchically?" \
  --top-k 5

# Requires the already-configured M4 embedding dependencies and cached model.
.venv/bin/graphrag-bench index-vector datasets/processed/my-paper-pilot/ingestion \
  --config configs/embedding.toml --output experiments/runs/my-paper-vectors
.venv/bin/graphrag-bench query-vector experiments/runs/my-paper-vectors \
  --source datasets/processed/my-paper-pilot/ingestion \
  --query "Which community detection algorithm does the GraphRAG pipeline apply hierarchically?" \
  --top-k 5
```

The M8 graph builder and M9 answer command accept this same ingestion representation. M10 does not run full-corpus model extraction or answer-quality evaluation. The M7 benchmark command still builds a deterministic rule graph internally; passing these papers into it does not automatically create the intended M11 LLM-graph experiment.

## Study one real question

Take `paper-c01`: which grouping algorithms do GraphRAG and RAPTOR describe in their respective methods sections? The annotation links to GraphRAG PDF page 5 and RAPTOR PDF page 3. The reference answer names Leiden community detection and Gaussian Mixture Models.

Reading only the GraphRAG passage gives half the comparison. Reading only RAPTOR gives the other half. Both are required by this annotation. But neither fact is reached through a hidden bridge entity. The question combines two direct statements, so its `reasoning_hops` remains 1.

Our earlier Scout question follows a different structure: Eren belongs to Lantern Squad; Lantern Squad guards the east gate. It follows two successive relationships through the shared squad. That is a two-hop chain. The number of documents and the number of hops describe different properties.

Software checks that the quotation exists. A reviewer must still check whether it supports the reference answer, whether another passage supplies an alternative sufficient answer, and whether a supposed two-document question has a shortcut. All 12 questions were authored and source-checked by the same AI agent. Their honest status remains `pending-independent-review`.

## Before the full comparative experiment

M10's acquisition, PDF provenance, draft-label compilation, and small pilot are implemented. Independent annotation review and corpus expansion remain open. Keep the M0 target—approximately 30 papers and 120 reviewed questions—as the planned study, subject to a practical expansion decision. Twelve related development questions cannot answer the main research hypothesis.

For expansion, select papers using a declared topic/date/license policy before seeing strategy scores. Record exclusions and avoid selecting only papers where our graph extractor succeeds. Keep related question templates and evidence chains in one split. Add genuine hidden-bridge two-hop questions and matched single-hop controls; inspect the full corpus for shortcuts. Review alternative sufficient evidence, entity ambiguity, and claims about numerical results. A claim made by a source paper is not a result reproduced by this project.

Freeze the held-out labels separately, with an independent reviewer and an audit trail of corrections. These already-inspected pilot questions must stay development material. Freeze retrieval/model settings before held-out evaluation. M11 will then connect the LLM-extracted graph to the comparison runner and measure actual retrieval and answer failures. No M11 experiment is included in M10.

## A short interview explanation

> I moved the pipeline from fictional tests to a reproducible three-paper pilot. I pinned the exact PDFs, tracked their authors and licenses, preserved physical pages and text coordinates, and compiled source quotations into development labels without exposing answers to retrieval. Real PDFs exposed an embedding token-limit failure, which I fixed through explicit chunk sizing and validation. The corpus is ready for review and expansion; I do not yet claim a graph-retrieval improvement.

Check your understanding: What is the difference between the reading box and the exam box? Why does an unchanged filename not guarantee unchanged content? Why can two documents still mean one-hop facts? What does an exact quote check leave unproven? Run the [short lesson](../scripts/study_m10.py), then inspect the [actual local observations](../reports/m10-pilot.md).
