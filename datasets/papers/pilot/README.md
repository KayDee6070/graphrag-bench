# Real-paper development pilot

This directory contains a curated acquisition catalog and 12 source-authored draft questions. Raw PDFs, parsed text, compiled labels, and retrieval artifacts are generated locally in ignored directories. No paper PDF is committed here.

The pilot uses three research papers about retrieval. They are **documents for our system to search**. Including a paper does not mean GraphRAG Bench implements its proposed algorithm.

| Source | Pinned arXiv version | PDF pages | Source license |
| --- | --- | ---: | --- |
| [GraphRAG, Edge et al.](https://arxiv.org/abs/2404.16130v2) | 2404.16130v2 | 26 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [RAPTOR, Sarthi et al.](https://arxiv.org/abs/2401.18059v1) | 2401.18059v1 | 23 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [LightRAG, Guo et al.](https://arxiv.org/abs/2410.05779v3) | 2410.05779v3 | 16 | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

The catalog lists every author, exact version, byte length, SHA-256, source license, and metadata-check date. Source licenses were checked through the versioned arXiv pages, separately from any software repository license. The total PDF download is **10,564,268 bytes**. Versioned PDFs may be regenerated upstream; a mismatch must be investigated and explicitly re-curated rather than silently accepted.

## Annotation status

All 12 questions are **development-only**, authored by an AI agent from the papers, and **pending independent review**. They were not created from the extracted graph, retrieval rankings, or generated answers. Selected passages were checked against the rendered PDF pages: GraphRAG page 5, RAPTOR page 3, and LightRAG page 4. This same-agent check is not an independent human annotation study.

There are six factual questions, three relationship questions, and three comparisons. Nine use one annotated source document; three use two. All have `reasoning_hops=1`: combining two directly stated facts in a comparison does not create a hidden two-edge reasoning chain. The pilot supplies no independently established two-hop test cases.

All questions share `rag-pilot-connected-methods` because the comparisons reuse facts from the individual questions. They must not be divided into apparent independent dev/test examples. Evidence labels cover 10 distinct source intervals, reused in 17 fact/span references. They are selected sufficient evidence sets, not exhaustive relevance judgments. Document counts are minima over those annotated sets; the compiler cannot prove that no unannotated shortcut exists elsewhere.

`annotations.json` stores paper ID, physical PDF page, and exact extracted quotation. The compiler locates each quotation exactly once on its page and produces normal `BenchmarkQuestion` records with document character offsets. It does not insert chunk IDs or gold graph entities. See the [walkthrough](../../../docs/real-paper-corpus.md) for commands and the [observation report](../../../reports/m10-pilot.md) for actual results.

## Licensing and changes

GraphRAG and RAPTOR excerpts retain **CC BY 4.0**. LightRAG excerpts retain **CC BY-NC-SA 4.0**. The combined annotation collection in `annotations.json`, including its reference paraphrases, is distributed under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**. The original GraphRAG and RAPTOR material remains available under its more permissive source license. Project code and original fictional fixtures remain MIT licensed.

Changes to source material: pypdf plain-text extraction, preservation of extracted line breaks/hyphenation, page separators, chunking, and selection of short supporting quotations. Questions, rationales, and reference answers were added by this project. No author endorsement is implied. The prepared bundle contains `attribution.md`; preserve it and this notice when sharing derived data. Full author credits appear in [catalog.json](catalog.json).
