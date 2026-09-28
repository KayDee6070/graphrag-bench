# Research-paper development corpus v1

This is the M10 expansion of the [three-paper pilot](../pilot/README.md): **30 papers and 40 draft development questions**. It supplies real reading material and a source-linked practice set. All annotations are `agent-source-authored`, `split=dev`, and `pending-independent-review`. There are **zero held-out questions**. The [study chapter](../../../docs/m10-expansion.md) explains preparation and review.

## Contents and intended use

- `catalog.json`: versioned arXiv identifiers, exact PDF hashes and byte sizes, titles, authors, source licenses, and metadata-check dates.
- `selection.json`: an ordered screening record for 58 additional candidates, with primary metadata URLs, cached-HTML hashes, and inclusion/exclusion reasons. The three original pilot papers are retained separately, not counted among those 58.
- `annotations.json`: the unchanged 12 pilot questions plus 28 new questions, with page-local exact quotations, reference answers, rationales, and related-question groups.
- `ATTRIBUTION.md`: human-readable authors, source URLs, licenses, fingerprints, and transformation notice for all 30 papers.

Raw PDFs, full extracted text, prepared bundles, and reviewer decision files are local ignored artifacts. Git tracks metadata, selected quotations, original annotations, configuration, software, and documentation. The question compiler does not consult retrieval ranks, graph proposals, or generated answers.

These papers describe retrieval, RAG, embeddings, and evaluation. Their methods and models are **subjects in the reading corpus**, not 30 implementations reproduced by GraphRAG Bench. Performance claims inside a source paper are not measurements from this project. This dataset is for development and a personal research portfolio; it is not a representative survey or an independently validated leaderboard.

## Selection protocol and bias

The authoring agent assembled an ordered, purposive list of English technical papers about retrieval/RAG, embedding methods/backbones, multi-hop QA, and evaluation. Retain GraphRAG, RAPTOR, and LightRAG from the pilot, then take the first 27 additional candidates that meet the declared source-license, acquisition, and parser bounds. The resulting works were first released in 2019–2025; exact versions are pinned as checked on **2026-09-28**. This is a convenience sample chosen before expanded-corpus retrieval experiments, not a systematic or random literature sample.

Source metadata is read from arXiv, including its posted paper license. Accept CC BY, BY-SA, BY-NC, or BY-NC-SA **4.0**. Exclude nonexclusive arXiv distribution-only licenses, no-derivatives licenses, and other license versions/types. A project's code license is not substituted for its paper license. The screening record includes 27 inclusions and 31 exclusions under this rule; no candidate was excluded because our retriever or graph extractor performed badly. No full-corpus extractor was run.

The screening includes important excluded works such as DPR, the original RAG paper, FiD, and ColBERT. Their omission is a licensing-policy consequence, not a scientific ranking. The initially confused identifier `2202.08904` was checked against its primary metadata and recorded correctly as SGPT; ColBERTv2 is `2112.01488`. Metadata hashes record the HTML inspected locally, not independent authentication of the source or an immutable arXiv HTML service.

Each PDF must be at most 16 MB, unencrypted, and contain 1–100 pages. The research-purpose catalog permits at most 40 papers and 128 MB total; the unchanged default pilot cap is 10 papers and 32 MB. The actual 30 PDFs total **43,300,527 bytes**. Existing pilot PDFs were reused. Downloads are serial, cached, size-bounded, and checked against the catalog fingerprint. The 30-paper sample is influenced by license availability, authoring convenience, English language, and readable extraction; none of these biases disappears because hashes are reproducible.

## Annotation composition

| Property | Current draft count |
| --- | ---: |
| Factual / relationship / comparison | 29 / 8 / 3 |
| One annotated document / two | 32 / 8 |
| One-hop / candidate two-hop | 35 / 5 |
| Related-question group IDs | 23 |
| Papers contributing reference evidence | 30 |
| Independently approved / held out | 0 / 0 |

New IDs `paper-b01`–`paper-b05` are bridge candidates; `paper-d01`–`paper-d23` are direct controls. The 12 original IDs stay unchanged. A two-document comparison remains one hop when it compares direct facts rather than following a hidden intermediate entity. Minimum document counts are relative to the selected sufficient evidence, not an independently proven global minimum.

All bridge candidates have one currently annotated sufficient evidence set. We have not established exhaustive alternatives. Explicit qualifications distinguish the original SBERT approach from RAPTOR's particular checkpoint, the original Contriever family from later supervised fine-tuning, and the original Mistral backbone from architectural changes in NV-Embed. Reviewers must check these distinctions and reject ambiguous questions.

The original pilot questions and RAPTOR/SBERT bridge share a group. REPLUG/Self-RAG/Contriever questions share another; NV-Embed/E5-mistral/Mistral questions share another. Other direct controls use paper-family groups. Reused facts are kept together. These 23 labels are conservative organizational metadata for this dev set, not evidence that the groups are independent. Broader topic/template overlap needs review before any future split. Never move these already-inspected labels into a held-out set.

## Author checks and remaining review

The same agent checked exact extracted source passages, used source context rather than model-generated answers, and visually inspected selected rendered pages. A corpus-wide term screen checked candidate root/answer co-occurrences. It found a Self-RAG/`uncased` hit in FlashRAG's appendix that actually refers to a different SimCSE checkpoint. A proposed REPLUG→Contriever pooling bridge was rejected because REPLUG page 3 already describes mean pooling. These are useful author checks, not independent review or proof that every shortcut is absent. See the [report](../../../reports/m10-expansion.md) for the screen and limitations.

The prepared `review.md` gives every question's quotations, links, and rationale. `export-paper-review` creates pending rows bound to the exact bundle; `verify-paper-review` validates their structure and fingerprint. Only an independent reviewer can supply actual decisions. These tools do not authenticate reviewers, silently approve annotations, or promote dev questions to held out.

## Licensing and changes

The Python software and original fictional fixtures use the repository's MIT license. **Paper text and quotations do not become MIT-licensed.** This directory is a collection of separately attributed works. Its 23 CC BY 4.0, three CC BY-SA 4.0, and four CC BY-NC-SA 4.0 papers retain their respective source terms. The four noncommercial/share-alike sources are LightRAG, SPLADE, FlashRAG, and jina-embeddings-v3; the BY-SA sources are SBERT, BEIR, and MultiHop-RAG.

Our original selection metadata and newly authored annotation prose are offered under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original pilot material remains available under its [previous collection notice](../pilot/README.md#licensing-and-changes); its underlying quotations keep their original paper licenses. No blanket license over this collection removes the attribution, noncommercial, or share-alike conditions of an included work. Preserve source copyright notices, author credits, license links, and change notices when reusing excerpts or derived text. Commercial users must not assume that this complete corpus is available for commercial use.

Changes to paper material: pypdf plain-text extraction, page separators, page-bounded chunking, and quotation selection. Whitespace, PDF reading order, font artifacts, and line-break hyphens are preserved in the canonical representation. Reference answers and questions are our paraphrases; source authors do not endorse this project. The [complete attribution](ATTRIBUTION.md) and each record's `paper_id` identify the relevant source terms. See Creative Commons' [collection/ShareAlike guidance](https://wiki.creativecommons.org/wiki/ShareAlike_interpretation) for the distinction between separately retained works and adaptations.
