# M10 research-corpus expansion: local observations

This report records the 30-paper development-corpus preparation completed on 2026-09-28. It does not report retrieval, graph-extraction, or answer-quality results. M11 has not started.

## Corpus and selection

The expanded catalog contains 30 versioned arXiv papers: the three pilot papers plus 27 papers selected from an ordered list of 58 additional candidates. Thirty-one candidates were excluded because their posted paper license fell outside the declared CC BY, BY-SA, BY-NC, or BY-NC-SA 4.0 policy. Selection did not use retrieval ranks, graph quality, or answer-generation results.

The PDFs total **43,300,527 bytes** and **558 physical pages**. Their source-license distribution is 23 CC BY 4.0, three CC BY-SA 4.0, and four CC BY-NC-SA 4.0. The collection is purposive and license-filtered; it is not a random or systematic literature sample. The [data card](../datasets/papers/research/README.md), [catalog](../datasets/papers/research/catalog.json), [screening record](../datasets/papers/research/selection.json), and [attribution](../datasets/papers/research/ATTRIBUTION.md) record these boundaries.

## Draft development questions

The annotation book contains the unchanged 12 pilot questions plus 28 new questions. All 40 are development labels authored from source text by the same agent and remain `pending-independent-review`.

| Property | Count |
| --- | ---: |
| Factual questions | 29 |
| Relationship questions | 8 |
| Comparison questions | 3 |
| One-hop questions | 35 |
| Candidate two-hop questions | 5 |
| One annotated document | 32 |
| Two annotated documents | 8 |
| Related-question groups | 23 |
| Papers with reference evidence | 30 |
| Independently approved questions | 0 |
| Held-out questions | 0 |

The candidate bridges are RAPTOR → SBERT, REPLUG → Contriever, Self-RAG → Contriever, NV-Embed → Mistral, and E5-mistral → Mistral. Their rationales explicitly limit claims to the cited original method or backbone. A corpus-wide normalized term screen found no paper containing both root and target terms for four candidates. It found `Self-RAG` and `uncased` together in FlashRAG, but the passage names an unrelated SimCSE checkpoint rather than Contriever's backbone. This is an author screening observation, not proof that the corpus contains no semantic shortcut.

A proposed REPLUG → Contriever question about representation pooling was rejected during authoring: REPLUG page 3 itself says the retrieval embedding uses mean pooling. That one-paper shortcut would defeat the intended bridge. The retained REPLUG question instead asks about random cropping and 10% token deletion from the original Contriever paper. Independent review still needs to check all retained questions, alternative sufficient evidence, ambiguity, minimum document counts, and hop labels.

## PDF extraction and visual checks

Preparation used Python 3.12.3 and `pypdf==6.19.0` in plain extraction mode. The parser preserves extracted characters, line-break hyphenation, and physical PDF pages. Selected rendered pages were visually inspected for the SBERT network-structure claim, Contriever augmentation, Mistral rolling-buffer cache, NV-Embed's Mistral starting point, and the Nomic font-extraction artifact. The source pages support the intended prose readings, subject to independent review of the full questions.

Pypdf emitted **84 optional-font warnings** while rebuilding the complete corpus. The optional `fontTools` package was absent. In affected papers, some glyph sequences may have awkward spaces or degraded mathematical text. Installing that optional decoder can change canonical text and evidence coordinates, so `plain-pages-v1` now refuses to parse when `fontTools` is present. Improving this representation requires a new parser recipe, rebuilt artifacts, and renewed annotation review. Offline verification of a saved bundle does not parse PDFs and therefore does not require this environment constraint.

## Chunk and tokenizer preflight

The initial expanded build reused the pilot's 600-character limit. All 6,136 chunks were within the cached embedding model's 384-token input limit; the maximum was 367 tokens including special tokens. The all-source audit nevertheless found `paper-d04` uncoverable because the INSTRUCTOR quotation crossed a hard chunk cut whose trimming omitted one separating space.

Three development configurations were checked before any comparative retrieval run:

| Maximum characters | Chunks | Maximum model tokens | Overlong chunks | Uncoverable questions |
| ---: | ---: | ---: | ---: | --- |
| 600 | 6,136 | 367 | 0 | `paper-d04` |
| 512 | 6,996 | 318 | 0 | None |
| 400 | 8,575 | 251 | 0 | `paper-d10` |

The final [research-paper configuration](../configs/papers-research.toml) uses 512 characters, five units, and one overlapping unit. Its 6,996 chunks represent all annotated evidence when the evaluator receives every source chunk. This checks representability only; it is not retrieval recall. The configuration is informed by visible development labels and must be frozen before a future held-out evaluation. The three-paper pilot retains its original 600-character setting.

## Bundle and review checks

The final local prepared bundle is `datasets/processed/m10-expanded-03`. It contains 30 documents, 558 page sections, 6,996 chunks, and 40 compiled development questions. Raw verification reproduced the canonical corpus from all pinned PDFs. The original three-paper bundle also continued to verify after the code changes.

The `audit-papers` command reports complete all-source evidence coverage and no uncovered questions. It reports no retrieval accuracy. The `export-paper-review` command created 40 pending rows in a separate ignored file. Verification reported zero approved, zero revised, 40 pending, zero held-out, and `reviewer_identity_authenticated=false`.

Review rows are tied to the exact SHA-256 of the bundle manifest and must cover its exact question inventory. Approval requires six explicit checks plus reviewer/date/notes and an independence attestation. The software validates these records but cannot establish that the named reviewer is real, qualified, or independent. A valid pending sheet returns successfully; consumers must inspect the decision counts rather than treating exit code zero as approval.

## Scientific boundary and next work

This expansion reaches the planned corpus size and draft development-set size. It does not complete the planned 120-question benchmark. An independent reviewer must inspect and correct the 40 dev questions. Eighty genuinely held-out questions—targeting 40 two-hop cases and 40 matched single-hop controls—must then be authored and reviewed without exposing their labels during tuning. The five current bridge candidates are insufficient for the held-out hypothesis test and can never become unseen labels because they have already been inspected.

No full-corpus LLM extraction, vector index, retrieval comparison, answer generation, statistical interval, or M11 experiment was run. Therefore, this work supports a reproducible corpus and review workflow, not a claim that graph, vector, hybrid, or lexical retrieval performs better.
