# Automated cross-check of the 40 draft question annotations

**This is not the independent review.** Two fresh agent instances read the blind
worksheet and the corpus, with results, reports, and score files explicitly withheld.
They are the same model family that authored the annotations, so their *agreement* with a
label is weak evidence — shared training data means shared blind spots and correlated
errors. Their *disagreement* is worth more, because finding a genuine defect is hard to do
by accident. Treat agreement as "unexamined", not "verified".

Its purpose is triage: it tells a human reviewer where to start. The blocker in
[project-status.md](project-status.md) stays open.

Written for whoever performs the human review described in
[annotation-review.md](../docs/annotation-review.md).

Raw output: `experiments/runs/m11-crosscheck-merged.jsonl`, 40 rows, one per question.

## Aggregate

| Field | Result |
| --- | --- |
| Questions assessed | 40 of 40 |
| Source support | 39 ok, 1 weak |
| Answerable from listed facts | 40 yes |
| Shortcut found | 1 |
| Declared hop count too high | 0 |
| Self-reported confidence | 32 high, 8 medium |
| Rows carrying any concern | 13 |

Both agents matched every quote programmatically against the corpus after Unicode and
whitespace normalisation, checking document identity, arXiv version, and page number.
Their shared finding: **every quote is verbatim text on the claimed page of the claimed
paper version.** No fabricated or misattributed quotation was found. That is the single
most reassuring result here, and it is also the kind of mechanical check where an
automated pass is genuinely reliable.

## Confirmed defects

I verified these three independently against the source documents rather than accepting
the agents' reports.

### 1. paper-b01 — the quote stops one sentence short (CONFIRMED)

The question asks about **RAPTOR's leaf construction**. The annotated span, offsets
10521–10642, reads:

> These texts are then embedded using SBERT, a BERT-based encoder (multi-qa-mpnet-base-cos-v1) (Reimers & Gurevych, 2019).

It begins with an anaphoric "These texts" and never mentions leaf nodes. The very next
sentence in the source is:

> The chunks and their corresponding SBERT embeddings form the leaf nodes of our tree structure.

So the quote does not prove the statement's "initial text chunks", nor the question's
"leaf construction" framing. Extending the span end from 10642 to **10737** captures both
sentences, 216 characters, and contains the word "leaf". The fix is one number; the
decision to make it belongs to the reviewer, not to me.

### 2. paper-l01 — the evidence set is not minimal, and a shortcut exists (CONFIRMED)

The question asks what LightRAG's low-level and high-level retrieval target. The
annotation requires two separate facts on page 4, at offsets 17287–17428 and
17562–17829. But a single sentence on page 2, at offset 5574, covers both:

> LightRAG employs efficient dual-level retrieval strategies: low-level retrieval, which focuses on precise information about specific entities and their relationships, and high-level retrieval, which encompasses broader topics and themes.

**This matters beyond the label.** Under these metrics, alternative unannotated support
earns no credit. A retriever that found the page-2 sentence and fully answered the
question would score **zero**. So at least one recorded zero may be a scoring artifact
rather than a retrieval failure. How many others share this shape is unknown, and the
cross-check only searched for shortcuts on the declared multi-hop questions.

### 3. paper-d13 — the source paper contradicts itself (CONFIRMED)

FlashRAG states two different dataset counts:

- "implemented 16 advanced RAG methods and gathered and organized **38** benchmark datasets" (abstract)
- "We collect and pre-process **32** benchmark datasets" (section 3.3.1)

Both are real text in the pinned version. The expected answer therefore depends on which
passage is cited, and the annotation picks one without noting the conflict. The method
count of 16 is uncontradicted. This is a defect in the annotation's certainty, not in the
quotation.

## Reported but not independently verified

Agent-reported, listed for the human reviewer to judge. I have not checked these.

- **paper-c01, paper-c02, paper-c03** — all declare 1 hop while requiring facts from two
  different documents. One agent flagged that these may be *under*-counted and noted it
  could not express that, because I only allowed `ok` or `too_high` in the output schema.
  **That is a flaw in my instructions, not a finding**, and it means under-counted hops
  went unsearched across all 40 questions. A reviewer should settle the hop convention
  first: chain length, or lookup count?
- **paper-c02** — the "how do they differ" framing compares unlike operations: GraphRAG's
  exact-string entity resolution against LightRAG's keyword-to-index vector lookup.
- **paper-d22** — "in the reported settings" is ambiguous. The cited page-7 sentence is
  the training/critic setup, while the inference default elsewhere is top-five documents.
- **paper-b03** — Self-RAG's default is Contriever-MS MARCO, but the architecture evidence
  concerns base Contriever. Holds only under the question's looser wording.
- **paper-b05** — the backbone quote never names the model, leaning on knowing that the
  paper title refers to E5-mistral. The same document prints "E5-mistral-7b" on page 6.
- **paper-d04, paper-d12, paper-d23, paper-l03** — minor: implicit subjects, a quote
  narrower than the question's wording, or a span split across two chunks.

## What this does not establish

It cannot supply the `independent_of_annotation_author` attestation in any meaningful
sense. It did not search for under-counted hops. It searched for shortcuts only on
questions already declared multi-hop, so single-hop questions were not checked the way
paper-l01 happened to be. It read no results, but it also cannot confirm that — the
claim rests on the agents' own reports of what they opened.

## Suggested order for the human pass

1. **paper-b01, paper-l01, paper-d13** — confirmed defects, decide the fix
2. **paper-c01, paper-c02, paper-c03** — settle the hop convention, then relabel if needed
3. **paper-d22, paper-b03, paper-b05** — entity and version precision
4. **paper-d04, paper-d12, paper-d23, paper-l03** — minor wording
5. The remaining 27 — quotes mechanically verified; spot-check rather than read in full

That reorders the work from 40 sequential reads into 13 focused judgements plus a
spot-check, which is the entire point of running this.
