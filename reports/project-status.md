# GraphRAG Bench: status at M11

A single honest statement of what exists, what was measured, and what is open. Written
for someone evaluating this work or deciding whether to continue it, who has not read the
800-line development report. Every claim here is reproducible from saved artifacts;
pointers follow each section.

## The question, and the answer so far

**Question.** When does graph-based retrieval beat vector retrieval for questions whose
evidence is spread across documents?

**Answer.** Not established, and not answerable from this work yet. What was measured is
narrower and still useful: on a 30-paper corpus under one evidence budget, **retrieval
depth and extraction coverage dominate the choice of retrieval method**. No method
completed a single multi-document question, and the reason is not that graphs are
unhelpful in principle.

## What exists

A complete, reproducible retrieval-comparison harness, offline by default:

- PDF acquisition with pinned versions and per-page provenance; 30 papers, 558 pages,
  6,996 chunks.
- Four retrieval methods — dense vector, BM25, bounded graph traversal, reciprocal rank
  fusion — under one shared token budget and one tokenizer.
- Two extraction paths: a deterministic rule grammar and local LLM-assisted extraction
  with source validation, response caching, and replay.
- Source-evidence metrics that score annotated spans, not answer text.
- Answer generation with exact citation checks, and offline replay of everything.

Design choices that matter more than the feature list: gold labels are evaluation inputs
only and never reach extraction, retrieval, or prompts; every run writes a manifest with
source hashes, configuration, and code provenance; and verification recomputes metrics
from saved files without a model. 701 tests pass offline with no network, API key, or
model weights.

## What was measured

**The comparison** (`experiments/runs/m11-research-comparison-03`, bundle
`m10-expanded-05`, 40 questions × 4 methods × 3 repeats = 480 records, fingerprint
`0d337e5d…`, verified offline). Complete evidence after the budget:

| Method | K=5 | K=10 |
| --- | ---: | ---: |
| BM25 | 23/40 | 26/40 |
| Hybrid | 15/40 | 17/40 |
| Vector | 13/40 | 20/40 |
| Graph | 7/40 | 8/40 |

Three findings, in descending confidence:

1. **Lexical retrieval beat dense retrieval** on these questions, at both cutoffs, under
   an identical budget. It leads every non-zero subgroup.
2. **No method completed any of the 8 multi-document questions, 7 of which are labelled
   two-hop and 1 three-hop.** A labelled
   oracle diagnostic explains why: the annotations are sound (0 of 17 facts uncoverable),
   but **19 of 26 reachable fact/method pairs sit deeper than rank 10**. These questions
   need two facts; typically one lands at rank 1–5 and its partner at 26–363, and half a
   set scores zero. The graph arm reached 1 of 17 facts and produced no seed at all on one
   question — with 111 assertions it cannot connect these entities.
3. **Fusion changes sign with budget**: hybrid gains 5.0 points over vector at K=5 and
   loses 7.5 at K=10. This is now **explained**: RRF hands graph roughly half the slots
   (measured 4.20 graph-only chunks in hybrid's top ten), and graph's useful contribution
   lives in a narrow band — evidence vector ranks 7th to 10th. At K=5 that band is outside
   the budget, so promoting it is near-free upside; at K=10 the budget already covers it,
   so the same mechanism evicts correct tail hits instead. See
   [m12-fusion-analysis.md](m12-fusion-analysis.md). Seven questions moved out of 40, so
   the mechanism is established and its magnitude is not.

**Reproducibility results worth recording.** Extraction moved from CPU float32 to GPU
bfloat16, cutting a 79-hour projection to 72 minutes. bfloat16 reproduced the CPU
reference exactly on eight cases, including both accepted assertions and their evidence;
float16 did not, dropping 2/6 to 1/6. Rebuilding the vector index under a different torch
build produced bit-identical vectors and reproduced a prior evaluation fingerprint exactly.

Detail: [M11 development report](m11-development.md). Reproduction commands:
[paper-comparison.md](../docs/paper-comparison.md).

## What is open

**Two blockers, neither of which is code.** Until both are resolved, every number above is
a development diagnostic and none supports a research claim:

1. **Independent annotation review.** 11 of 40 questions now carry a recorded decision,
   made by the project owner and applied to the labels; the other 29 are pending. That
   review is recorded as `project-owner-review-not-blind-to-results`, and every note
   carries a provenance string saying so: quote-to-page verification was machine-performed
   rather than re-checked by hand, an agent recorded the entries, and the reviewer had
   already seen the aggregate results. **It is not independent human review**, which would
   require someone who did not author the labels and had not seen the numbers. An external
   model review was attempted and rejected as unusable — 35 identical high-confidence
   verdicts, missing all three pre-named canaries. The tooling is ready: `export-paper-review`
   writes 40 pending rows, `scripts/blind_review_sheet.py` strips the author's reasoning
   from the worksheet, and `verify-paper-review` validates the filled result. What is
   missing is a reviewer who did not author the labels. Procedure and its limits:
   [annotation-review.md](../docs/annotation-review.md). An automated cross-check has
   already triaged the 40 questions down to 13 worth reading closely, and confirmed that
   every quote is verbatim text on its claimed page: [annotation-crosscheck.md](annotation-crosscheck.md).
   That is triage, not review — the agents share a model family with the annotation
   author, so their agreement carries little weight and only their disagreement does.
2. **Held-out questions.** There are zero. All 40 were visible throughout. The plan calls
   for 80 sealed questions, and they should be written after review, not before, or they
   inherit whatever review finds.

**Known limitations of the graph arm.** The extractor recovers 2 of 6 sampled draft facts
and produced 111 assertions from 6,996 chunks. The graph column therefore measures this
extractor at least as much as it measures graph retrieval. No assertion has been reviewed.

**Two label fixes applied to source, not yet reflected in any measured result.** An
automated cross-check found both defects and each was verified against the source
documents; see [annotation-crosscheck.md](annotation-crosscheck.md). Both are now in
`datasets/papers/research/annotations.json`, on the authorisation of the project owner:

- `paper-b01` — the `raptor-encoder` quote stopped one sentence short of the line tying
  SBERT embeddings to leaf nodes, which is what the question asks about. The quote now
  runs to the end of that sentence, 121 to 216 characters, and the statement says so.
- `paper-l01` — a single page-2 sentence covers both annotated page-4 facts. Because the
  metrics credit only annotated spans, a retriever that found it scored zero. That
  sentence is now a **second sufficient evidence set**, so either route scores.

**The comparison has been re-run on reviewed labels** as
`experiments/runs/m11-research-comparison-03`, bundle `m10-expanded-05`, fingerprint
`0d337e5d…`, offline verification passed. Eleven of the forty questions now carry a
recorded decision; see "What is open" below for what that review is and is not.

**Every aggregate is byte-for-byte unchanged from the pre-review run.** BM25 23/40,
hybrid 15/40, vector 13/40, graph 7/40 at K=5; 26, 17, 20, 8 at K=10. The fusion sign
flip is unchanged at +5.0 pp (3/36/1) and −7.5 pp (0/37/3). No question is unreachable,
every repeat is stable.

That stability is itself worth recording. Eleven annotations were revised or confirmed,
three hop labels were changed, and one question was reworded — and not one aggregate
moved. The labelling work mattered for the honesty of the record, not for the numbers.

The one visible change is structural: relabelling `paper-c01` and `paper-c02` to 2 hops
moved them into the `hops:2` subgroup, which grew from 5 questions to 7, and
`paper-c03` at 3 hops created a `hops:3` group of one. All three subgroups score
**0.000 for every method at both cutoffs**, so the central finding now rests on a wider
and more consistently labelled set than before.

Derivation of the earlier label fixes and the defects behind them is in
[annotation-crosscheck.md](annotation-crosscheck.md). `paper-b01` and `paper-l01` remain
**pending** in the review file on purpose: those two fixes were proposed by an agent of
the same model family that authored the original labels, so a reviewer should confirm them
rather than inherit them.

**The fusion sign flip is resolved.** Six configurations left the K=10 penalty at exactly
−7.5 pp, because vector and graph retrieve nearly disjoint candidates and the decisive
comparisons therefore contain no agreement for RRF to reward. Adding per-method weights
fixes it: **any `graph_weight` below 1.0 removes the K=10 penalty and keeps the K=5 gain
in full**, making hybrid weakly dominant over vector. The threshold is derivable rather
than tuned, and weighting works by stripping graph's power to nominate candidates vector
never found while preserving its 25 agreement lifts. It makes graph harmless at K=10, not
helpful, and does not touch the multi-document zeros. See
[m12-fusion-analysis.md](m12-fusion-analysis.md). The five draft two-hop questions still need a shortcut check; a
heuristic sweep found no unannotated alternative route for any of the 8 multi-document
questions, which is evidence but not proof. The fusion sign flip is unexplained. Nobody
has checked whether any hop count is *under*-stated — the cross-check could only report
labels as correct or too high, so that direction went unexamined across all 40 questions.
M12 ablations, M13–M14 polish, charts, and the optional demonstration UI are untouched, though M12's error analysis is now partly done:
[m12-fusion-analysis.md](m12-fusion-analysis.md) and the multi-hop diagnostic cover its
first two questions.

**Not claimed anywhere.** No general ranking of retrieval methods, no answer-quality or
abstention evaluation, no confidence intervals, no significance tests. Forty questions with
three identical repeats are not independent statistical samples, and alternative
unannotated support earns no credit under these metrics.

## If this continues

The diagnostic points at a lever that is not method choice. Nineteen of 26 findable
fact/method pairs sit below rank 10, so a larger candidate pool with reranking, or
retrieval per fact rather than per question, would move these numbers considerably more
than swapping retrievers. Denser extraction would give the graph arm a real chance to
participate. Both are engineering, and both are cheaper than the two blockers above.

If the blockers cannot be resolved, this work still stands as a reproducible benchmark
harness with a documented negative development result. That is a smaller claim than the
original question, and it is the one the evidence currently supports.
