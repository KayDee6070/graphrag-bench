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

**The comparison** (`experiments/runs/m11-research-comparison-01`, 40 questions × 4
methods × 3 repeats = 480 records, verified offline). Complete evidence after the budget:

| Method | K=5 | K=10 |
| --- | ---: | ---: |
| BM25 | 22/40 | 25/40 |
| Hybrid | 14/40 | 16/40 |
| Vector | 12/40 | 19/40 |
| Graph | 7/40 | 8/40 |

Three findings, in descending confidence:

1. **Lexical retrieval beat dense retrieval** on these questions, at both cutoffs, under
   an identical budget. It leads every non-zero subgroup.
2. **No method completed any of the 8 multi-document questions, 5 of which are also labelled two-hop.** A labelled
   oracle diagnostic explains why: the annotations are sound (0 of 17 facts uncoverable),
   but **19 of 26 reachable fact/method pairs sit deeper than rank 10**. These questions
   need two facts; typically one lands at rank 1–5 and its partner at 26–363, and half a
   set scores zero. The graph arm reached 1 of 17 facts and produced no seed at all on one
   question — with 111 assertions it cannot connect these entities.
3. **Fusion changes sign with budget**: hybrid gains 5.0 points over vector at K=5 and
   loses 7.5 at K=10. Mechanically plausible — reserved slots are cheap insurance in five
   positions and a net cost in ten — but 3 wins and 3 losses is a direction, not an effect.

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

1. **Independent annotation review.** All 40 questions were authored by an AI agent from
   paper text and checked by nobody else. The tooling is ready: `export-paper-review`
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

**The measured results in this report predate both fixes.** Every number above, including
the fingerprint `30bf9bf4…`, was produced from the previous annotations. Making them
current requires re-running `prepare-papers` into a fresh bundle and re-running the
comparison against it. Until that happens, treat this report's tables as describing the
pre-fix labels, and expect `paper-l01` to move: adding a valid route can only leave a
score equal or better, so the comparison's floor rises slightly.

Independent review should confirm both fixes rather than inherit them. They were proposed
by an agent of the same model family that authored the original labels, which is precisely
the correlation the review exists to break.

**Smaller open items.** The five draft two-hop questions still need a shortcut check; a
heuristic sweep found no unannotated alternative route for any of the 8 multi-document
questions, which is evidence but not proof. The fusion sign flip is unexplained. Nobody
has checked whether any hop count is *under*-stated — the cross-check could only report
labels as correct or too high, so that direction went unexamined across all 40 questions.
M12 ablations, M13–M14 polish, charts, and the optional demonstration UI are untouched.

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
