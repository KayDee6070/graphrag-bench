# The held-out question set: how it is built and what it is worth

Written for a reader deciding how much weight to put on any result reported against this
set. It describes the authoring protocol, the blinding that was and was not achieved, and
the specific ways the set could still be biased.

The 40 development questions have a known weakness: they were authored by an AI agent, and
every one of them was visible to the agents, the retrieval runs, and the project owner
before any number was reported. Nothing measured on them can be called held out. This set
exists to fix that, and it only works if the protocol below is followed exactly.

## The protocol

**Authored by agents blinded to results.** Each drafting agent starts with no project
context and is given read access to three things only: the pinned papers' full text, the
page-aware chunks, and the catalog that maps `paper_id` to title and arXiv version. It is
explicitly forbidden from reading `reports/`, `experiments/runs/`, `docs/`, the README, the
existing questions, or any file named for a report, summary, comparison or diagnostic.

The point is not that an agent reads better than a person. It is that an agent with no
access to the findings **cannot** shape questions around them, and that is checkable from
its instructions rather than taken on trust. The project owner and the orchestrating agent
have both seen every result, so neither could author this set credibly.

**Two strata, written independently.** One batch of two-paper bridge questions, where the
reader must learn an intermediate entity from paper A to know what to look for in paper B.
One batch of single-paper factual lookups as a control. Separate agents, so neither batch's
author saw the other's.

**Mechanically verified before acceptance.** Every draft passes
`scripts/verify_draft_questions.py`, which confirms each quote exists in the named paper,
occurs exactly once there, and sits on the declared page; that some run of adjacent chunks
can satisfy each fact; that `reasoning_hops` equals the facts in the smallest sufficient
set; that `required_document_count` equals the papers that set spans; and that no
identifier collides with the drafts or with the 40 already shipped.

**Sealed before evaluation.** Questions are written, verified, and committed. Nothing is
run against them until the result is reported once. The moment a number from this set is
used to choose a configuration, fix an annotation, or tune a parameter, the set stops being
held out and a fresh one is required.

## What this does not achieve

**It is not independent human annotation.** The questions are agent-authored. The blinding
addresses *results* contamination, not authorship quality. A reader who distrusts
agent-authored labels should discount this set accordingly, and the honest description is
"agent-authored, results-blinded, mechanically verified" rather than "held out" alone.

**Mechanical verification does not check whether a quote proves its statement.** That is
the one defect class that needed human judgement on the development set, and it still does
here. Two term-overlap heuristics were tried and rejected; `scripts/verify_questions.py`
records why. A spot-check of a sample is the mitigation, and a sample is not a review: if a
systematic bias runs through the whole set, reading ten of eighty will probably miss it.

**The corpus is shared with the development set.** These questions are new, but they are
drawn from the same 30 papers the retrieval system has been developed against. Held-out
*questions* over a familiar *corpus* is weaker than a held-out corpus, and some of the
system's behaviour may already be fitted to these documents.

**Agents share a model family with the annotation author.** The development annotations were
written by an AI agent whose identity the repository does not record. If it belongs to the
same family as the drafting agents, their blind spots correlate, and a defect pattern in
the 40 may reappear here. The blinding prevents results contamination; it does nothing
about shared priors.

## Current state and the promotion decision

`datasets/papers/research/held-out-drafts.json` holds **80 verified questions**, written by
four separate results-blinded agents in four strata:

| Stratum | n | Shape |
| --- | ---: | --- |
| `held-a` | 20 | two-paper bridge: paper A names a component, paper B states its property |
| `held-b` | 20 | single-paper lookup |
| `held-c` | 20 | two-paper comparison: both facts independently findable, no hidden bridge |
| `held-d` | 20 | single-paper lookup drawn from body text rather than abstracts |

40 questions at one hop, 40 at two. All 80 pass every mechanical check, each verified by
this session independently rather than on the drafting agent's word. **Nothing has been run
against them.**

It is deliberately not an `AnnotationBook`. `prepare-papers` rejects any annotation file
declaring `split = "test"`, and that guard exists to stop dev-era labels being relabelled
as held-out — exactly the move a held-out set must not make casually. Promoting this draft
therefore needs a deliberate choice:

- widen `AnnotationBook.split` and require an explicit opt-in on `prepare-papers`, keeping
  the guard's intent while allowing a genuine held-out book; or
- accept them as a second `dev` book and report them as a separate stratum, which is more
  honest about the authorship but gives up the `test` label.

Either is defensible. Neither should be chosen by an agent on the owner's behalf, because
the guard is the mechanism that makes a `test` label mean anything.

### Known weak points, self-reported by the drafting agents

Recorded because a spot-check should start here rather than reading at random:

- **`held-c-20`** is the loosest pairing in the comparison batch: SBERT pools a sentence
  embedding while M3-Embedding describes a query's dense representation. Both are "how one
  dense vector is derived", which is arguably fair and arguably a stretch.
- **`held-c-03`** compares a dense embedding model against BM25. Both papers state their own
  retrieval model, so the question is sound, but the two answers are different kinds of
  object.
- **`held-c-02`** quotes GraphRAG saying "when using GPT-4 as the LLM" while the paper
  elsewhere mentions a `gpt-4-turbo` endpoint. A grader expecting the exact endpoint could
  quibble.
- **`held-a-07`** requires mapping an ordered dataset list onto an ordered list of point
  gains. Exact, but the grader must do that mapping.
- **`held-a-10`, `held-a-11`, `held-a-12`** identify the bridge target through an
  author-year citation string printed in paper A. Genuine bridges, but a model with strong
  parametric knowledge might shortcut without reading paper A.
- **Several quotes carry PDF damage verbatim** — `B/e.sc /r.sc/t.sc` for BERT in
  `held-c-05`, `text-embeddi ng-ada-002` in `held-a-18`, `INSTRUCT OR-Base` in `held-b-09`.
  Byte-exact and unique, but any downstream normaliser that rejoins hyphenation or strips
  ligatures will break the exact-match assumption.

Papers each stratum avoided for stated reasons are listed in the commit history: FlashRAG
was dropped from the comparison batch entirely because it reports 38 and 32 benchmark
datasets in different places, and several parameter counts were avoided for the same
inconsistency reason.

## Using the set

Report it as its own split, never pooled with the 40. State the stratum breakdown, because
bridge and single-paper questions behave very differently and an aggregate over both hides
that. Report it once.

If a result on this set disagrees with the development result, the held-out number is the
one to believe, and the disagreement itself is worth reporting as the more interesting
finding.

## A corpus observation, checked and dismissed

A drafting agent reported that `documents.jsonl` "appears to have been assembled from
overlapping chunks", because many sentences occur twice and so fail the quote-uniqueness
rule. Its practical handling was correct — it selected only sentences occurring exactly
once — but the inference about the pipeline was wrong, and it is worth recording so the
observation does not resurface as a suspected bug.

Checked across all 30 papers: **every one of the 6,996 chunks has text identical to the
document slice at its own offsets**, and each document's length matches its last chunk's
end. Documents are authoritative; chunks are overlapping views of them, which is the
intended design.

The repeated sentences are genuine repetition inside the papers: appendices restating
examples, figure captions reprinted near their figure. Corpus-wide, 99 of 11,641 sentences
longer than 60 characters are redundant repeats, 0.9%. That is a property of academic PDFs,
not of the ingestion.

It does constrain question authoring, which is why the drafting prompts warn about it: a
quote must occur exactly once for `prepare-papers` to resolve it, so roughly one sentence
in a hundred is unusable as evidence no matter how good it reads.

## Measured grounding, and a third failure of the same heuristic

The drafting prompts warned explicitly about the one defect class that needed human
judgement on the development set: a statement naming an entity its quote never mentions.
That warning measurably worked. Applying the same term-overlap check to both sets:

| Set | Statements whose distinctive terms are absent from their own quote |
| --- | ---: |
| 40 development questions | 35 of 51 facts (69%) |
| 80 held-out questions | 57 of 120 facts (48%) |

Excluding each fact's own `paper_id` — because a statement worded "the method in the
pinned `jina2` paper" legitimately contains a token the quote will never carry — leaves 25
of 80 flagged.

Reading those 25 shows they are **not defects**. They are PDF extraction artifacts:

- `held-b-08` statement says `E5-small`, the extracted quote says `E5 small, E5base` because
  the PDF lost subscript formatting
- `held-b-09` statement says `INSTRUCTOR-Base`, the quote says `INSTRUCT OR-Base` because the
  extractor split a small-caps word
- `held-a-18` statement says `text-embedding-ada-002`, the quote says `text-embeddi ng-ada-002`
  with an injected space

In each case the quote plainly proves its statement to a human reader. This is the third
distinct way this heuristic fails — after "papers call their own method *the model*" and
"questions name a model the paper refers to obliquely" — and it closes the question of
whether grounding can be checked mechanically. It cannot. `scripts/verify_questions.py`
records the first two failures; this is the third.

**The practical consequence for a spot-check:** ignore hyphenation, spacing and case
mismatches between a statement and its quote. Every one examined so far has been extraction
damage rather than a labelling error. Spend the attention on whether the quote's *claim*
matches the statement's claim.
