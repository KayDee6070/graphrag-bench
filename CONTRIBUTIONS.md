# Who did what

This project was built with heavy use of AI coding assistance, which is worth stating
plainly rather than leaving a reader to guess. What follows is the division of labour,
because on a benchmark the question "who decided this" matters more than "who typed it".

## Kuntal Dive — project owner

**The research question and the design.** The hypothesis, the choice to measure retrieval
independently of answer generation, the shared-evidence-budget comparison, the milestone
structure, and the rule that gold labels never become query seeds, extraction inputs, or
prompt hints.

**The definitions.** `reasoning_hops` counts the facts that must be combined, via the
smallest sufficient evidence set. That convention did not exist until it was decided here,
and it relabelled four questions. A model cannot settle what a benchmark measures; see
[annotation-review.md](docs/annotation-review.md).

**The annotation decisions.** Eleven of the forty development questions carry a recorded
decision with a stated reason, including whether Contriever-MS MARCO is close enough to
base Contriever for a question about the retriever family, and how to label a paper that
reports both 38 and 32 benchmark datasets. Records:
[owner-review-decisions.json](datasets/papers/research/owner-review-decisions.json).

**Three defects nothing automated caught.** A spot-check of ten held-out questions found six
needing revision. Three were genuine labelling errors that four drafting agents and two
mechanical verifiers had all passed:

- `held-c-01` — the GraphRAG quote stops before the clause saying what Leiden partitions.
- `held-c-20` — the M3 quote covers only queries and never says "dense".
- `held-a-10` — the evidence proves two of the three things the question asks.

Each is the same shape: a quote that is true but does not prove the specific claim. That is
the check two separate attempts at automation failed to reproduce, and the reason the
spot-check was worth doing. Records:
[held-out-spot-check.json](datasets/papers/research/held-out-spot-check.json).

**The judgement calls with consequences.** Accepting a weak extractor and reporting it as a
limitation rather than blocking the study; moving extraction to the GPU; and declining to
repair the six defective held-out questions once their scores were known.

## AI assistance

**Implementation.** Most of the code, tests and documentation were written with an AI
assistant working from the owner's direction and reviewed against the test suite, the
linter and offline verification.

**The held-out questions.** All 80 were drafted by agents given the papers, the chunks and
the catalogue, and explicitly denied access to every report, result and existing question.
**This is deliberate and must not be presented otherwise.** The owner and the orchestrating
assistant had both seen every measured result, so neither could author a held-out set
credibly. An author who cannot see the findings cannot shape questions around them, and
that is checkable from the instructions rather than taken on trust. It is the entire
argument for why those 80 questions are worth anything.

The questions are therefore agent-authored and results-blinded, not independently
human-authored, and [held-out-questions.md](docs/held-out-questions.md) lists what that is
and is not worth. Seventy of the eighty have had no human read them.

**An external model review was attempted and rejected.** A different model family was asked
to review the annotations and returned 35 identical high-confidence verdicts, missing all
three defects named in advance. It is recorded as a failed attempt in
[annotation-crosscheck.md](reports/annotation-crosscheck.md) and no row of it entered any
decision file.

## What this means for the results

Every number reproduces offline from saved artifacts, independently of who wrote the code.
`verify-paper-benchmark` recomputes the metrics, the context selection and the report from
the saved records without a model, and every comparison in this repository passes it. That
property is what the results rest on — not the authorship of the pipeline.

The open requirement is unchanged and is stated wherever results appear: independent human
review of the annotations. Reviewer tooling exists and is unused by anyone outside this
project.
