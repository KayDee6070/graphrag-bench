# Reviewing the 40 draft question annotations

Written for the person doing the review. You do not need to know how retrieval works.
You need to read English, open a linked PDF page, and be willing to write "this quote
does not prove that."

All 40 questions in `datasets/papers/research/` were authored by an AI agent reading
paper text. Nobody else has checked them. Until someone does, every measured result in
this repository stays labelled a development diagnostic. That check is this document.

## Who may do this

Anyone who did not author the annotations. The existing labels were written by an agent,
so a project owner reading the quotes fresh is a genuine second judgement and may
truthfully attest `independent_of_annotation_author: true`.

**Record it honestly if you have already seen the comparison results.** Knowing that all
13 multi-document questions failed creates quiet pressure to revise exactly those 13. A
results-aware review is worth far more than no review and less than a blind one. Write
which you did in your notes; do not imply the stronger claim.

## Set up

```bash
.venv/bin/graphrag-bench export-paper-review datasets/processed/m10-expanded-03 \
  --output experiments/runs/m11-question-review.json
.venv/bin/python scripts/blind_review_sheet.py \
  datasets/processed/m10-expanded-03/review.md \
  --output experiments/runs/m11-question-review-blind.md
```

The first command writes 40 `pending` rows; it refuses to overwrite an existing file, so
your decisions can never be clobbered by re-running it. The second removes the original
author's reasoning from the worksheet, keeping every question, reference answer, declared
label, exact quote, and page link. Blinding removes persuasion, not information.

Read the blind worksheet. Record decisions in the JSON file.

## The five judgements

For each question, in file order:

1. **Source support.** Open the linked page. Does the quoted text actually prove the
   stated fact? Not approximately — actually. Sets `source_checked` and
   `answer_supported`.
2. **Completeness.** Could someone hold every listed fact and still not answer the
   question? Then the evidence set is incomplete. Sets `evidence_complete`.
3. **Shortcuts.** Does somewhere else in the 30 papers answer this question directly, in
   one step? If so the question is not really multi-hop. This is the slowest check and the
   most valuable one. Sets `alternatives_checked`.
4. **Hop count.** Is the declared `reasoning_hops` right, or too high? The worksheet's
   original screening was done by the same agent that wrote the labels, which is not
   proof. Sets `reasoning_hops_checked`.
5. **Document count.** Is `required_document_count` right? Sets
   `document_count_checked`.

Then set `decision` to `approved` or `revise`, and fill `reviewer`, `reviewed_on`, and
`notes`. Approval requires **all six checks true and the independence attestation**; the
schema rejects an approval that is missing either. A `pending` row must stay completely
empty — the schema also rejects half-filled pending rows.

## Habits that protect the result

- Go in file order. Do not start with the questions you know failed.
- Do not open `report.md`, `summary.json`, or the diagnostic while reviewing.
- Finish all 40 and save before comparing against any score.
- Judge the label, never the outcome. Lowering a hop count is correct when the label is
  wrong, and corrupt when the reason is that nothing retrieved it. If you could not have
  given the reason before seeing a score, it is not a reason.

## Finish

```bash
.venv/bin/graphrag-bench verify-paper-review datasets/processed/m10-expanded-03 \
  --review experiments/runs/m11-question-review.json
```

This validates that your rows cover exactly the frozen bundle, that every decided row
carries a reviewer, date, and notes, and that no approval is missing a check. It reports
`reviewer_identity_authenticated: false` always: it validates **self-reported**
attestations and cannot confirm who you are or that you were genuinely independent.

It also reports `held_out_questions: 0`. Review does not create held-out data. These 40
questions have been visible throughout and stay development labels no matter how the
review turns out. The 80 held-out questions are separate work, and should be written
after this review so they do not inherit whatever it finds.

Expect roughly 2 to 3 hours. Any `revise` decision is a useful result, not a failure —
especially among the 13 multi-document questions, where a mislabelled hop count would
change what the comparison means.
