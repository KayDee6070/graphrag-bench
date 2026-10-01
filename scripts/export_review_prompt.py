"""Build a paste-ready prompt asking an external model to judge annotation quotes.

Produces batches small enough to answer reliably. Each batch contains the blind
worksheet text for its questions and asks for one compact JSON line per question, so the
replies can be checked and applied without re-reading the worksheet.

This is for readings, not decisions. Conventions such as what reasoning_hops counts are
project design choices and are stated here as given, not put up for judgement.
"""

import argparse
import importlib.util
from pathlib import Path

HEADER = """You are checking draft research annotations for a retrieval benchmark.

For each question below you are given: the question, the reference answer, declared
labels, and one or more facts. Each fact has a statement and an exact quote from a named
paper and page. Every quote has already been machine-verified as real text on that page,
so do not re-check that. Judge only the reasoning.

Answer these four things per question:

1. PROVES: does each quote actually prove its statement? Be strict. "Topically related"
   or "implies with background knowledge" is NOT proof. Flag a quote whose subject is
   implicit, that is about a different model or version than the statement claims, or
   that is a figure or table fragment whose meaning is unclear out of context.
2. ANSWERABLE: holding every listed fact and nothing else, can the question as worded be
   answered? If something is missing, say what.
3. HOPS: this project defines reasoning_hops as THE NUMBER OF FACTS THAT MUST BE
   COMBINED, counted via the smallest sufficient evidence set. That definition is fixed;
   do not argue with it. Just say whether the declared number matches it.
4. CONFIDENCE: high, medium or low. Low is a useful answer. Guessing is not.

Reply with ONE JSON object per line and nothing else. No preamble, no markdown fence:

{"question_id":"...","proves":"ok|weak|fail","answerable":"yes|no","hops":"ok|wrong",\
"confidence":"high|medium|low","concern":"one sentence, or empty string"}

Use "weak" when a quote needs a leap to support its statement, and "fail" when it does
not support it at all. If you are unsure, say so in "concern" rather than picking a
confident answer.

"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worksheet", type=Path, help="blind review worksheet")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip", help="comma-separated question IDs already decided")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location(
        "review_questions", Path(__file__).parent / "review_questions.py"
    )
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)

    skip = {q.strip() for q in args.skip.split(",")} if args.skip else set()
    parts = {k: v for k, v in sorted(review.sections(args.worksheet).items()) if k not in skip}
    if not parts:
        raise SystemExit("no questions left after --skip")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    ids = list(parts)
    batches = [ids[i : i + args.batch_size] for i in range(0, len(ids), args.batch_size)]
    for number, batch in enumerate(batches, start=1):
        body = "\n\n".join(parts[qid] for qid in batch)
        path = args.output_dir / f"batch-{number:02d}.txt"
        path.write_text(f"{HEADER}Questions in this batch: {len(batch)}\n\n{body}\n", "utf-8")
    print(
        f'{{"questions": {len(parts)}, "batches": {len(batches)}, '
        f'"skipped": {len(skip)}, "output": "{args.output_dir}"}}'
    )


if __name__ == "__main__":
    main()
