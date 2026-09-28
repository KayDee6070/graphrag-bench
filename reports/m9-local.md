# M9 local answers: working provenance, limited model reliability

Date: 2026-09-28. Package: `graphrag-bench 0.9.0`.

M9 connects the four existing retrievers to a shared local answer pipeline. It records evidence selection, full prompts, completions, and exact source citations, then supports offline replay. **The small example model is not reliable enough to treat these answers as verified facts.** In this controlled development pilot it invented a score absent from every source, failed to abstain on incomplete evidence, and omitted a necessary citation from a two-source answer.

These failures are retained alongside the successful direct and negated answers. A status of `answered` records an accepted model proposal with resolvable references; it is not an answer-correctness score. Read the [M9 walkthrough](../docs/answer-generation.md) or run `.venv/bin/python scripts/study_m9.py` for the Scout explanation.

## Setup

| Setting | Value |
| --- | --- |
| Answer model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Runtime | CPU float32, 4 threads, greedy decoding, one beam |
| Limits | 1,500 evidence tokens; 4,096 total input tokens; 768 generated tokens |
| Python / PyTorch | 3.12.3 / 2.14.0+cpu |
| Transformers / tokenizers | 5.17.0 / 0.23.2 |
| huggingface-hub / Pydantic | 1.33.0 / 2.13.5 |
| Source corpus | The three original fictional M8 passages |
| Graph | The verified M8 local model-extracted graph, with 3 entities and 2 unreviewed assertions |

The already approved and cached M8 model was reused. No additional model download or paid API was used. Inference ran with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. The [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) describes the publisher's model; this pilot assesses only the local integration and observed responses.

The exact input ingestion is `datasets/processed/m8-local-input`; the graph is `experiments/runs/m8-final-02`. The source passages are tracked in `datasets/examples/llm-extraction/texts/`. New generation uses [configs/generation.toml](../configs/generation.toml). The prompt's fictional Helix/Comet/Tundra examples are independently written development examples, not benchmark labels.

## Final prompt: observed cases

The following cases use `experiments/runs/m9-reviewed-<case>`. The directory name refers to this report's inspection; the persisted answer claims still carry `review_status: unreviewed`. The table is a manual development diagnostic, not a held-out evaluation.

| Case | Question and retrieval | Observed response | Assessment |
| --- | --- | --- | --- |
| `bridge` | Which dataset evaluates the model used by Orion? Graph, K=5 | Names Harbor; cites only the Nova/Harbor passage | Answer names the expected dataset, but citation support omits the Orion/Nova connection |
| `missing` | Same question. Graph, K=1 supplies only the Orion/Nova passage | Restates that Orion uses Nova; status `answered` | Does not answer the dataset question and fails to abstain |
| `unknown` | What was Orion's exact score on Harbor? Graph, K=5 | Invents `0.99`, citing both real passages | Unsupported answer passes exact-reference checks |
| `negation` | Was Nova evaluated on Mirage? BM25, K=5 | Says Nova was not evaluated on Mirage; quotes the negated passage | Correct on this development example |
| `empty` | What is Zorbax? Graph, K=5 retrieves nothing | `insufficient_evidence`; completion is null | Software abstention, with no generation call; not evidence of model abstention skill |
| `direct` | Which dataset was Nova evaluated on? Graph, K=5 | Says Harbor; quotes the Nova/Harbor passage | Correct on this development example |

The `unknown` response included this claim verbatim:

> Orion's exact score on Harbor was 0.99.

Its citations were exact copies of “The Nova model was evaluated on the Harbor dataset.” and “The Orion method uses the Nova model as its backbone.” Neither passage contains a score. Both citations resolve correctly, so a string/coordinate check cannot reject the conclusion on semantic grounds.

Similarly, requiring at least one citation per claim does not establish that every necessary step is cited. The `bridge` answer omitted the source connecting Orion to Nova. The implementation and study lesson explicitly preserve this distinction instead of reporting perfect “citation accuracy” from successful string matches.

The negative Mirage passage was excluded from the M8 graph because its extraction response was rejected. BM25 still retrieves it from the independent source corpus. This illustrates why generation uses original source text rather than treating graph edges as a complete or authoritative substitute.

Additional real local runs exercised vector and hybrid answering on the bridge question: `experiments/runs/m9-real-vector` and `m9-real-hybrid`. Both selected all three passages, including both necessary affirmative facts, but the answer model merely restated that Orion uses Nova. These are observed generation failures despite sufficient retrieved context, not evidence that the retrievers missed the necessary facts.

Those runs used the existing cached M4 encoder, `sentence-transformers/msmarco-MiniLM-L6-cos-v5` at revision `14ca9be4bbcf1402eac0f43a2e2ccb6e0f994ba3`, with a freshly built 384-dimensional index at `experiments/runs/m9-local-vectors`. No embedding-model download was needed. One development question per strategy is an integration smoke check, not a comparative experiment.

## Prompt development did not solve the failure

The initial prompt included one two-source answer demonstration and one missing-source abstention demonstration. A trial added stronger instructions to answer the requested value, never invent numbers, and cite the connecting source, plus a separate missing-score abstention demonstration.

That trial still invented `0.99` and still failed to abstain with one missing report. It also regressed on the two-source question, restating the Orion/Nova fact instead of naming Harbor. The simpler original prompt was retained. Trial artifacts (`m9-final-*`) remain local diagnostics and use a different prompt; the supported final replay examples are the `m9-reviewed-*` runs.

All these questions were used during development. Prompt selection therefore used development observations, not held-out evidence. The observations do not establish a general answer-quality rate or improvement over another model or retrieval strategy.

## Timing and repeatability

| Case | Evidence tokens | Full input tokens | Generated tokens | Provider call |
| --- | ---: | ---: | ---: | ---: |
| `bridge` | 41 | 436 | 57 | 6.548 s |
| `missing` | 22 | 417 | 55 | 5.347 s |
| `unknown` | 41 | 436 | 81 | 6.487 s |
| `negation` | 61 | 453 | 55 | 4.848 s |
| `empty` | 0 | No call | No call | No call |
| `direct` | 41 | 434 | 53 | 5.430 s |
| `repeat` | 41 | 436 | 57 | 5.465 s |

Provider-call durations include prompt formatting, tokenization, generation, and decoding. They exclude retrieval, model loading, artifact writing, and initial downloads. Generated counts include the end-of-sequence token. These are individual local observations, not benchmark latency estimates.

`m9-reviewed-repeat` made a fresh generation call for the same question and selected evidence as `bridge`; M9 has no response cache. The requests, generated text, token counts, stopping reasons, and structured answer matched. Retrieval and generation durations differed. This is a repeat on one machine, not a cross-hardware determinism guarantee.

`m9-reviewed-replay` then replayed the saved ranking, budget decisions, prompt reconstruction, and citation checks. All five artifact hashes matched `bridge` exactly; the new manifest records replay mode. Replay kept the original completion/timings and did not rerun retrieval, tokenization, or generation.

## Trace the recorded result

For `experiments/runs/m9-reviewed-bridge`:

| Record | SHA-256 |
| --- | --- |
| Package Python sources | `f5ee8cbe2f3fe353e3b4186b0a479c091b4c2dfd5bacffe626624737d42cc138` |
| `request.json` | `0d9dd073855a2e8927894cb9ad31ed6f1cc96f1c0d9c4809c9740a4260cbfafe` |
| `receipt.json` | `fdaaf0a88b0c7663a35ff26c2f086744f0f34f43eee3ce10bba65278303cfb39` |
| `answer.json` | `8604ab47a0ded437e237ab0125e3fd1771959e420a5d5635d112817c59120e8a` |
| `answer.md` | `2dd77dc5e3fc36cb30f1911f9ce78c78f140769ea28d560629d235fa1b12e50c` |
| `retrieval.json` | `a3696f1c4d74c05acf87d6ce5619671cb2db9d9d51fd5300eec8fca7cd669b9e` |

These are consistency checks, not cryptographic signatures authenticating the experiment. Generated runs and weights remain outside Git. This report, configuration, code, tests, and study materials are tracked.

## Software validation and remaining work

- **475 tests passed**, including 51 new generation tests and two additions for the shared local provider. The original M8 and M7 tests still pass.
- All four retrieval strategies were exercised through the answer CLI with controlled providers, followed by model-free replay. These tests verify integration, not model quality.
- All four also completed actual local answer calls using cached models: graph and BM25 in the case table, and vector/hybrid in the additional smoke checks above. Their saved runs passed source/citation replay verification.
- Tests cover malformed/duplicate JSON, unknown sources, invented/wrong/ambiguous quotes, Unicode offsets, overlap removal, exact evidence-budget boundaries, empty context, output limits, inconsistent receipts, corrupted artifacts, and output protection.
- The executable M9 Scout lesson passed. Ruff lint and formatting passed.
- The 0.9.0 wheel was installed outside the repository and replayed the real answer run with neural imports deliberately blocked. All five output artifacts matched the original.

M9's generation/provenance pipeline is implemented. Independent answer correctness, citation entailment/completeness, reliable abstention, and comparative quality on real papers are not established. M10's real-paper corpus and subsequent comparative work have not started. No remote push was made.
