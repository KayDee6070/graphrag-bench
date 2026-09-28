# M8 local extraction: observations and limits

Date: 2026-09-28. Package: `graphrag-bench 0.8.0`.

M8 adds real local language-model inference for entity and relation extraction, source-reference checks, saved inference receipts, and offline replay. On three fictional development passages, the final configuration retained the two expected affirmative relationships and rejected the model's response to a negated statement. This is an integration check, not a real-paper benchmark or an extraction-accuracy estimate.

For the beginner-friendly explanation and commands, read [the M8 lesson](../docs/llm-extraction.md). For a short demonstration that does not load a model, run `.venv/bin/python scripts/study_m8.py`.

## What ran

| Setting | Recorded value |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Immutable revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Weight file size | 988,097,824 bytes |
| Device / precision | CPU / float32 |
| CPU threads | 4 |
| Decoding | Greedy, one beam, repetition penalty 1.0 |
| Limits | 4,096 input tokens; 768 generated tokens |
| Python / PyTorch | 3.12.3 / 2.14.0+cpu |
| Transformers / tokenizers | 5.17.0 / 0.23.2 |
| huggingface-hub | 1.33.0 |
| Pydantic / NetworkX | 2.13.5 / 3.6.1 |
| Corpus | Three original fictional passages, one chunk each |

The user approved the free model download. The final inference runs used cached weights with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. No hosted inference or paid API was used. Model information: [publisher's model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

The ontology and three independently written prompt examples are in [the configuration](../configs/llm-extraction.toml). The examples use Helix, Comet, Tundra, and Marsh; the development passages use Orion, Nova, Harbor, and Mirage. Benchmark questions, answers, and gold graph annotations are not inputs to extraction.

## Actual final output

| Source passage | Model proposal | Validation outcome |
| --- | --- | --- |
| The Orion method uses the Nova model as its backbone. | Orion (`Method`) `USES` Nova (`Model`) | Retained with the exact source quote |
| The Nova model was evaluated on the Harbor dataset. | Nova (`Model`) `EVALUATED_ON` Harbor (`Dataset`) | Retained with the exact source quote |
| The Nova model was not evaluated on the Mirage dataset. | Nova `NOT EVALUATED ON` Mirage | Entire response rejected: predicate violates the identifier schema |

Final counts: **3 entities, 2 assertions, 1 issue, 1 rejected chunk**. Both retained assertions have `review_status: unreviewed`.

The negated passage did **not** produce the requested empty relation list. Its actual relation object was:

```json
{
  "subject": "e1",
  "predicate": "NOT EVALUATED ON",
  "object": "e2",
  "quote": "The Nova model was not evaluated on the Mirage dataset."
}
```

`NOT EVALUATED ON` contains spaces and is not an allowed predicate. The initial schema check rejects the response before ontology checking. The issue code is `invalid_json`, which currently covers both JSON parsing errors and response-schema failures. In this case the JSON syntax is valid; the schema is not.

Do not interpret this rejection as proof that the software understands negation. A wrong positive `EVALUATED_ON` proposal with both names and a real negated quote could pass the mechanical checks. The study script and tests demonstrate the related case of a reversed relationship with a real quote. Semantic interpretation still needs independent review.

## Fresh inference and replay are different checks

Two final runs, `experiments/runs/m8-final-02` and `experiments/runs/m8-final-03`, each called the local model for all three chunks **without a response cache**. Their requests, generated text, token counts, stopping reasons, graph bytes, and issue bytes matched. Their elapsed times differed, as expected. Every response ended at an end-of-sequence token; none hit the generation limit.

| Passage | Input tokens | Output tokens | Run 02 generation call | Run 03 generation call |
| --- | ---: | ---: | ---: | ---: |
| Nova / Harbor | 905 | 89 | 16.994 s | 15.758 s |
| Orion / Nova | 906 | 86 | 15.089 s | 15.223 s |
| Nova / Mirage, negated | 906 | 90 | 16.119 s | 15.542 s |

These durations include prompt formatting, tokenization, generation, and decoding within the provider call. They exclude model loading, ingestion, graph construction, and artifact writing. They are observations from one machine, not a latency benchmark or a guarantee across hardware and library versions. Output counts include the generated end-of-sequence token.

`experiments/runs/m8-final-replay` then rebuilt run 02 without loading a model. Its graph, issues, and response receipts were byte-identical to run 02. Its manifest records `execution_mode: replay`; replay keeps the original inference durations and does not measure a new generation.

The existing graph CLI also retrieved both affirmative source passages for:

> Which dataset evaluates the model used by Orion?

The returned evidence included Orion's use of Nova and Nova's evaluation on Harbor. This verifies graph retrieval across the shared Nova entity. It does not generate a natural-language answer; that remains M9.

## Development failures are part of the result

The prompt was developed against these three passages, so the passages are not held-out evaluation data.

1. The initial prompt supplied a fuller schema without demonstrations. All three responses failed schema validation; examples of failures included copying ontology definitions into the output and using undeclared type descriptions.
2. A simpler response-format prompt without demonstrations still rejected all three responses. The model invented relation names containing spaces.
3. Three independently authored examples improved the two affirmative passages. The negated passage still produced the invalid predicate shown above.
4. The provider was corrected to replace the model publisher's generation defaults with the explicitly recorded local configuration. The two final runs above use that configuration.

Earlier development runs remain in ignored local experiment directories. Their prompts and intermediate manifest formats differ from the final implementation; they are diagnostic history, not supported final replay fixtures. The final runs retain the full requests and raw responses, including the rejected one.

## Local evidence and verification

Run 02 uses `datasets/processed/m8-local-input`. Its graph, issues, responses, and manifest are in `experiments/runs/m8-final-02`. Run 03 and the replay have separate directories; no run was overwritten.

| Run 02 record | SHA-256 |
| --- | --- |
| Extraction recipe | `772c88c61e580da5f46096b722195643ab6cf6d0da8c9900b879de1e887a9bdb` |
| Package Python sources | `23871e294dff2babdc4f73cdcc6910019feafd5e62f2d48f34e30104392e968e` |
| `graph.json` | `b50fdcf9b0b0ffb30dc8442e68098d285f61e329017557844979d13d7c495425` |
| `issues.jsonl` | `29e852d733ccb61dff2b54753af5a2045738e91bf31c1469b41f4b557842ff73` |
| `responses.jsonl` | `0a01682102d942e06060eaf19f4de1a8bb5b74b758a501ce323275d790d5355d` |

These are consistency checks, not signatures proving who produced the files. Generated artifacts and model weights stay outside Git. This report and the fictional source passages are tracked.

The example passages now live in `datasets/examples/llm-extraction/texts/` so ingestion does not include their explanatory README. A fresh ingestion from that directory produced the same document and chunk bytes as the validation corpus. Its ingestion manifest differs because the source directory is recorded. To replay the recorded final run, use its original `m8-local-input` ingestion artifacts; to perform a new experiment, follow the fresh-directory commands in the lesson.

Completed software checks:

- Full suite: **422 tests passed**, including 57 M8 tests. These test software behaviour using prewritten responses and stubbed neural calls; they do not measure model accuracy.
- Ruff lint and formatting passed.
- The offline M8 lesson passed its assertions.
- Both fresh local inference runs, offline artifact replay, and graph CLI retrieval passed the checks described above.
- Version 0.8.0 built as a wheel and source distribution. The source distribution includes the configuration, passages, lesson, report, and tests; neither distribution includes model weights or generated run artifacts.
- The wheel was installed into a temporary directory and exercised outside the repository. Replay and graph CLI retrieval passed with imports of `torch`, `transformers`, and `sentence_transformers` deliberately blocked.

## What this does and does not establish

M8 establishes a working local extraction path with explicit configuration, source alignment, inspectable failures, integration with existing graph/hybrid retrieval, and model-free replay. The automated hybrid integration test uses controlled embedding and extraction providers; the actual-model smoke check above exercises graph retrieval.

The model is small and remains unreliable even on this tiny development set. Real research papers, held-out extraction annotations, entity-resolution errors, precision/recall, retrieval comparisons using these graphs, and answer-quality evaluation remain future work. The existing M7 retrieval benchmark continues to use its deterministic extractor. M9 and later milestones have not been started by this work.
