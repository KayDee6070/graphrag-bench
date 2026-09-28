# M8: a local language model proposes the graph

M8 adds an optional local language model to propose entities and relationships. It saves the complete requests and responses, validates the proposals against the source, and builds a graph that the existing graph and hybrid retrievers can use. The deterministic M3 extractor remains a separate baseline.

## Start with the Scout story

Imagine a junior Scout reading mission reports and drafting a map of names and connections. M3 gave that Scout a strict set of sentence templates. A sentence written differently could be missed. M8 asks a language model to draft the map from less rigid language.

The Scout must attach a receipt to every connection: an exact quote from the report. Our software checks those receipts and the shape of the proposed map. A reviewer must still check whether a quote actually supports the connection. A real quote can be misinterpreted.

M8's language model is being used to extract a graph. Writing an answer to a user's question remains M9 work.

## A concrete example

Source one:

> The Orion method uses the Nova model as its backbone.

Source two:

> The Nova model was evaluated on the Harbor dataset.

The intended map has Orion `USES` Nova and Nova `EVALUATED_ON` Harbor. This is a teaching example; a model's actual output may differ. The [local observation report](../reports/m8-local.md) records actual responses and failures.

The model returns proposals in this form:

```json
{
  "entities": [
    {"id": "e1", "name": "Nova", "entity_type": "Model"},
    {"id": "e2", "name": "Harbor", "entity_type": "Dataset"}
  ],
  "relations": [
    {
      "subject": "e1",
      "predicate": "EVALUATED_ON",
      "object": "e2",
      "quote": "The Nova model was evaluated on the Harbor dataset."
    }
  ]
}
```

`e1` and `e2` are temporary names within one response. Our code assigns stable graph IDs after validation. It computes character positions from the quote instead of asking the model to count characters.

## Three things to keep separate

1. **Valid format:** Is the output one JSON object with the required fields and no unexpected fields?
2. **Valid source reference:** Do the names and quote occur in the source at the claimed location?
3. **Valid interpretation:** Does the source actually support the proposed relationship, direction, and types?

The implementation checks the first two and validates ontology constraints. It does not establish the third. Every retained assertion is marked `review_status: unreviewed`. This marker is audit metadata; it does not filter the graph out of retrieval. Retrieval using these graphs is experimental until the extraction quality is independently reviewed.

For example, a model could propose Nova `USES` Orion while quoting “Orion uses Nova.” Both names and the quote exist. The direction is still wrong. The offline lesson deliberately shows this limitation rather than hiding it behind a confidence number.

## Run the short offline lesson

From the repository root:

```bash
.venv/bin/python scripts/study_m8.py
```

It uses explicitly handwritten responses to demonstrate a valid source reference, an invented quote, malformed output, and a wrong interpretation with a real quote. It needs neither model weights nor neural dependencies. It is a validator lesson, not a model-quality result.

## Configure the model and ontology

[configs/llm-extraction.toml](../configs/llm-extraction.toml) records the model ID and immutable revision, CPU settings, token limits, allowed entity types, directed relation types, and independently authored fictional prompt examples. The examples demonstrate the response format and abstention on a negated statement. They are development material and must never be constructed from held-out benchmark questions or answers.

The provided small model is `Qwen/Qwen2.5-0.5B-Instruct`, pinned to `7ae557604adf67be50417f59c2c2f167def9a775`. Its weights are approximately 988 MB. It is a practical integration example, not a model selected for proven extraction accuracy. See the [publisher's model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and [weight file](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/model.safetensors).

The provider uses CPU float32, four CPU threads by default, greedy decoding, one beam, and an explicit output limit. Inputs include the complete chat template, instructions, ontology, examples, and source passage. An oversized prompt raises an error; no source text is silently truncated. An output that exhausts its generation limit is saved but rejected, even if its text happens to parse as JSON.

The tokenizer's chat template formats the messages. We tokenize that rendered string without adding a second set of special tokens. See the [Transformers chat-template guidance](https://huggingface.co/docs/transformers/main/chat_templating).

## Run actual local extraction

The optional CPU dependencies are shared with M4. If they are not installed, follow the CPU setup in the [README](../README.md), using `requirements-embeddings-cpu.txt`. Core tests and graph replay do not require this optional stack. Installation and the initial model download use the network; inference runs locally and makes no paid API call.

```bash
.venv/bin/graphrag-bench ingest datasets/examples/llm-extraction/texts \
  --config configs/sentence-fixture.toml \
  --output datasets/processed/my-m8-input

.venv/bin/graphrag-bench build-llm-graph datasets/processed/my-m8-input \
  --config configs/llm-extraction.toml \
  --output experiments/runs/my-m8-graph \
  --response-cache experiments/runs/my-m8-cache
```

The model is loaded from the local cache by default. If you choose to download the pinned model, add `--allow-download` explicitly to `build-llm-graph`. An optional `--cache-folder` selects the model cache. `--response-cache` is a different directory: it holds requests and generated responses, not model weights.

Always choose a fresh output directory. A failed provider call aborts the run; it is not silently replaced with deterministic extraction or an empty successful response. Completed requests in an optional response cache can be reused on a subsequent attempt. Malformed model responses instead produce recorded per-chunk issues, allowing the other chunks to be processed. `built_with_issues` is a warning to inspect the saved results, not a quality score.

## Replay and query the resulting graph

```bash
.venv/bin/graphrag-bench replay-llm-graph experiments/runs/my-m8-graph \
  --source datasets/processed/my-m8-input \
  --output experiments/runs/my-m8-replay

.venv/bin/graphrag-bench query-graph experiments/runs/my-m8-graph \
  --source datasets/processed/my-m8-input \
  --query "Which dataset evaluates the model used by Orion?" \
  --top-k 5
```

Replay needs no model. It verifies hashes and source artifacts, reconstructs the expected requests, reruns proposal validation, and checks that the graph and issues match the saved outputs. It then writes a fresh graph directory with `execution_mode: replay`.

The existing `query-hybrid` command also accepts an M8 graph. Build the vector index from the same ingestion directory, then supply the M8 graph with `--graph`. The M7 `benchmark` command still builds its deterministic graph baseline internally; this milestone does not silently change that established experiment. Broader comparisons using model-extracted graphs belong to the later comparative study.

## What validation does

- Reject malformed JSON, duplicate JSON keys, duplicate local entity IDs, missing fields, unknown fields, and oversized entity/relation lists.
- Require every entity type to be declared and every entity name to occur verbatim with word boundaries in the source chunk. A name fragment does not count as a mention.
- Require relation endpoints to refer to accepted local entities. Check predicate names, endpoint types, and disallow self-relations.
- Require an exact contiguous quote occurring exactly once in the chunk and containing both endpoint names. Ambiguous repeated quotes are rejected instead of assigning an arbitrary occurrence.
- Convert chunk-relative positions to absolute Unicode character offsets and attach every source chunk containing the complete quote.
- Deduplicate the same assertion at the same document coordinates across overlapping chunks. Repeated statements at different source locations remain separate assertions.
- Merge entities only by normalized name plus type. Different types remain separate candidates. No fuzzy matching or automatic alias inference is added.

This policy is conservative about references. It can reject a correctly understood relation that uses a pronoun, a repeated quote, or an abbreviation. Conversely, it can retain a wrong interpretation with a real quote. Equal names and types can still conflate different real entities; abbreviations can split one entity into several nodes. These are explicit targets for later error analysis.

## What each run saves

| File | Purpose |
| --- | --- |
| `graph.json` | Source-checked entities, assertions, evidence, and unreviewed status |
| `issues.jsonl` | Rejected outputs or proposals with their source chunk |
| `responses.jsonl` | Exact messages, source chunks, raw completions, token counts, stopping reason, and inference duration |
| `manifest.json` | Python/package/library versions, package source hash, model, ontology, prompt examples, extraction fingerprint, counts, source/artifact hashes, and execution mode |

Request-cache keys include the complete model specification and request. Changes to source coordinates, text, examples, ontology, model revision, or recorded settings invalidate reuse. Cached malformed responses stay malformed; there is no silent retry to obtain a nicer result. Corrupt cache entries raise an error.

`execution_mode: inference` means the provider path was used, possibly with response-cache hits. Cached response timings are the durations of the original calls. They are not measurements of cache lookup speed or a new generation. A replay likewise retains the original receipts. Hash checks detect inconsistency, not malicious replacement of every artifact and hash.

Greedy decoding and pinned versions make behaviour easier to inspect, but do not guarantee fresh inference is identical across hardware or library changes. Replaying saved responses is a different and stronger determinism claim. The observation report distinguishes these checks.

The adapter replaces the publisher's generation defaults with the explicit local decoding configuration and records it. This prevents model-specific sampling or repetition settings from silently changing a supposedly controlled run. The package source hash is a provenance record; the loader does not require the entire installed package to remain byte-identical when replaying an otherwise compatible extraction recipe.

## Follow the code

| File | Responsibility |
| --- | --- |
| `extraction/llm/config.py` | Model settings, ontology, and prompt-example validation |
| `extraction/llm/contracts.py` | Proposals, messages, responses, issues, and provider interface |
| `extraction/llm/prompt.py` | Source-only requests and extraction/request fingerprints |
| `extraction/llm/provider.py` | Optional local Transformers inference |
| `extraction/llm/cache.py` | Inference-receipt reuse and checksum checks |
| `extraction/llm/extractor.py` | JSON validation, source alignment, identity, and assertion construction |
| `extraction/llm/pipeline.py` | Artifact export, verification, and replay |
| `graph/pipeline.py` | Load either a deterministic or model-extracted graph |

Paths are relative to `src/graphrag_bench/`. No gold benchmark loader is called by extraction or graph construction. The model has no tools; source text is passed as data with explicit instructions not to follow embedded commands. That prompt is a mitigation, not proof against prompt injection or bad extraction.

## Tests and exercises

Run `.venv/bin/python -m pytest tests/llm -q` for focused checks. Tests use prewritten model responses and a stubbed neural backend. They cover offsets, overlap, unknown entities, quote failures, schema errors, ambiguous names, cache invalidation, provider failure, offline replay, corrupt artifacts, and existing graph/hybrid CLI integration. Actual model behaviour is checked separately.

1. **The quote exists, but the subject and object are reversed. Is the relation proven?** No. Mechanical checks can pass; a reviewer must assess interpretation.
2. **Why let the code calculate offsets?** It can locate exact strings reliably and reject ambiguous occurrences without trusting model arithmetic.
3. **Why do we keep raw rejected responses?** They explain failures and allow inspection without another model call.
4. **Does replay measure whether the model generates the same answer twice?** No. Replay reuses one saved response. A fresh inference comparison must bypass the response cache.
5. **Can a model response contain zero relations?** Yes. Abstention is valid and useful when a passage lacks supported relations.
6. **Can we use test answers as prompt examples?** No. That leaks evaluation labels into graph construction.
7. **Why is a name plus type only a starting identity policy?** Different real things can share a name and type; one real thing can have multiple names.
8. **Is the whole GraphRAG answer system finished?** No. M8 extracts graph proposals; cited answer generation and real-paper evaluation remain later milestones.
