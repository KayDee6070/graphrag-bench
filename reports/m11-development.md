# M11 development comparison and extraction feasibility

Work began 2026-09-28 and continued through 2026-09-30 (Europe/Berlin). Packages 0.11.0–0.11.4.
The full four-method study is **incomplete**. This report separates the completed
baseline from source-only extraction diagnostics. All 40 questions are draft dev
labels pending independent review; there are zero held-out questions.

## Completed full-corpus baseline

The runner reused `datasets/processed/m10-expanded-03`: 30 pinned papers, 558 PDF
pages, 6,996 page-aware chunks, and 40 questions in 23 related groups. The vector
index contains 6,996 × 384 values. Indexing took about 203.46 seconds on the local
CPU, including the work timed by the index CLI. No model or PDF download occurred.

The embedding model was `sentence-transformers/msmarco-MiniLM-L6-cos-v5` at revision
`14ca9be4bbcf1402eac0f43a2e2ccb6e0f994ba3`. Configuration: CPU, four PyTorch threads,
batch size 16, 384-token encoder limit, empty query and document prefixes. The
adapter rejects oversized embedding inputs rather than silently truncating them.

`configs/benchmark-papers-baseline.toml` selected vector and BM25, K=5/10, a
2,000-token context budget, three repeats, and seed 0. Graph and hybrid did not run.
There are 240 question/strategy/repeat records, each containing both cutoffs.
Rankings and evidence results were stable across repeats. All draft evidence was
reachable when the entire chunk collection was supplied to the coverage scorer.

| Method | K | Complete evidence after budget | Mean fraction of fully covered facts |
| --- | ---: | ---: | ---: |
| Vector | 5 | 12/40 = 30.0% | 33.33% |
| BM25 | 5 | 22/40 = 55.0% | 61.25% |
| Vector | 10 | 19/40 = 47.5% | 50.83% |
| BM25 | 10 | 25/40 = 62.5% | 68.75% |

No chunks were skipped for the token budget in this run, so raw and budgeted
evidence scores match. Mean context size was about 720–725 tokens at K=5 and
1,440–1,465 at K=10. These use the embedding tokenizer to measure the whole rendered
context; the context is not passed through the embedding encoder.

Both methods retrieved complete annotated evidence for **0/5 candidate two-hop
questions**, at both cutoffs. Mean fact coverage on those questions was 10% for
vector and 40% for BM25. This identifies an engineering challenge. It does not
show that a graph would solve it or establish that the draft hop labels are right.

| Method | Warm retrieval median | Warm retrieval p95 |
| --- | ---: | ---: |
| Vector | 35.66 ms | 91.62 ms |
| BM25 | 38.57 ms | 60.70 ms |

Each timing is the largest-K retrieval call, reused for both cutoff evaluations.
One warmup query per strategy is excluded. Context assembly is separately recorded.
Model loading, artifact verification, index construction, and extraction are
excluded. These are descriptive observations on a shared desktop, not service
latency guarantees. Repeated questions are not independent statistical samples.

BM25 scored higher on these known draft questions. This is not a general ranking
of retrieval methods: exact annotated coordinates, question wording, corpus
composition, and encoder choice all affect the outcome. Alternative unannotated
support receives no coordinate credit. There are no significance claims.

## Saved baseline and verification

The index is `experiments/runs/m11-research-vectors`. The self-contained comparison
is `experiments/runs/m11-research-baseline`; its outer manifest records exact source
hashes, model settings and library versions, comparison policy, and code provenance.
It includes copies of the paper bundle and vector index. Generated artifacts and
third-party source text remain Git-ignored; the code, configuration, and this
report are tracked.

The evaluation fingerprint is:

```text
ae3e03a22f921a966a1b497a731e82d6f6e57a21d4caa392eaeaa62f4606b93e
```

Offline verification reproduced the source metrics, summary, and report. It does
not rerun neural search, recount tokenizer tokens, or independently verify timings.
The run records the working-tree code state at execution, before the final local
commit; later extraction work does not rewrite this saved result.

## Original extraction recipe: bounded source sample

Eight chunks were selected by evenly spaced positions after sorting the full
corpus by document ID and ordinal. This is a deterministic source-only diagnostic;
benchmark questions, answers, and evidence labels were not selection inputs.
It is not a representative extraction-quality evaluation.

The cached `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775` ran with CPU float32, four threads,
4,096 input tokens, and at most 768 generated tokens. The original configuration
contains three fictional format examples.

| Sample | PDF page | Input tokens | Output tokens | Completion seconds | Source validation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 1,023 | 179 | 26.42 | All proposed entities rejected |
| 2 | 8 | 1,059 | 516 | 72.26 | Invalid JSON/schema |
| 3 | 14 | 1,005 | 768 | 85.61 | Output limit; whole chunk rejected |
| 4 | 19 | 1,180 | 751 | 76.84 | Invalid JSON/schema |
| 5 | 5 | 1,013 | 177 | 24.82 | All proposed entities rejected |
| 6 | 9 | 955 | 90 | 12.01 | All proposed entities rejected |
| 7 | 12 | 939 | 768 | 85.05 | Output limit; whole chunk rejected |
| 8 | 6 | 1,062 | 768 | 84.49 | Output limit; whole chunk rejected |

Total accepted entities: **0**. Total accepted relations: **0**. Five responses
were rejected outright; the other three parsed but all proposed entities failed
source/type checks. Two of those responses copied fictional names such as Comet
and Tundra from the format examples into unrelated paper passages. Existing source
checks prevented those names from becoming graph evidence.

The mean recorded completion time was about 58.44 seconds. Multiplying by 6,996
projects **113.57 hours, about 4.7 days**. This is a rough planning estimate from
eight heterogeneous chunks on a shared machine, not a runtime guarantee. Early
diagnostic retries briefly overlapped, so the sample is not an isolated speed
benchmark. Cached replays preserve original generation durations.

Full extraction was not launched. The user selected improving the existing local
extractor before scaling. No larger model, hosted service, or paid API was used.

Original receipts remain in `experiments/runs/m11-research-extraction-cache`.
`m11-extraction-preflight-validated.jsonl` records source validation, exact chunk
IDs, request fingerprints, model settings, and input hashes. A sample cannot be
loaded as a complete paper graph: complete graph validation still requires every
source chunk exactly once.

## Local extractor changes and controlled checks

The user selected improving the existing local extractor before any full-corpus
run. We added opt-in prompt profiles and source-only diagnostic replay while
preserving the original default recipe and its fingerprint.

The first compact prototype removed the fictional examples and shortened the
ontology formatting. All four completed samples hit the 320-token output limit;
the model copied ontology labels into the entity list. The run was stopped during
the fifth sample. Those four receipts remain in
`m11-extraction-compact-preflight.jsonl`; this is an incomplete diagnostic, not an
eight-chunk comparison or a valid graph.

The first focused prototype moved the ontology/schema into the system message,
leaving only the passage in each user message. It used four fictional examples,
including an empty-result example. On the same eight paper chunks, seven outputs
hit the limit and one parsed but yielded no accepted entities or relations.
Its average completion time was 35.29 seconds. The lower time mostly reflects
earlier output cutoffs; it is not a successful quality/cost improvement.

Replay exposed an implementation bug in the new prompt builder: ontology type
definitions were initially rendered in dictionary insertion order, whereas saved
JSON sorts dictionary keys. The final builder sorts type definitions explicitly.
Tests require identical requests and recipe hashes before and after configuration
serialization. The initial `m11-focused-controls` artifact belongs to that prototype
and is superseded; it cannot be replayed by the final builder. Historical prototype
receipts retain their exact original prompts. Final candidate measurements use
the canonical builder, with their own fresh paths.

After canonicalization, the small model proposed `" Harbor"` in the positive
Nova/Harbor control. Strict parsing correctly rejected the padded name. The new
optional `strip_entity_whitespace` setting removes leading/trailing name whitespace,
records `normalized_entity`, and then applies all the existing exact source,
endpoint-type, and quote checks. It does not repair spelling, invent names, merge
aliases, or alter source quotations. The raw completion is preserved. This setting
defaults to false; the focused candidate enables it and records it in the recipe.

The final controlled graph is `experiments/runs/m11-focused-controls-normalized`.
It reuses the already generated canonical-prompt receipts from
`m11-focused-controls-v2`; normalization itself requires no additional inference.
It contains **4 entities, 2 assertions, 1 normalization diagnostic, and 0 rejected
chunks**. Its only relations are Orion `USES` Nova and Nova `EVALUATED_ON` Harbor.
The negated Nova/Mirage passage yields an empty relation list, while preserving
both name mentions. These are three known fictional development passages, not
real-paper evaluation questions.

In the earlier M8 controlled run, the negated passage produced a malformed
predicate and was rejected. The final focused configuration instead produces the
requested empty relation list on this one control. That is a limited observed
improvement; it does not establish general negation reliability or paper accuracy.

`m11-focused-controls-normalized-replay` reconstructs the final graph, issue file,
and response file byte for byte, without inference. The old `m8-final-02` graph
also still verifies under the new implementation.

| Final control artifact | SHA-256 |
| --- | --- |
| Extraction recipe | `4836c1a01b77f9b90b179ac1dafd051262187e0e219bd7d7c186f367e9c4a1b4` |
| `graph.json` | `2025fe00ff899e569ea32338d62918abed9972b5148dfc738a81dafe3efe249e` |
| `issues.jsonl` | `c5e3d8f8df69eee53e38a04f02d17630ff7e75c54672090712596a0c93f7f23b` |
| `responses.jsonl` | `23354df932bf3fc6b6c87a0a34da908b75fce04d7cfd6693d2e21a92bfc28794` |

## Final focused profile on the same eight paper chunks

After fixing canonical prompt order and enabling recorded name trimming, we ran
the same eight real-paper source chunks again. The final results are saved in
`experiments/runs/m11-extraction-focused-final.jsonl`.

| Observation | Original recipe | Final focused candidate |
| --- | ---: | ---: |
| Sampled paper chunks | 8 | 8 |
| Generated-token limit | 768 | 320 |
| Schema-valid responses | 3 | 1 |
| Responses stopped at output limit | 3 | 7 |
| Entire responses rejected | 5 | 7 |
| Accepted entities | 0 | 0 |
| Accepted relations | 0 | 0 |
| Mean recorded completion time | 58.44 s | 42.74 s |
| Rough full-corpus extrapolation | 113.57 h | 83.06 h |

The only final focused response that parsed proposed names/types that failed source
validation. No normalization could repair those failures. The candidate therefore
**fails the real-paper readiness check** and is not promoted to the default.
Reduced output allowance cuts generation short more often; these timings do not
demonstrate a useful extraction speedup. Runs used a shared desktop on different
dates and are not an isolated performance experiment.

The outcome is deliberately narrow: the comparison infrastructure, real baseline,
sample diagnostics, deterministic prompt replay, and opt-in name normalization are
implemented and verified. The current small model's real-paper extraction remains
unsuitable for scaling on this evidence. Further extraction design/model work and
a broader source-reviewed development check should precede full graph construction.
No multi-day extraction job is running, and no larger model was downloaded.

## Engineering findings

The first baseline attempt exposed a previously unsupported case in shared context
assembly. After overlapping passages have been selected, a later passage may add
only a space between them. That is a valid uncovered source interval. The earlier
context-piece contract rejected whitespace-only text and aborted the run.

Context pieces now preserve that exact nonempty whitespace interval, while keeping
the inherited offset/length checks. Documents, chunks, and annotated fact spans
still require nonblank text. A regression test covers the gap between `a` and `b`
in `a b`. The source-union algorithm and scoring policy are unchanged. The successful
baseline above uses this fix. Token counting also suppresses the encoder-length
warning only for context that is counted without neural encoding; input length
checks for actual embeddings remain active.

The new comparison tests exercise nonempty graph paths, all four strategies,
source-only search behavior, frozen inputs, offline verification after original
folders are removed, configuration/provider mismatches, and altered results.
Fake providers in those tests validate software wiring, not model quality.

Validation: **565 tests pass**, Ruff lint and formatting checks pass. An installed
0.11.0 wheel was checked outside the repository. It verified the 40-question,
240-record baseline and the final control graph with imports of PyTorch,
Transformers, Sentence Transformers, and pypdf explicitly blocked. This checks
that offline verification does not require those optional inference/PDF libraries.

## Cue-filter trial (0.11.1)

The next bounded local trial keeps the focused prompt, normalization, model,
revision, CPU settings, and 320-token output limit. It adds the opt-in
`relation-cues-v1` passage selector. The original extraction config remains the
default. This is M11 feasibility work, not the planned M12 retrieval ablation study.

The selector splits on punctuation followed by whitespace/end, strips outer
whitespace, and retains the first two sentences matching its case-insensitive
cue pattern. Cues include “based on,” “built on,” “builds on,” “use/uses,” “used by,”
“evaluated on/using,” “measured by,” “propose/proposes/proposed,” and
“introduce/introduces/introduced.” The patterns, flags, sentence cap, algorithm
identifier, and validation policy are included in the opt-in recipe fingerprint.
Selection reads source text only; no questions, answers, or gold relationships.

This is an intentionally incomplete heuristic. It omits some ontology predicates,
can select generic verbs without named relationships, can split abbreviations,
and discards later matching sentences. Overlapping chunks are counted separately.
No claim is made about the fraction of useful facts retained. Full-corpus source
accounting does not make a filtered graph equivalent in evidence coverage.

| Corpus accounting | Count |
| --- | ---: |
| Original source chunks | 6,996 |
| Chunks with selected text, eligible for a model call | 1,416 (20.24%) |
| Chunks receiving a recorded deterministic skip | 5,580 (79.76%) |
| Actual model responses sampled | 8 |

The eligible chunks were sorted by original document ID and chunk ordinal, then
sampled at positions 0, 202, 404, 606, 809, 1011, 1213, and 1415. Position numbers
refer to the filtered pool, not the full corpus. The final source-validated
diagnostic is `experiments/runs/m11-extraction-cued-final.jsonl`.

| Observation | Final cue-filter sample |
| --- | ---: |
| Provider schema-valid responses | 2/8 |
| Responses stopped at output limit | 5/8 |
| Entire responses rejected | 6/8 |
| Mechanically accepted entity names | 4 |
| Accepted relations | **0** |
| Mean historical provider completion time | 30.39 s |
| Rough projected inference time for 1,416 calls | 11.95 h |

One response had empty entities/relations. The other parseable response named
`batch`, `learning_rate`, `small`, `base`, and `large`. The invented underscore in
`learning_rate` failed exact-source validation. The other four names occurred in
the selected sentence and passed mechanical checks. Calling `batch` a Dataset,
or treating size labels as named models, does not establish correct extraction.
The proposed relation referenced the rejected name and produced no assertion.
Another response failed schema validation; the other five were truncated.

The final validation reused the eight cached completions from
`m11-extraction-cued-model-sample-v2.jsonl`; it did not generate a second independent
set of responses. The final file binds the hardened validation/selection recipe:
`b00a1e28eb83a0d5f0e8326d63f1010c97010228abd089ec0c30c22e0145e136`.

The 11.95-hour projection is mean sampled **provider** time multiplied by eligible
chunks. It excludes model loading, cache/validation overhead, and machine variation.
Eight systematically selected chunks are not a representative runtime study.
These chunks differ from the prior focused sample; 30.39 versus 42.74 seconds is
not a matched prompt speedup. No full extraction was launched. The trial fails
the real-paper readiness check and is not promoted to the default.

### Accounting and prototype corrections

An initial eight-chunk sample over the whole corpus selected only cue-less chunks.
The prototype incorrectly counted their well-formed empty records as model schema
successes and reported a zero-hour projection. Those are not model results. The
corrected diagnostic (`m11-extraction-cued-all-final.jsonl`) reports eight
deterministic abstentions, zero provider samples, and a `null` runtime projection
because 1,416 other chunks still need model calls. Mixed samples now average only
provider times, avoiding a second reduction for the skip fraction.

The earlier broader selector included words such as “supports,” which selected
the noun phrase “customer support.” That trial was stopped after its first
completed response. Its files (`m11-extraction-cued-model-sample.jsonl` and
`m11-extraction-cued-preflight.jsonl`) are prototypes, not the final policy or
an additional independent comparison. The early `m11-cued-controls` manifest also
predates selection-policy fingerprinting; use the final control artifact below.

### Audit checks and fictional controls

Provider receipts keep their legacy serialized shape. Deterministic skip records
carry an explicit origin and an exact empty completion with zero tokens/time.
Replay checks both directions: only cue-less chunks may skip, and cue-less chunks
must carry a skip receipt. Altered completions or origins are rejected. Provider
cache entries cannot masquerade as deterministic skips. Failures never silently
become successful empty model results.

Names/mentions must lie in selected source spans. Each relation quote must occur
uniquely in the original chunk and fit wholly inside one selected span, with both
endpoint names present. This prevents accidentally accepting material omitted
from the prompt or treating the join between selected sentences as source evidence.
It still does not prove semantic truth, negation handling, or correct entity types.

The three fictional controls in `m11-cued-controls-final` retain four entities,
two assertions, one whitespace-normalization issue, and zero rejected responses.
The positive controls produce Orion USES Nova and Nova EVALUATED_ON Harbor; the
negated control produces no edge. Their prompts equal the final focused controls,
so cached model completions were reused. This is a replay/control check, not fresh
independent model-quality evidence.

| Final cue control artifact | SHA-256 |
| --- | --- |
| `graph.json` | `e75afed342efd1efd807adaf616e29683dd2527be4099311bb999c5cc5d42524` |
| `issues.jsonl` | `c5e3d8f8df69eee53e38a04f02d17630ff7e75c54672090712596a0c93f7f23b` |
| `responses.jsonl` | `23354df932bf3fc6b6c87a0a34da908b75fce04d7cfd6693d2e21a92bfc28794` |

The final cue controls, existing focused controls, and original `m8-final-02` all
replay graph, issues, and response files byte for byte into fresh `-v0111-replay`
directories. Generated corpora, caches, and run directories remain local and
gitignored; the implementation, config, tests, and study report are committed.

Validation: **585 tests pass**, including altered skip receipts, incorrect origins,
out-of-prompt names/quotes, cache-origin corruption, legacy serialization, stable
config round trips, and runtime accounting with zero/mixed provider samples.

Next extraction work should address output correctness and semantic quality on a
bounded source-reviewed sample before a full graph build. Reducing call counts
alone has not solved this. Independent review and held-out evaluation remain
separate work; no new model, paid API, push, or M12 run was used for this trial.

## Numbered-sentence extraction and draft checks (0.11.2)

The next experiment simplifies what the existing local model must generate.
`indexed-v1` numbers exact source sentences and asks for endpoint names/types,
a predicate, and a sentence ID. Deterministic code creates local IDs and obtains
the quote from the referenced source slice. Existing type, exact-name, unique-quote,
and endpoint checks still apply. The parser accepts at most four named or positional
six-field rows, with strict positive integer sentence IDs. Raw responses remain
unchanged. The profile requests relationships only, so standalone entity coverage
can decline. It remains experimental and is not the default.

The final config retains the cached Qwen2.5-0.5B-Instruct revision and CPU settings
listed above, all eight ontology predicates, all source sentences, four fictional
examples, and a 320-token output limit. It disables whitespace normalization.
The prompt uses a shorter type list, numbered text, and a revised response schema.
This is a bundled extraction-design comparison, not an isolated measurement of
one prompt feature and not the planned M12 retrieval ablation study.

### Positional-array prototype

The first prototype asked for six-value arrays. Its three fictional controls
produced two assertions, three entities, no issues, and no rejected responses
(`m11-indexed-controls-01`). On the original eight systematically sampled paper
chunks, it produced zero schema-valid responses and zero assertions; four hit the
output limit. Mean recorded completion time was 26.68 seconds
(`m11-indexed-systematic-01.jsonl`). This again demonstrates that fictional control
success does not establish paper readiness.

Inspection showed named objects where arrays were requested, undeclared types such
as Person/Number/Conference, ontology/example copying, and too many rows. On the
eight targeted development cases described below, the prototype also produced zero
schema-valid responses and zero assertions; three hit the output limit, with a
26.51-second mean (`m11-indexed-checks-01.json`). These raw results explain the move
to named fields in the final prompt. The final contract also accepts six-value
arrays; its schema, conversion policy, and prompt are fingerprinted.

The `-01` artifacts predate the final prompt/contract. They are archived prototypes,
not artifacts that claim compatibility with the final indexed recipe. Their full
requests/responses remain in the local cache. Use the final artifacts for replay.

### Same-source development diagnostic

`datasets/examples/extraction-study/m11-source-checks.json` pins eight original
chunks and their ingestion hashes. Five positive cases contain six draft reference
facts; three cases are negative. The positives cover RAGAS/GPT-3.5, RAG/DPR/BART,
jina-embeddings-v3/XLM-RoBERTa, BERT-FTbase/BERTbase, and Sentence-BERT/BERT. The
negatives cover generic hyperparameters, a comparison, and task unsuitability.
The reference file explains each interpretation and accepted name spelling.

These checks were drafted after reading the source. They are **not independently
reviewed or held out**. Selection was fixed before
inspecting candidate outputs on these target cases, but the hyperparameter negative
was already a known failure and the broader corpus was already used for development.
The sample is deliberately small, positive-enriched, and does not cover all predicates.
Interpreting initialization as BASED_ON needs a second review.

`scripts/check_extraction_sample.py` supplies the same full original chunks to each
profile. Labels never enter the inference interface. A matched fact checks endpoint
names, direction, and predicate against a draft reference. It does not establish
exact entity-type accuracy or full semantic correctness. Unmatched assertions are
saved for review rather than automatically called false positives. The script makes
no full-corpus projection and creates no paper graph.

Reports bind the case-file hash, source hashes, recipe, provider, raw completions,
accepted assertions/evidence, issues, and scores. `--replay` rebuilds requests and
recomputes validation and scores without loading a model. It detects inconsistent
saved results; it does not independently authenticate model provenance or prove
the reference labels true.

### Final paired observations

| Observation on the same eight chunks | Focused baseline | Final indexed candidate |
| --- | ---: | ---: |
| Draft reference facts matched | 0/6 | 0/6 |
| Mechanically accepted assertions | 0 | 1 |
| Accepted assertions unmatched by draft references | 0 | 1 |
| Schema-valid responses | 1/8 | 1/8 |
| Entire responses rejected | 7/8 | 7/8 |
| Responses stopped at output limit | 7/8 | 4/8 |
| Negative cases with no assertion and no whole-response rejection | 0/3 | 0/3 |
| Mean recorded completion time | 33.10 s | 23.47 s |

The candidate's one accepted assertion is **Sentence-BERT USES BERT**, citing the
sentence describing Sentence-BERT as a modification of pretrained BERT. The draft
reference asks for BASED_ON. This is an ontology interpretation requiring review;
the unmatched count does not automatically prove the assertion false. We did not
change the reference to reward the observed output. A second proposed assertion,
SBERT EVALUATED_ON BERT, failed because EVALUATED_ON requires a dataset object.

The other candidate responses show continued failures to follow instructions:
repeated self-relations, fictional example names copied into paper responses,
undeclared type labels, more than four rows, and invalid sentence references.
For example, the RAG component response copied Comet from a fictional example and
proposed component self-relations. The Jina figure response produced a chain through
layout words instead of the stated backbone relation. Truncated responses remain
rejected as a whole; we do not salvage an apparently useful prefix.

The measured reduction in truncations is real for these outputs, but **readiness
still fails**. Seven of eight responses fail whole-response validation, none of six
draft target facts match, and all three negatives fail to yield a usable empty
response. This does not establish useful full-paper extraction or general model
accuracy. The final named-field recipe has not been run across the full corpus or
the earlier systematic sample. Its targeted selection cannot estimate corpus recall.

Times are descriptive receipt values from sequential profile runs on a shared
desktop, with lightweight development checks during execution. There are no repeated
timing trials, confidence intervals, or claims of isolated performance gains.

| Local artifact | SHA-256 |
| --- | --- |
| Draft case file | `4f475cf0ad71b328cfa33721131db2ee6ffa7bcdff2f1c7670d6f9a3ea4f8e2a` |
| `m11-focused-checks-public.json` | `02e1a4299fcac79d2936d9e1f4420a1b62186313aa1dcc8a1fc49cd1fac796f7` |
| `m11-indexed-checks-public.json` | `d77588c7c351c61d050c7ccc9d32c40b5ef464ea0e5dbae8b3167bc3e66ad460` |
| Final indexed recipe | `ced901bd37ec50a9752f2e20faf02433899975a9f6efc358fd89e8487fb9ff4e` |

Both final diagnostics replay successfully without inference. The existing M8,
focused, and cue-filter control artifacts also still verify. Validation includes
**615 passing tests**, source-only request checks, strict named/positional row
contracts, exact quote offsets, invalid references, incompatible endpoint types,
legacy recipe compatibility, graph replay, and tampered diagnostic scores.

The [numbered-sentence lesson](../docs/indexed-extraction.md) explains the change
with a scout analogy and gives reproduction commands. New work should prioritize
semantic extraction quality and source-reviewed labels before scaling. A stronger
local model or a materially different extraction design is the next decision;
schema constraints alone would not fix the observed wrong relationships. No new
model download, paid API, full-corpus extraction, push, or M12 study occurred.

## Larger local-model trial (0.11.3, 2026-09-30)

Repeated 0.5B failures motivate testing model capacity on the unchanged indexed
recipe before another prompt redesign. The candidate config,
`configs/llm-extraction-indexed-3b.toml`, pins Qwen2.5-3B-Instruct at
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Only model ID/revision differ from the
final 0.5B indexed config; source messages, fictional examples, ontology, CPU float32,
four threads, and input/output limits stay fixed. Tokenizer and chat template come
with the model, so this is not a parameter-count-only causal experiment.

`check_extraction_sample.py --preflight` now checks config/source hashes and chunk
IDs without loading a model or writing artifacts. The optional
`--min-available-gib` gate stops both preflight and inference before provider
construction when available RAM is insufficient or unknown. Source bindings are
also checked before provider construction on normal inference commands. The new
preflight does not change extraction recipes or saved diagnostic formats.
Offline replay remains independent of the model and memory check.

The real-source preflight validated all eight cases and the unchanged case-file
hash `4f475cf0ad71b328cfa33721131db2ee6ffa7bcdff2f1c7670d6f9a3ea4f8e2a`.
It reported 1,721,425,920 available bytes (1.60 GiB) and correctly exited with code 2
against an 18 GiB threshold, without initializing a model. This threshold is an
engineering headroom estimate, not a measured requirement or a memory reservation.
Model metadata indicates approximately 11.50 GiB for float32 parameters alone;
other runtime allocations will add to that.

The approximately 6.18 GB pinned download was explicitly approved before it began.
The runner verified both weight-file SHA-256 hashes, waited for 18 GiB available
RAM, ran only the eight frozen cases, then replayed the saved diagnostic offline.
It used CPU float32 with four threads. Inference plus model loading took 334.07
seconds wall time; recorded peak child RSS was 18,562,104 KiB. These one-run host
measurements are descriptive.

| Same eight paper chunks | 0.5B indexed | 3B indexed |
| --- | ---: | ---: |
| Draft facts matched | 0/6 | 2/6 |
| Mechanically accepted assertions | 1 | 2 |
| Accepted assertions unmatched by draft references | 1 | 0 |
| Schema-valid responses | 1/8 | 7/8 |
| Entire responses rejected | 7/8 | 1/8 |
| Responses stopped at output limit | 4/8 | 0/8 |
| Negative cases with no accepted assertion or whole rejection | 0/3 | 3/3 |
| Mean recorded completion time | 23.47 s | 40.71 s |

The accepted assertions are `jina-embeddings-v3 BASED_ON XLM-RoBERTa` and
`Sentence-BERT BASED_ON BERT`. Their exact evidence states that the Jina architecture
is based on XLM-RoBERTa and that SBERT modifies the pretrained BERT network. They
match the draft endpoint names, direction, and predicate. This source inspection is
development review; independent annotation review remains pending.

The RAG-components response was rejected because all four proposed rows omitted the
required sentence ID. It also used `BASED_ON` instead of the draft `USES` predicate
for DPR/BART and included an ontology-incompatible SUPPORTS_TASK proposal. The raw
response remains in the diagnostic. No prefix or plausible row was salvaged.
RAGAS and BERT initialization yielded valid empty responses, leaving four positive
draft facts undetected.

Two 3B negative responses were clean empty results. The third proposed an unsupported
row, which validation discarded, so it had no accepted assertion but was not clean.
The 3B reader markedly improved schema compliance, but 2/6 draft-fact recall is not
sufficient for a complete graph. The sample
is small, purposive, previously used for development, and lacks independent labels.
No corpus-wide accuracy, retrieval gain, or causal parameter-count claim follows.

The replay-verified diagnostic SHA-256 is
`1cd04b767961755ef0fab6edf76e74692f7f52afd34586b4ea5aa148c86edbba`;
the extraction recipe SHA-256 is
`70e26d7258535535cacc3d6b495cbaa8fe96fcc380bd0c66df8a54ab1547c1e9`.
The [study lesson and commands](../docs/local-model-trial.md) document the bounded
runner, memory controls, exact outputs, and interpretation. No paid API, full-corpus
extraction, push, or M12 work occurred.

Validation: **643 tests pass**, including low/unknown RAM stopping before model
construction, a read-only preflight, invalid source rejection before loading,
unchanged messages across model configs, separate model cache fingerprints, and
model-free replay. Runner tests cover frozen config/check copies, cache-only inference,
weight checks, memory waiting, timeout/failure status, and offline replay. Ruff passes.
Both saved 0.5B diagnostics and the new 3B diagnostic replay against their source
and draft case file.

### Follow-up candidates: richer examples and relation cues

Two further bounded 3B candidates were evaluated after inspecting the 2/6 result.
Both are development adaptation, not held-out experiments. They use the same pinned
model and no paid service.

The richer-examples candidate replaces four short fictional demonstrations with five
fictional examples covering two components in one sentence, initialization, late
evidence, comparisons, negation, and generic hyperparameters. It changed only the
examples. On the original eight cases it still matched **2/6** draft facts; schema
validity fell from 7/8 to 6/8, whole-response rejection rose from 1/8 to 2/8, and
mean recorded completion time rose from 40.71 to 47.61 seconds. The RAGAS response
copied nearby ARES text, and the RAG-components response still omitted sentence IDs.
The Jina and SBERT matches remained. This candidate is rejected. Its replayed
report SHA-256 is
`82d66bc9d823c2fcbc7687d1480144645a4dbe2c9b52509b3b5a6fa651130275`, and
its recipe SHA-256 is
`383bd8ad5be9b52a14442688a3c0be8b62f2ae2c853c5f6004be8d592148a66a`.
The [example lesson](../docs/extraction-examples.md) records the exact change.

The relation-cues-v2 candidate retained only the first two sentences with explicit
relationship wording. It adds `initializes from` and `modification of` to the
unchanged v1 cue list. The original generic-hyperparameters negative could not be
used because v2 exposes a real initialization statement in that chunk. The trial
therefore used a five-positive-case slice, which measures draft-fact recovery only.

This candidate matched **1/6** facts. It recovered RAGAS–GPT-3.5 but lost the Jina
match after PDF figure layout became the entire model input. It proposed
`BERT-FTbase USES E5` rather than the draft BERTbase ancestry relationship; the
source supports the words used but the ontology interpretation needs review. It
also failed the RAG-components schema and used an incompatible predicate for SBERT.
This candidate is rejected. Its report SHA-256 is
`fb6c3bbb84350e34ed361979e2e6b11d2071dd2d57ad2ece098bd925cad8e705`, and
its recipe SHA-256 is
`f6a846cc6bde07415f826f453ef9615f58723377d16cd2f544dd4e4b8f77e73b`.
The [cue-selection lesson](../docs/relation-cue-trial.md) explains why its negative
labels need recipe-specific review.

The original full-context 3B indexed recipe remains the best observed development
candidate at 2/6 draft-fact recovery. Its recall remains too low for a full graph.
Further prompt-only changes should not proceed without a source-label review plan
and an extractor design that addresses PDF layout and semantic ontology decisions.

## Moving the accepted recipe to a local GPU (0.11.6, 2026-09-30)

The accepted recipe is CPU float32 at four threads, chosen for a bounded eight-chunk
trial. Projected over 6,996 chunks at its 40.71-second mean it needs about **79 hours**.
The host has an RTX 4070 Laptop with 8 GiB VRAM that the pinned CPU-only torch wheel
could not address, and `LocalModelConfig` forbade any device but `cpu` and any dtype
but `float32`. Both restrictions are now widened; CPU still refuses reduced precision,
and a CUDA recipe records provider `transformers-causal-cuda-v1`.

Installing `torch==2.14.0+cu130` in place of `2.14.0+cpu` left all offline tests
passing. Saved vector artifacts are unaffected because verification never re-embeds,
but later embedding runs would use this different build.

Both reduced-precision recipes were measured on the **same eight cases and the same
draft checks file** (`4f475cf0…`) as the CPU reference, with no prompt, ontology, or
selection change:

| Same eight paper chunks | CPU float32 | CUDA float16 | CUDA bfloat16 |
| --- | ---: | ---: | ---: |
| Draft facts matched | 2/6 | 1/6 | 2/6 |
| Accepted assertions | 2 | 1 | 2 |
| Schema-valid responses | 7/8 | 6/8 | 7/8 |
| Rejected responses | 1/8 | 2/8 | 1/8 |
| Negative cases without assertions | 3 | 3 | 3 |
| Mean completion seconds | 40.71 | 2.31 | **1.81** |

float16 lost the `Sentence-BERT BASED_ON BERT` match and gained a rejection. bfloat16
reproduced the reference exactly: the same two accepted assertions,
`jina-embeddings-v3 BASED_ON XLM-RoBERTa` and `Sentence-BERT BASED_ON BERT`, with the
same supporting chunk IDs, and every recorded count identical. bfloat16 keeps
float32's exponent range and only reduces mantissa bits, which is the plausible
explanation, but eight cases cannot establish that the two recipes agree in general.
float16 is retained only as this recorded negative result and must not be used.

bfloat16 is therefore **22.5x faster than the reference with no observed quality
change on these cases**, and its recipe fingerprint is
`93adda02c94a944f1ab30c47dc1de72fb20fc3d5a67d32d17bcb3ad0d61bb27c` against the
reference `70e26d7258535535cacc3d6b495cbaa8fe96fcc380bd0c66df8a54ab1547c1e9`.
Two GPU-recipe caveats remain: 2/6 recovery is still the accepted recipe's weak
recall, not an improvement, and identical results on eight hand-picked chunks do not
predict agreement across 6,996.

## Four-method run with a deterministic graph floor (0.11.5, 2026-09-30)

The comparison runner now accepts either graph kind and records which one it used.
A deterministic rule graph is identified by its own `extractor_version` receipt and
copied as three artifacts without `responses.jsonl`; the outer manifest stores
`graph_status = "rule-based-unreviewed"`. Relabelling a saved run's graph kind fails
offline verification because the recorded artifact file list no longer matches.

Running `build-graph` with `configs/extraction.toml` over the 6,996 real chunks took
about 3 seconds and produced **0 entities, 0 assertions, and 36,583 unsupported
statements**. The M3 grammar requires one complete controlled-English statement per
line with a final period. Wrapped PDF prose lines do not match any rule. This is a
measured transfer result for that extractor, not a property of graph retrieval.

The full four-method run over the same frozen bundle and vector index is
`experiments/runs/m11-rule-graph-comparison-01`: 40 questions, 4 strategies, 3
repeats, 480 records, about 48 seconds of wall time after model loading. Offline
verification reproduced its source metrics, summary, and report.

| Method | K | Complete evidence after budget | Mean fraction of fully covered facts |
| --- | ---: | ---: | ---: |
| Vector | 5 | 12/40 = 30.0% | 33.3% |
| BM25 | 5 | 22/40 = 55.0% | 61.3% |
| Graph | 5 | 0/40 = 0.0% | 0.0% |
| Hybrid | 5 | 12/40 = 30.0% | 33.3% |
| Vector | 10 | 19/40 = 47.5% | 50.8% |
| BM25 | 10 | 25/40 = 62.5% | 68.8% |
| Graph | 10 | 0/40 = 0.0% | 0.0% |
| Hybrid | 10 | 19/40 = 47.5% | 50.8% |

Evaluation fingerprint:

```text
70eabebe2beb9205136dc336178cb7459dd5bff0429cf6910e4b41abdc25541b
```

Three observations, none of which is a retrieval-method result:

The vector and BM25 columns reproduce the earlier two-method baseline exactly,
which is the intended behaviour of adding strategies to a frozen comparison.

The graph arm scores zero because its graph is empty. Graph retrieval returned no
candidates and assembled no context, so its 0.13 ms p95 measures an empty traversal.
This run therefore establishes a floor for this extractor, not for graph retrieval.

Hybrid equals vector on every question at both cutoffs: 0 wins, 40 ties, 0 losses.
Reciprocal rank fusion with one empty component correctly degenerates to the other
component's ranking. That is a wiring check, not evidence about fusion.

The one substantive pattern remains the lexical advantage: BM25 recovers complete
annotated evidence for 25 percentage points more questions than dense retrieval at
K=5 under an identical 2,000-token budget. The draft annotations, their exact
quoted coordinates, the question wording, and the pinned encoder all shape that
number, and no alternative unannotated support receives credit. It is a development
observation on unreviewed labels with no held-out split and no significance test.

## Remaining study requirements

A usable complete real-paper graph, a four-method comparison with a non-empty graph,
independently reviewed annotations, and 80 genuinely held-out questions remain. The
deterministic graph arm above does not satisfy the graph requirement.
Producing an LLM graph over the 1,416 cue-eligible chunks is projected at about
11.95 hours of local provider time at 2/6 sampled draft-fact recovery; that run has
not been launched and its quality gate is undecided.
The five draft two-hop questions still need shortcut checks. No M12 ablation study has started.
The beginner lesson and reproduction commands are in [paper-comparison.md](../docs/paper-comparison.md).
