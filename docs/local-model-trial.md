# M11: testing a stronger local reader

**Status: completed and replay-verified on 2026-09-30.** The 3B model recovered
2/6 draft facts across eight cases, compared with 0/6 for the 0.5B candidate.
This remains a bounded M11 development experiment, not a full graph build or M12.

## Why change the reader?

Think of the extractor as a scout reading reports and drawing connections on a map.
We made its form simpler: it now names two things, chooses their relationship, and
points to a numbered sentence. The existing scout still filled out seven of eight
paper forms incorrectly. It matched none of the six draft facts we expected.

The next question is whether a larger reader handles the same form better. The
candidate is **Qwen2.5-3B-Instruct**, compared with the existing 0.5B model. “3B”
means roughly three billion learned numerical parameters. More parameters give the
model more capacity, but do not guarantee correct answers. This trial measures its
behavior on the same small development sample before considering a larger run.

## What changes and what stays fixed?

The new config is [llm-extraction-indexed-3b.toml](../configs/llm-extraction-indexed-3b.toml).
Compared with the existing indexed config, only the model ID and pinned revision
change. Both use the same:

- Eight original paper chunks: five positive cases containing six draft facts and
  three negative cases, from the unchanged [case file](../datasets/examples/extraction-study/m11-source-checks.json).
- Full passages, numbered sentences, eight relationship types, and four fictional examples.
- CPU float32 execution, four CPU threads, greedy generation, 4,096 input tokens,
  and a 320-token output allowance.
- Source validation, rejection rules, scoring, and raw-response receipts.

The messages supplied to the provider are identical. The model repository also
supplies its tokenizer and chat template, which may differ. This is a comparison
of two model packages, not proof that parameter count alone causes a difference.
New provider and extraction fingerprints prevent reuse of the smaller model's
cached responses as larger-model outputs.

The reference facts stay outside the model requests. They remain draft labels
awaiting independent review. These cases have already informed development, so
they cannot establish held-out accuracy or performance across the entire corpus.

## What happened

| Same eight paper chunks | 0.5B indexed | 3B indexed |
| --- | ---: | ---: |
| Draft facts matched | 0/6 | 2/6 |
| Accepted assertions | 1 | 2 |
| Unmatched accepted assertions | 1 | 0 |
| Schema-valid responses | 1/8 | 7/8 |
| Rejected responses | 7/8 | 1/8 |
| Output-limit responses | 4/8 | 0/8 |
| Negative cases with no accepted assertion or whole rejection | 0/3 | 3/3 |
| Mean recorded completion time | 23.47 s | 40.71 s |

The two 3B assertions were:

1. `jina-embeddings-v3 BASED_ON XLM-RoBERTa`, supported by the sentence stating
   that its architecture is based on XLM-RoBERTa.
2. `Sentence-BERT BASED_ON BERT`, supported by the sentence describing SBERT as a
   modification of the pretrained BERT network.

Both match the draft endpoint names, direction, and predicate. Inspection confirms
that their quoted source text supports those relationships. This is development
review, not the pending independent annotation review.

The rejected RAG-components response omitted every required `sentence_id`. It also
used `BASED_ON` where the draft labels use `USES`, and proposed an invalid
`Model SUPPORTS_TASK Task` row. The program rejected the complete response; it did
not salvage plausible fragments. RAGAS and BERT initialization were valid empty
responses, so four positive draft facts were still missed.

The generic-hyperparameters and comparison negatives were clean empty responses.
The unsuitable-tasks response proposed an unsupported row, which validation
discarded; it therefore had no accepted assertion but was not a clean empty answer.

The larger reader therefore improved instruction following and bounded precision,
but recall remains insufficient: it recovered only one third of the six positive
facts. Eight development cases cannot establish corpus-wide accuracy. A full graph,
retrieval comparison, held-out score, and general model advantage remain unproven.

The saved report SHA-256 is
`1cd04b767961755ef0fab6edf76e74692f7f52afd34586b4ea5aa148c86edbba`.
Its extraction recipe SHA-256 is
`70e26d7258535535cacc3d6b495cbaa8fe96fcc380bd0c66df8a54ab1547c1e9`.
Inference plus model loading took 334.07 seconds wall time. Peak child RSS was
18,562,104 KiB. These are single-machine observations, not performance benchmarks.

## Download and memory budget

The [official model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
describes the candidate. The selected revision is
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Metadata inspected on 2026-09-30 lists
3,085,938,688 parameters stored as BF16 and two weight files totaling
6,171,926,992 bytes. The explicit download list below totals **6,183,458,486 bytes**
(about 6.18 GB or 5.76 GiB) before cache/transfer overhead. Existing cached files
would reduce the transfer.

Downloaded file size and working memory are different. The current provider loads
float32 weights: parameters alone require about **11.50 GiB**. Loading, attention,
generation state, and Python need additional memory. For this trial, use an
**18 GiB available-RAM threshold** before model loading. This is a conservative
engineering estimate, not a measured peak or a guarantee against running out of RAM.
The check reads Linux `MemAvailable`; it does not reserve memory or monitor later use.

The development machine has 32 GiB nominal RAM and an 8 GB NVIDIA laptop GPU.
The installed PyTorch environment is CPU-only. This first comparison keeps the
existing CPU provider; moving to GPU or quantized weights would be a separate recipe.
At preparation time only 1.60 GiB RAM was available, so the check correctly refused
to proceed. Close memory-heavy applications, then rerun the check. The script never
closes applications itself. Swap space does not count toward the threshold.

The model has its own
[Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE).
The repository's MIT license does not relicense model weights. The download list
includes the model license; weights remain outside this Git repository.

## Run the bounded experiment

Run commands from the project root with the existing optional local-model
environment and M10 paper ingestion. Rebuilding those prerequisites is explained
in [LLM extraction](llm-extraction.md) and [paper ingestion](real-paper-corpus.md).

### 1. Check readiness without loading or downloading a model

```bash
.venv/bin/python scripts/check_extraction_sample.py \
  datasets/processed/m10-expanded-03/ingestion \
  --checks datasets/examples/extraction-study/m11-source-checks.json \
  --config configs/llm-extraction-indexed-3b.toml \
  --preflight --min-available-gib 18
```

The command validates the config, ingestion hashes, and unique known chunk IDs,
then prints the pinned model, source/case hashes, case count, and available RAM.
It creates no model cache or experiment output. Insufficient or unreadable available
RAM produces `memory_gate_passed: false` and exit code 2. Invalid source artifacts
also stop execution before a provider is constructed.

This preflight does **not** verify model-cache completeness, tokenizer input lengths,
model quality, or actual peak memory. Without `--min-available-gib`, it reports RAM
without enforcing a threshold. Always include the threshold for this 3B trial.
The same threshold works on an inference command and is checked before loading.

`build-llm-graph` accepts the same `--min-available-gib` and refuses before
constructing a provider. Use it on any long run: the measured 18.1 GiB peak child RSS
exceeds what a busy desktop leaves free, and without the gate a full-corpus run starts
anyway and can be killed after hours of completed work. A threshold is headroom
guidance, not a reservation; another process can take the memory a moment later.

### Why the reference recipe is slow, and the two ways off it

The accepted recipe is CPU, float32, four threads. That combination was chosen for a
bounded eight-chunk trial, where a fixed, reproducible arithmetic path mattered more
than throughput. It is a poor choice for 6,996 chunks: at the measured 40.71-second
mean it projects to about **79 hours**. Neither available RAM nor chunk count is the
bottleneck there; single-precision matrix arithmetic on CPU cores is.

Two recipes exist for long runs. Both change the fingerprint, so both need the
eight-case diagnostic re-run before they are trusted, and neither is described by the
2/6 result measured on the reference recipe.

### Thread count and the full-corpus recipe

[llm-extraction-indexed-3b-threads16.toml](../configs/llm-extraction-indexed-3b-threads16.toml)
differs from the accepted 3B recipe in exactly one line, `cpu_threads = 16`, to make a
full-corpus run tractable. Because `cpu_threads` is part of the recipe fingerprint,
this is a **new recipe**: the 2/6 eight-chunk result was measured at four threads and
does not describe it. Re-run the eight-case diagnostic under this config and compare
matched facts, schema validity, and rejections before launching a full extraction.
CPU reduction order can differ with thread count, so identical outputs are not
guaranteed and must be observed rather than assumed.

### Local GPU with reduced precision

[llm-extraction-indexed-3b-cuda-fp16.toml](../configs/llm-extraction-indexed-3b-cuda-fp16.toml)
differs from the accepted recipe in two lines, `device = "cuda"` and
`dtype = "float16"`. `LocalModelConfig` accepts `cpu`, `cuda`, and `float32`,
`float16`, `bfloat16`, and still refuses reduced precision on CPU so the reference
path cannot drift. A CUDA recipe records provider version
`transformers-causal-cuda-v1` rather than `transformers-causal-cpu-v1`, so any saved
run states which arithmetic produced it. Requesting `cuda` from a CPU-only torch
build fails before the model loads.

This needs a CUDA torch wheel; the pinned `requirements-embeddings-cpu.txt` build has
no CUDA support. Qwen2.5-3B in float16 is about 6.2 GiB of weights plus a small
grouped-query KV cache, so it fits an 8 GiB device. Weights sit in VRAM, which is why
`--min-available-gib` does not apply to this recipe.

**float16 is a larger change than thread count.** Halving mantissa precision can change
generated tokens, and extraction here depends on exact quoted substrings, so a
degraded response fails validation rather than degrading quietly. Compare matched
facts, schema validity, and rejections against the CPU reference on the same eight
cases, and keep the CPU float32 recipe as the reference regardless of the outcome.

### 2. Download the pinned files only after approval

The approximately 6.18 GB download was explicitly approved before it began.
Reading model metadata during initial preparation did not fetch weights. The
background runner later completed and verified the pinned download through the Hub
cache. This is the equivalent standalone download command:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
    allow_patterns=[
        "LICENSE", "config.json", "generation_config.json",
        "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    ],
    max_workers=2,
)
PY
```

### 3. Run only the eight cases

Choose a fresh output path. No paid service or full-corpus job is involved.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/check_extraction_sample.py \
  datasets/processed/m10-expanded-03/ingestion \
  --checks datasets/examples/extraction-study/m11-source-checks.json \
  --config configs/llm-extraction-indexed-3b.toml \
  --min-available-gib 18 \
  --response-cache experiments/runs/m11-research-extraction-cache \
  --output experiments/runs/m11-indexed-3b-checks-01.json
```

The existing provider is cache-only by default; the environment flags also disable
Hub access. Missing files cause failure. The command does not silently download
them, truncate inputs, or switch to a different model. The report stores actual
outputs and their validation, not the preflight result. Use a fresh output path
when repeating the completed trial.

### 4. Replay and inspect

```bash
.venv/bin/python scripts/check_extraction_sample.py \
  datasets/processed/m10-expanded-03/ingestion \
  --checks datasets/examples/extraction-study/m11-source-checks.json \
  --replay experiments/runs/m11-indexed-3b-checks-01.json
```

Replay needs neither a model nor the RAM threshold. Compare the new receipt with
`experiments/runs/m11-indexed-checks-public.json`, whose measured baseline is
0/6 draft facts, 1/8 schema-valid responses, seven rejected responses, four output
limit stops, and zero usable empty responses on the three negatives. The
[development report](../reports/m11-development.md) records that baseline's hashes.

Inspect every accepted assertion against its sentence, including direction, type,
predicate, and negation. Report missing draft facts and unmatched assertions
separately; an unmatched assertion is not automatically false. Preserve raw outputs
and labels, including disagreements such as `USES` versus `BASED_ON`.

For a promising result, look for eight well-formed responses, no output-limit stops,
three correct empty negative responses, and source-supported positive assertions.
Draft fact matches remain a diagnostic, not a substitute for semantic review.
Even a clean eight-case result earns only a broader bounded feasibility check.
It does not authorize a 6,996-chunk job or establish a retrieval advantage.

## Running after VS Code closes

VS Code edits files; Python runs the experiment. The editor is not a dependency.
However, the current interactive coding session is a child of VS Code's extension
host. Closing that host disconnects the session. A normal child shell command is
therefore insufficient for a reliable handoff.

`scripts/run_local_model_trial.py` runs under the systemd user manager, independently
of the editor. It saves exact copies of the config and draft checks, validates the
source, downloads only when `--allow-download` is explicit, and checks both weight
files against their pinned SHA-256 hashes. It then waits up to 30 minutes for
18 GiB available RAM, runs only the eight cases with a one-hour inference timeout,
and replays the result offline. It never closes applications or publishes changes.

For this approved run, the service and artifact directory are:

```text
graphrag-m11-3b-trial.service
experiments/runs/m11-indexed-3b-trial-01/
```

To launch a fresh run after approval, choose a new unit name and output directory
if those already exist. Substitute your absolute project path for `PROJECT`:

```bash
systemd-run --user --unit=graphrag-m11-3b-trial \
  --working-directory=PROJECT \
  --property=RuntimeMaxSec=3h \
  --property=MemoryMax=20G --property=MemorySwapMax=0 \
  --setenv=HF_HUB_DISABLE_PROGRESS_BARS=1 \
  PROJECT/.venv/bin/python -u PROJECT/scripts/run_local_model_trial.py \
  --allow-download \
  --output PROJECT/experiments/runs/m11-indexed-3b-trial-01
```

The service has a three-hour total limit and a 20 GiB cgroup memory limit, with
swap disabled for the job. These limits bound the background job; they do not
guarantee that the model fits. Failure leaves logs and a resumable Hub/response
cache. Do not overwrite a partial run directory; use a new one to retry.

Save your open work, then close VS Code normally. Keep the computer awake and stay
logged in. The job survives closing the editor; it is not configured to survive
logout, shutdown, or reboot. Avoid reopening memory-heavy applications until the
trial finishes.

Check it from a separate terminal:

```bash
systemctl --user status graphrag-m11-3b-trial --no-pager
journalctl --user -u graphrag-m11-3b-trial -n 30 --no-pager
cat experiments/runs/m11-indexed-3b-trial-01/status.json
```

`status.json` records the last stage. `completed` means inference and replay both
succeeded; independent semantic review remains pending. `diagnostic.json` contains
the raw responses, evidence, scores, and provider settings. `inference.log` and `replay.log`
record each command's output. Inference wall time includes loading and validation;
peak child RSS is recorded in KiB on Linux. Neither is an isolated model benchmark.
If the service is killed externally, the last status may be stale; check systemd
as well. Stop the job with `systemctl --user stop graphrag-m11-3b-trial`.

Running on another computer or a hosted notebook is possible because the project
uses Python and files. A GPU, different numerical precision, or different library
versions would need their own recorded recipe and comparison. They are not silently
substituted into this CPU trial.

**Check your understanding:** If the larger scout produces perfect forms but links
the wrong people, has the experiment succeeded? No. Correct formatting and correct
meaning must be checked separately.
