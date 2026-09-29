# M11: testing a stronger local reader

**Status: prepared, not run.** The 3B model has not been downloaded or evaluated.
This is a bounded extraction experiment within M11, not a full graph build or M12.

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

### 2. Download the pinned files only after approval

This is the approximately 6.18 GB download requiring approval under the established
large-download preference. Reading model metadata during preparation did not fetch
weights. The following command has **not** been executed:

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
outputs and their validation, not the preflight result. Run time remains unmeasured.

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

**Check your understanding:** If the larger scout produces perfect forms but links
the wrong people, has the experiment succeeded? No. Correct formatting and correct
meaning must be checked separately.
