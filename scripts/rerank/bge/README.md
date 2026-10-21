# Reproduce the BGE validation

Run from the repository root on Apple Silicon. The production addon needs no
Python; this directory is an offline acceptance harness. The pinned model,
datasets, environment and thresholds live in `fixtures/rerank/bge/manifest.json`.
`reference-lock.txt` records the full Python environment used for the report.
Never compare different quantized checkpoints to establish implementation parity.

```sh
python3.11 -m venv .cache/bge-reference
.cache/bge-reference/bin/pip install -r scripts/rerank/bge/requirements.txt
.cache/bge-reference/bin/python scripts/rerank/bge/prepare.py .cache/bge-validation
yarn build
node scripts/rerank/bge/convert.mjs .cache/bge-validation/fp32 .cache/bge-validation
```

Conversion deliberately refuses an existing nonempty destination. Use a new
root when changing conversion settings. Keep enough free disk and memory for
the FP32 reference and the converted checkpoints.

## Independent numerical references

The Torch chain executes the official Transformers architecture in FP32. The
functional Python MLX chain reads tensors directly and implements XLM-R without
importing the addon, native loader or native inference. Dense projections use
explicit matmul followed by bias; GELU uses standard compiled `mlx.nn.gelu`.
Packed projection boundaries restore the activation dtype. These conventions
fix low-precision evaluation order; thresholds remain frozen in the manifest.

```sh
.cache/bge-reference/bin/python scripts/rerank/bge/reference.py torch \
  .cache/bge-validation/fp32 .cache/bge-validation/fp32-reference.json
.cache/bge-reference/bin/python scripts/rerank/bge/reference.py mlx \
  .cache/bge-validation/fp16 .cache/bge-validation/fp16-reference.json
MLX_BGE_GOLDEN_MODEL=.cache/bge-validation/fp16 \
MLX_BGE_GOLDEN_REFERENCE=.cache/bge-validation/fp16-reference.json \
  cargo test -p mlx-core --lib golden_checkpoint_logits -- --ignored --nocapture --test-threads=1
```

Repeat for FP32/Torch and BF16, affine, MXFP4/MLX. Use `--cases
fixtures/rerank/bge/long-cases.json` and set `MLX_BGE_GOLDEN_CASES` to the same
file for 512/2048/8192-token cases. The harness compares exact token IDs,
positions, masks, usage and truncation; records per-layer CLS differences; then
enforces both internal-logit and public-sigmoid thresholds and ranking margins.
Internal logits exist only in these offline reference files and test output.

For mixed weights and a quantized classifier:

```sh
.cache/bge-reference/bin/python scripts/rerank/bge/mixed-fixture.py \
  .cache/bge-validation/affine .cache/bge-validation/mixed
```

Run the same MLX/reference test for `mixed`. To verify the compiler-disabled
execution mode, set `MLX_DISABLE_COMPILE=1` in **both** independent processes
and use a separate reference output. This does not alter production defaults.

## Fixed-subset quality

```sh
.cache/bge-reference/bin/python scripts/rerank/bge/quality-inputs.py \
  .cache/bge-validation/fp32 .cache/bge-validation/datasets .cache/bge-validation/quality-inputs.json
node scripts/rerank/bge/quality.mjs .cache/bge-validation/fp32 \
  .cache/bge-validation/quality-inputs.json .cache/bge-validation/quality-fp32.json
node scripts/rerank/bge/quality.mjs .cache/bge-validation/affine \
  .cache/bge-validation/quality-inputs.json .cache/bge-validation/quality-affine.json \
  .cache/bge-validation/quality-fp32.json
```

Repeat the last command for MXFP4. Each dataset contributes 256 queries sorted
by SHA-256, with original positive/negative candidates and recorded skips. Both
quantized formats must meet the nDCG@10 drop gate separately on each dataset.
Results describe this fixed subset, not a full MTEB benchmark. Per-query progress
is persisted; the same command resumes only when input, model and addon hashes
match. The recovery key includes the dataset and original sample index, so
repeated query text with different candidates remains distinct. A changed artifact
requires a new output path.

Quality acceptance pins the official FP32 file hashes in
[`fp32-checksums.json`](../../../fixtures/rerank/bge/fp32-checksums.json). It checks
actual quantization format and policy, binds the FP32 baseline report, addon,
Metal library and execution environment, and recomputes metrics from all candidate
indices. Schema-3 journals cannot resume older evidence missing that identity.
`MLX_DISABLE_COMPILE` must be absent. `quality-summary.mjs ROOT` verifies evidence
without importing the addon; `report.py` uses this verification and leaves missing,
partial, wrong-revision or inconsistent results unaccepted.

Run the CPU evidence-contract regression tests with:

```sh
node --test scripts/rerank/bge/quality-contract.test.mjs
```

## Performance and regression

```sh
node scripts/rerank/bge/benchmark.mjs .cache/bge-validation/fp16 .cache/bge-validation/performance-fp16.json
```

Run performance jobs alone, without builds or other inference jobs. Each case
warms up three times and measures ten times. The required 512-token cases run
first for both batch sizes; a failed dense FP16 throughput target stops before
the longer matrix and leaves the report incomplete. Passing that precheck does
not replace the full matrix or resource checks. Completed reports set `complete`
only after all scenarios and checks finish; reports contain all samples,
median/P95, padded workload, active/cache/peak memory and cancellation cycles.
Each completed scenario is saved atomically. Repeating the same command resumes
only when the model, tokenizer, addon, Metal library, script, manifest and host
identity and execution-environment fingerprint match. Performance acceptance
requires compilation enabled (`MLX_DISABLE_COMPILE` must be absent, including an
empty value). Interrupted cancellation cycles restart; completed timing samples
retain their timestamps. The optional final argument `512` runs only the batching
acceptance scenarios and needs a separate output path.
Cold-model figures include loading and the first request; integrity hashing warms
the filesystem cache, so they are not cold-storage measurements. Do not infer
speed from memory reduction. Keep baseline-addon comparisons and
unloaded/idle/concurrent measurements separate from numerical tests.

Freeze a copy of the baseline native addon **before** building this branch, with
its adjacent `mlx.metallib` and `paged_attn.metallib`. `NAPI_RS_NATIVE_LIBRARY_PATH`
selects the exact binary for each run. Also prepare an independent archive of the
frozen baseline commit, build its TypeScript packages, and write the full baseline
SHA to `source-revision.txt` in that archive. Its workspace dependency links must
resolve inside the archive; only third-party dependencies may be shared. Pass that
root through `MLX_BGE_SOURCE_ROOT`. Switching the addon alone does not measure SDK
or Server regressions. The sampler verifies package resolution and records source,
built-code, lockfile, checkpoint, tokenizer and execution-environment fingerprints.
Both runs use the same sampling script and compare actual chat output token counts.
Use the same checkpoint paths throughout:

```sh
MLX_BGE_SOURCE_ROOT=/absolute/baseline-source \
NAPI_RS_NATIVE_LIBRARY_PATH=/absolute/baseline/mlx-core.darwin-arm64.node \
  yarn exec oxnode scripts/rerank/bge/regression.ts CHAT_MODEL JINA_MODEL baseline.json
NAPI_RS_NATIVE_LIBRARY_PATH=/absolute/current/mlx-core.darwin-arm64.node \
  yarn exec oxnode scripts/rerank/bge/regression.ts CHAT_MODEL JINA_MODEL current.json BGE_MODEL
node scripts/rerank/bge/compare-regression.mjs baseline.json current.json comparison.json
yarn exec oxnode scripts/rerank/bge/surfaces.ts BGE_MODEL surfaces.json
```

`regression.ts` collects three rounds; only `compare-regression.mjs` applies the
frozen regression gate and returns failure when all three rounds exceed it.
Idle residency and concurrent inference measurements are reported separately.
`benchmark.mjs` also returns failure when either required dense FP16 batching
scenario misses the throughput target; other precisions report measurements with
that target explicitly marked not applicable. The surface test uses real weights through SDK, HTTP and
CLI and checks identical normalized results and BGE-only service startup.

```sh
python3 scripts/rerank/bge/report.py .cache/bge-validation docs/validation/bge-results.json
```

The compact report records checkpoint/tokenizer, reference and addon SHA-256
hashes. It deliberately keeps missing or partial gates pending. Raw per-query
quality rankings, timings and stage references remain in the validation folder.

The report at `docs/validation/bge-rerank.md` distinguishes completed gates,
baseline failures, resource contention and any remaining acceptance work.
