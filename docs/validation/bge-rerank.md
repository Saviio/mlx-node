# BGE-v2-M3 validation

This report accompanies the `feat/rerank` implementation based on `1fc3baee`.
The machine is an Apple M4 (Mac16,12), 32 GiB unified memory. No merge or release
is authorized by a successful build alone. Gates still in progress below are
not accepted support claims.

The [manifest](../../fixtures/rerank/bge/manifest.json) freezes the checkpoint,
dataset revisions, reference packages and thresholds. The
[reproduction guide](../../scripts/rerank/bge/README.md) describes each command.
The compact evidence file records model/tokenizer, reference and addon hashes;
full stage tensors, rankings and raw timing samples stay in `.cache/bge-validation`.

## Numerical correctness

| Storage / reference                                 | Short inputs | 512 / 2048 / 8192 tokens | Maximum logit error |
| --------------------------------------------------- | ------------ | ------------------------ | ------------------- |
| FP32 / official Transformers                        | Pass         | Pass                     | `4.04715538e-5`     |
| FP16 / same-checkpoint Python MLX                   | Pass         | Pass                     | `0`                 |
| BF16 / same-checkpoint Python MLX                   | Pass         | Pass                     | `0`                 |
| affine 4-bit, group 64 / same-checkpoint Python MLX | Pass         | Pass                     | `0`                 |
| MXFP4, group 32 / same-checkpoint Python MLX        | Pass         | Pass                     | `0`                 |

FP32's largest case mean logit error is `2.07424164e-5`; its largest public-score
error is `8.50104225e-6`. Every frozen maximum/mean threshold and margin-qualified
ranking assertion passed. Tokens, masks, position IDs, truncation and usage match
exactly. Coverage includes Chinese/English, empty candidates/documents, duplicates,
special-token text and literal pad tokens. A mixed checkpoint with FP32 affine
sidecars and a quantized classifier also passes; FP16 with compilation disabled
passes against the independently disabled reference.

The Python MLX implementation loads safetensors directly, without the native
loader or inference. It uses standard `mlx.nn.gelu`; dense projections use explicit
matmul then bias, matching the documented native projection rounding boundary.
The official Transformers FP32 chain independently checks the architecture.
An initially observed low-precision GELU difference was fixed with a model-owned
compiled exact GELU expression using the existing compiled-graph API. No shared
activation, chat transformer or Metal kernel changed. Thresholds were not widened.

Batch sizes 1 and 8 each match their same-policy reference exactly. Changing the
batch shape has ordinary low-precision rounding effects in **both** independent
implementations: maximum FP16/BF16 logit changes are `0.00390625` / `0.0625`, and
public-score changes are `4.4663e-6` / `1.8011e-4`. No margin-qualified inversion
occurred in the frozen batch fixture; reversing candidate order produced zero
logit change after restoring indices. This is not a claim of bitwise invariance
across batch policies.

All eight existing Jina fixed-checkpoint golden tests pass against their unchanged
references, including community affine/MXFP4 layouts. Jina remains cosine-scored.

## Quantization quality

In progress. T2Reranking and AskUbuntu each use 256 queries ordered by query
SHA-256 with all original candidates retained. The selected subsets contain
4,131 and 5,120 candidates respectively. T2 selection records eight rejected
samples; AskUbuntu records none. Inputs and labels are hashed, and baseline
comparison rejects any input-identity mismatch before loading a model.

Both 4-bit formats must pass the maximum `0.02` nDCG@10 drop independently on
each dataset. No partial result is treated as a pass or as a full MTEB score.

## Lifecycle, interfaces and regressions

Completed checks include family dispatch and policy rejection, native-free imports,
sigmoid-only SDK/HTTP contracts, the host residency matrix for chat/ASR/TTS plus
Jina/BGE, and synthetic native loading/conversion/cancellation cases. The complete
pure-TS suite passed: 2,115 tests in 108 files, run serially after parallel timing
failures in existing supervisor/coordinator tests. TypeScript and clippy pass.

The first full Rust core run passed 3,989 tests, ignored 161, and failed two: an
expected conversion-registry list was updated for the new family; the existing
Qwen3.5 segmented-attention calibration exceeded its 100 ms timing bound under
concurrent build load. The latter also exceeded its bound on an immediate retest.
It remains unresolved until a quiet-machine retest; unchanged source alone does
not establish a baseline pass. Full final native/integration checks remain pending.

Independent pair audit found and corrected incomplete-download revision provenance,
quality-baseline input identity and performance-script failure signaling. Partial
download markers invalidate older config/Hugging Face revision hints; unknown
provenance remains null. Explicit document policies exceeding effective context
are rejected. Audit is evidence of review, not a guarantee of no defects.

## Performance and acceptance

Pending. Performance runs must execute without simultaneous builds or other
inference jobs. The frozen matrix uses three warmups and ten measurements, with
median/P95, padding, active/cache/peak memory, cancellation and unload checks.
Required FP16 125/200-candidate, approximately 512-token cases must improve
throughput by at least 20% with default batching.

Separate baseline/current chat and Jina runs enforce the three-round 5% regression
rule. BGE unloaded, idle resident and concurrently executing are measured separately
so resource contention is not described as zero overhead. Until quality, lifecycle,
regression and performance gates finish, this branch is not ready for merge.
