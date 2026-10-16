// One keyed categorical draw from a logits row: temperature, top-k filter,
// softmax and the Gumbel-max draw in one dispatch. Bit-identical to the
// composed MLX chain `divide(x, temperature)` -> `where(x >= kth, x, -inf)`
// -> `softmax` -> `mlx_array_sample_probabilities` on the same key:
//   - softmax mirrors `softmax_single_row` (N_READS=4 contiguous chunks, a
//     threadgroup of 32 * ceil(V / 128), the two simd_max / simd_sum folds,
//     `fast::exp`, `1 / sum`, then the product);
//   - the k-th largest value is exact (radix select over order-preserving
//     keys, every NaN above all numbers as in MLX's sort) and is applied
//     with a float `>=`, as the sort threshold is;
//   - uniforms reproduce `rbitsc` (threefry2x32; word j < G comes from
//     count (j, j + G) and word j >= G from (j - G, j), G = ceil(V / 2), the
//     last low word of an odd row from (G - 1, 0)) and `random::uniform`
//     (divide by 2^32, minimum with the float below one), then
//     `maximum(u, FLT_MIN)`, `-log(-log(u))`, `precise::log(p)` on finite
//     positive p, and argmax with the lowest index on ties.
//
// Bound names (fast::metal_kernel generates the signature):
//   logits      : const device T*            [1, V] row-contiguous
//   key         : const constant uint32_t*   [2] (high word, low word)
//   temperature : const constant float&
//   top_k       : const constant int&        filter when 0 < top_k < V
//   token       : device int32_t*            [1, 1]; -1 without finite positive mass
//
// Template: T (float / bfloat16_t / half), V (row width, V <= 4096).
// Grid: (32 * ceil(V / 128), 1, 1); threadgroup the same.
R"(
  {
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    constexpr int TG = SIMD_SIZE * ((V + N_READS * SIMD_SIZE - 1) / (N_READS * SIMD_SIZE));
    constexpr uint HALF = (V + 1) / 2;
    const int lid = int(thread_position_in_threadgroup.x);
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = simdgroup_index_in_threadgroup;
    const int first = lid * N_READS;

    threadgroup atomic_uint hist[256];
    threadgroup uint pick[2];
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];
    threadgroup float local_best[SIMD_SIZE];
    threadgroup uint local_index[SIMD_SIZE];

    float ld[N_READS];
    for (int i = 0; i < N_READS; i++) {
      ld[i] = first + i < V
          ? static_cast<float>(logits[first + i]) / temperature
          : Limits<float>::min;
    }

    if (top_k > 0 && top_k < V) {
      uint keys[N_READS];
      for (int i = 0; i < N_READS; i++) {
        const uint bits = as_type<uint>(ld[i]);
        keys[i] = isnan(ld[i]) ? 0xffffffffu
            : (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
      }
      uint prefix = 0;
      uint mask = 0;
      uint remaining = uint(top_k);
      for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = lid; b < 256; b += TG) {
          atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < N_READS; i++) {
          if (first + i < V && (keys[i] & mask) == prefix) {
            atomic_fetch_add_explicit(
                &hist[(keys[i] >> shift) & 0xffu], 1u, memory_order_relaxed);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
          uint counts[8];
          uint sum = 0;
          for (int j = 0; j < 8; j++) {
            counts[j] = atomic_load_explicit(&hist[lane * 8 + j], memory_order_relaxed);
            sum += counts[j];
          }
          const uint above = simd_sum(sum) - simd_prefix_inclusive_sum(sum);
          if (above < remaining && remaining <= above + sum) {
            uint seen = above;
            for (int j = 7; j >= 0; j--) {
              if (seen + counts[j] >= remaining) {
                pick[0] = lane * 8 + j;
                pick[1] = remaining - seen;
                break;
              }
              seen += counts[j];
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= pick[0] << shift;
        mask |= 0xffu << shift;
        remaining = pick[1];
      }
      const float kth =
          as_type<float>((prefix & 0x80000000u) ? (prefix & 0x7fffffffu) : ~prefix);
      for (int i = 0; i < N_READS; i++) {
        ld[i] = ld[i] >= kth ? ld[i] : Limits<float>::min;
      }
    }

    if (sgid == 0) {
      local_max[lane] = Limits<float>::min;
      local_normalizer[lane] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxval = Limits<float>::finite_min;
    for (int i = 0; i < N_READS; i++) {
      maxval = (maxval < ld[i]) ? ld[i] : maxval;
    }
    maxval = simd_max(maxval);
    if (lane == 0) {
      local_max[sgid] = maxval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
      maxval = simd_max(local_max[lane]);
      if (lane == 0) {
        local_max[0] = maxval;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];
    float normalizer = 0;
    for (int i = 0; i < N_READS; i++) {
      const float exp_x = metal::fast::exp(ld[i] - maxval);
      ld[i] = exp_x;
      normalizer += exp_x;
    }
    normalizer = simd_sum(normalizer);
    if (lane == 0) {
      local_normalizer[sgid] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
      normalizer = simd_sum(local_normalizer[lane]);
      if (lane == 0) {
        local_normalizer[0] = normalizer;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = 1 / local_normalizer[0];

    const uint k0 = key[0];
    const uint k1 = key[1];
    const uint ks[3] = {k0, k1, k0 ^ k1 ^ 0x1BD11BDAu};
    const uint rotations[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
    const float below_one = as_type<float>(0x3f7fffffu);
    const float smallest = as_type<float>(0x00800000u);
    float best = Limits<float>::min;
    uint best_index = 0;
    for (int i = 0; i < N_READS; i++) {
      const uint j = uint(first + i);
      if (j < uint(V)) {
        const float p = ld[i] * normalizer;
        const float logit = (isfinite(p) && p > 0) ? metal::precise::log(p) : Limits<float>::min;
        const bool low = j < HALF;
        uint2 v = low ? uint2(j, (V % 2 == 1 && j == HALF - 1) ? 0u : j + HALF)
                      : uint2(j - HALF, j);
        v.x += ks[0];
        v.y += ks[1];
        for (int r = 0; r < 5; r++) {
          for (int s = 0; s < 4; s++) {
            const uint rot = rotations[r % 2][s];
            v.x += v.y;
            v.y = (v.y << rot) | (v.y >> (32 - rot));
            v.y ^= v.x;
          }
          v.x += ks[(r + 1) % 3];
          v.y += ks[(r + 2) % 3] + uint(r) + 1;
        }
        float u = static_cast<float>(low ? v.x : v.y) / 4294967296.0f;
        u = u < below_one ? u : below_one;
        u = u > smallest ? u : smallest;
        const float score = -metal::precise::log(-metal::precise::log(u)) + logit;
        if (score > best) {
          best = score;
          best_index = j;
        }
      }
    }
    const float simd_best = simd_max(best);
    const uint simd_index = simd_min(best == simd_best ? best_index : 0xffffffffu);
    if (lane == 0) {
      local_best[sgid] = simd_best;
      local_index[sgid] = simd_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
      const bool live = lane < uint(TG / SIMD_SIZE);
      const float value = live ? local_best[lane] : Limits<float>::min;
      const uint index = live ? local_index[lane] : 0xffffffffu;
      const float top = simd_max(value);
      const uint winner = simd_min(value == top ? index : 0xffffffffu);
      if (lane == 0) {
        token[0] = top > Limits<float>::min ? int(winner) : -1;
      }
    }
  }
)"
