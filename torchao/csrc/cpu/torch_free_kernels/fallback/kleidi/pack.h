// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace torchao::kernels::cpu::fallback::kleidi::
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p {

namespace internal {

inline size_t roundup(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

inline uint16_t get_bf16_from_float(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

inline float get_float_from_bf16(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline void check_layout(int k, int group_size, int nr, int kr, int sr) {
  if (group_size % 32 != 0) {
    throw std::runtime_error(
        "Group size must be a multiple of 32, but got group_size=" +
        std::to_string(group_size));
  }
  if (k % group_size != 0) {
    throw std::runtime_error(
        "k must be a multiple of group size, but got k=" + std::to_string(k) +
        " and group_size=" + std::to_string(group_size));
  }
  if (group_size % kr != 0 || kr % sr != 0 || nr % 4 != 0) {
    throw std::runtime_error("Invalid KleidiAI packing parameters");
  }
}

inline size_t packed_panel_size(int k, int group_size, int nr) {
  const size_t num_blocks_per_row = k / group_size;
  const size_t bytes_per_block = group_size / 2 + sizeof(uint16_t);
  return nr *
         (num_blocks_per_row * bytes_per_block + sizeof(float) + sizeof(float));
}

inline bool is_zero_padding_row(int row, int n) {
  // The KleidiAI packer first pads odd N to an even value with a zero row,
  // then repeats its final row while filling an NR-sized panel.
  return n % 2 != 0 && row >= n;
}

inline int source_row(int row, int n) { return std::min(row, n - 1); }

inline int8_t get_qval(const int8_t *weight_qvals, int row, int k_idx, int n,
                       int k) {
  if (is_zero_padding_row(row, n)) {
    // The intermediate unsigned int4 buffer is zero-filled. After the
    // KleidiAI packer removes its zero point of 8, this is encoded as -8.
    return -8;
  }
  return weight_qvals[source_row(row, n) * k + k_idx];
}

inline float get_scale(const float *weight_scales, int row, int block, int n,
                       int blocks_per_row) {
  if (is_zero_padding_row(row, n)) {
    return 0.0f;
  }
  return weight_scales[source_row(row, n) * blocks_per_row + block];
}

inline float get_bias(const float *bias, int row, int n) {
  if (bias == nullptr || is_zero_padding_row(row, n)) {
    return 0.0f;
  }
  return bias[source_row(row, n)];
}

} // namespace internal

inline size_t packed_weights_size(int n, int k, int group_size, int weight_nbit,
                                  bool has_weight_zeros, bool has_bias, int nr,
                                  int kr, int sr) {
  (void)has_weight_zeros;
  (void)has_bias;
  if (weight_nbit != 4) {
    throw std::runtime_error("KleidiAI packing only supports 4-bit weights");
  }
  internal::check_layout(k, group_size, nr, kr, sr);
  return internal::roundup(n, nr) / nr *
         internal::packed_panel_size(k, group_size, nr);
}

inline size_t packed_weights_offset(int n_idx, int k, int group_size,
                                    int weight_nbit, bool has_weight_zeros,
                                    bool has_bias, int nr, int kr, int sr) {
  (void)has_weight_zeros;
  (void)has_bias;
  if (weight_nbit != 4) {
    throw std::runtime_error("KleidiAI packing only supports 4-bit weights");
  }
  internal::check_layout(k, group_size, nr, kr, sr);
  if (n_idx % nr != 0) {
    throw std::runtime_error("n_idx must be a multiple of nr");
  }
  return n_idx / nr * internal::packed_panel_size(k, group_size, nr);
}

inline void pack_weights(void *packed_weights, int n, int k, int group_size,
                         const int8_t *weight_qvals, const float *weight_scales,
                         const int8_t *weight_zeros, const float *bias, int nr,
                         int kr, int sr) {
  internal::check_layout(k, group_size, nr, kr, sr);
  if (n < 1) {
    throw std::runtime_error("n must be at least 1");
  }

  const int blocks_per_row = k / group_size;
  if (weight_zeros != nullptr) {
    for (int i = 0; i < n * blocks_per_row; ++i) {
      if (weight_zeros[i] != 0) {
        throw std::runtime_error(
            "KleidiAI packing does not support weight zeros");
      }
    }
  }

  const int block_length_in_bytes = kr / sr;
  const size_t panel_size = internal::packed_panel_size(k, group_size, nr);
  auto *output = static_cast<uint8_t *>(packed_weights);

  for (int panel_row = 0; panel_row < n; panel_row += nr) {
    uint8_t *panel = output + panel_row / nr * panel_size;
    uint8_t *dst = panel;
    std::vector<float> sums(nr, 0.0f);

    for (int block = 0; block < blocks_per_row; ++block) {
      uint8_t *packed_scales = dst + nr * group_size / 2;

      for (int row_in_panel = 0; row_in_panel < nr; ++row_in_panel) {
        const int row = panel_row + row_in_panel;
        const float scale =
            internal::get_scale(weight_scales, row, block, n, blocks_per_row);
        const uint16_t scale_bf16 = internal::get_bf16_from_float(scale);
        std::memcpy(packed_scales + row_in_panel * sizeof(scale_bf16),
                    &scale_bf16, sizeof(scale_bf16));
      }

      int k_base = block * group_size;
      for (int byte_idx = 0; byte_idx < group_size / 2; byte_idx += 16) {
        for (int segment = 0; segment < 16 / block_length_in_bytes; ++segment) {
          for (int row_in_panel = 0; row_in_panel < nr; ++row_in_panel) {
            const int row = panel_row + row_in_panel;
            int k0 = k_base;
            int k1 = k_base + 16;
            int32_t partial_sum = 0;

            for (int block_byte = 0; block_byte < block_length_in_bytes;
                 block_byte += 2) {
              const int8_t q0 = internal::get_qval(weight_qvals, row, k0, n, k);
              const int8_t q1 = internal::get_qval(weight_qvals, row, k1, n, k);
              const int8_t q2 =
                  internal::get_qval(weight_qvals, row, k0 + 1, n, k);
              const int8_t q3 =
                  internal::get_qval(weight_qvals, row, k1 + 1, n, k);

              const uint16_t packed = (static_cast<uint8_t>(q0) & 0x0F) |
                                      ((static_cast<uint8_t>(q1) & 0x0F) << 4) |
                                      ((static_cast<uint8_t>(q2) & 0x0F) << 8) |
                                      ((static_cast<uint8_t>(q3) & 0x0F) << 12);
              std::memcpy(dst, &packed, sizeof(packed));
              dst += sizeof(packed);

              partial_sum += q0 + q1 + q2 + q3;
              k0 += 2;
              k1 += 2;
            }

            uint16_t scale_bf16;
            std::memcpy(&scale_bf16,
                        packed_scales + row_in_panel * sizeof(scale_bf16),
                        sizeof(scale_bf16));
            sums[row_in_panel] += static_cast<float>(partial_sum) *
                                  internal::get_float_from_bf16(scale_bf16);
          }
          k_base += block_length_in_bytes;
        }
        k_base += 16;
      }
      dst = packed_scales + nr * sizeof(uint16_t);
    }

    std::memcpy(dst, sums.data(), nr * sizeof(float));
    dst += nr * sizeof(float);
    for (int row_in_panel = 0; row_in_panel < nr; ++row_in_panel) {
      const float value = internal::get_bias(bias, panel_row + row_in_panel, n);
      std::memcpy(dst, &value, sizeof(value));
      dst += sizeof(value);
    }
    assert(dst == panel + panel_size);
  }
}

inline size_t get_preferred_alignement() { return 16; }

} // namespace
  // torchao::kernels::cpu::fallback::kleidi::kai_matmul_clamp_f32_qai8dxp_qsi4c32p
  // kai_matmul_clamp_f32_qai8dxp_qsi4c32p
