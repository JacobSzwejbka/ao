// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <torchao/csrc/cpu/torch_free_kernels/fallback/kleidi/pack.h>

#if !defined(__aarch64__)
#include <torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight/kernel_selector.h>
#endif

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace packing = torchao::kernels::cpu::fallback::kleidi::
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p;

template <typename T> T read(const std::vector<uint8_t> &data, size_t offset) {
  T value;
  std::memcpy(&value, data.data() + offset, sizeof(value));
  return value;
}

TEST(KleidiPacking, PacksKleidiLayoutAndPadsOddRowsWithZero) {
  constexpr int n = 1;
  constexpr int k = 32;
  constexpr int group_size = 32;
  constexpr int nr = 8;
  constexpr int kr = 8;
  constexpr int sr = 2;

  std::vector<int8_t> qvals(k);
  for (int i = 0; i < k; ++i) {
    qvals[i] = static_cast<int8_t>(i % 16 - 8);
  }
  const std::vector<float> scales = {1.0f};
  const std::vector<float> bias = {2.0f};

  const size_t packed_size =
      packing::packed_weights_size(n, k, group_size,
                                   /*weight_nbit=*/4,
                                   /*has_weight_zeros=*/false,
                                   /*has_bias=*/true, nr, kr, sr);
  ASSERT_EQ(packed_size, 208);
  std::vector<uint8_t> packed(packed_size, 0xFF);
  packing::pack_weights(packed.data(), n, k, group_size, qvals.data(),
                        scales.data(),
                        /*weight_zeros=*/nullptr, bias.data(), nr, kr, sr);

  // The first two packed words contain q[0], q[16], q[1], q[17] and
  // q[2], q[18], q[3], q[19], respectively. The other seven rows in this
  // portion of the NR panel are the -8 nibble pattern produced by KleidiAI's
  // zero-filled intermediate unsigned-int4 buffer.
  EXPECT_EQ(read<uint16_t>(packed, 0), 0x9988);
  EXPECT_EQ(read<uint16_t>(packed, 2), 0xBBAA);
  for (size_t offset = 4; offset < 32; ++offset) {
    EXPECT_EQ(packed[offset], 0x88);
  }

  constexpr size_t scales_offset = nr * group_size / 2;
  EXPECT_EQ(read<uint16_t>(packed, scales_offset), 0x3F80);
  for (size_t offset = scales_offset + sizeof(uint16_t);
       offset < scales_offset + nr * sizeof(uint16_t); ++offset) {
    EXPECT_EQ(packed[offset], 0);
  }

  constexpr size_t sums_offset = scales_offset + nr * sizeof(uint16_t);
  EXPECT_FLOAT_EQ(read<float>(packed, sums_offset), -16.0f);
  for (int row = 1; row < nr; ++row) {
    EXPECT_FLOAT_EQ(read<float>(packed, sums_offset + row * sizeof(float)),
                    0.0f);
  }

  constexpr size_t bias_offset = sums_offset + nr * sizeof(float);
  EXPECT_FLOAT_EQ(read<float>(packed, bias_offset), 2.0f);
  for (int row = 1; row < nr; ++row) {
    EXPECT_FLOAT_EQ(read<float>(packed, bias_offset + row * sizeof(float)),
                    0.0f);
  }
}

#if !defined(__aarch64__)
TEST(KleidiPacking, SelectsPackingOnlyConfigOnNonArmHosts) {
  using namespace torchao::ops::linear_8bit_act_xbit_weight;
  auto default_format = select_packed_weights_format<4>(
      std::nullopt, /*has_weight_zeros=*/false, /*has_bias=*/false);
  EXPECT_EQ(
      default_format.type,
      torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal);

  auto format =
      select_packed_weights_format<4>(std::optional<std::string>("kleidiai"),
                                      /*has_weight_zeros=*/false,
                                      /*has_bias=*/false);

  EXPECT_EQ(
      format.type,
      torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_kleidi_ai);
  EXPECT_EQ(format.nr, 8);
  EXPECT_EQ(format.kr, 8);
  EXPECT_EQ(format.sr, 2);

  auto config = select_ukernel_config<4>(format);
  EXPECT_EQ(config.pack_weights, &packing::pack_weights);
  EXPECT_THROW(config.linear_configs[0].packed_activations_size(
                   /*m=*/1,
                   /*k=*/32,
                   /*group_size=*/32,
                   /*has_weight_zeros=*/false,
                   /*mr=*/1,
                   /*kr=*/8,
                   /*sr=*/2),
               std::runtime_error);
}
#endif // !__aarch64__

} // namespace
