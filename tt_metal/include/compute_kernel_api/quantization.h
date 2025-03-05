// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_quant.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

#include <debug/dprint.h>
#include <debug/dprint_tile.h>
#include <debug/dprint_tensix.h>

namespace ckernel {

void print_tile(
    uint8_t cb, int tile, uint8_t max_h, uint8_t max_w, bool endl_rows, bool print_untilized, const char* src_loc) {
    DPRINT << "+++++++ print_tile(" << static_cast<int>(cb) << ',' << tile << ") +++++++ " << src_loc << ENDL();
    // If the number of entries is small, construct a single SliceRange or just use SliceRange::hw041()
    for (uint8_t r = 0; r < max_h; r++) {
        const auto sr = SliceRange{.h0 = r, .h1 = static_cast<uint8_t>(r + 1), .hs = 1, .w0 = 0, .w1 = max_w, .ws = 1};
        DPRINT << static_cast<int>(r) << ": " << TileSlice<64>(cb, tile, sr, endl_rows, print_untilized) << ENDL();
    }
}

// clang-format off
/**
 * Performs an elementwise per-tensor affine quantization operation on the first operand using the scaling factor in the second operand.
 * Output overwrites first operand in DST.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void quant_tile(uint32_t idst0, uint32_t idst1) {
    UNPACK(print_tile(0, 0, 3, 3, true, true, __PRETTY_FUNCTION__);
           print_tile(1, 0, 3, 3, true, true, __PRETTY_FUNCTION__);
           print_tile(1, 1, 3, 3, true, true, __PRETTY_FUNCTION__);
           DPRINT << RAISE(0););
    MATH(DPRINT << WAIT(0); llk_math_eltwise_binary_sfpu_quant_int32<APPROX>(idst0, idst1); DPRINT << RAISE(1););
    PACK(DPRINT << WAIT(1););
}

ALWI void requant_tile(uint32_t idst0, uint32_t idst1) {
    UNPACK(print_tile(0, 0, 3, 3, true, true, __PRETTY_FUNCTION__);
           print_tile(1, 0, 3, 3, true, true, __PRETTY_FUNCTION__);
           print_tile(1, 1, 3, 3, true, true, __PRETTY_FUNCTION__);
           DPRINT << RAISE(0););
    MATH(DPRINT << WAIT(0); llk_math_eltwise_binary_sfpu_requant_int32<APPROX>(idst0, idst1); DPRINT << RAISE(1));
    PACK(DPRINT << WAIT(1);)
}

ALWI void dequant_tile(uint32_t idst0, uint32_t idst1) {
    UNPACK(print_tile(0, 0, 3, 3, true, true, __PRETTY_FUNCTION__);
           print_tile(1, 0, 3, 3, true, true, __PRETTY_FUNCTION__);
           print_tile(1, 1, 3, 3, true, true, __PRETTY_FUNCTION__);
           DPRINT << RAISE(0););
    MATH(DPRINT << WAIT(0); llk_math_eltwise_binary_sfpu_dequant_int32<APPROX>(idst0, idst1); DPRINT << RAISE(1););
    PACK(DPRINT << WAIT(1););
}

// clang-format off
/**
 * Initialize the sfpu with the zero point argument of the quantization Op.
 * To be called once at beginning of a kernel.
 *
 * Return value: None
 *
 * | Argument   | Description                           | Data type | Valid range | Required |
 * |------------|---------------------------------------|-----------|-------------|----------|
 * | zero_point | The zero point of the quantization Op | uint32_t  | Any number  | Yes      |
 * */
// clang-format on
ALWI void quant_tile_init(const uint32_t zero_point) {
    MATH((llk_math_eltwise_binary_sfpu_quant_int32_init<APPROX>(zero_point)));
}
ALWI void requant_tile_init(const uint32_t zero_point) {
    MATH((llk_math_eltwise_binary_sfpu_requant_int32_init<APPROX>(zero_point)));
}
ALWI void dequant_tile_init(const uint32_t zero_point) {
    MATH((llk_math_eltwise_binary_sfpu_dequant_int32_init<APPROX>(zero_point)));
}

}  // namespace ckernel
