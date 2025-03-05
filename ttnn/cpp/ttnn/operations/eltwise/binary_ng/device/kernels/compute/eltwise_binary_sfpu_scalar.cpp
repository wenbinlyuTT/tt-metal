// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/binary_shift.h"
#include "compute_kernel_api/add_int32_sfpu.h"
#include "compute_kernel_api/quantization.h"

#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

#include "debug/dprint.h"
#include <debug/dprint_tile.h>
#include <debug/dprint_tensix.h>

namespace NAMESPACE {

void print_tile(
    uint8_t cb, int tile, uint8_t max_h, uint8_t max_w, bool endl_rows, bool print_untilized, const char* src_loc) {
    DPRINT << "+++++++ print_tile(" << static_cast<int>(cb) << ',' << tile << ") +++++++ " << src_loc << ENDL();
    // If the number of entries is small, construct a single SliceRange or just use SliceRange::hw041()
    for (uint8_t r = 0; r < max_h; r++) {
        const auto sr = SliceRange{.h0 = r, .h1 = static_cast<uint8_t>(r + 1), .hs = 1, .w0 = 0, .w1 = max_w, .ws = 1};
        DPRINT << static_cast<int>(r) << ": " << TileSlice<64>(cb, tile, sr, endl_rows, print_untilized) << ENDL();
    }
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    UNPACK(print_tile(1, 0, 3, 3, true, true, __PRETTY_FUNCTION__));
    UNPACK(print_tile(1, 1, 3, 3, true, true, __PRETTY_FUNCTION__));

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    UNPACK(DPRINT << "+ SFPU Ker: n_tiles " << num_tiles << " n_tiles/cycle " << num_tiles_per_cycle << " CB"
                  << static_cast<int>(cb_post_lhs) << "->dst0 CB" << static_cast<int>(cb_post_rhs) << "->dst1"
                  << ENDL(););

    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS))
    BINARY_SFPU_INIT
#endif

    PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
    cb_wait_front(cb_post_rhs, num_tiles_per_cycle);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);

        cb_reserve_back(cb_out, num_tiles_per_cycle);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)
        BINARY_SFPU_INIT
#endif
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs, 0, true);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            dprint_tensix_dest_reg(i * 2, " start");
            dprint_tensix_dest_reg(i * 2 + 1, " start");
            copy_tile(cb_post_lhs, i, i * 2, true);
        }
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs, 0, true);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            dprint_tensix_dest_reg(i * 2, " after lhs copy");
            copy_tile(cb_post_rhs, i, i * 2 + 1, true);
            dprint_tensix_dest_reg(i * 2, " after rhs copy");
            dprint_tensix_dest_reg(i * 2 + 1, " after rhs copy");
            BINARY_SFPU_OP(i * 2, i * 2 + 1);
            dprint_tensix_dest_reg(i * 2, " after sfpu op");
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        tile_regs_commit();

        tile_regs_wait();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);
        }
        tile_regs_release();

        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
        cb_push_back(cb_out, num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
