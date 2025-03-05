// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"
#include "debug/dprint.h"
#include "debug/dprint_tile.h"

void print_tile(
    uint8_t cb, int tile, uint8_t max_h, uint8_t max_w, bool endl_rows, bool print_untilized, const char* src_loc) {
    DPRINT << "+++++++ print_tile(" << static_cast<int>(cb) << ',' << tile << ") +++++++ " << src_loc << ENDL();
    for (uint8_t r = 0; r < max_h; r++) {
        const auto sr = SliceRange{.h0 = r, .h1 = static_cast<uint8_t>(r + 1), .hs = 1, .w0 = 0, .w1 = max_w, .ws = 1};
        DPRINT << static_cast<int>(r) << ": "
               << TileSlice<64>(cb, tile, sr, TSLICE_INPUT_CB, TSLICE_RD_PTR, endl_rows, print_untilized) << ENDL();
    }
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    const uint32_t src_num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(4);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(5);
    const uint32_t nD_stride = get_arg_val<uint32_t>(6);
    const uint32_t n_stride = get_arg_val<uint32_t>(7);
    const uint32_t c_stride = get_arg_val<uint32_t>(8);
    const uint32_t N = get_arg_val<uint32_t>(9);
    const uint32_t C = get_arg_val<uint32_t>(10);
    const uint32_t Ht = get_arg_val<uint32_t>(11);
    const uint32_t Wt = get_arg_val<uint32_t>(12);
    const uint32_t cND = get_arg_val<uint32_t>(13);  // collapsed dims > 4
    const uint32_t HtWt = Ht * Wt;

    constexpr uint32_t onetile = 1;

    constexpr auto cb_id_src = tt::CBIndex::c_1;
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const DataFormat src_data_format = get_dataformat(cb_id_src);

    const InterleavedAddrGenFast<src_is_dram> src = {
        .bank_base_address = src_addr, .page_size = src_tile_bytes, .data_format = src_data_format};

    constexpr auto cb_id_dst = tt::CBIndex::c_2;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    const uint32_t dst_tile_bytes = get_tile_size(cb_id_dst);
    const DataFormat dst_data_format = get_dataformat(cb_id_dst);

    const InterleavedAddrGenFast<dst_is_dram> dst = {
        .bank_base_address = dst_addr, .page_size = dst_tile_bytes, .data_format = dst_data_format};

    const uint32_t tiles_per_depth = N * C * HtWt;
    uint32_t start_d = start_tile_id / tiles_per_depth;  // ND index
    uint32_t start_remaining_1 = start_tile_id % tiles_per_depth;
    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_remaining_1 / tiles_per_batch;  // N index
    uint32_t start_remaining_2 = start_remaining_1 % tiles_per_batch;
    uint32_t tiles_per_channel = HtWt;
    uint32_t start_c = start_remaining_2 / tiles_per_channel;  // C index
    uint32_t start_t = start_remaining_2 % tiles_per_channel;

    // this is the INPUT tile offset
    uint32_t tile_offset = start_d * nD_stride + start_n * n_stride + start_c * c_stride;
    uint32_t next_batch_shift = n_stride - c_stride * C;
    uint32_t next_depth_shift = nD_stride - (n_stride * N);

    if ((src_num_tiles != 0) || (dst_num_tiles != 0)) {
        DPRINT << "Writer SB: src_num_tiles " << src_num_tiles << " dst_num_tiles " << dst_num_tiles << ENDL();
    }

    uint32_t num_tiles_written = 0;
    for (uint32_t nd = start_d; nd < cND && num_tiles_written < dst_num_tiles; ++nd, start_n = 0) {
        for (uint32_t n = start_n; n < N && num_tiles_written < dst_num_tiles; ++n, start_c = 0) {
            for (uint32_t c = start_c; c < C && num_tiles_written < dst_num_tiles; ++c, start_t = 0) {
                // read a tile from src
                cb_reserve_back(cb_id_src, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_src);
                noc_async_read_tile(tile_offset, src, l1_write_addr);
                noc_async_read_barrier();
                print_tile(cb_id_src, 0, 8, 8, true, true, " before Writer SB fill");
                FILL_TILE_WITH_FIRST_ELEMENT(cb_id_src);
                print_tile(cb_id_src, 0, 8, 8, true, true, " after Writer SB fill");
                cb_push_back(cb_id_src, onetile);

                for (uint32_t t = start_t; t < HtWt && num_tiles_written < dst_num_tiles; ++t, ++num_tiles_written) {
                    // write a tile to dst, since the dst shape is full, the tile offset simply grows linearly
                    cb_wait_front(cb_id_dst, onetile);
                    uint32_t l1_read_addr = get_read_ptr(cb_id_dst);
                    noc_async_write_tile(start_tile_id + num_tiles_written, dst, l1_read_addr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_dst, onetile);
                }
                tile_offset += c_stride;
            }
            tile_offset += next_batch_shift;
        }
        tile_offset += next_depth_shift;
    }

    if ((src_num_tiles != 0) || (dst_num_tiles != 0) || (num_tiles_written != 0)) {
        DPRINT << "Writer SB: num_tiles_written " << num_tiles_written << ENDL();
    }
}
