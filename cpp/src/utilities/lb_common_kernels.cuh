/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <raft/core/device_span.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t>
__device__ __forceinline__ void get_sub_warp_bin(i_t* id_warp_beg,
                                                 i_t* id_range_end,
                                                 i_t* t_p_v,
                                                 raft::device_span<const i_t> warp_offsets,
                                                 raft::device_span<const i_t> warp_id_offsets,
                                                 i_t sub_warp_count)
{
  i_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
  i_t lane_id = threadIdx.x & 31;
  bool pred   = false;
  if (lane_id < warp_offsets.size()) { pred = (warp_id >= warp_offsets[lane_id]); }
  unsigned int m  = __ballot_sync(0xffffffff, pred);
  i_t seg         = 31 - __clz(m);
  i_t it_per_warp = (1 << (5 - seg));  // item per warp = 32/(2^seg)
  if ((5 - seg < 0) || (warp_id >= sub_warp_count)) {
    *t_p_v = 0;
    return;
  }
  i_t beg       = warp_id_offsets[seg] + (warp_id - warp_offsets[seg]) * it_per_warp;
  i_t end       = warp_id_offsets[seg + 1];
  *id_warp_beg  = beg;
  *id_range_end = end;
  *t_p_v        = (1 << seg);
}

template <typename i_t>
__device__ __forceinline__ void get_block_bin(i_t* id_block_beg,
                                              i_t* id_range_end,
                                              i_t* t_p_v,
                                              raft::device_span<const i_t> block_offsets,
                                              raft::device_span<const i_t> block_id_offsets,
                                              i_t sub_warp_block_count,
                                              i_t med_block_count)
{
  i_t lane_id       = threadIdx.x & 31;
  auto med_block_id = blockIdx.x - sub_warp_block_count;
  bool pred         = false;
  if (lane_id < block_offsets.size()) { pred = (med_block_id >= block_offsets[lane_id]); }
  unsigned int m      = __ballot_sync(0xffffffff, pred);
  i_t seg             = 31 - __clz(m);
  i_t threads_per_row = (32 << seg);
  // heavy
  if (threads_per_row > 256) {
    *t_p_v = threads_per_row;
    //*id_block_beg = sub_warp_block_count + med_block_count;
    //*id_range_end = gridDim.x;
    *id_block_beg = -1;
    *id_range_end = -1;
    return;
  } else {
    i_t beg =
      block_id_offsets[seg] + (med_block_id - block_offsets[seg]) * (blockDim.x / threads_per_row);
    i_t end = block_id_offsets[seg + 1];
    // if (threadIdx.x == 0) {
    //   printf("seg %d block_offsets[seg] %d block_id_offsets[seg] %d beg %d end %d\n", seg,
    //   block_offsets[seg], block_id_offsets[seg], beg, end);
    // }
    *id_block_beg = beg;
    *id_range_end = end;
    *t_p_v        = threads_per_row;
  }
}

}  // namespace cuopt::linear_programming::detail
