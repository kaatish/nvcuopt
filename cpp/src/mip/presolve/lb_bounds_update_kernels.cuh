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

#include <mip/utils.cuh>
#include <raft/core/device_span.hpp>
#include <utilities/lb_common_kernels.cuh>

namespace cuopt::linear_programming::detail {

// template <typename i_t, typename f_t, i_t BDIM, typename view_t, typename update_view_t>
//__global__ void cnst_slack_kernel(view_t view,
//                                   update_view_t upd0,
//                                   update_view_t upd1,
//                                   i_t sub_warp_count,
//                                   i_t sub_warp_blocks_end,
//                                   i_t med_blocks_end,
//                                   raft::device_span<const i_t> warp_offsets,
//                                   raft::device_span<const i_t> warp_id_offsets,
//                                   raft::device_span<const i_t> block_offsets,
//                                   raft::device_span<const i_t> block_id_offsets)
//{
//   if (blockIdx.x < sub_warp_blocks_end) {
//     // sub warps
//     cnst_slack_sub_warp<i_t, f_t, BDIM>(
//       view, upd0, upd1, view.warp_offsets, view.warp_id_offsets, view.sub_warp_count);
//   } else if (blockIdx.x < block_offsets[1]) {
//     // medium blocks - 64 threads per row
//     cnst_slack_block_64<i_t, f_t, BDIM>(blockIdx.x - block_offsets[0],  // pseudo block id
//                                         block_id_offsets[0],            // beginning of segment
//                                         block_id_offsets[1],            // end of segment
//                                         view,
//                                         upd0,
//                                         upd1);
//   } else if (blockIdx.x < block_offsets[2]) {
//     // medium blocks - 256 (BDIM) threads per row
//     cnst_slack_block<i_t, f_t, BDIM>(
//       blockIdx.x - block_offsets[1], block_id_offsets[1], view, upd0, upd1);
//   }
// }

template <typename i_t, typename upd_view_t>
inline __device__ thrust::pair<bool, bool> skip_cnst(upd_view_t upd_0,
                                                     upd_view_t upd_1,
                                                     i_t cnst_idx)
{
  return thrust::make_pair((upd_0.changed_constraints[cnst_idx] == i_t{0}),
                           (upd_1.changed_constraints[cnst_idx] == i_t{0}));
}

inline __device__ bool skip_both(thrust::pair<bool, bool>& skip_flag)
{
  return thrust::get<0>(skip_flag) && thrust::get<1>(skip_flag);
}

inline __device__ bool both_valid(thrust::pair<bool, bool>& skip_flag)
{
  return !thrust::get<0>(skip_flag) && !thrust::get<1>(skip_flag);
}

template <typename i_t, typename upd_view_t>
inline __device__ bool get_valid(thrust::pair<bool, bool>& skip_flag,
                                 upd_view_t upd_0,
                                 upd_view_t upd_1)
{
  return !thrust::get<0>(skip_flag) && !thrust::get<1>(skip_flag);
}

template <typename f_t, int MAX_EDGE_PER_CNST>
struct warp_reduce_t {
  using f_t2        = typename type_2<f_t>::type;
  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_CNST>;
  using storage_t   = typename warp_reduce::TempStorage[4];

  storage_t& temp_storage;

  __device__ warp_reduce_t(storage_t& storage_) : temp_storage(storage_) {}

  inline __device__ thrust::pair<f_t2, f_t2> sum(thrust::pair<f_t2, f_t2>& in)
  {
    f_t2 out0, out1;
    out0.x = warp_reduce(temp_storage[0]).Sum(thrust::get<0>(in).x);
    out0.y = warp_reduce(temp_storage[1]).Sum(thrust::get<0>(in).y);
    out1.x = warp_reduce(temp_storage[2]).Sum(thrust::get<1>(in).x);
    out1.y = warp_reduce(temp_storage[3]).Sum(thrust::get<1>(in).y);
    return thrust::make_pair(out0, out1);
  }

  inline __device__ thrust::pair<f_t2, f_t2> sum(thrust::pair<f_t2, f_t2>& in, int valid_items)
  {
    f_t2 out0, out1;
    out0.x = warp_reduce(temp_storage[0]).Sum(thrust::get<0>(in).x, valid_items);
    out0.y = warp_reduce(temp_storage[1]).Sum(thrust::get<0>(in).y, valid_items);
    out1.x = warp_reduce(temp_storage[2]).Sum(thrust::get<1>(in).x, valid_items);
    out1.y = warp_reduce(temp_storage[3]).Sum(thrust::get<1>(in).y, valid_items);
    return thrust::make_pair(out0, out1);
  }

  inline __device__ f_t2 sum(f_t2& in)
  {
    f_t2 out;
    out.x = warp_reduce(temp_storage[0]).Sum(in.x);
    out.y = warp_reduce(temp_storage[1]).Sum(in.y);
    return out;
  }

  inline __device__ f_t2 sum(f_t2& in, int valid_items)
  {
    f_t2 out;
    out.x = warp_reduce(temp_storage[0]).Sum(in.x, valid_items);
    out.y = warp_reduce(temp_storage[1]).Sum(in.y, valid_items);
    return out;
  }
};

template <typename f_t, int BDIM>
struct block_reduce_t {
  using f_t2         = typename type_2<f_t>::type;
  using block_reduce = cub::BlockReduce<f_t, BDIM>;
  using storage_t    = typename block_reduce::TempStorage;

  storage_t& temp_storage;

  __device__ block_reduce_t(storage_t& storage_) : temp_storage(storage_) {}

  inline __device__ f_t2 sum(f_t2& in)
  {
    f_t2 out;
    out.x = block_reduce(temp_storage).Sum(in.x);
    __syncthreads();
    out.y = block_reduce(temp_storage).Sum(in.y);
    return out;
  }

  inline __device__ thrust::pair<f_t2, f_t2> sum(thrust::pair<f_t2, f_t2>& in)
  {
    f_t2 out0, out1;
    out0.x = block_reduce(temp_storage).Sum(thrust::get<0>(in).x);
    __syncthreads();
    out0.y = block_reduce(temp_storage).Sum(thrust::get<0>(in).y);
    __syncthreads();
    out1.x = block_reduce(temp_storage).Sum(thrust::get<1>(in).x);
    __syncthreads();
    out1.y = block_reduce(temp_storage).Sum(thrust::get<1>(in).y);
    return thrust::make_pair(out0, out1);
  }
};

template <typename f_t, int BDIM, int PSEUDO_BDIM>
struct partial_block_reduce_t {
  using f_t2 = typename type_2<f_t>::type;

  using reduce_t = warp_reduce_t<f_t, 32>;

  struct storage_t {
    using warp_reduce_storage_t = typename reduce_t::storage_t;
    warp_reduce_storage_t warp_storage[BDIM / raft::WarpSize];
    f_t2 act0[BDIM / raft::WarpSize];
    f_t2 act1[BDIM / raft::WarpSize];
  };

  storage_t& temp_storage;

  __device__ partial_block_reduce_t(storage_t& storage_) : temp_storage(storage_) {};

  inline __device__ bool is_aggregated_thread() { return (threadIdx.x & (PSEUDO_BDIM - 1)) == 0; }

  inline __device__ int pseudo_thread_id() { return (threadIdx.x & (PSEUDO_BDIM - 1)); }

  inline __device__ thrust::pair<f_t2, f_t2> sum(thrust::pair<f_t2, f_t2>& in)
  {
    int warp_id = threadIdx.x / raft::WarpSize;

    reduce_t reduce(temp_storage.warp_storage[warp_id]);
    auto warp_sum = reduce.sum(in);

    // write temps to shared memory
    if ((threadIdx.x & (raft::WarpSize - 1)) == 0) {
      temp_storage.act0[warp_id] = thrust::get<0>(warp_sum);
      temp_storage.act1[warp_id] = thrust::get<1>(warp_sum);
    }
    __syncthreads();

    auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});
    // in 'head warp' of each pseudo block, load temp results of children warps
    // and sum them up
    if ((threadIdx.x & (PSEUDO_BDIM - 1)) / raft::WarpSize == 0) {
      constexpr int valid_item_count = PSEUDO_BDIM / raft::WarpSize;
      static_assert(valid_item_count < raft::WarpSize,
                    "number of valid items cannot exceed warpsize");
      auto lane_id = threadIdx.x & (raft::WarpSize - 1);
      if (lane_id < valid_item_count) {
        auto temp_warp_result_idx = lane_id + warp_id;
        act                       = thrust::make_pair(temp_storage.act0[temp_warp_result_idx],
                                temp_storage.act1[temp_warp_result_idx]);
      }
      act = reduce.sum(act, valid_item_count);
    }
    return act;
  }

  inline __device__ f_t2 sum(f_t2& in)
  {
    int warp_id = threadIdx.x / raft::WarpSize;

    reduce_t reduce(temp_storage.warp_storage[warp_id]);
    auto warp_sum = reduce.sum(in);

    // write temps to shared memory
    if ((threadIdx.x & (raft::WarpSize - 1)) == 0) { temp_storage.act0[warp_id] = warp_sum; }
    __syncthreads();

    auto act = f_t2{0., 0.};
    // in 'head warp' of each pseudo block, load temp results of children warps
    // and sum them up
    if ((threadIdx.x & (PSEUDO_BDIM - 1)) / raft::WarpSize == 0) {
      constexpr int valid_item_count = PSEUDO_BDIM / raft::WarpSize;
      static_assert(valid_item_count < raft::WarpSize,
                    "number of valid items cannot exceed warpsize");
      auto lane_id = threadIdx.x & (raft::WarpSize - 1);
      if (lane_id < valid_item_count) {
        auto temp_warp_result_idx = lane_id + warp_id;
        act                       = temp_storage.act0[temp_warp_result_idx];
      }
      act = reduce.sum(act, valid_item_count);
    }
    return act;
  }
};

template <typename f_t,
          int MAX_EDGE_PER_CNST,
          typename i_t,
          typename csr_view_t,
          typename upd_view_t>
__device__ typename type_2<f_t>::type calc_act(
  csr_view_t view, upd_view_t upd, i_t tid, i_t beg, i_t end)
{
  using f_t2 = typename type_2<f_t>::type;
  auto act   = f_t2{0., 0.};
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_CNST) {
    auto coeff = view.coefficients[i];
    auto var   = view.col_elem[i];

    atomicExch(&upd.changed_variables[var], 1);

    auto bounds      = upd.vars_bnd[var];
    auto min_contrib = bounds.x;
    auto max_contrib = bounds.y;
    if (coeff < 0.0) {
      min_contrib = bounds.y;
      max_contrib = bounds.x;
    }
    act.x += coeff * min_contrib;
    act.y += coeff * max_contrib;
  }
  return act;
}

template <typename f_t,
          int MAX_EDGE_PER_CNST,
          typename i_t,
          typename csr_view_t,
          typename upd_view_t>
__device__ thrust::pair<typename type_2<f_t>::type, typename type_2<f_t>::type> calc_act(
  csr_view_t view, upd_view_t upd_0, upd_view_t upd_1, i_t tid, i_t beg, i_t end)
{
  using f_t2 = typename type_2<f_t>::type;
  auto act_0 = f_t2{0., 0.};
  auto act_1 = f_t2{0., 0.};
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_CNST) {
    auto coeff = view.coefficients[i];
    auto var   = view.col_elem[i];

    atomicExch(&upd_0.changed_variables[var], 1);
    atomicExch(&upd_1.changed_variables[var], 1);

    auto bounds_0      = upd_0.vars_bnd[var];
    auto bounds_1      = upd_1.vars_bnd[var];
    auto min_contrib_0 = bounds_0.x;
    auto max_contrib_0 = bounds_0.y;
    auto min_contrib_1 = bounds_1.x;
    auto max_contrib_1 = bounds_1.y;
    if (coeff < 0.0) {
      min_contrib_0 = bounds_0.y;
      max_contrib_0 = bounds_0.x;
      min_contrib_1 = bounds_1.y;
      max_contrib_1 = bounds_1.x;
    }
    act_0.x += coeff * min_contrib_0;
    act_0.y += coeff * max_contrib_0;
    act_1.x += coeff * min_contrib_1;
    act_1.y += coeff * max_contrib_1;
  }
  return thrust::make_pair(act_0, act_1);
}

template <bool erase_inf_cnst, typename i_t, typename f_t, typename f_t2, typename upd_view_t>
inline __device__ void write_cnst_slack(
  upd_view_t view, i_t cnst_idx, f_t2 cnst_lb_ub, f_t2 act, f_t eps)
{
  auto cnst_prop = f_t2{cnst_lb_ub.y - act.x, cnst_lb_ub.x - act.y};
  if constexpr (erase_inf_cnst) {
    if ((0 > cnst_prop.x + eps) || (eps < cnst_prop.y)) {
      cnst_prop.x = std::numeric_limits<f_t>::quiet_NaN();
    }
  }
  view.cnst_slack[cnst_idx] = cnst_prop;
}

template <typename f_t, int BDIM, typename i_t, typename csr_view_t, typename upd_view_t>
__device__ void cnst_heavy(i_t id_block_beg,
                           i_t id_range_end,
                           i_t work_per_block,
                           csr_view_t view,
                           upd_view_t upd0,
                           upd_view_t upd1)
{
  auto idx       = view.heavy_vertex_ids[blockIdx.x - id_block_beg] + view.heavy_beg_id;
  auto cnst_idx  = view.reorg_ids[idx];
  auto skip_calc = skip_cnst(upd0, upd1, cnst_idx);

  if (skip_both(skip_calc)) { return; }

  auto pseudo_block_id = view.heavy_pseudo_block_ids[blockIdx.x];
  i_t item_off_beg     = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end     = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  using reduce_t  = block_reduce_t<f_t, BDIM>;
  using storage_t = typename reduce_t::storage_t;
  __shared__ storage_t storage;
  block_reduce_t<f_t, BDIM> reduce(storage);

  if (both_valid(skip_calc)) {
    auto act = calc_act<f_t, BDIM>(view, upd0, upd1, threadIdx.x, item_off_beg, item_off_end);
    reduce.sum(act);
    if (threadIdx.x == 0) {
      upd0.tmp_act[blockIdx.x] = thrust::get<0>(act);
      upd1.tmp_act[blockIdx.x] = thrust::get<1>(act);
    }
  } else {
    auto& upd = get_valid(skip_calc, upd0, upd1);
    auto act  = calc_act<f_t, BDIM>(view, upd, threadIdx.x, item_off_beg, item_off_end);
    reduce.sum(act);
    if (threadIdx.x == 0) { upd.tmp_act[blockIdx.x] = act; }
  }
}

template <bool erase_inf_cnst,
          typename f_t,
          int BDIM,
          typename i_t,
          typename csr_view_t,
          typename upd_view_t>
__global__ void finalize_cnst_heavy(csr_view_t view, upd_view_t upd0, upd_view_t upd1)
{
  using f_t2 = typename type_2<f_t>::type;

  auto idx        = blockIdx.x + view.heavy_beg_id;
  i_t cnst_idx    = view.cnst_reorg_ids[idx];
  auto cnst_lb_ub = view.cnst_bnd[idx];

  auto skip_calc = skip_cnst(upd0, upd1, cnst_idx);
  if (skip_both(skip_calc)) { return; }

  [[maybe_unused]] f_t eps = {};
  if constexpr (erase_inf_cnst) {
    eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                       cnst_lb_ub.y,
                                       view.tolerances.absolute_tolerance,
                                       view.tolerances.relative_tolerance);
  }

  using reduce_t  = warp_reduce_t<f_t, 32>;
  using storage_t = typename reduce_t::storage_t;
  __shared__ storage_t storage;
  reduce_t reduce(storage);

  i_t item_off_beg = view.heavy_block_segments[idx];
  i_t item_off_end = view.heavy_block_segments[idx + 1];
  if (both_valid(skip_calc)) {
    auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});
    for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
      auto act0 = upd0.tmp_act[i];
      auto act1 = upd1.tmp_act[i];
      thrust::get<0>(act).x += act0.x;
      thrust::get<0>(act).y += act0.y;

      thrust::get<1>(act).x += act1.x;
      thrust::get<1>(act).y += act1.y;
    }
    act = reduce.sum(act);
    if (threadIdx.x == 0) {
      write_cnst_slack<erase_inf_cnst>(upd0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
      write_cnst_slack<erase_inf_cnst>(upd1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
    }
  } else {
    auto& upd = get_valid(skip_calc, upd0, upd1);
    auto act  = f_t2{0., 0.};
    for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
      auto act_load = upd.tmp_act[i];
      act.x += act_load.x;
      act.y += act_load.y;
    }
    act = reduce.sum(act);
    if (threadIdx.x == 0) { write_cnst_slack<erase_inf_cnst>(upd, cnst_idx, cnst_lb_ub, act, eps); }
  }
}

template <bool erase_inf_cnst,
          typename f_t,
          int BDIM,
          int MAX_EDGE_PER_CNST,
          typename i_t,
          typename csr_view_t,
          typename upd_view_t>
__device__ void cnst_sub_warp(
  i_t id_warp_beg, i_t id_range_end, csr_view_t view, upd_view_t upd0, upd_view_t upd1)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t lane_id = (threadIdx.x & 31);
  i_t idx     = id_warp_beg + (lane_id / MAX_EDGE_PER_CNST);
  i_t cnst_idx;
  f_t2 cnst_lb_ub;
  [[maybe_unused]] f_t eps = {};

  thrust::pair<bool, bool> skip_warp;
  bool valid_item                    = (idx < id_range_end);
  thrust::pair<bool, bool> skip_calc = thrust::make_pair(valid_item, valid_item);
  if (valid_item) {
    cnst_idx  = view.reorg_ids[idx];
    skip_calc = skip_cnst(upd0, upd1, cnst_idx);
    skip_warp = thrust::make_pair(__all_sync(0xfffffff, thrust::get<0>(skip_calc)),
                                  __all_sync(0xfffffff, thrust::get<1>(skip_calc)));
    if (skip_both(skip_warp)) { return; }

    cnst_lb_ub = view.cnst_bnd[idx];
    if constexpr (erase_inf_cnst) {
      eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                         cnst_lb_ub.y,
                                         view.tolerances.absolute_tolerance,
                                         view.tolerances.relative_tolerance);
    }
  }

  i_t p_tid      = lane_id & (MAX_EDGE_PER_CNST - 1);
  bool head_flag = (p_tid == 0);

  using reduce_t  = warp_reduce_t<f_t, MAX_EDGE_PER_CNST>;
  using storage_t = typename reduce_t::storage_t;
  __shared__ storage_t storage;
  reduce_t reduce(storage);

  auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});

  if (valid_item && both_valid(skip_calc)) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    act = calc_act<f_t, MAX_EDGE_PER_CNST>(view, upd0, upd1, p_tid, item_off_beg, item_off_end);
  } else if (valid_item) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    if (thrust::get<0>(skip_calc)) {
      thrust::get<1>(act) =
        calc_act<f_t, MAX_EDGE_PER_CNST>(view, upd1, p_tid, item_off_beg, item_off_end);
    } else {
      thrust::get<0>(act) =
        calc_act<f_t, MAX_EDGE_PER_CNST>(view, upd0, p_tid, item_off_beg, item_off_end);
    }
  }

  act = reduce.sum(act);

  if (valid_item && head_flag && !thrust::get<0>(skip_calc)) {
    write_cnst_slack<erase_inf_cnst>(upd0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
  }
  if (valid_item && head_flag && !thrust::get<1>(skip_calc)) {
    write_cnst_slack<erase_inf_cnst>(upd1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
  }
}

template <bool erase_inf_cnst,
          typename f_t,
          int BDIM,
          typename i_t,
          typename csr_view_t,
          typename upd_view_t>
__device__ void cnst_warp(
  i_t id_block_beg, i_t id_range_end, csr_view_t view, upd_view_t upd0, upd_view_t upd1)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t lane_id = (threadIdx.x & 31);
  i_t idx     = id_block_beg + lane_id;
  i_t cnst_idx;
  f_t2 cnst_lb_ub;
  [[maybe_unused]] f_t eps = {};

  bool valid_item                    = (idx < id_range_end);
  thrust::pair<bool, bool> skip_calc = thrust::make_pair(valid_item, valid_item);
  if (valid_item) {
    cnst_idx  = view.reorg_ids[idx];
    skip_calc = skip_cnst(upd0, upd1, cnst_idx);
    if (skip_both(skip_calc)) { return; }

    cnst_lb_ub = view.cnst_bnd[idx];
    if constexpr (erase_inf_cnst) {
      eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                         cnst_lb_ub.y,
                                         view.tolerances.absolute_tolerance,
                                         view.tolerances.relative_tolerance);
    }
  }

  i_t p_tid      = (threadIdx.x & 31);
  bool head_flag = (p_tid == 0);

  using reduce_t  = warp_reduce_t<f_t, 32>;
  using storage_t = typename reduce_t::storage_t;
  __shared__ storage_t storage;
  reduce_t reduce(storage);

  auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});

  if (valid_item && both_valid(skip_calc)) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    act              = calc_act<f_t, 32>(view, upd0, upd1, p_tid, item_off_beg, item_off_end);
  } else if (valid_item) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    if (thrust::get<0>(skip_calc)) {
      thrust::get<1>(act) = calc_act<f_t, 32>(view, upd1, p_tid, item_off_beg, item_off_end);
    } else {
      thrust::get<0>(act) = calc_act<f_t, 32>(view, upd0, p_tid, item_off_beg, item_off_end);
    }
  }

  act = reduce.sum(act);

  if (valid_item && head_flag) {
    write_cnst_slack<erase_inf_cnst>(upd0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
    write_cnst_slack<erase_inf_cnst>(upd1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
  }
}

template <bool erase_inf_cnst,
          typename f_t,
          int BDIM,
          int PSEUDO_BDIM,
          typename i_t,
          typename csr_view_t,
          typename upd_view_t>
__device__ void cnst_block(
  i_t id_block_beg, i_t id_range_end, csr_view_t view, upd_view_t upd0, upd_view_t upd1)
{
  using f_t2 = typename type_2<f_t>::type;

  i_t id_within_block = (threadIdx.x / PSEUDO_BDIM);
  i_t idx             = id_block_beg + id_within_block;
  i_t cnst_idx;
  f_t2 cnst_lb_ub;
  [[maybe_unused]] f_t eps = {};

  bool valid_item                    = (idx < id_range_end);
  thrust::pair<bool, bool> skip_calc = thrust::make_pair(valid_item, valid_item);
  if (valid_item) {
    cnst_idx  = view.reorg_ids[idx];
    skip_calc = skip_cnst(upd0, upd1, cnst_idx);
    if (skip_both(skip_calc)) { return; }

    cnst_lb_ub = view.cnst_bnd[idx];
    if constexpr (erase_inf_cnst) {
      eps = get_cstr_tolerance<i_t, f_t>(cnst_lb_ub.x,
                                         cnst_lb_ub.y,
                                         view.tolerances.absolute_tolerance,
                                         view.tolerances.relative_tolerance);
    }
  }

  using reduce_t  = partial_block_reduce_t<f_t, BDIM, PSEUDO_BDIM>;
  using storage_t = typename reduce_t::storage_t;
  __shared__ storage_t storage;
  reduce_t reduce(storage);

  i_t item_off_beg = view.offsets[idx];
  i_t item_off_end = view.offsets[idx + 1];

  auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});
  if (valid_item && both_valid(skip_calc)) {
    auto act = calc_act<f_t, PSEUDO_BDIM>(
      view, upd0, upd1, reduce.pseudo_thread_id(), item_off_beg, item_off_end);
    act = reduce.sum(act);
    if (reduce.is_aggregated_thread()) {
      write_cnst_slack<erase_inf_cnst>(upd0, cnst_idx, cnst_lb_ub, thrust::get<0>(act), eps);
      write_cnst_slack<erase_inf_cnst>(upd1, cnst_idx, cnst_lb_ub, thrust::get<1>(act), eps);
    }
  } else if (valid_item) {
    auto& upd = get_valid(skip_calc, upd0, upd1);
    auto act =
      calc_act<f_t, PSEUDO_BDIM>(view, upd, reduce.pseudo_thread_id(), item_off_beg, item_off_end);
    act = reduce.sum(act);
    if (reduce.is_aggregated_thread()) {
      write_cnst_slack<erase_inf_cnst>(upd, cnst_idx, cnst_lb_ub, act, eps);
    }
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          int BDIM,
          typename csr_view_t,
          typename upd_view_t>
__device__ void call_cnst_sub_warp(csr_view_t view, upd_view_t upd0, upd_view_t upd1)
{
  i_t id_warp_beg, id_range_end, t_p_v;
  get_sub_warp_bin<i_t>(&id_warp_beg,
                        &id_range_end,
                        &t_p_v,
                        view.warp_offsets,
                        view.warp_id_offsets,
                        view.sub_warp_count);

  if (t_p_v == 1) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 1>(id_warp_beg, id_range_end, view, upd0, upd1);
  } else if (t_p_v == 2) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 2>(id_warp_beg, id_range_end, view, upd0, upd1);
  } else if (t_p_v == 4) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 4>(id_warp_beg, id_range_end, view, upd0, upd1);
  } else if (t_p_v == 8) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 8>(id_warp_beg, id_range_end, view, upd0, upd1);
  } else if (t_p_v == 16) {
    cnst_sub_warp<erase_inf_cnst, f_t, BDIM, 16>(id_warp_beg, id_range_end, view, upd0, upd1);
  }
}

template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          int BDIM,
          typename csr_view_t,
          typename upd_view_t>
__device__ void call_cnst_block(csr_view_t view, upd_view_t upd0, upd_view_t upd1)
{
  i_t id_block_beg, id_block_end, t_p_v;
  get_block_bin<i_t>(&id_block_beg,
                     &id_block_end,
                     &t_p_v,
                     view.block_offsets,
                     view.block_id_offsets,
                     view.sub_warp_block_count,
                     view.med_block_count);

  if (t_p_v == 32) {
    cnst_warp<erase_inf_cnst, f_t, BDIM>(id_block_beg, id_block_end, view, upd0, upd1);
  } else if (t_p_v == 64) {
    cnst_block<erase_inf_cnst, f_t, BDIM, 64>(id_block_beg, id_block_end, view, upd0, upd1);
  } else if (t_p_v == 128) {
    cnst_block<erase_inf_cnst, f_t, BDIM, 128>(id_block_beg, id_block_end, view, upd0, upd1);
  } else if (t_p_v == 256) {
    cnst_block<erase_inf_cnst, f_t, BDIM, 256>(id_block_beg, id_block_end, view, upd0, upd1);
  } else {
    cnst_heavy<f_t, BDIM>(id_block_beg, id_block_end, view.work_per_block, view, upd0, upd1);
  }
}

// TODO : call_constraint_slack_kernel
template <bool erase_inf_cnst,
          typename i_t,
          typename f_t,
          int BDIM,
          typename csr_view_t,
          typename upd_view_t>
__global__ void call_cnst_slack(csr_view_t view, upd_view_t upd0, upd_view_t upd1)
{
  if (blockIdx.x < view.sub_warp_block_count) {
    call_cnst_sub_warp<erase_inf_cnst, i_t, f_t, BDIM>(view, upd0, upd1);
  } else {
    call_cnst_block<erase_inf_cnst, i_t, f_t, BDIM>(view, upd0, upd1);
  }
}

}  // namespace cuopt::linear_programming::detail
