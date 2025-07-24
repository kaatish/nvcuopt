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

#include <mip/mip_constants.hpp>

#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <utilities/copy_helpers.hpp>
#include <utilities/device_utils.cuh>

#include <cub/cub.cuh>
#include "lb_bounds_update_kernels.cuh"
#include "lb_multi_probe.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
lb_multi_probe_t<i_t, f_t>::lb_multi_probe_t(mip_solver_context_t<i_t, f_t>& context_,
                                             lb_problem_t<i_t, f_t>& problem,
                                             settings_t in_settings)
  : context(context_), upd_0(problem), upd_1(problem), settings(in_settings)
{
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::resize(lb_problem_t<i_t, f_t>& problem)
{
  upd_0.resize(problem);
  upd_1.resize(problem);
}

template <typename i_t, typename f_t>
void lb_multi_probe_t<i_t, f_t>::calculate_constraint_slack_iter(lb_problem_t<i_t, f_t>& problem,
                                                                 const raft::handle_t* handle_ptr)
{
  auto num_blocks = problem.cnst_csr.sub_warp_block_count + problem.cnst_csr.med_block_count +
                    problem.cnst_csr.num_blocks_heavy;
  call_cnst_slack<true, i_t, f_t, 256><<<num_blocks, 256, 0, handle_ptr->get_stream()>>>(
    problem.cnst_csr.view(), upd_0.view(), upd_1.view());
}

#if MIP_INSTANTIATE_FLOAT
template class lb_multi_probe_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class lb_multi_probe_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
