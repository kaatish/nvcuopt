/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <thrust/sort.h>
#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/presolve/bounds_presolve.cuh>
#include <mip/presolve/lb_multi_probe.cuh>
#include <mip/presolve/lb_problem.cuh>
#include <mip/presolve/multi_probe.cuh>
#include <mip/presolve/trivial_presolve.cuh>
#include <mps_parser/parser.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>
#include <utilities/timer.hpp>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

template <typename i_t, typename f_t>
std::vector<i_t> rand_id(const raft::handle_t* handle_ptr,
                         rmm::device_uvector<i_t>& reorg_ids,
                         const std::vector<i_t>& bin_offsets,
                         int64_t seed)
{
  std::mt19937 rng(seed);
  std::vector<i_t> r_id;

  for (size_t i = 0; i < bin_offsets.size() - 1; ++i) {
    if (bin_offsets[i] != bin_offsets[i + 1]) {
      std::uniform_int_distribution<i_t> dist(bin_offsets[i], bin_offsets[i + 1] - 1);
      auto id = reorg_ids.element(dist(rng), handle_ptr->get_stream());
      r_id.push_back(id);
    }
  }
  return r_id;
}

template <typename i_t, typename f_t>
std::vector<i_t> get_rand_items(detail::problem_t<i_t, f_t>& lb_problem,
                                bool disp_cnst,
                                int64_t seed)
{
  if (disp_cnst) {
    return rand_id(lb_problem.cnst_csr.reorg_ids, lb_problem.cnst_csr.bin_offsets, seed);
  } else {
    return rand_id(lb_problem.vars_csr.reorg_ids, lb_problem.vars_csr.bin_offsets, seed);
  }
}

#if 1
template <typename i_t, typename f_t>
std::tuple<std::vector<i_t>, std::vector<i_t>, std::vector<i_t>> display_degree_dist(
  detail::problem_t<i_t, f_t>& problem, bool disp_cnst)
{
  rmm::device_uvector<i_t>& offsets = disp_cnst ? problem.offsets : problem.reverse_offsets;
  rmm::device_uvector<i_t> degrees(offsets.size() - 1, problem.handle_ptr->get_stream());
  thrust::transform(problem.handle_ptr->get_thrust_policy(),
                    offsets.begin() + 1,
                    offsets.end(),
                    offsets.begin(),
                    degrees.begin(),
                    thrust::minus<i_t>{});
  thrust::sort(problem.handle_ptr->get_thrust_policy(), degrees.begin(), degrees.end());
  // auto count = thrust::count_if(handle_.get_thrust_policy(),
  //                  degrees.begin(), degrees.end(),
  //                  [] __device__ (auto i) {
  //                  return i == 1;
  //                  });
  // std::cout<<"single variable constraint count : "<<count<<"\n";
  std::vector<i_t> lb_dist;
  std::vector<i_t> ub_dist;
  std::vector<i_t> count_dist;
  {
    i_t lb     = -1;
    i_t ub     = 0;
    auto count = thrust::count_if(problem.handle_ptr->get_thrust_policy(),
                                  degrees.begin(),
                                  degrees.end(),
                                  [lb, ub] __device__(auto i) { return (lb < i) && (i <= ub); });
    if (count != 0) {
      lb_dist.push_back(lb);
      ub_dist.push_back(ub);
      count_dist.push_back(count);
    }
  }
  {
    i_t lb     = 0;
    i_t ub     = 1;
    auto count = thrust::count_if(problem.handle_ptr->get_thrust_policy(),
                                  degrees.begin(),
                                  degrees.end(),
                                  [lb, ub] __device__(auto i) { return (lb < i) && (i <= ub); });
    if (count != 0) {
      lb_dist.push_back(lb);
      ub_dist.push_back(ub);
      count_dist.push_back(count);
    }
  }
  for (i_t i = 0; i < 32; ++i) {
    auto lb    = std::pow(2, i);
    auto ub    = std::pow(2, i + 1);
    auto count = thrust::count_if(problem.handle_ptr->get_thrust_policy(),
                                  degrees.begin(),
                                  degrees.end(),
                                  [lb, ub] __device__(auto i) { return (lb < i) && (i <= ub); });
    if (count != 0) {
      lb_dist.push_back(lb);
      ub_dist.push_back(ub);
      count_dist.push_back(count);
    }
  }
  return std::make_tuple(std::move(lb_dist), std::move(ub_dist), std::move(count_dist));
}

template <typename i_t, typename f_t>
void display_cnst_dist(detail::problem_t<i_t, f_t>& problem)
{
  auto [cnst_lb_dist, cnst_ub_dist, cnst_count_dist] = display_degree_dist(problem, true);
  // display max degree
  auto max_count = cnst_count_dist[0];
  auto max_bin   = 0;
  for (size_t i = 0; i < cnst_count_dist.size(); ++i) {
    if (max_count < cnst_count_dist[i]) {
      max_count = cnst_count_dist[i];
      max_bin   = i;
    }
  }
  std::cout << "\ncnst dist max count ";
  std::cout << cnst_ub_dist[max_bin] << " " << max_count << "\n";
  for (size_t i = 0; i < cnst_lb_dist.size(); ++i) {
    std::cout << cnst_lb_dist[i] << " < degree <= " << cnst_ub_dist[i] << "\t" << cnst_count_dist[i]
              << "\n";
    //<< " " << ((cnst_count_dist[i] * cnst_ub_dist[i]) + 31) / 32 << "\n";
  }
}

template <typename i_t, typename f_t>
void display_vars_dist(detail::problem_t<i_t, f_t>& problem)
{
  auto [vars_lb_dist, vars_ub_dist, vars_count_dist] = display_degree_dist(problem, false);
  // display max degree
  auto max_count = vars_count_dist[0];
  auto max_bin   = 0;
  for (size_t i = 0; i < vars_count_dist.size(); ++i) {
    if (max_count < vars_count_dist[i]) {
      max_count = vars_count_dist[i];
      max_bin   = i;
    }
  }
  std::cout << "\nvars dist max count ";
  std::cout << vars_ub_dist[max_bin] << " " << max_count << "\n";
  for (size_t i = 0; i < vars_lb_dist.size(); ++i) {
    std::cout << vars_lb_dist[i] << " < degree <= " << vars_ub_dist[i] << "\t" << vars_count_dist[i]
              << "\n";
    //<< " " << ((vars_count_dist[i] * vars_ub_dist[i]) + 31) / 32 << "\n";
  }
}
#endif

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

std::tuple<std::vector<int>, std::vector<double>, std::vector<double>> select_k_random(
  detail::problem_t<int, double>& problem, int sample_size)
{
  auto seed = std::random_device{}();
  std::cerr << "Tested with seed " << seed << "\n";
  problem.compute_n_integer_vars();
  auto v_lb       = host_copy(problem.variable_lower_bounds);
  auto v_ub       = host_copy(problem.variable_upper_bounds);
  auto int_var_id = host_copy(problem.integer_indices);
  int_var_id.erase(std::remove_if(int_var_id.begin(),
                                  int_var_id.end(),
                                  [v_lb, v_ub](auto id) {
                                    return !(std::isfinite(v_lb[id]) && std::isfinite(v_ub[id]));
                                  }),
                   int_var_id.end());
  sample_size = std::min(sample_size, static_cast<int>(int_var_id.size()));
  std::vector<int> random_int_vars;
  std::mt19937 m{seed};
  std::sample(
    int_var_id.begin(), int_var_id.end(), std::back_inserter(random_int_vars), sample_size, m);
  std::vector<double> probe_0(sample_size);
  std::vector<double> probe_1(sample_size);
  for (int i = 0; i < static_cast<int>(random_int_vars.size()); ++i) {
    if (i % 2) {
      probe_0[i] = v_lb[random_int_vars[i]];
      probe_1[i] = v_ub[random_int_vars[i]];
    } else {
      probe_1[i] = v_lb[random_int_vars[i]];
      probe_0[i] = v_ub[random_int_vars[i]];
    }
  }
  return std::make_tuple(std::move(random_int_vars), std::move(probe_0), std::move(probe_1));
}

std::pair<std::vector<thrust::pair<int, double>>, std::vector<thrust::pair<int, double>>>
convert_probe_tuple(std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>& probe)
{
  std::vector<thrust::pair<int, double>> probe_first;
  std::vector<thrust::pair<int, double>> probe_second;
  for (size_t i = 0; i < std::get<0>(probe).size(); ++i) {
    probe_first.emplace_back(thrust::make_pair(std::get<0>(probe)[i], std::get<1>(probe)[i]));
    probe_second.emplace_back(thrust::make_pair(std::get<0>(probe)[i], std::get<2>(probe)[i]));
  }
  return std::make_pair(std::move(probe_first), std::move(probe_second));
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
bounds_probe_results(detail::bound_presolve_t<int, double>& bnd_prb_0,
                     detail::bound_presolve_t<int, double>& bnd_prb_1,
                     detail::problem_t<int, double>& problem,
                     const std::pair<std::vector<thrust::pair<int, double>>,
                                     std::vector<thrust::pair<int, double>>>& probe)
{
  auto& probe_first  = std::get<0>(probe);
  auto& probe_second = std::get<1>(probe);
  rmm::device_uvector<double> b_lb_0(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_ub_0(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_lb_1(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_ub_1(problem.n_variables, problem.handle_ptr->get_stream());
  bnd_prb_0.solve(problem, probe_first);
  bnd_prb_0.set_updated_bounds(problem.handle_ptr, make_span(b_lb_0), make_span(b_ub_0));
  bnd_prb_1.solve(problem, probe_second);
  bnd_prb_1.set_updated_bounds(problem.handle_ptr, make_span(b_lb_1), make_span(b_ub_1));

  auto h_lb_0 = host_copy(b_lb_0);
  auto h_ub_0 = host_copy(b_ub_0);
  auto h_lb_1 = host_copy(b_lb_1);
  auto h_ub_1 = host_copy(b_ub_1);
  return std::make_tuple(
    std::move(h_lb_0), std::move(h_ub_0), std::move(h_lb_1), std::move(h_ub_1));
}

// std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
// multi_probe_results(
//   detail::lb_multi_probe_t<int, double>& prb,
//   detail::problem_t<int, double>& problem,
//   const std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>& probe_tuple)
//{
//   prb.solve(problem, probe_tuple);
//   rmm::device_uvector<double> m_lb_0(problem.n_variables, problem.handle_ptr->get_stream());
//   rmm::device_uvector<double> m_ub_0(problem.n_variables, problem.handle_ptr->get_stream());
//   rmm::device_uvector<double> m_lb_1(problem.n_variables, problem.handle_ptr->get_stream());
//   rmm::device_uvector<double> m_ub_1(problem.n_variables, problem.handle_ptr->get_stream());
//   prb.set_updated_bounds(problem.handle_ptr, make_span(m_lb_0), make_span(m_ub_0), 0);
//   prb.set_updated_bounds(problem.handle_ptr, make_span(m_lb_1), make_span(m_ub_1), 1);
//
//   auto h_lb_0 = host_copy(m_lb_0);
//   auto h_ub_0 = host_copy(m_ub_0);
//   auto h_lb_1 = host_copy(m_lb_1);
//   auto h_ub_1 = host_copy(m_ub_1);
//   return std::make_tuple(
//     std::move(h_lb_0), std::move(h_ub_0), std::move(h_lb_1), std::move(h_ub_1));
// }

void test_multi_probe(std::string path)
{
  auto memory_resource = make_async();
  rmm::mr::set_current_device_resource(memory_resource.get());
  const raft::handle_t handle_{};
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);
  detail::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  detail::trivial_presolve(problem);

  detail::lb_problem_t<int, double> lb_problem(problem);

  // display_cnst_dist(problem);
  // display_vars_dist(problem);
  {
    std::cout << "cnst bin offsets\n";
    for (auto& o : lb_problem.cnst_csr.bin_offsets) {
      std::cout << o << "\n";
    }
    std::cout << "\n\n";
  }
  mip_solver_settings_t<int, double> default_settings{};
  detail::pdhg_solver_t<int, double> pdhg_solver(problem.handle_ptr, problem);
  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(&handle_,
                                                               problem,
                                                               10,
                                                               1.0,
                                                               pdhg_solver,
                                                               problem.reverse_coefficients,
                                                               problem.reverse_offsets,
                                                               problem.reverse_constraints,
                                                               true);
  detail::mip_solver_t<int, double> solver(problem, default_settings, scaling, cuopt::timer_t(0));
  detail::bound_presolve_t<int, double> bnd_prb(solver.context);
  // detail::multi_probe_t<int, double> multi(solver.context);
  // multi.upd_0.init_changed_constraints(problem.handle_ptr);
  // multi.upd_1.init_changed_constraints(problem.handle_ptr);

  handle_.sync_stream();
  RAFT_CHECK_CUDA(handle_.get_stream());
  detail::lb_multi_probe_t<int, double> lb_multi(solver.context, lb_problem);

  handle_.sync_stream();
  RAFT_CHECK_CUDA(handle_.get_stream());
  lb_multi.upd_0.copy(lb_problem);

  handle_.sync_stream();
  RAFT_CHECK_CUDA(handle_.get_stream());
  lb_multi.upd_1.copy(lb_problem);

  handle_.sync_stream();
  RAFT_CHECK_CUDA(handle_.get_stream());
  lb_multi.upd_0.init_changed_constraints(problem.handle_ptr);

  handle_.sync_stream();
  RAFT_CHECK_CUDA(handle_.get_stream());
  lb_multi.upd_1.init_changed_constraints(problem.handle_ptr);

  handle_.sync_stream();
  RAFT_CHECK_CUDA(handle_.get_stream());
  lb_multi.calculate_constraint_slack_iter(lb_problem, problem.handle_ptr);

  bnd_prb.calculate_activity_on_problem_bounds(problem);
  handle_.sync_stream();
  RAFT_CHECK_CUDA(handle_.get_stream());

  double tol = 1e-8;
  std::cout << lb_multi.upd_0.cnst_slack.size() << "\n";
  std::cout << lb_multi.upd_1.cnst_slack.size() << "\n";
  std::cout << bnd_prb.upd.min_activity.size() << "\n";
  std::cout << bnd_prb.upd.max_activity.size() << "\n";
  std::cout << lb_problem.n_constraints << "\n";

  std::cerr << "pt 0\n";
  auto min_act = host_copy(bnd_prb.upd.min_activity);
  std::cerr << "pt 1\n";
  auto max_act = host_copy(bnd_prb.upd.max_activity);
  std::cerr << "pt 2\n";
  auto c_lb = host_copy(problem.constraint_lower_bounds);
  std::cerr << "pt 3\n";
  auto c_ub = host_copy(problem.constraint_upper_bounds);
  std::cerr << "pt 4\n";
  auto off = host_copy(problem.offsets);
  std::cerr << "start multi data move\n";

  auto c_sl_0 = host_copy(lb_multi.upd_0.cnst_slack);
  auto c_sl_1 = host_copy(lb_multi.upd_1.cnst_slack);
  for (int i = 0; i < lb_problem.n_constraints; ++i) {
    auto bnd_min_slack_0 = c_ub[i] - min_act[i];
    auto bnd_max_slack_0 = c_lb[i] - max_act[i];
    // auto lb_slack_0 = c_sl_0[2*i];
    // auto lb_slack_1 = c_sl_0[2*i+1];

    auto lb_min_slack_0 = c_sl_0[2 * i];
    auto lb_max_slack_0 = c_sl_0[2 * i + 1];
    auto lb_min_slack_1 = c_sl_1[2 * i];
    auto lb_max_slack_1 = c_sl_1[2 * i + 1];
    if ((abs(lb_min_slack_0 - bnd_min_slack_0) / bnd_min_slack_0 > tol) ||
        (std::isinf(lb_min_slack_0) ^ std::isinf(bnd_min_slack_0))) {
      auto deg = off[i + 1] - off[i];
      std::cout << "min mismatch " << i << " " << bnd_min_slack_0 << " " << lb_min_slack_0
                << "\tdiff = " << abs(bnd_min_slack_0 - lb_min_slack_0) << " " << deg << "\n";
    }
    if ((abs(lb_max_slack_0 - bnd_max_slack_0) / bnd_max_slack_0 > tol) ||
        (std::isinf(lb_max_slack_0) ^ std::isinf(bnd_max_slack_0))) {
      auto deg = off[i + 1] - off[i];
      std::cout << "min mismatch " << i << " " << bnd_max_slack_0 << " " << lb_max_slack_0
                << "\tdiff = " << abs(bnd_max_slack_0 - lb_max_slack_0) << " " << deg << "\n";
    }
    if (i < 2) {
      auto deg = off[i + 1] - off[i];
      std::cout << "deg " << deg << "\n";
      std::cout << "min " << i << " " << bnd_min_slack_0 << " " << lb_min_slack_0 << " "
                << lb_min_slack_1 << "\n";
      std::cout << "max " << i << " " << bnd_max_slack_0 << " " << lb_max_slack_0 << " "
                << lb_max_slack_1 << "\n";
    }
  }
  // for (int i = 0; i < lb_problem.n_constraints; ++i) {
  //   //auto bnd_slack_0 = c_ub[i] - min_act[i];
  //   //auto bnd_slack_1 = c_lb[i] - max_act[i];
  //   //auto lb_slack_0 = c_sl_0[2*i];
  //   //auto lb_slack_1 = c_sl_0[2*i+1];

  //  //auto lb_min_act_0 = c_ub[i] - c_sl_0[2*i];
  //  //auto lb_max_act_0 = c_lb[i] - c_sl_0[2*i+1];
  //  //auto lb_min_act_1 = c_ub[i] - c_sl_1[2*i];
  //  //auto lb_max_act_1 = c_lb[i] - c_sl_1[2*i+1];
  //  auto lb_min_act_0 = c_sl_0[2*i];
  //  auto lb_max_act_0 = c_sl_0[2*i+1];
  //  auto lb_min_act_1 = c_sl_1[2*i];
  //  auto lb_max_act_1 = c_sl_1[2*i+1];
  //  //if (abs(lb_slack_0 - bnd_slack_0)/bnd_slack_0 > tol) {
  //  if ((abs(lb_min_act_0 - min_act[i])/min_act[i] > tol) || (std::isinf(lb_min_act_0) ^
  //  std::isinf(min_act[i]))) {
  //    auto deg = off[i+1] - off[i];
  //    std::cout<<"min mismatch "<<i<<" "<<min_act[i]<<" "<<lb_min_act_0<<"\tdiff =
  //    "<<abs(min_act[i] - lb_min_act_0)<<" "<<deg<<"\n";
  //  }
  //  //if (abs(lb_slack_1 - bnd_slack_1)/bnd_slack_1 > tol) {
  //  if ((abs(lb_max_act_0 - max_act[i])/max_act[i] > tol) || (std::isinf(lb_max_act_0) ^
  //  std::isinf(max_act[i]))) {
  //    auto deg = off[i+1] - off[i];
  //    std::cout<<"max mismatch "<<i<<" "<<max_act[i]<<" "<<lb_max_act_0<<"\tdiff =
  //    "<<abs(max_act[i] - lb_max_act_0)<<" "<<deg<<"\n";
  //  }
  //  if (i < 2)
  //  {
  //    auto deg = off[i+1] - off[i];
  //    std::cout<<"deg "<<deg<<"\n";
  //    std::cout<<"min "<<i<<" "<<min_act[i]<<" "<<lb_min_act_0<<" "<<lb_min_act_1<<"\n";
  //    std::cout<<"max "<<i<<" "<<max_act[i]<<" "<<lb_max_act_0<<" "<<lb_max_act_1<<"\n";
  //  }
  //}

  // auto lb_sl = lb_multi.
  //  detail::bound_presolve_t<int, double> bnd_prb_1(solver.context);
  //  detail::multi_probe_t<int, double> multi_probe_prs(solver.context);

  // auto probe_tuple       = select_k_random(problem, 100);
  // auto bounds_probe_vals = convert_probe_tuple(probe_tuple);

  // auto [bnd_lb_0, bnd_ub_0, bnd_lb_1, bnd_ub_1] =
  //   bounds_probe_results(bnd_prb_0, bnd_prb_1, problem, bounds_probe_vals);
  // auto [m_lb_0, m_ub_0, m_lb_1, m_ub_1] =
  //   multi_probe_results(multi_probe_prs, problem, probe_tuple);

  // auto bnd_min_act_0 = host_copy(bnd_prb_0.upd.min_activity);
  // auto bnd_max_act_0 = host_copy(bnd_prb_0.upd.max_activity);
  // auto bnd_min_act_1 = host_copy(bnd_prb_1.upd.min_activity);
  // auto bnd_max_act_1 = host_copy(bnd_prb_1.upd.max_activity);

  // auto mlp_min_act_0 = host_copy(multi_probe_prs.upd_0.min_activity);
  // auto mlp_max_act_0 = host_copy(multi_probe_prs.upd_0.max_activity);
  // auto mlp_min_act_1 = host_copy(multi_probe_prs.upd_1.min_activity);
  // auto mlp_max_act_1 = host_copy(multi_probe_prs.upd_1.max_activity);

  // for (int i = 0; i < (int)bnd_min_act_0.size(); ++i) {
  //   EXPECT_DOUBLE_EQ(bnd_min_act_0[i], mlp_min_act_0[i]);
  //   EXPECT_DOUBLE_EQ(bnd_max_act_0[i], mlp_max_act_0[i]);
  //   EXPECT_DOUBLE_EQ(bnd_min_act_1[i], mlp_min_act_1[i]);
  //   EXPECT_DOUBLE_EQ(bnd_max_act_1[i], mlp_max_act_1[i]);
  // }

  // for (int i = 0; i < (int)bnd_lb_0.size(); ++i) {
  //   EXPECT_DOUBLE_EQ(bnd_lb_0[i], m_lb_0[i]);
  //   EXPECT_DOUBLE_EQ(bnd_ub_0[i], m_ub_0[i]);
  //   EXPECT_DOUBLE_EQ(bnd_lb_1[i], m_lb_1[i]);
  //   EXPECT_DOUBLE_EQ(bnd_ub_1[i], m_ub_1[i]);
  // }
}

TEST(presolve, multi_probe)
{
  // std::vector<std::string> test_instances = {"mip/50v-10-free-bound.mps",
  //                                            "mip/neos5-free-bound.mps"};
  ////"mip/50v-10-free-bound.mps", "mip/neos5-free-bound.mps", "mip/neos5.mps"};
  // for (const auto& test_instance : test_instances) {
  //   std::cout << "Running: " << test_instance << std::endl;
  //   auto path = make_path_absolute(test_instance);
  //   test_multi_probe(path);
  // }
  std::string path = "/home/aatish/rapids/mip_files/miplib/square47.mps";
  // std::string path = "/home/aatish/rapids/mip_files/miplib/neos17.mps";
  // std::string path = "/home/aatish/rapids/mip_files/miplib/sing44.mps";
  // std::string path = "/home/aatish/rapids/mip_files/miplib/neos-3402454-bohle.mps";
  test_multi_probe(path);
}

}  // namespace cuopt::linear_programming::test
