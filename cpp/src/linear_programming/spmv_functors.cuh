/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct identity_functor {
  __device__ __forceinline__ void operator()(i_t idx, f_t x, raft::device_span<f_t> output) const
  {
    output[idx] = x;
  }
};

}  // namespace cuopt::linear_programming::detail