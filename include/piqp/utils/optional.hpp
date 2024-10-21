// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_OPTIONAL_HPP
#define PIQP_UTILS_OPTIONAL_HPP

#include <piqp/fwd.hpp>
#include <optional>

namespace piqp {

template <class T>
using optional = std::optional<T>;
using in_place_t = std::in_place_t;
using nullopt_t = std::nullopt_t;
inline constexpr nullopt_t nullopt = std::nullopt;


}  // namespace piqp

#endif  // PIQP_UTILS_OPTIONAL_HPP
