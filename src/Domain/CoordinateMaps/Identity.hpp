// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class Identity.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
namespace CoordinateMaps {

/// \ingroup CoordinateMapsGroup
/// Identity  map from \f$\xi \rightarrow x\f$.
template <size_t Dim>
class Identity {
 public:
  static constexpr size_t dim = Dim;

  Identity() = default;
  ~Identity() = default;
  Identity(const Identity&) = default;
  Identity(Identity&&) noexcept = default;  // NOLINT
  Identity& operator=(const Identity&) = default;
  Identity& operator=(Identity&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> operator()(
      const std::array<T, Dim>& source_coords) const noexcept;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, Dim>> inverse(
      const std::array<double, Dim>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> jacobian(
      const std::array<T, Dim>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> inv_jacobian(
      const std::array<T, Dim>& source_coords) const noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& /*p*/) {}  // NOLINT

  bool is_identity() const noexcept { return true; }
};

template <size_t Dim>
inline constexpr bool operator==(
    const CoordinateMaps::Identity<Dim>& /*lhs*/,
    const CoordinateMaps::Identity<Dim>& /*rhs*/) noexcept {
  return true;
}

template <size_t Dim>
inline constexpr bool operator!=(
    const CoordinateMaps::Identity<Dim>& lhs,
    const CoordinateMaps::Identity<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace CoordinateMaps
}  // namespace domain
