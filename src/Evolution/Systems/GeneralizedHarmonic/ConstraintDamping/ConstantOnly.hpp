// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
/// \endcond

namespace GeneralizedHarmonic::ConstraintDamping {
/*!
 * \brief A constant: \f$f = C
 *
 * \details Input file options are: `Constant` \f$C\f$.
 * of type `tnsr::I<T, VolumeDim, Fr>`, where `T` is e.g. `double`.
 */
template <size_t VolumeDim, typename Fr>
class ConstantOnly : public DampingFunction<VolumeDim, Fr> {
 public:
  struct Constant {
    using type = double;
    static constexpr Options::String help = {"The constant."};
  };

  using options = tmpl::list<Constant>;

  static constexpr Options::String help = {"Returns a constant."};

  /// \cond
  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(DampingFunction<VolumeDim, Fr>),
                                     ConstantOnly);  // NOLINT

  explicit ConstantOnly(CkMigrateMessage* /*unused*/) noexcept {}
  /// \endcond

  ConstantOnly(double constant) noexcept;

  ConstantOnly() = default;
  ~ConstantOnly() override = default;
  ConstantOnly(const ConstantOnly& /*rhs*/) = default;
  ConstantOnly& operator=(const ConstantOnly& /*rhs*/) = default;
  ConstantOnly(ConstantOnly&& /*rhs*/) noexcept = default;
  ConstantOnly& operator=(ConstantOnly&& /*rhs*/) noexcept = default;

  void operator()(const gsl::not_null<Scalar<double>*> value_at_x,
                  const tnsr::I<double, VolumeDim, Fr>& x, double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time) const noexcept override;
  void operator()(const gsl::not_null<Scalar<DataVector>*> value_at_x,
                  const tnsr::I<DataVector, VolumeDim, Fr>& x, double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time) const noexcept override;

  auto get_clone() const noexcept
      -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> override;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override;  // NOLINT

 private:
  friend bool operator==(const ConstantOnly& lhs,
                         const ConstantOnly& rhs) noexcept {
    return lhs.constant_ == rhs.constant_
  }

  double constant_ = std::numeric_limits<double>::signaling_NaN();

  template <typename T>
  void apply_call_operator(const gsl::not_null<Scalar<T>*> value_at_x,
                           const tnsr::I<T, VolumeDim, Fr>& x) const noexcept;
};

template <size_t VolumeDim, typename Fr>
bool operator!=(const ConstantOnly<VolumeDim, Fr>& lhs,
                const ConstantOnly<VolumeDim, Fr>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace GeneralizedHarmonic::ConstraintDamping

/// \cond
template <size_t VolumeDim, typename Fr>
PUP::able::PUP_ID GeneralizedHarmonic::ConstraintDamping::ConstantOnly<
    VolumeDim, Fr>::my_PUP_ID = 0;  // NOLINT
/// \endcond
