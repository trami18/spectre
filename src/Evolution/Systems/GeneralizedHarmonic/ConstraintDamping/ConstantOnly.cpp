// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/ConstantOnly.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
/// \endcond

namespace GeneralizedHarmonic::ConstraintDamping {

template <size_t VolumeDim, typename Fr>
ConstantOnly<VolumeDim, Fr>::ConstantOnly(
    const double constant) noexcept
  : constant_(constant)
  ) {}

template <size_t VolumeDim, typename Fr>
template <typename T>
void ConstantOnly<VolumeDim, Fr>::apply_call_operator(
    const gsl::not_null<Scalar<T>*> value_at_x,
    const tnsr::I<T, VolumeDim, Fr>& x) get(*value_at_x) = constant_;
}  // namespace GeneralizedHarmonic::ConstraintDamping

template <size_t VolumeDim, typename Fr>
void ConstantOnly<VolumeDim, Fr>::operator()(
    const gsl::not_null<Scalar<double>*> value_at_x,
    const tnsr::I<double, VolumeDim, Fr>& x, const double /*time*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/) const noexcept {
  apply_call_operator(value_at_x, x);
}
template <size_t VolumeDim, typename Fr>
void ConstantOnly<VolumeDim, Fr>::operator()(
    const gsl::not_null<Scalar<DataVector>*> value_at_x,
    const tnsr::I<DataVector, VolumeDim, Fr>& x, const double /*time*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/) const noexcept {
  destructive_resize_components(value_at_x, get<0>(x).size());
  apply_call_operator(value_at_x, x);
}

template <size_t VolumeDim, typename Fr>
void ConstantOnly<VolumeDim, Fr>::pup(PUP::er& p) {
  DampingFunction<VolumeDim, Fr>::pup(p);
  p | constant_;
}

template <size_t VolumeDim, typename Fr>
auto ConstantOnly<VolumeDim, Fr>::get_clone() const noexcept
    -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> {
  return std::make_unique<ConstantOnly<VolumeDim, Fr>>(*this);
}
}  // namespace GeneralizedHarmonic::ConstraintDamping

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                                 \
  template GeneralizedHarmonic::ConstraintDamping::ConstantOnly<             \
      DIM(data), FRAME(data)>::ConstantOnly(const double constant) noexcept; \
  template void GeneralizedHarmonic::ConstraintDamping::ConstantOnly<        \
      DIM(data), FRAME(data)>::pup(PUP::er& p);                              \
  template auto GeneralizedHarmonic::ConstraintDamping::ConstantOnly<        \
      DIM(data), FRAME(data)>::get_clone() const noexcept                    \
      ->std::unique_ptr<DampingFunction<DIM(data), FRAME(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))
#undef DIM
#undef FRAME
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void GeneralizedHarmonic::ConstraintDamping::                   \
      ConstantOnly<DIM(data), FRAME(data)>::operator()(                    \
          const gsl::not_null<Scalar<DTYPE(data)>*> value_at_x,            \
          const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x,           \
          const double /*time*/,                                           \
          const std::unordered_map<                                        \
              std::string,                                                 \
              std::unique_ptr<domain::FunctionsOfTime::                    \
                                  FunctionOfTime>>& /*functions_of_time*/) \
          const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef FRAME
#undef DIM
#undef DTYPE
#undef INSTANTIATE

/// \endcond
