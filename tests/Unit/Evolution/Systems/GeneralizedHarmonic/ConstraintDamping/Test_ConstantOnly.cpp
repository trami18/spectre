// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/ConstantOnly.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t VolumeDim, typename DataType, typename Fr>
void test_constant_only_random(const DataType& used_for_size) noexcept {
  Parallel::register_derived_classes_with_charm<
      GeneralizedHarmonic::ConstraintDamping::ConstantOnly<VolumeDim, Fr>>();
  // Generate the amplitude and width
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  // std::uniform_real_distribution<> positive_dis(0, 1);

  const double constant = real_dis(gen);

  GeneralizedHarmonic::ConstraintDamping::ConstantOnly<VolumeDim, Fr>
      const_only{constant};

  TestHelpers::GeneralizedHarmonic::ConstraintDamping::check(
      std::move(const_only), "constant_only", used_for_size, {{{-1.0, 1.0}}},
      "IgnoredFunctionOfTime", constant);

  std::unique_ptr<
      GeneralizedHarmonic::ConstraintDamping::ConstantOnly<VolumeDim, Fr>>
      const_only_unique_ptr = std::make_unique<
          GeneralizedHarmonic::ConstraintDamping::ConstantOnly<VolumeDim, Fr>>(
          constant);

  TestHelpers::GeneralizedHarmonic::ConstraintDamping::check(
      std::move(const_only_unique_ptr->get_clone()), "constant_only",
      used_for_size, {{{-1.0, 1.0}}}, "IgnoredFunctionOfTime", constant);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.ConstraintDamp.ConstOnly",
    "[PointwiseFunctions][Unit]") {
  const DataVector dv{5};

  pypp::SetupLocalPythonEnvironment{
      "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Python"};

  using VolumeDims = tmpl::integral_list<size_t, 1, 2, 3>;
  using Frames = tmpl::list<Frame::Grid, Frame::Inertial>;

  tmpl::for_each<VolumeDims>([&dv](auto dim_v) {
    using VolumeDim = typename decltype(dim_v)::type;
    tmpl::for_each<Frames>([&dv](auto frame_v) {
      using Fr = typename decltype(frame_v)::type;
      test_constant_only_random<VolumeDim::value, DataVector, Fr>(dv);
      test_constant_only_random<VolumeDim::value, double, Fr>(
          std::numeric_limits<double>::signaling_NaN());
    });
  });

  TestHelpers::test_factory_creation<GeneralizedHarmonic::ConstraintDamping::
                                         DampingFunction<1, Frame::Inertial>>(
      "GaussianPlusConstant:\n"
      "  Constant: 4.0");

  const double constant_3d{5.0};
  const double amplitude_3d{4.0};
  const double width_3d{1.5};
  const std::array<double, 3> center_3d{{1.1, -2.2, 3.3}};
  const GeneralizedHarmonic::ConstraintDamping::GaussianPlusConstant<
      3, Frame::Inertial>
      gauss_plus_const_3d{constant_3d, amplitude_3d, width_3d, center_3d};
  const auto created_gauss_plus_const =
      TestHelpers::test_creation<GeneralizedHarmonic::ConstraintDamping::
                                     GaussianPlusConstant<3, Frame::Inertial>>(
          "Constant: 5.0\n"
          "Amplitude: 4.0\n"
          "Width: 1.5\n"
          "Center: [1.1, -2.2, 3.3]");
  CHECK(created_gauss_plus_const == gauss_plus_const_3d);
  const auto created_gauss_gh_damping_function =
      TestHelpers::test_factory_creation<
          GeneralizedHarmonic::ConstraintDamping::DampingFunction<
              3, Frame::Inertial>>(
          "GaussianPlusConstant:\n"
          "  Constant: 5.0\n"
          "  Amplitude: 4.0\n"
          "  Width: 1.5\n"
          "  Center: [1.1, -2.2, 3.3]");

  test_serialization(gauss_plus_const_3d);
}
