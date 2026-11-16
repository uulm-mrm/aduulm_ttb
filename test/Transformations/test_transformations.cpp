#include "gtest/gtest.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Transformations/Transformation.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include <aduulm_logger/aduulm_logger.hpp>
#include <filesystem>
#include <random>
#include <figcone/configreader.h>
#include <cmath>
#include <numbers>
#include <optional>

using namespace std::literals;
std::filesystem::path const file{ "test/StateModels/ctrv.yaml" };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(Transformations, transformation_matrix)
{
  ttb::Components const to({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y });
  ttb::Components const from({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y, ttb::COMPONENT::POS_Z });
  auto const trans = ttb::transformation::transformationMatrix<double>(from, to);
  ASSERT_TRUE(trans.has_value());
  ASSERT_TRUE(not ttb::transformation::transformationMatrix<double>(to, from).has_value());
}

TEST(Transformations, transformation)
{
  ttb::Components from({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y });
  ttb::Components to({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y, ttb::COMPONENT::POS_Z });
  ttb::Vector<double> state{ { 0, 0 } };
  ttb::Matrix<double> cov = ttb::Matrix<double>::Identity(2, 2);
  {
    auto sigma_points_2 = ttb::transformation::createSigmaPoints(state, cov);
  }
  auto trans = ttb::transformation::transform(state, cov, from, to);
  ASSERT_TRUE(not trans.has_value());
}

TEST(Transformations, rp_transformation)
{
  ttb::Components from({ ttb::COMPONENT::POS_X,
                         ttb::COMPONENT::POS_Y,
                         ttb::COMPONENT::ROT_Z,
                         ttb::COMPONENT::LENGTH,
                         ttb::COMPONENT::WIDTH });
  for (std::size_t i = 0; i < 50; ++i)
  {
    ttb::Vector<double> state = ttb::Vector<double>::Random(5);
    state(2) *= std::numbers::pi;
    state(3) *= state(3);
    state(4) *= state(4);

    auto trans_bl =
        ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::BACK_LEFT);
    auto trans_br = ttb::transformation::transform(
        trans_bl, from, ttb::REFERENCE_POINT::BACK_LEFT, ttb::REFERENCE_POINT::BACK_RIGHT);
    auto trans_fl = ttb::transformation::transform(
        trans_br, from, ttb::REFERENCE_POINT::BACK_RIGHT, ttb::REFERENCE_POINT::FRONT_LEFT);
    auto trans_fr = ttb::transformation::transform(
        trans_fl, from, ttb::REFERENCE_POINT::FRONT_LEFT, ttb::REFERENCE_POINT::FRONT_RIGHT);
    auto trans_c =
        ttb::transformation::transform(trans_fr, from, ttb::REFERENCE_POINT::FRONT_RIGHT, ttb::REFERENCE_POINT::CENTER);
    ASSERT_TRUE((trans_c - state).norm() < 1e-7);
  }
}

TEST(Transformations, rp_transformation_edges)
{
  ttb::Components from({ ttb::COMPONENT::POS_X,
                         ttb::COMPONENT::POS_Y,
                         ttb::COMPONENT::ROT_Z,
                         ttb::COMPONENT::LENGTH,
                         ttb::COMPONENT::WIDTH });
  for (std::size_t i = 0; i < 50; ++i)
  {
    ttb::Vector<double> state = ttb::Vector<double>::Random(5);
    state(2) *= std::numbers::pi;
    state(3) *= state(3);
    state(4) *= state(4);

    auto trans_b =
        ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::BACK);
    auto trans_l =
        ttb::transformation::transform(trans_b, from, ttb::REFERENCE_POINT::BACK, ttb::REFERENCE_POINT::LEFT);
    auto trans_r =
        ttb::transformation::transform(trans_l, from, ttb::REFERENCE_POINT::LEFT, ttb::REFERENCE_POINT::RIGHT);
    trans_b = ttb::transformation::transform(trans_r, from, ttb::REFERENCE_POINT::RIGHT, ttb::REFERENCE_POINT::BACK);
    auto trans_c =
        ttb::transformation::transform(trans_b, from, ttb::REFERENCE_POINT::BACK, ttb::REFERENCE_POINT::CENTER);
    ASSERT_TRUE((trans_c - state).norm() < 1e-7);
  }
}

TEST(Transformations, rp_transformation_edges_values)
{
  ttb::Components from({ ttb::COMPONENT::POS_X,
                         ttb::COMPONENT::POS_Y,
                         ttb::COMPONENT::ROT_Z,
                         ttb::COMPONENT::LENGTH,
                         ttb::COMPONENT::WIDTH });

  ttb::Vector<double> state{ { 0, 0, 0, 1, 2 } };

  ASSERT_TRUE((ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::BACK) -
               ttb::Vector<double>{ { -0.5, 0, 0, 1, 2 } })
                  .norm() < 1e-7);
  ASSERT_TRUE((ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::FRONT) -
               ttb::Vector<double>{ { 0.5, 0, 0, 1, 2 } })
                  .norm() < 1e-7);
  ASSERT_TRUE((ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::LEFT) -
               ttb::Vector<double>{ { 0, 1, 0, 1, 2 } })
                  .norm() < 1e-7);
  ASSERT_TRUE((ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::RIGHT) -
               ttb::Vector<double>{ { 0, -1, 0, 1, 2 } })
                  .norm() < 1e-7);
  ASSERT_TRUE(
      (ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::BACK_RIGHT) -
       ttb::Vector<double>{ { -0.5, -1, 0, 1, 2 } })
          .norm() < 1e-7);
  ASSERT_TRUE(
      (ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::FRONT_LEFT) -
       ttb::Vector<double>{ { 0.5, 1, 0, 1, 2 } })
          .norm() < 1e-7);
  ASSERT_TRUE(
      (ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::FRONT_RIGHT) -
       ttb::Vector<double>{ { 0.5, -1, 0, 1, 2 } })
          .norm() < 1e-7);
  ASSERT_TRUE(
      (ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::BACK_LEFT) -
       ttb::Vector<double>{ { -0.5, 1, 0, 1, 2 } })
          .norm() < 1e-7);

  state = ttb::Vector<double>{ { 0, 0, std::numbers::pi, 1, 2 } };
  ASSERT_TRUE(
      (ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::FRONT_LEFT) -
       ttb::Vector<double>{ { -0.5, -1, std::numbers::pi, 1, 2 } })
          .norm() < 1e-7);
}

namespace
{
constexpr double kPi = std::numbers::pi;

double angleDifference(double a, double b)
{
  return std::remainder(a - b, 2.0 * kPi);
}

std::optional<ttb::Vector<double>> wrapYaw(ttb::Vector<double> const& x)
{
  ttb::Vector result = x;
  result(0) = std::remainder(result(0), 2.0 * kPi);
  return result;
}
}  // namespace

TEST(TransformationsUT, SigmaPointWeightsAndNegativeCentralWeight)
{
  ttb::Vector<double> const mean = ttb::Vector<double>::Zero(4);
  ttb::Matrix<double> const cov = ttb::Matrix<double>::Identity(4, 4);
  auto const sigma = ttb::transformation::createSigmaPoints(mean, cov);

  ASSERT_TRUE(sigma.has_value());
  EXPECT_EQ(sigma->sigma_points.rows(), 4);
  EXPECT_EQ(sigma->sigma_points.cols(), 9);
  EXPECT_NEAR(sigma->weights(0), -1.0 / 3.0, 1e-12);
  EXPECT_NEAR(sigma->weights.sum(), 1.0, 1e-12);
  for (ttb::Index i = 1; i < sigma->weights.size(); ++i)
  {
    EXPECT_NEAR(sigma->weights(i), 1.0 / 6.0, 1e-12);
  }
}

TEST(TransformationsUT, ModifiedUTMatchesLinearGroundTruth)
{
  ttb::Vector<double> mean(4);
  mean << 2.0, -1.0, 0.5, 3.0;

  ttb::Matrix<double> cov = ttb::Matrix<double>::Identity(4, 4);
  cov(0, 0) = 0.4;
  cov(1, 1) = 0.7;
  cov(2, 2) = 0.2;
  cov(3, 3) = 0.9;
  cov(0, 2) = cov(2, 0) = 0.1;

  ttb::Matrix<double> A(2, 4);
  A << 1.0, 2.0, -0.5, 0.0, 0.0, -1.0, 1.0, 3.0;
  ttb::Vector<double> b(2);
  b << 3.0, -4.0;

  auto const transformed = ttb::transformation::unscentTransform(
      mean, cov, [&A, &b](ttb::Vector<double> const& x) -> std::optional<ttb::Vector<double>> {
        return ttb::Vector<double>(A * x + b);
      });

  ASSERT_TRUE(transformed.has_value());
  ASSERT_TRUE(transformed->cross_cov.has_value());
  EXPECT_TRUE(transformed->mean.isApprox(A * mean + b, 1e-10));
  EXPECT_TRUE(transformed->cov.isApprox(A * cov * A.transpose(), 1e-10));
  EXPECT_TRUE(transformed->cross_cov->isApprox(cov * A.transpose(), 1e-10));
}

TEST(TransformationsUT, ModifiedUTAvoidsNegativeVarianceForNonlinearMap)
{
  ttb::Vector<double> const mean = ttb::Vector<double>::Zero(4);
  ttb::Matrix<double> const cov = ttb::Matrix<double>::Identity(4, 4);
  auto const sigma = ttb::transformation::createSigmaPoints(mean, cov);
  ASSERT_TRUE(sigma.has_value());

  ttb::Matrix<double> y(1, sigma->sigma_points.cols());
  for (ttb::Index i = 0; i < y.cols(); ++i)
  {
    y(0, i) = sigma->sigma_points.col(i).squaredNorm() / 3.0;
  }

  auto const standard = ttb::transformation::SigmaPoints2Moments(y, sigma->weights);
  auto const modified = ttb::transformation::modifiedSigmaPoints2Moments(y, sigma->weights);

  EXPECT_NEAR(standard.first(0), 4.0 / 3.0, 1e-10);
  EXPECT_NEAR(modified.first(0), 4.0 / 3.0, 1e-10);
  EXPECT_NEAR(standard.second(0, 0), -4.0 / 9.0, 1e-10);
  EXPECT_NEAR(modified.second(0, 0), 4.0 / 3.0, 1e-10);

  // The actual unscentTransform() must select the modified version.
  auto const result = ttb::transformation::unscentTransform(
      mean, cov, [](ttb::Vector<double> const& x) -> std::optional<ttb::Vector<double>> {
        ttb::Vector<double> tmp(1);
        tmp(0) = x.squaredNorm() / 3.0;
        return tmp;
      });
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->cov(0, 0), 4.0 / 3.0, 1e-10);
}

// TEST(TransformationsUT, AngleWrappingWithPositiveCentralWeight)
// {
//   ttb::Vector<double> mean = ttb::Vector<double>::Zero(2);
//   mean(0) = 3.1;
//   ttb::Matrix<double> cov = ttb::Matrix<double>::Identity(2, 2) * 0.01;
//   cov(0, 0) = 0.04;
//
//   auto const result = ttb::transformation::unscentTransform(mean, cov, wrapYaw, std::optional<ttb::Index>{ 0 });
//   ASSERT_TRUE(result.has_value());
//   ASSERT_TRUE(result->cross_cov.has_value());
//
//   EXPECT_NEAR(angleDifference(result->mean(0), mean(0)), 0.0, 1e-6);
//   // Expected to FAIL on the current implementation: unwrapped differences
//   // are used in SigmaPoints2Moments() before the circular mean is applied.
//   EXPECT_NEAR(result->cov(0, 0), 0.04, 1e-6) << "Yaw covariance must use wrapped angle residuals (+pi/-pi).";
//   EXPECT_NEAR((*result->cross_cov)(0, 0), 0.04, 1e-6) << "Cross-covariance must use wrapped transformed angle "
//                                                          "residuals.";
// }
//
// TEST(TransformationsUT, AngleWrappingWithNegativeCentralWeight)
// {
//   ttb::Vector<double> mean = ttb::Vector<double>::Zero(4);
//   mean(0) = 3.1;
//   ttb::Matrix<double> cov = ttb::Matrix<double>::Identity(4, 4) * 0.01;
//   cov(0, 0) = 0.04;
//
//   auto const result = ttb::transformation::unscentTransform(mean, cov, wrapYaw, std::optional<ttb::Index>{ 0 });
//   ASSERT_TRUE(result.has_value());
//   ASSERT_TRUE(result->cross_cov.has_value());
//
//   EXPECT_NEAR(angleDifference(result->mean(0), mean(0)), 0.0, 1e-6);
//   // Expected to FAIL until the yaw residuals have been wrapped.
//   EXPECT_NEAR(result->cov(0, 0), 0.04, 1e-6) << "Modified UT currently uses raw yaw differences near +/-pi.";
//   EXPECT_NEAR((*result->cross_cov)(0, 0), 0.04, 1e-6) << "Cross-covariance currently uses raw yaw differences.";
// }

namespace
{
double utSquaredYawCircularReference(ttb::transformation::SigmaPoints<double> const& sigma)
{
  double weighted_sin = 0.0;
  double weighted_cos = 0.0;
  for (ttb::Index i = 0; i < sigma.weights.size(); ++i)
  {
    double const x0 = sigma.sigma_points(0, i);
    double const angle = std::remainder(x0 * x0, 2.0 * std::numbers::pi);
    weighted_sin += sigma.weights(i) * std::sin(angle);
    weighted_cos += sigma.weights(i) * std::cos(angle);
  }
  return std::atan2(weighted_sin, weighted_cos);
}

std::optional<ttb::Vector<double>> utSquaredYaw(ttb::Vector<double> const& x)
{
  ttb::Vector<double> angle(1);
  angle(0) = std::remainder(x(0) * x(0), 2.0 * std::numbers::pi);
  return angle;
}
}  // namespace

TEST(TransformationsUT, ModifiedCovarianceDiffersFromStandardForNonlinearMap)
{
  ttb::Vector<double> const mean = ttb::Vector<double>::Zero(4);
  ttb::Matrix<double> const cov = ttb::Matrix<double>::Identity(4, 4);
  auto const sigma = ttb::transformation::createSigmaPoints(mean, cov);
  ASSERT_TRUE(sigma.has_value());

  ttb::Matrix<double> outputs(1, sigma->sigma_points.cols());
  for (ttb::Index i = 0; i < outputs.cols(); ++i)
  {
    outputs(0, i) = sigma->sigma_points(0, i) * sigma->sigma_points(0, i);
  }
  auto const standard = ttb::transformation::SigmaPoints2Moments(outputs, sigma->weights);
  auto const modified = ttb::transformation::modifiedSigmaPoints2Moments(outputs, sigma->weights);

  EXPECT_NEAR(standard.first(0), 1.0, 1e-12);
  EXPECT_NEAR(modified.first(0), 1.0, 1e-12);
  EXPECT_NEAR(standard.second(0, 0), 2.0, 1e-12);
  EXPECT_NEAR(modified.second(0, 0), 3.0, 1e-12);
  EXPECT_NEAR(modified.second(0, 0) - standard.second(0, 0), std::pow(modified.first(0) - outputs(0, 0), 2), 1e-12);
}

TEST(TransformationsUT, NonlinearCircularMeanUsesAllWeightsForPositiveCentralWeight)
{
  ttb::Vector<double> const mean = ttb::Vector<double>::Zero(2);
  ttb::Matrix<double> cov = ttb::Matrix<double>::Identity(2, 2) * 0.01;
  cov(0, 0) = 0.25;
  auto const sigma = ttb::transformation::createSigmaPoints(mean, cov);
  ASSERT_TRUE(sigma.has_value());
  double const expected = utSquaredYawCircularReference(*sigma);

  auto const result = ttb::transformation::unscentTransform(mean, cov, utSquaredYaw, std::optional<ttb::Index>{ 0 });
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(std::remainder(result->mean(0) - expected, 2.0 * std::numbers::pi), 0.0, 0.01) << "Angle mean appears to "
                                                                                                "ignore the positive "
                                                                                                "central sigma-point "
                                                                                                "weight";
}

TEST(TransformationsUT, NonlinearCircularMeanUsesSignedWeightsForNegativeCentralWeight)
{
  ttb::Vector<double> const mean = ttb::Vector<double>::Zero(9);
  ttb::Matrix<double> cov = ttb::Matrix<double>::Identity(9, 9) * 0.01;
  cov(0, 0) = 0.04;
  auto const sigma = ttb::transformation::createSigmaPoints(mean, cov);
  ASSERT_TRUE(sigma.has_value());
  double const expected = utSquaredYawCircularReference(*sigma);

  auto const result = ttb::transformation::unscentTransform(mean, cov, utSquaredYaw, std::optional<ttb::Index>{ 0 });
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(std::remainder(result->mean(0) - expected, 2.0 * std::numbers::pi), 0.0, 0.001) << "Angle mean appears "
                                                                                                 "to normalize away a "
                                                                                                 "negative central UT "
                                                                                                 "weight";
}

// TEST(TransformationsUT, AugmentedNoiseYawCrossesBranchCut)
// {
//   ttb::Vector<double> mean = ttb::Vector<double>::Zero(2);
//   mean(0) = 3.1;
//   ttb::Matrix<double> cov = ttb::Matrix<double>::Identity(2, 2) * 0.01;
//   cov(0, 0) = 0.04;
//   ttb::Matrix<double> noise_cov = ttb::Matrix<double>::Identity(2, 2) * 0.01;
//   noise_cov(0, 0) = 0.09;
//
//   auto const result = ttb::transformation::unscentTransform(
//       mean,
//       cov,
//       noise_cov,
//       [](ttb::Vector<double> const& x, ttb::Vector<double> const& noise) -> std::optional<ttb::Vector<double>> {
//         ttb::Vector<double> output(1);
//         output(0) = std::remainder(x(0) + noise(0), 2.0 * std::numbers::pi);
//         return output;
//       },
//       std::optional<ttb::Index>{ 0 });
//
//   ASSERT_TRUE(result.has_value());
//   ASSERT_TRUE(result->cross_cov.has_value());
//   EXPECT_NEAR(std::remainder(result->mean(0) - mean(0), 2.0 * std::numbers::pi), 0.0, 1e-8);
//   EXPECT_NEAR(result->cov(0, 0), 0.13, 1e-8);
//   EXPECT_NEAR((*result->cross_cov)(0, 0), 0.04, 1e-8);
//   EXPECT_NEAR((*result->cross_cov)(1, 0), 0.0, 1e-8);
// }
