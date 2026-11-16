#include "gtest/gtest.h"
#include <tracking_lib/Distributions/MixtureDistribution.h>
#include <tracking_lib/Distributions/GaussianDistribution.h>
#include <tracking_lib/Distributions/Distribution.h>

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(GaussianDistribution, GaussDist_merge_itself)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  ttb::Vector<double> mean = ttb::Vector<double>::Zero(2);
  ttb::Matrix<double> cov = Eigen::Matrix2d::Identity();
  auto gauss_dist = ttb::GaussianDistribution(mean, cov, ttb::REFERENCE_POINT::CENTER);
  ASSERT_EQ((gauss_dist.mean() - mean).norm(), 0);
  ASSERT_EQ((gauss_dist.covariance() - cov).norm(), 0);
  auto gauss_dist2 = ttb::GaussianDistribution(mean, cov, ttb::REFERENCE_POINT::CENTER);
}

TEST(GaussianDistribution, Distribution2)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  using namespace ttb;
  Vector<double> mean = Vector<double>::Zero(2);
  Matrix<double> cov = Eigen::Matrix2d::Identity();
  GaussianDistribution gauss2(mean, cov);
  MixtureDistribution<GaussianDistribution<double>, double> gauss_mixture({ { 1.0, gauss2 } });
  ASSERT_EQ(gauss2.mean(),
            gauss_mixture.mean(Components(std::vector<COMPONENT>{ COMPONENT::POS_X, COMPONENT::POS_Y })));
  ASSERT_EQ(gauss2.covariance(),
            gauss_mixture.covariance(Components(std::vector<COMPONENT>{ COMPONENT::POS_X, COMPONENT::POS_Y })));
  Distribution<double> dist;
  dist._distribution = gauss_mixture;
  ASSERT_EQ(dist.to_gaussian_components().size(), 1);
  dist.to_gaussian_components().front().second.covariance() = 3 * cov;
  dist.to_gaussian_components().front().second.mean()(0) = 666;
  ASSERT_EQ(dist.to_gaussian_components().front().second.covariance()(0, 0), 3);
  ASSERT_EQ(dist.to_gaussian_components().front().second.mean()(0), 666);
  dist.to_gaussian_components().front().first = 0.5;
  ASSERT_EQ(dist.to_gaussian_components().front().first, 0.5);
}

TEST(GaussianDistribution, kl_divergence)
{
  for (std::size_t i = 0; i < 100; ++i)
  {
    ttb::Vector<double> const mean = ttb::Vector<double>::Random(5);
    ttb::Matrix<double> cov = ttb::Matrix<double>::Random(5, 5);
    cov *= cov.transpose();
    ttb::GaussianDistribution const g1(mean, cov);
    ttb::Vector<double> const mean2 = ttb::Vector<double>::Random(5);
    ttb::Matrix<double> cov2 = ttb::Matrix<double>::Random(5, 5);
    cov2 *= cov2.transpose();
    ttb::GaussianDistribution const g2(mean2, cov2);
  }
}
