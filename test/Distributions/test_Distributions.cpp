//
// Created by alex on 10/17/23.
//
#include "gtest/gtest.h"
#include "tracking_lib/Distributions/MixtureDistribution.h"
#include <tracking_lib/Distributions/BaseDistribution.h>
#include <tracking_lib/Distributions/GaussianDistribution.h>

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(GaussianDistribution, GaussDist_merge_itself)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  ttb::Vector mean = ttb::Vector::Zero(2);
  ttb::Matrix cov = Eigen::Matrix2d::Identity();
  LOG_FATAL(".");
  auto gauss_dist = std::make_unique<ttb::GaussianDistribution>(mean, cov, 1, ttb::REFERENCE_POINT::CENTER);
  LOG_FATAL(".");
  ASSERT_EQ((gauss_dist->mean() - mean).norm(), 0);
  LOG_FATAL(".");

  ASSERT_EQ((gauss_dist->covariance() - cov).norm(), 0);
  LOG_FATAL(".");

  auto gauss_dist2 = std::make_unique<ttb::GaussianDistribution>(mean, cov, 1, ttb::REFERENCE_POINT::CENTER);
  gauss_dist->merge(std::move(gauss_dist2));
  LOG_FATAL(".");

  ASSERT_EQ((gauss_dist->mean() - mean).norm(), 0);
  ASSERT_EQ((gauss_dist->covariance() - cov).norm(), 0);

  auto gauss_dist_empty = std::make_unique<ttb::GaussianDistribution>();
  ASSERT_EQ(gauss_dist_empty->dists().size(), 0);
  auto gauss_dist_full = std::make_unique<ttb::GaussianDistribution>(mean, cov);
  gauss_dist_empty->merge(std::move(gauss_dist_full));
  ASSERT_EQ((gauss_dist_empty->mean() - mean).norm(), 0);
  ASSERT_EQ((gauss_dist_empty->covariance() - cov).norm(), 0);
  LOG_DEB(gauss_dist_empty->toString());
  ASSERT_EQ(gauss_dist_empty->dists().size(), 1);
}

TEST(GaussianDistribution, GaussDist_merge_other)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  ttb::Vector mean0 = ttb::Vector::Zero(2);
  ttb::Vector mean1 = ttb::Vector::Ones(2);
  ttb::Matrix eye = ttb::Matrix::Identity(2, 2);
  auto gauss_dist = std::make_unique<ttb::GaussianDistribution>(mean0, eye, 1, ttb::REFERENCE_POINT::CENTER);
  auto id_old = gauss_dist->id();

  auto gauss_dist2 = std::make_unique<ttb::GaussianDistribution>(mean1, eye, 1, ttb::REFERENCE_POINT::CENTER);
  gauss_dist->merge(std::move(gauss_dist2));

  ASSERT_NE(id_old, gauss_dist->id());
  ASSERT_EQ((gauss_dist->mean() - (mean0 + mean1) / 2).norm(), 0);
  std::cout << "Merged Covariance: " << gauss_dist->covariance();
  // TODO: check if this is right:
  // ASSERT_EQ(arma::norm(gauss_dist->covariance()-cov, "fro"), 0);
}

TEST(MixtureDistribution, Mixture)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  ttb::Vector mean0 = ttb::Vector::Zero(2);
  ttb::Matrix eye = ttb::Matrix::Identity(2, 2);
  auto gauss_dist = std::make_unique<ttb::GaussianDistribution>(mean0, eye, 1, ttb::REFERENCE_POINT::CENTER);
  auto mixture = std::make_unique<ttb::MixtureDistribution>();
  auto id_old = mixture->id();
  for (std::size_t i = 0; i < 10; ++i)
  {
    auto dist = gauss_dist->clone();
    dist->set(static_cast<double>(i) / 10 + 0.01);
    mixture->merge(std::move(dist));
    ASSERT_NE(id_old, mixture->id());
    id_old = mixture->id();
  }
  ASSERT_EQ(mixture->dists().size(), 10);
  id_old = mixture->id();
  mixture->pruneWeight(0.6);
  ASSERT_EQ(mixture->dists().size(), 4);
  ASSERT_NE(id_old, mixture->id());
  mixture->pruneWeight(10);
  ASSERT_EQ(mixture->dists().size(), 0);
}

TEST(MixtureDistribution, Mixture_prune)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  ttb::Vector mean0 = ttb::Vector::Zero(2);
  ttb::Matrix var0 = ttb::Matrix::Identity(2, 2);

  auto dist0 = std::make_unique<ttb::GaussianDistribution>(mean0, var0, 0.6, ttb::REFERENCE_POINT::CENTER);

  ttb::Vector mean1 = ttb::Vector::Ones(2);
  ttb::Matrix var1 = ttb::Matrix::Identity(2, 2);
  var1 *= 100;

  auto dist1 = std::make_unique<ttb::GaussianDistribution>(mean1, var1, 0.4, ttb::REFERENCE_POINT::CENTER);

  auto mixture = std::make_unique<ttb::MixtureDistribution>();
  mixture->merge(std::move(dist1));
  mixture->merge(std::move(dist0));

  mixture->pruneWeight(0.5);

  ASSERT_EQ(mixture->dists().size(), 1);
  ASSERT_TRUE(mixture->dists().front()->mean()(0) == 0);
}

TEST(MixtureDistribution, Best_component)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  ttb::Vector mean0 = ttb::Vector::Zero(2);
  ttb::Matrix var0 = ttb::Matrix::Identity(2, 2);

  auto dist0 = std::make_unique<ttb::GaussianDistribution>(mean0, var0, 0.6, ttb::REFERENCE_POINT::CENTER);
  auto bestId = dist0->id();
  LOG_DEB("Best id should be: " << bestId);
  ttb::Vector mean1 = ttb::Vector::Ones(2);
  ttb::Matrix var1 = ttb::Matrix::Identity(2, 2);
  var1 *= 100;

  auto dist1 = std::make_unique<ttb::GaussianDistribution>(mean1, var1, 0.4, ttb::REFERENCE_POINT::CENTER);

  auto mixture = std::make_unique<ttb::MixtureDistribution>();
  mixture->merge(std::move(dist1));
  mixture->merge(std::move(dist0));

  LOG_DEB("mixture: " << mixture->toString());

  auto const& bestComp = mixture->bestComponent();
  LOG_DEB("Best comp: " << bestComp.toString());
  ASSERT_EQ(bestComp.id(), bestId);
  ASSERT_EQ(mixture->dists().size(), 2);
}