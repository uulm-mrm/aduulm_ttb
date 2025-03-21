#include "gtest/gtest.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include <tracking_lib/MeasurementModels/BaseMeasurementModel.h>
#include <tracking_lib/Measurements/Measurement.h>
#include <tracking_lib/States/Innovation.h>

#include <tracking_lib/TTBManager/TTBManager.h>

#include <aduulm_logger/aduulm_logger.hpp>
#include <filesystem>
#include <random>
#include <figcone/configreader.h>
#include <tracking_lib/Misc/AngleNormalization.h>

using namespace std::literals;
std::filesystem::path const file{ "test/configs/tracking.yaml" };
constexpr double TOL{ 1e-4 };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(StateModels, predict)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager manager(params);
  EXPECT_TRUE(true) << "Invalid config";
  std::random_device rd;
  std::mt19937 gen(rd());
  for (std::size_t i = 0; i < 1000; ++i)
  {
    ttb::Vector mean = ttb::Vector::Random(ttb::ALL_COMPONENTS.size());
    ttb::Matrix cov = ttb::Matrix::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();
    ttb::Measurement meas(
        std::make_unique<ttb::GaussianDistribution>(mean, cov), ttb::Time::max(), ttb::Components(ttb::ALL_COMPONENTS));
    std::optional<ttb::State> state = manager.getMeasModelMap().at(ttb::MeasModelId{ "0" })->createState(meas);
    ASSERT_TRUE(state.has_value()) << "Can not create State of Meas with all possible Components";
    mean = ttb::Vector::Random(ttb::ALL_COMPONENTS.size());
    cov = ttb::Matrix::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();
    ttb::EgoMotionDistribution egoMotion(std::make_unique<ttb::GaussianDistribution>(mean, cov),
                                         ttb::Components(ttb::ALL_COMPONENTS));
    state.value().predict(std::chrono::milliseconds(std::uniform_int_distribution<>(1, 5000)(gen)), egoMotion);
    ASSERT_TRUE(true);
  }
}

TEST(CTRVModels, predict)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager manager(params);
  EXPECT_TRUE(true) << "Invalid config";
  std::random_device rd;
  std::mt19937 gen(rd());
  for (std::size_t i = 0; i < 1000; ++i)
  {
    ttb::Vector mean = ttb::Vector::Random(ttb::ALL_COMPONENTS.size());
    ttb::Matrix cov = ttb::Matrix::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();
    ttb::Measurement meas(
        std::make_unique<ttb::GaussianDistribution>(mean, cov), ttb::Time::max(), ttb::Components(ttb::ALL_COMPONENTS));
    std::optional<ttb::State> state = manager.getMeasModelMap().at(ttb::MeasModelId{ "0" })->createState(meas);
    ASSERT_TRUE(state.has_value()) << "Can not create State of Meas with all possible Components";
    mean = ttb::Vector::Random(ttb::ALL_COMPONENTS.size());
    cov = ttb::Matrix::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();
    ttb::EgoMotionDistribution egoMotion(std::make_unique<ttb::GaussianDistribution>(mean, cov),
                                         ttb::Components(ttb::ALL_COMPONENTS));
    auto const rot_ind =
        manager.getStateModel(ttb::StateModelId{ 4 }).state_comps().indexOf(ttb::COMPONENT::ROT_Z).value();
    double angle_before = state.value()._state_dist.at(ttb::StateModelId{ 4 })->mean()(rot_ind);
    state.value().predict(std::chrono::milliseconds(1), egoMotion);
    double angle_after = state.value()._state_dist.at(ttb::StateModelId{ 4 })->mean()(rot_ind);
    if (ttb::angles::smaller_diff(angle_before, angle_after) > 0.1)
    {
      LOG_FATAL("big change in angle in prediction. before: " << angle_before << " after: " << angle_after);
      EXPECT_TRUE(false);
    }
    ASSERT_TRUE(true);
  }
}