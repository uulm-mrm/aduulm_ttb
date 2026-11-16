#include "gtest/gtest.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/TTBManager/TTBManager.h"
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
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager<double> manager(params);
  EXPECT_TRUE(true) << "Invalid config";
  std::random_device rd;
  std::mt19937 gen(rd());
  for (std::size_t i = 0; i < 100; ++i)
  {
    ttb::Vector<double> mean = ttb::Vector<double>::Random(ttb::ALL_COMPONENTS.size());
    ttb::Matrix<double> cov = ttb::Matrix<double>::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();
    ttb::State meas =
        ttb::create_meas_state<double>(
            &manager, ttb::GaussianDistribution(mean, cov), ttb::Time::max(), ttb::Components(ttb::ALL_COMPONENTS))
            .value();
    std::optional<ttb::State<double>> state = manager.getMeasModel(ttb::SourceId{ "0" }).create_state(meas);
    ASSERT_TRUE(state.has_value()) << "Can not create State of Meas with all possible Components";
    mean = ttb::Vector<double>::Random(ttb::ALL_COMPONENTS.size());
    cov = ttb::Matrix<double>::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();
    ttb::EgoMotionDistribution egoMotion(ttb::GaussianDistribution(mean, cov), ttb::Components(ttb::ALL_COMPONENTS));
    state.value().predict(std::chrono::milliseconds(std::uniform_int_distribution<>(1, 5000)(gen)), egoMotion);
    ASSERT_TRUE(true);
  }
}

TEST(CTRVModels, predict)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager<double> manager(params);
  EXPECT_TRUE(true) << "Invalid config";
  std::random_device rd;
  std::mt19937 gen(rd());
  for (std::size_t i = 0; i < 100; ++i)
  {
    ttb::Vector<double> mean = ttb::Vector<double>::Random(ttb::ALL_COMPONENTS.size());
    ttb::Matrix<double> cov = ttb::Matrix<double>::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();
    ttb::State<double> meas =
        ttb::create_meas_state(
            &manager, ttb::GaussianDistribution(mean, cov), ttb::Time::max(), ttb::Components(ttb::ALL_COMPONENTS))
            .value();
    std::optional<ttb::State<double>> state = manager.getMeasModel(ttb::SourceId{ "0" }).create_state(meas);
    ASSERT_TRUE(state.has_value()) << "Can not create State of Meas with all possible Components";
    mean = ttb::Vector<double>::Random(ttb::ALL_COMPONENTS.size());
    cov = ttb::Matrix<double>::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();
    ttb::EgoMotionDistribution egoMotion(ttb::GaussianDistribution(mean, cov), ttb::Components(ttb::ALL_COMPONENTS));
    auto const rot_ind =
        manager.getStateModel(ttb::StateModelId{ 4 }).state_comps().indexOf(ttb::COMPONENT::ROT_Z).value();
    double angle_before = state.value()
                              ._state_dist.at(ttb::StateModelId{ 4 })
                              .second.to_gaussian_components()
                              .front()
                              .second.mean()(rot_ind);
    state.value().predict(std::chrono::milliseconds(1), egoMotion);
    double angle_after = state.value()
                             ._state_dist.at(ttb::StateModelId{ 4 })
                             .second.to_gaussian_components()
                             .front()
                             .second.mean()(rot_ind);
    if (ttb::angles::smaller_diff(angle_before, angle_after) > 0.1)
    {
      LOG_FATAL("big change in angle in prediction. before: " << angle_before << " after: " << angle_after);
      EXPECT_TRUE(false);
    }
    ASSERT_TRUE(true);
  }
}
