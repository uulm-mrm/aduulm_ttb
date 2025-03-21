#include "gtest/gtest.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include <tracking_lib/Measurements/MeasurementContainer.h>
#include <tracking_lib/MeasurementModels/BaseMeasurementModel.h>
#include <tracking_lib/StateModels/BaseStateModel.h>
#include <tracking_lib/Measurements/Measurement.h>
#include <tracking_lib/States/Innovation.h>

#include <tracking_lib/TTBManager/TTBManager.h>
#include <tracking_lib/PersistenceModels/BasePersistenceModel.h>
#include <tracking_lib/BirthModels/BaseBirthModel.h>
#include <tracking_lib/Trackers/BaseTracker.h>

#include <aduulm_logger/aduulm_logger.hpp>
#include <filesystem>
#include <random>
#include <figcone/configreader.h>

using namespace std::literals;
std::filesystem::path const file{ "test/configs/tracking.yaml" };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(GenericBoxModel, createState)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager manager(params);
  EXPECT_TRUE(true) << "Invalid config";
  for (std::size_t i = 0; i < 10000; ++i)
  {
    ttb::Vector mean = ttb::Vector::Random(ttb::ALL_COMPONENTS.size());
    ttb::Matrix cov = ttb::Matrix::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();
    ttb::Measurement meas(
        std::make_unique<ttb::GaussianDistribution>(mean, cov), ttb::Time::max(), ttb::Components(ttb::ALL_COMPONENTS));
    std::optional<ttb::State> state = manager.getMeasModelMap().at(ttb::MeasModelId{ "0" })->createState(meas);
    ASSERT_TRUE(state.has_value()) << "Can not create State of Meas with all possible Components";
  }
}

TEST(GenericBoxModel, innovate)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager manager(params);
  EXPECT_TRUE(true) << "Invalid config";
  std::random_device rd;
  std::mt19937 gen(rd());
  for (std::size_t i = 0; i < 100; ++i)
  {
    ttb::Vector mean = ttb::Vector::Random(ttb::ALL_COMPONENTS.size());
    ttb::Matrix cov = ttb::Matrix::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();

    std::optional<ttb::State> state =
        manager.getMeasModelMap()
            .at(ttb::MeasModelId{ "0" })
            ->createState(ttb::Measurement(std::make_unique<ttb::GaussianDistribution>(mean, cov),
                                           ttb::Time::max(),
                                           ttb::Components(ttb::ALL_COMPONENTS)));
    ASSERT_TRUE(state.has_value()) << "Can not create State of Meas with all possible Components";
    ttb::MeasurementContainer container{ ._id = ttb::MeasModelId{ "0" }, ._time = ttb::Time::max() };
    for (std::size_t numMeas = 0; numMeas < 10; ++numMeas)
    {
      auto compsMeas = ttb::ALL_COMPONENTS;
      std::shuffle(compsMeas.begin(), compsMeas.end(), gen);
      std::size_t numCompsMeasured = std::uniform_int_distribution<>(1, compsMeas.size() - 1)(gen);
      mean = ttb::Vector::Random(numCompsMeasured);
      cov = ttb::Matrix::Random(numCompsMeasured, numCompsMeasured);
      cov *= cov.transpose();
      ttb::Measurement meas(std::make_unique<ttb::GaussianDistribution>(mean, cov),
                            ttb::Time::max(),
                            ttb::Components(std::vector(compsMeas.begin(), compsMeas.begin() + numCompsMeasured)));
      container._data.push_back(std::move(meas));
    }
    state.value().innovate(container);
    EXPECT_TRUE(true);
  }
}

TEST(GenericBoxModel, innovateRP)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager manager(params);
  EXPECT_TRUE(true) << "Invalid config";

  ttb::Vector mean{ { 0, 0, 0, 0, 2, 1 } };
  ttb::Matrix cov = ttb::Matrix::Identity(6, 6);
  std::optional<ttb::State> state = manager.createState();
  state->_state_dist.at(ttb::StateModelId{ 1 })->merge(std::make_unique<ttb::GaussianDistribution>(mean, cov));

  ttb::Vector meas_mean{ { 1, 0 } };
  ttb::Matrix meas_cov = 0.1 * ttb::Matrix::Identity(2, 2);
  ttb::Measurement meas_front(std::make_unique<ttb::GaussianDistribution>(meas_mean, meas_cov),
                              ttb::Time::max(),
                              ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas_front._ref_point_measured = true;
  meas_front._dist->set(ttb::REFERENCE_POINT::FRONT);

  ttb::MeasurementContainer container{ ._id = ttb::MeasModelId{ "0" },
                                       ._data = { meas_front },
                                       ._time = ttb::Time::max() };

  state.value().innovate(container);
  LOG_FATAL(state->toString("Updated State"));
  EXPECT_TRUE(true);
}