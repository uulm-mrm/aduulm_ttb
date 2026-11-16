#include "gtest/gtest.h"
#include <tracking_lib/MeasurementModels/BaseMeasurementModel.h>
#include <tracking_lib/MultiObjectStateDistributions/Utils.h>
#include <tracking_lib/TTBManager/TTBManager.h>
#include <aduulm_logger/aduulm_logger.hpp>
#include <filesystem>
#include <random>
#include <figcone/configreader.h>

using namespace std::literals;
std::filesystem::path const file{ "test/configs/tracking.yaml" };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(GenericBoxModel, createState)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager<double> manager(params);
  EXPECT_TRUE(true) << "Invalid config";

  ttb::Vector<double> mean = ttb::Vector<double>::Random(ttb::ALL_COMPONENTS.size());
  ttb::Matrix<double> cov = ttb::Matrix<double>::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
  cov *= cov.transpose();
  ttb::State<double> meas =
      ttb::create_meas_state(
          &manager, ttb::GaussianDistribution(mean, cov), ttb::Time::max(), ttb::Components(ttb::ALL_COMPONENTS))
          .value();
  std::optional<ttb::State<double>> state = manager.getMeasModel(ttb::SourceId{ "0" }).create_state(meas);
  ASSERT_TRUE(state.has_value()) << "Can not create State of Meas with all possible Components";
}

TEST(GenericBoxModel, innovate_multiple_states_one_meas_container_with_utils)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager<double> manager(params);
  EXPECT_TRUE(true) << "Invalid config";
  std::random_device rd;
  std::mt19937 gen(rd());
  ttb::DataSourceContainer<double> container{ ._source_id = ttb::SourceId{ "0" }, ._time = ttb::Time::max() };
  for (std::size_t numMeas = 0; numMeas < 1000; ++numMeas)
  {
    auto compsMeas = ttb::ALL_COMPONENTS;
    std::shuffle(compsMeas.begin(), compsMeas.end(), gen);
    std::size_t numCompsMeasured = std::uniform_int_distribution<>(1, compsMeas.size() - 1)(gen);
    ttb::Vector<double> mean = ttb::Vector<double>::Random(numCompsMeasured);
    ttb::Matrix<double> cov = ttb::Matrix<double>::Random(numCompsMeasured, numCompsMeasured);
    cov *= cov.transpose();
    ttb::State<double> meas =
        ttb::create_meas_state(&manager,
                               ttb::GaussianDistribution(mean, cov),
                               ttb::Time::max(),
                               ttb::Components(std::vector(compsMeas.begin(), compsMeas.begin() + numCompsMeasured)))
            .value();
    container._measurements.push_back(std::move(meas));
  }
  std::vector<ttb::StateId> state_ids;
  for (std::size_t i = 0; i < 1000; ++i)
  {
    ttb::Vector<double> mean = ttb::Vector<double>::Random(ttb::ALL_COMPONENTS.size());
    ttb::Matrix<double> cov = ttb::Matrix<double>::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();

    std::optional<ttb::State<double>> state =
        manager.getMeasModel(ttb::SourceId{ "0" })
            .create_state(ttb::create_meas_state(&manager,
                                                 ttb::GaussianDistribution(mean, cov),
                                                 ttb::Time::max(),
                                                 ttb::Components(ttb::ALL_COMPONENTS))
                              .value());
    ASSERT_TRUE(state.has_value()) << "Can not create State of Meas with all possible Components";
    manager.database().states.insert(state.value());
    state_ids.push_back(state.value()._id);
  }
  ttb::utils::innovate_db(&manager, state_ids, container);
  ASSERT_TRUE(true);
}
