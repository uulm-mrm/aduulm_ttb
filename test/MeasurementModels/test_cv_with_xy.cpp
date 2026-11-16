#include "gtest/gtest.h"
#include <tracking_lib/MeasurementModels/BaseMeasurementModel.h>
#include <tracking_lib/TTBManager/TTBManager.h>
#include <tracking_lib/MultiObjectStateDistributions/Utils.h>
#include <aduulm_logger/aduulm_logger.hpp>
#include <filesystem>
#include <random>
#include <figcone/configreader.h>

using namespace std::literals;
std::filesystem::path const file{ "test/MeasurementModels/cv_update_xy.yaml" };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(GenericBoxModel, cv_update_xy)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager<double> manager(params);
  EXPECT_TRUE(true) << "Invalid config";
  for (std::size_t i = 0; i < 100; ++i)
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
    ttb::DataSourceContainer<double> container{ ._source_id = ttb::SourceId{ "0" }, ._time = ttb::Time::max() };
    mean = ttb::Vector<double>::Random(2);
    cov = ttb::Matrix<double>::Random(2, 2);
    cov *= cov.transpose();
    ttb::State<double> meas = ttb::create_meas_state(&manager,
                                                     ttb::GaussianDistribution(mean, cov),
                                                     ttb::Time::max(),
                                                     ttb::Components({
                                                         ttb::COMPONENT::POS_X,
                                                         ttb::COMPONENT::POS_Y,
                                                     }))
                                  .value();
    container._measurements.push_back(std::move(meas));
    auto t = state.value().innovate(container);
    EXPECT_TRUE(true);
  }
}

TEST(GenericBoxModel, parallel_update)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager<double> manager(params);
  EXPECT_TRUE(true) << "Invalid config";

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
  ttb::DataSourceContainer<double> container{ ._source_id = ttb::SourceId{ "0" }, ._time = ttb::Time::max() };

  mean = ttb::Vector<double>::Random(2);
  cov = ttb::Matrix<double>::Random(2, 2);
  cov *= cov.transpose();
  ttb::State meas = ttb::create_meas_state(&manager,
                                           ttb::GaussianDistribution(mean, cov),
                                           ttb::Time::max(),
                                           ttb::Components({
                                               ttb::COMPONENT::POS_X,
                                               ttb::COMPONENT::POS_Y,
                                           }))
                        .value();
  for (auto i : std::views::iota(1, 100))
  {
    container._measurements.push_back(meas);
  }
  ttb::utils::innovate_db(&manager, std::vector{ state.value()._id }, container);
  EXPECT_TRUE(true);
}
