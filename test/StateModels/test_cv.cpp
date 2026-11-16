#include "gtest/gtest.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include <aduulm_logger/aduulm_logger.hpp>
#include <filesystem>
#include <random>
#include <figcone/configreader.h>

using namespace std::literals;
std::filesystem::path const file{ "test/StateModels/cv.yaml" };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(Prediction, cv)
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
    state.value().predict(ttb::Duration{ std::chrono::milliseconds(100) }, ttb::EgoMotionDistribution::zero());
    EXPECT_TRUE(true);
  }
}
