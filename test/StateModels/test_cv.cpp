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
std::filesystem::path const file{ "test/StateModels/cv.yaml" };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(Prediction, cv)
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

    std::optional<ttb::State> state =
        manager.getMeasModelMap()
            .at(ttb::MeasModelId{ "0" })
            ->createState(ttb::Measurement(std::make_unique<ttb::GaussianDistribution>(mean, cov),
                                           ttb::Time::max(),
                                           ttb::Components(ttb::ALL_COMPONENTS)));
    ASSERT_TRUE(state.has_value()) << "Can not create State of Meas with all possible Components";
    ttb::MeasurementContainer container{ ._id = ttb::MeasModelId{ "0" }, ._time = ttb::Time::max() };

    state.value().predict(ttb::Duration{ std::chrono::milliseconds(100) }, ttb::EgoMotionDistribution::zero());
    EXPECT_TRUE(true);
  }
}