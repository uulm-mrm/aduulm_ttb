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
#include <tracking_lib/Misc/AngleNormalization.h>

using namespace std::literals;
std::filesystem::path const file{ "test/MeasurementModels/ctrv_box_update_box.yaml" };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(GenericBoxModel, ctrv_box_update_box)
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

    mean = ttb::Vector::Random(7);
    cov = ttb::Matrix::Random(7, 7);
    ttb::angles::normalizeAngle(mean(6));
    cov *= cov.transpose();
    ttb::Measurement meas(std::make_unique<ttb::GaussianDistribution>(mean, cov),
                          ttb::Time::max(),
                          ttb::Components({ ttb::COMPONENT::POS_X,
                                            ttb::COMPONENT::POS_Y,
                                            ttb::COMPONENT::POS_Z,
                                            ttb::COMPONENT::LENGTH,
                                            ttb::COMPONENT::WIDTH,
                                            ttb::COMPONENT::HEIGHT,
                                            ttb::COMPONENT::ROT_Z }));
    meas._ref_point_measured = true;

    container._data.push_back(std::move(meas));

    state.value().innovate(container);
    EXPECT_TRUE(true);
  }
}