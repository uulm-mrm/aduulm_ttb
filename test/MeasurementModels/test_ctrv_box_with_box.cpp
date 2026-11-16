#include "gtest/gtest.h"
#include <tracking_lib/MeasurementModels/BaseMeasurementModel.h>
#include <tracking_lib/TTBManager/TTBManager.h>
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

    mean = ttb::Vector<double>::Random(7);
    cov = ttb::Matrix<double>::Random(7, 7);
    ttb::angles::normalizeAngle(mean(6));
    cov *= cov.transpose();
    ttb::State meas = ttb::create_meas_state(&manager,
                                             ttb::GaussianDistribution(mean, cov),
                                             ttb::Time::max(),
                                             ttb::Components({ ttb::COMPONENT::POS_X,
                                                               ttb::COMPONENT::POS_Y,
                                                               ttb::COMPONENT::POS_Z,
                                                               ttb::COMPONENT::LENGTH,
                                                               ttb::COMPONENT::WIDTH,
                                                               ttb::COMPONENT::HEIGHT,
                                                               ttb::COMPONENT::ROT_Z }))
                          .value();

    container._measurements.push_back(std::move(meas));

    auto t = state.value().innovate(container);
    EXPECT_TRUE(true);
  }
}
