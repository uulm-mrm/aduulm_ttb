#include "gtest/gtest.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/MultiObjectStateDistributions/Utils.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include <aduulm_logger/aduulm_logger.hpp>
#include <filesystem>
#include <figcone/configreader.h>

using namespace std::literals;
std::filesystem::path const file{ "test/StateModels/ctrv.yaml" };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(Prediction, ctrv_utils)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager<double> manager(params);
  ASSERT_TRUE(manager.state_comps(ttb::StateModelId{ 0 }) == ttb::CTRV_state_comps);
  ASSERT_TRUE(manager.noise_comps(ttb::StateModelId{ 0 }) == ttb::CTRV_noise_comps);
  EXPECT_TRUE(true) << "Invalid config";
  std::vector<ttb::StateId> state_ids;
  for (std::size_t i = 0; i < 10000; ++i)
  {
    ttb::Vector<double> mean = ttb::Vector<double>::Random(ttb::ALL_COMPONENTS.size());
    ttb::Matrix<double> cov = ttb::Matrix<double>::Random(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
    cov *= cov.transpose();

    std::optional<ttb::State<double>> state = manager.getMeasModel(ttb::SourceId{ "0" })
                                          .create_state(ttb::create_meas_state(&manager,
                                                                              ttb::GaussianDistribution(mean, cov),
                                                                              ttb::Time::max(),
                                                                              ttb::Components(ttb::ALL_COMPONENTS))
                                                           .value());
    ASSERT_TRUE(state.has_value()) << "Can not create State of Meas with all possible Components";
    state_ids.push_back(state.value()._id);
    manager.database().states.insert(state.value());
    EXPECT_TRUE(true);
  }
  auto t = ttb::utils::predict_db(&manager, state_ids, 100ms, ttb::EgoMotionDistribution::zero());
  EXPECT_TRUE(t.size() > 0);
}
