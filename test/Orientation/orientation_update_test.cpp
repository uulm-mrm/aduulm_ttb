#include "gtest/gtest.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include <tracking_lib/MeasurementModels/BaseMeasurementModel.h>
#include <tracking_lib/StateModels/BaseStateModel.h>
#include <tracking_lib/TTBManager/TTBManager.h>
#include <aduulm_logger/aduulm_logger.hpp>
#include <filesystem>
#include <random>
#include <figcone/configreader.h>

using namespace std::literals;
using namespace ttb;
std::filesystem::path const file{ "test/Orientation/ctrv_box_innovate.yaml" };

TEST(OrientationUpdate, WestwardYawUpdate)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Warn);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYamlFile<ttb::Params>(file);
  ttb::TTBManager<double> _manager(params);

  ttb::Vector<double> mean = ttb::Vector<double>::Zero(ttb::ALL_COMPONENTS.size());
  ttb::Matrix<double> cov = ttb::Matrix<double>::Identity(ttb::ALL_COMPONENTS.size(), ttb::ALL_COMPONENTS.size());
  cov *= cov.transpose();

  State<double> meas = create_meas_state<double>(&_manager,
                                 { { 1.0, ttb::GaussianDistribution(mean, cov) } },
                                 ttb::Time::max(),
                                 ttb::Components(ttb::ALL_COMPONENTS))
                   .value();

  std::optional<ttb::State<double>> state = _manager.getMeasModel(ttb::SourceId{ "0" }).create_state(meas);

  ASSERT_TRUE(state.has_value()) << "Can not create State of Meas with all possible Components";
  int randomVelocity = 10;
  for (StateModelId modelId : _manager.params()->state.multi_model.use_state_models)
  {
    StateModel<double> const& sm = _manager.getStateModelMap().at(modelId);
    if (modelId == StateModelId{ 4 })
    {
      Index vel_abs_idx = sm.state_comps().indexOf(COMPONENT::VEL_ABS).value();
      Index rot_z_idx = sm.state_comps().indexOf(COMPONENT::ROT_Z).value();
      ASSERT_GE(vel_abs_idx, 0);
      ASSERT_GE(rot_z_idx, 0);
      assert(state->_state_dist.at(StateModelId{ 4 }).second.to_gaussian_components().size() == 1);
      mean = state->_state_dist.at(modelId).second.to_gaussian_components().front().second.mean();
      mean(vel_abs_idx) = randomVelocity;
      mean(rot_z_idx) = M_PI;  // 0 M_PI;
      mean(sm.state_comps().indexOf(COMPONENT::LENGTH).value()) = 4;
      mean(sm.state_comps().indexOf(COMPONENT::WIDTH).value()) = 2;
      mean(sm.state_comps().indexOf(COMPONENT::HEIGHT).value()) = 1;
      assert(state->_state_dist.at(StateModelId{ 4 }).second.to_gaussian_components().size() == 1);
      state->_state_dist.at(modelId).second.to_gaussian_components().front().second.mean() = mean;
    }
  }

  double lastYaw = M_PI;  // 0 M_PI;

  auto const& measModel = _manager.getMeasModel(SourceId{ "0" });
  auto measModelComps = measModel.meas_model_comps();
  Index ind_z_rot = measModelComps.indexOf(COMPONENT::ROT_Z).value();
  Vector<double> m = Vector<double>::Zero(5);
  Matrix<double> c = Matrix<double>::Identity(5, 5);

  std::vector<double> diffs;
  for (int i = 0; i < 100; i++)
  {
    double newYaw = (rand() % 10) / 50.0;
    double sign = rand() % 2;
    double sign2 = rand() % 2;
    if (sign && sign2)
    {
      newYaw = -M_PI - newYaw;
    }
    else if ((!sign && sign2))
    {
      newYaw = -M_PI + newYaw;
    }
    else if ((sign && !sign2))
    {
      newYaw = M_PI - newYaw;
    }
    else
    {
      newYaw = M_PI + newYaw;
    }
    state->predict(Duration{ std::chrono::milliseconds{ 1000 } },
                   EgoMotionDistribution{ EgoMotionDistribution::zero() });
    m = Vector<double>::Zero(5);
    c = Matrix<double>::Identity(5, 5);

    m(0) = -10 * (i + 1);  // -3.3 * (index+1);

    m(ind_z_rot) = newYaw;
    m(3) = 4;
    m(4) = 2;

    c(ind_z_rot, ind_z_rot) = 0.1;

    DataSourceContainer<double> cont;
    cont._source_id = SourceId{ "0" };
    cont._time = ttb::Time::max();

    State mbox = ttb::create_meas_state(
                     &_manager,
                     { { 1.0, GaussianDistribution(m, c) } },
                     ttb::Time::max(),
                     ttb::Components(
                         { COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::ROT_Z, COMPONENT::LENGTH, COMPONENT::WIDTH }))
                     .value();
    MeasurementId measId = mbox._id;

    cont._measurements.push_back(std::move(mbox));

    // Innovate state
    auto const [state_detection, meas_updates, clutter_info] = state->innovate(cont);
    ASSERT_FALSE(meas_updates.empty());
    std::optional<ttb::State<double>> next_state;

    for (auto& [old_state_id, meas_id, updated_state, meas_log_likeli] : meas_updates)
    {
      if (meas_id == measId)
      {
        next_state = updated_state;
      }
    }

    if (next_state.has_value())
    {
      state = std::move(next_state);
    }
    else
    {
      assert(false);
    }

    auto res = state->getEstimate();

    // extract new yaw
    assert(state->_state_dist.at(StateModelId{ 4 }).second.to_gaussian_components().size() == 1);
    mean = state->_state_dist.at(StateModelId{ 4 }).second.to_gaussian_components().front().second.mean();
    double yaw;

    yaw = mean(_manager.getStateModel(StateModelId{ 4 }).state_comps().indexOf(COMPONENT::ROT_Z).value());
    EXPECT_TRUE(std::isfinite(yaw));

    double diff = yaw - lastYaw;
    while (diff > M_PI)
      diff -= 2 * M_PI;
    while (diff < -M_PI)
      diff += 2 * M_PI;

    diffs.emplace_back(diff);

    EXPECT_LT(std::abs(diff), M_PI / 8.0)
        << "Yaw jump detected when transitioning near West! " << " prev=" << lastYaw << " new=" << yaw;
  }
}
