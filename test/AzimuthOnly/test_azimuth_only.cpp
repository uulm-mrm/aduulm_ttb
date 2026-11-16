#include "gtest/gtest.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include <tracking_lib/TTBManager/TTBManager.h>
#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
#include "tracking_lib/MultiObjectStateDistributions/Utils.h"

#include <aduulm_logger/aduulm_logger.hpp>
#include <filesystem>
#include <figcone/configreader.h>

using namespace std::literals;
std::string const params_string{ "thread_pool_size: 0\n"
                                 "version: 1.0.0\n"
                                 "meas_models:\n"
                                 "  gaussian_models:\n"
                                 "    -\n"
                                 "        gating_prob: 0.99\n"
                                 "        id: meas_model_0\n"
                                 "        components: [AZIMUTH]\n"
                                 "        can_init: true\n"
                                 "        clutter:\n"
                                 "          intensity: 1.11408460164\n"
                                 "        detection:\n"
                                 "          prob: 0.67\n"
                                 "          prob_min: 0.001\n"
                                 "        occlusion:\n"
                                 "          model: NO_OCCLUSION\n"
                                 "        default_values:\n"
                                 "          - type: [ NOT_CLASSIFIED ]\n"
                                 "            mean:\n"
                                 "              AZIMUTH: 0\n"
                                 "            var:\n"
                                 "              AZIMUTH: 0.00030462\n"
                                 "    -\n"
                                 "        id: 1\n"
                                 "        components: [AZIMUTH]\n"
                                 "        can_init: true\n"
                                 "        clutter:\n"
                                 "          intensity: 1.11408460164\n"
                                 "        detection:\n"
                                 "          prob: 0.67\n"
                                 "          prob_min: 0.001\n"
                                 "        occlusion:\n"
                                 "          model: NO_OCCLUSION\n"
                                 "        default_values:\n"
                                 "          - type: [ NOT_CLASSIFIED ]\n"
                                 "            mean:\n"
                                 "              AZIMUTH: 0\n"
                                 "            var:\n"
                                 "              AZIMUTH: 0.00030462\n"
                                 "        gating_prob: 0.99\n"
                                 "state:\n"
                                 "  multi_model:\n"
                                 "    use_state_models: [1]\n"
                                 "    birth_weights:    [1]\n"
                                 "    markov_transition:\n"
                                 "      - type: [ NOT_CLASSIFIED ]\n"
                                 "        transition_matrix: \"1\"\n"
                                 "  classification:\n"
                                 "    use_meas_models: [0]\n"
                                 "    classes: [NOT_CLASSIFIED]\n"
                                 "    use_union_classes: true\n"
                                 "    discount_factor_prediction: 0.98\n"
                                 "  estimation:\n"
                                 "    transform_output_state: false\n"
                                 "    output_state_model: 1\n"
                                 "    type: BEST_STATE_MODEL\n"
                                 "state_models:\n"
                                 "    -\n"
                                 "      id: 1\n"
                                 "      type: CV\n"
                                 "      extent: NONE\n"
                                 "      distribution:\n"
                                 "        type: GAUSSIAN\n"
                                 "        mixture: true\n"
                                 "        extraction_type: BEST_COMPONENT\n"
                                 "        post_process:\n"
                                 "          enable: true\n"
                                 "          max_components: 10\n"
                                 "          merging_distance: 0.0000001\n"
                                 "          min_weight: 0.001\n"
                                 "          max_variance: 1000\n"
                                 "      model_noise_std_dev:\n"
                                 "        ACC_X: 0.2\n"
                                 "        ACC_Y: 0.2\n"
                                 "        VEL_Z: 0.001\n"
                                 "        WIDTH_CHANGE: 0.001\n"
                                 "        LENGTH_CHANGE: 0.001\n"
                                 "        HEIGHT_CHANGE: 0.001\n"
                                 "birth_model:\n"
                                 "  type: DYNAMIC\n"
                                 "  allow_overlapping: true\n"
                                 "  min_mhd_4_overlapping_wo_extent: 1\n"
                                 "  default_birth_existence_prob: 0.05\n"
                                 "  dynamic_model: \n"
                                 "    mean_num_birth: 2.\n"
                                 "    birth_threshold: 0.1\n"
                                 "persistence_model:\n"
                                 "  type: CONSTANT\n"
                                 "  constant:\n"
                                 "    persistence_prob: 0.99\n"
                                 "lmb_distribution:\n"
                                 "  update_method: LBP\n"
                                 "  post_process_prediction:\n"
                                 "    enable: false\n"
                                 "    max_tracks: 1000000\n"
                                 "    pruning_threshold: 0\n"
                                 "    max_last_assoc_duration_ms: 50\n"
                                 "  post_process_update:\n"
                                 "    enable: true\n"
                                 "    max_tracks: 1000000000\n"
                                 "    pruning_threshold: 1e-8\n"
                                 "  extraction:\n"
                                 "    type: EXISTENCE_PROBABILITY  #Cardinality\n"
                                 "    threshold: 0.5\n"
                                 "filter:\n"
                                 "  enable: true\n"
                                 "  type: LMB_IC\n" };

constexpr double TOL{ 1e-5 };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(Azimuth_Only, SensorsMultipleMeasurements)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYaml<ttb::Params>(params_string);
  auto manager = ttb::TTBManager<double>(params);
  ttb::SourceId measModelId{ "meas_model_0" };
  ttb::SourceId measModelId2{ "1" };
  ttb::StateModelId stateModelId{ 1 };

  // Create Measurements for both sensors
  // sensor rot and pos
  ttb::Vector<double> pos1(3);
  pos1(0) = 600.0;
  pos1(1) = 600.0;
  pos1(2) = -0;
  ttb::Matrix33 rot1;
  rot1 = ttb::Matrix33::Identity(3, 3);
  LOG_ERR("pos1: " << pos1);
  LOG_ERR("rot1:" << rot1);
  ttb::SE3Trafo sensor_pose1 = Eigen::Translation3d(pos1) * Eigen::Quaterniond(rot1);

  ttb::Vector<double> pos2(3);
  pos2(0) = 600.0;
  pos2(1) = -600.0;
  pos2(2) = -0.0;
  ttb::Matrix33 rot2;
  rot2 = ttb::Matrix33::Identity(3, 3);
  rot2(0, 0) = -1.0;
  rot2(1, 1) = -1.0;
  LOG_ERR("pos2: " << pos2);
  LOG_ERR("rot2:" << rot2);
  ttb::SE3Trafo sensor_pose2 = Eigen::Translation3d(pos2) * Eigen::Quaterniond(rot2);

  auto egoDist = ttb::EgoMotionDistribution::zero();

  ttb::DataSourceContainer<double> measContainer{ ._source_id = measModelId, ._time = ttb::Time(0s) };

  ttb::Vector<double> meas_mean = ttb::Vector<double>::Zero(1);
  meas_mean(0) = 0.209144;

  ttb::Matrix<double> meas_cov = ttb::Matrix<double>::Identity(1, 1);
  meas_cov(0, 0) = 0.00030462;
  ttb::State meas1 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean, meas_cov, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas: " + meas1.toString());
  measContainer._measurements.push_back(std::move(meas1));

  ttb::Vector<double> meas_mean2 = ttb::Vector<double>::Zero(1);
  meas_mean2(0) = 1.35088;

  ttb::Matrix<double> meas_cov2 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov2(0, 0) = 0.00030462;
  ttb::State meas2 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean2, meas_cov2, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas2: " + meas2.toString());
  measContainer._measurements.push_back(std::move(meas2));

  ttb::Vector<double> meas_mean3 = ttb::Vector<double>::Zero(1);
  meas_mean3(0) = 2.15773;

  ttb::Matrix<double> meas_cov3 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov3(0, 0) = 0.00030462;
  ttb::State meas3 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean3, meas_cov3, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas3: " + meas3.toString());
  measContainer._measurements.push_back(std::move(meas3));

  ttb::Vector<double> meas_mean4 = ttb::Vector<double>::Zero(1);
  meas_mean4(0) = 2.54272;

  ttb::Matrix<double> meas_cov4 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov4(0, 0) = 0.00030462;
  ttb::State meas4 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean4, meas_cov4, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas4: " + meas4.toString());
  measContainer._measurements.push_back(std::move(meas4));

  ttb::Vector<double> meas_mean5 = ttb::Vector<double>::Zero(1);
  meas_mean5(0) = -2.10669;

  ttb::Matrix<double> meas_cov5 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov5(0, 0) = 0.00030462;
  ttb::State meas5 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean5, meas_cov5, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas5: " + meas5.toString());
  measContainer._measurements.push_back(std::move(meas5));

  ttb::Vector<double> meas_mean6 = ttb::Vector<double>::Zero(1);
  meas_mean6(0) = -1.91881;

  ttb::Matrix<double> meas_cov6 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov6(0, 0) = 0.00030462;
  ttb::State meas6 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean6, meas_cov6, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas6: " + meas6.toString());
  measContainer._measurements.push_back(std::move(meas6));

  ttb::Vector<double> meas_mean7 = ttb::Vector<double>::Zero(1);
  meas_mean7(0) = -0.856559;

  ttb::Matrix<double> meas_cov7 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov7(0, 0) = 0.00030462;
  ttb::State meas7 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean7, meas_cov7, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas7: " + meas7.toString());
  measContainer._measurements.push_back(std::move(meas7));

  ttb::Vector<double> meas_mean8 = ttb::Vector<double>::Zero(1);
  meas_mean8(0) = -0.648161;
  ttb::Matrix<double> meas_cov8 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov8(0, 0) = 0.00030462;
  ttb::State meas8 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean8, meas_cov8, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas8: " + meas8.toString());
  measContainer._measurements.push_back(std::move(meas8));

  ttb::Vector<double> meas_mean9 = ttb::Vector<double>::Zero(1);
  meas_mean9(0) = -2.20981;
  ttb::Matrix<double> meas_cov9 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov9(0, 0) = 0.00030462;
  ttb::State meas9 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean9, meas_cov9, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas9: " + meas9.toString());
  measContainer._measurements.push_back(std::move(meas9));
  measContainer._sensorInfo._to_sensor_cs = sensor_pose1;

  auto egoDist2 = ttb::EgoMotionDistribution::zero();

  LOG_WARN(measContainer.toString());

  ttb::DataSourceContainer<double> measContainer2{ ._source_id = measModelId2, ._time = ttb::Time(0s) };

  ttb::Vector<double> meas_mean11 = ttb::Vector<double>::Zero(1);
  meas_mean11(0) = -0.514243;

  ttb::Matrix<double> meas_cov11 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov11(0, 0) = 0.00030462;
  ttb::State meas11 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean11, meas_cov11, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas11: " + meas11.toString());
  measContainer2._measurements.push_back(std::move(meas11));

  ttb::Vector<double> meas_mean21 = ttb::Vector<double>::Zero(1);
  meas_mean21(0) = -0.778418;

  ttb::Matrix<double> meas_cov21 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov21(0, 0) = 0.00030462;
  ttb::State meas21 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean21, meas_cov21, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas21: " + meas21.toString());
  measContainer2._measurements.push_back(std::move(meas21));

  ttb::Vector<double> meas_mean31 = ttb::Vector<double>::Zero(1);
  meas_mean31(0) = -0.318594;

  ttb::Matrix<double> meas_cov31 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov31(0, 0) = 0.00030462;
  ttb::State meas31 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean31, meas_cov31, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas31: " + meas31.toString());
  measContainer2._measurements.push_back(std::move(meas31));

  ttb::Vector<double> meas_mean41 = ttb::Vector<double>::Zero(1);
  meas_mean41(0) = -1.87417;

  ttb::Matrix<double> meas_cov41 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov41(0, 0) = 0.00030462;
  ttb::State meas41 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean41, meas_cov41, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas41: " + meas41.toString());
  measContainer2._measurements.push_back(std::move(meas41));

  ttb::Vector<double> meas_mean51 = ttb::Vector<double>::Zero(1);
  meas_mean51(0) = -0.610508;

  ttb::Matrix<double> meas_cov51 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov51(0, 0) = 0.00030462;
  ttb::State meas51 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean51, meas_cov51, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas51: " + meas51.toString());
  measContainer2._measurements.push_back(std::move(meas51));

  ttb::Vector<double> meas_mean61 = ttb::Vector<double>::Zero(1);
  meas_mean61(0) = -1.73718;

  ttb::Matrix<double> meas_cov61 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov61(0, 0) = 0.00030462;
  ttb::State meas61 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean61, meas_cov61, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas61: " + meas61.toString());
  measContainer2._measurements.push_back(std::move(meas61));

  ttb::Vector<double> meas_mean71 = ttb::Vector<double>::Zero(1);
  meas_mean71(0) = -1.62158;

  ttb::Matrix<double> meas_cov71 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov71(0, 0) = 0.00030462;
  ttb::State meas71 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean71, meas_cov71, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas71: " + meas71.toString());
  measContainer2._measurements.push_back(std::move(meas71));

  ttb::Vector<double> meas_mean81 = ttb::Vector<double>::Zero(1);
  meas_mean81(0) = 1.54007;

  ttb::Matrix<double> meas_cov81 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov81(0, 0) = 0.00030462;
  ttb::State meas81 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean81, meas_cov81, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas81: " + meas81.toString());
  measContainer2._measurements.push_back(std::move(meas81));

  ttb::Vector<double> meas_mean91 = ttb::Vector<double>::Zero(1);
  meas_mean91(0) = 1.41364;

  ttb::Matrix<double> meas_cov91 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov91(0, 0) = 0.00030462;
  ttb::State meas91 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean91, meas_cov91, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas91: " + meas91.toString());
  measContainer2._measurements.push_back(std::move(meas91));

  ttb::Vector<double> meas_mean101 = ttb::Vector<double>::Zero(1);
  meas_mean101(0) = -3.14197;

  ttb::Matrix<double> meas_cov101 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov101(0, 0) = 0.00030462;
  ttb::State meas101 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean101, meas_cov101, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas101: " + meas101.toString());
  measContainer2._measurements.push_back(std::move(meas101));

  ttb::Vector<double> meas_mean111 = ttb::Vector<double>::Zero(1);
  meas_mean111(0) = 0.906631;

  ttb::Matrix<double> meas_cov111 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov111(0, 0) = 0.00030462;
  ttb::State meas111 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean111, meas_cov111, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas111: " + meas111.toString());
  measContainer2._measurements.push_back(std::move(meas111));

  ttb::Vector<double> meas_mean121 = ttb::Vector<double>::Zero(1);
  meas_mean121(0) = -0.960077;

  ttb::Matrix<double> meas_cov121 = ttb::Matrix<double>::Identity(1, 1);
  meas_cov121(0, 0) = 0.00030462;
  ttb::State meas121 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean121, meas_cov121, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::AZIMUTH }))
          .value();
  LOG_DEB("Meas121: " + meas121.toString());
  measContainer2._measurements.push_back(std::move(meas121));
  measContainer2._sensorInfo._to_sensor_cs = sensor_pose2;

  LOG_WARN(measContainer2.toString());

  // create 6 tracks
  ttb::StateModel<double> const& state_model = manager.getStateModel(stateModelId);
  // track1

  ttb::Vector<double> x = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x(0) = -400;   // x
  x(1) = 400;    // y
  x(2) = 0.001;  // z
  x(3) = 0;      // vx
  x(4) = 0;      // vy

  ttb::Matrix<double> P =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P(0, 0) = 225;    // x,x
  P(1, 1) = 225;    // y,y
  P(2, 2) = 0.001;  // z,z
  P(3, 3) = 25;     // vx, vx
  P(4, 4) = 25;     // vy, vy

  ttb::BernoulliState state1 = {
    ttb::State(&manager, { { stateModelId, { { 1.0, ttb::GaussianDistribution(x, P) } } } }), 0.05
  };
  state1.state._label = ttb::Label{ 1 };

  ttb::Vector<double> x2 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x2(0) = -400;  // x
  x2(1) = 0;     // y
  x2(2) = 0;     // z
  x2(3) = 0;     // vx
  x2(4) = 0;     // vy

  ttb::Matrix<double> P2 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P2(0, 0) = 225;    // x,x
  P2(1, 1) = 225;    // y,y
  P2(2, 2) = 0.001;  // z,z
  P2(3, 3) = 25;     // vx, vx
  P2(4, 4) = 25;     // vy, vy

  ttb::BernoulliState state2 = {
    ttb::State(&manager, { { stateModelId, { { 1.0, ttb::GaussianDistribution(x2, P2) } } } }), 0.05
  };
  state2.state._label = ttb::Label{ 2 };

  // track3

  ttb::Vector<double> x3 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x3(0) = -400.0;  // x
  x3(1) = -400.0;  // y
  x3(2) = 0;       // z
  x3(3) = 0;       // vx
  x3(4) = 0;       // vy

  ttb::Matrix<double> P3 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P3(0, 0) = 225;    // x,x
  P3(1, 1) = 225;    // y,y
  P3(2, 2) = 0.001;  // z,z
  P3(3, 3) = 25;     // vx, vx
  P3(4, 4) = 25;     // vy, vy

  ttb::BernoulliState state3 = {
    ttb::State(&manager, { { stateModelId, { { 1.0, ttb::GaussianDistribution(x3, P3) } } } }), 0.05
  };
  state3.state._label = ttb::Label{ 3 };

  // track4

  ttb::Vector<double> x4 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x4(0) = 400.0;   // x
  x4(1) = -400.0;  // y
  x4(2) = 0;       // z
  x4(3) = 0;       // vx
  x4(4) = 0;       // vy

  ttb::Matrix<double> P4 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P4(0, 0) = 225;    // x,x
  P4(1, 1) = 225;    // y,y
  P4(2, 2) = 0.001;  // z,z
  P4(3, 3) = 25;     // vx, vx
  P4(4, 4) = 25;     // vy, vy

  ttb::BernoulliState state4 = {
    ttb::State(&manager, { { stateModelId, { { 1.0, ttb::GaussianDistribution(x4, P4) } } } }), 0.05
  };
  state4.state._label = ttb::Label{ 4 };

  // track5

  ttb::Vector<double> x5 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x5(0) = 400.0;  // x
  x5(1) = 0;      // y
  x5(2) = 0;      // z
  x5(3) = 0;      // vx
  x5(4) = 0;      // vy

  ttb::Matrix<double> P5 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P5(0, 0) = 225;    // x,x
  P5(1, 1) = 225;    // y,y
  P5(2, 2) = 0.001;  // z,z
  P5(3, 3) = 25;     // vx, vx
  P5(4, 4) = 25;     // vy, vy

  ttb::BernoulliState state5 = {
    ttb::State(&manager, { { stateModelId, { { 1.0, ttb::GaussianDistribution(x5, P5) } } } }), 0.05
  };

  state5.state._label = ttb::Label{ 5 };

  // track6

  ttb::Vector<double> x6 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x6(0) = 400.0;  // x
  x6(1) = 400;    // y
  x6(2) = 0;      // z
  x6(3) = 0;      // vx
  x6(4) = 0;      // vy

  ttb::Matrix<double> P6 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P6(0, 0) = 225;    // x,x
  P6(1, 1) = 225;    // y,y
  P6(2, 2) = 0.001;  // z,z
  P6(3, 3) = 25;     // vx, vx
  P6(4, 4) = 25;     // vy, vy

  ttb::BernoulliState state6 = {
    ttb::State(&manager, { { stateModelId, { { 1.0, ttb::GaussianDistribution(x6, P6) } } } }), 0.05
  };

  state6.state._label = ttb::Label{ 6 };

  ttb::LMBDistribution lmb(&manager);
  lmb.add_tracks({ state1, state2, state3, state4, state5, state6 });

  lmb.update(measContainer);

  std::map<ttb::Label, double> label2gtExProb;
  label2gtExProb.emplace((ttb::Label)1, 0.26515158913996939);
  label2gtExProb.emplace((ttb::Label)2, 0.017071908949818925);
  label2gtExProb.emplace((ttb::Label)3, 0.017071908949818925);
  label2gtExProb.emplace((ttb::Label)4, 0.33637527963697128);
  label2gtExProb.emplace((ttb::Label)5, 0.017071908949818925);
  label2gtExProb.emplace((ttb::Label)6, 0.017071908949818925);

  for (ttb::StateId id : lmb._tracks)
  {
    EXPECT_FLOAT_EQ(manager.get_state_existence_probability(id), label2gtExProb.at(manager.get_state(id)._label));
  }

  lmb.update(measContainer2);

  std::map<ttb::Label, double> label2gtExProb2;
  label2gtExProb2.emplace(ttb::Label{ 1 }, 0.58668579489046457);
  label2gtExProb2.emplace(ttb::Label{ 2 }, 0.09041256659485751);
  label2gtExProb2.emplace(ttb::Label{ 3 }, 0.0056989151651848039);
  label2gtExProb2.emplace(ttb::Label{ 4 }, 0.40349670714244701);
  label2gtExProb2.emplace(ttb::Label{ 5 }, 0.0056989151651848039);
  label2gtExProb2.emplace(ttb::Label{ 6 }, 0.0056989151651848039);

  for (ttb::StateId id : lmb._tracks)
  {
    EXPECT_FLOAT_EQ(manager.get_state_existence_probability(id), label2gtExProb2.at(manager.get_state(id)._label));
  }
}
