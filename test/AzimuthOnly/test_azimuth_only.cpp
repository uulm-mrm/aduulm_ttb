//
// Created by hermann on 2/9/24.
//
#include "gtest/gtest.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include <tracking_lib/Measurements/MeasurementContainer.h>
#include <tracking_lib/Distributions/GaussianDistribution.h>
#include <tracking_lib/Measurements/Measurement.h>

#include <tracking_lib/TTBManager/TTBManager.h>
#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
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
                                 "        id: 0\n"
                                 "        components: [POS_X, POS_Y, LENGTH, WIDTH, ROT_Z]\n"
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
                                 "        components: [POS_X, POS_Y, LENGTH, WIDTH, ROT_Z]\n"
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
                                 "    output_state_model: CTRV\n"
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

TEST(Azimuth_Only, 2SensorsMultipleMeasurements)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYaml<ttb::Params>(params_string);
  auto manager = ttb::TTBManager(params);
  ttb::MeasModelId measModelId{ "0" };
  ttb::MeasModelId measModelId2{ "1" };
  ttb::StateModelId stateModelId{ 1 };

  // Create Measurements for both sensors
  // sensor rot and pos
  ttb::Vector3 pos1(3);
  pos1(0) = 600.0;
  pos1(1) = 600.0;
  pos1(2) = -0;
  ttb::Matrix33 rot1;
  rot1 = ttb::Matrix33::Identity(3, 3);
  LOG_ERR("pos1: " << pos1);
  LOG_ERR("rot1:" << rot1);
  ttb::SE3Trafo sensor_pose1 = Eigen::Translation3d(pos1) * Eigen::Quaterniond(rot1);

  ttb::Vector3 pos2(3);
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

  ttb::MeasurementContainer measContainer{ ._id = measModelId,
                                           ._egoMotion = std::move(egoDist),
                                           ._time = ttb::Time(0s) };

  ttb::Vector meas_mean = ttb::Vector::Zero(1);
  meas_mean(0) = 0.209144;

  ttb::Matrix meas_cov = ttb::Matrix::Identity(1, 1);
  meas_cov(0, 0) = 0.00030462;
  ttb::Measurement meas1(
      std::make_unique<ttb::GaussianDistribution>(meas_mean, meas_cov, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas1._ref_point_measured = true;
  LOG_DEB("Meas: " + meas1.toString());
  measContainer._data.push_back(std::move(meas1));

  ttb::Vector meas_mean2 = ttb::Vector::Zero(1);
  meas_mean2(0) = 1.35088;

  ttb::Matrix meas_cov2 = ttb::Matrix::Identity(1, 1);
  meas_cov2(0, 0) = 0.00030462;
  ttb::Measurement meas2(
      std::make_unique<ttb::GaussianDistribution>(meas_mean2, meas_cov2, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas2._ref_point_measured = true;
  LOG_DEB("Meas2: " + meas2.toString());
  measContainer._data.push_back(std::move(meas2));

  ttb::Vector meas_mean3 = ttb::Vector::Zero(1);
  meas_mean3(0) = 2.15773;

  ttb::Matrix meas_cov3 = ttb::Matrix::Identity(1, 1);
  meas_cov3(0, 0) = 0.00030462;
  ttb::Measurement meas3(
      std::make_unique<ttb::GaussianDistribution>(meas_mean3, meas_cov3, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas3._ref_point_measured = true;
  LOG_DEB("Meas3: " + meas3.toString());
  measContainer._data.push_back(std::move(meas3));

  ttb::Vector meas_mean4 = ttb::Vector::Zero(1);
  meas_mean4(0) = 2.54272;

  ttb::Matrix meas_cov4 = ttb::Matrix::Identity(1, 1);
  meas_cov4(0, 0) = 0.00030462;
  ttb::Measurement meas4(
      std::make_unique<ttb::GaussianDistribution>(meas_mean4, meas_cov4, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas4._ref_point_measured = true;
  LOG_DEB("Meas4: " + meas4.toString());
  measContainer._data.push_back(std::move(meas4));

  ttb::Vector meas_mean5 = ttb::Vector::Zero(1);
  meas_mean5(0) = -2.10669;

  ttb::Matrix meas_cov5 = ttb::Matrix::Identity(1, 1);
  meas_cov5(0, 0) = 0.00030462;
  ttb::Measurement meas5(
      std::make_unique<ttb::GaussianDistribution>(meas_mean5, meas_cov5, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas5._ref_point_measured = true;
  LOG_DEB("Meas5: " + meas5.toString());
  measContainer._data.push_back(std::move(meas5));

  ttb::Vector meas_mean6 = ttb::Vector::Zero(1);
  meas_mean6(0) = -1.91881;

  ttb::Matrix meas_cov6 = ttb::Matrix::Identity(1, 1);
  meas_cov6(0, 0) = 0.00030462;
  ttb::Measurement meas6(
      std::make_unique<ttb::GaussianDistribution>(meas_mean6, meas_cov6, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas6._ref_point_measured = true;
  LOG_DEB("Meas6: " + meas6.toString());
  measContainer._data.push_back(std::move(meas6));

  ttb::Vector meas_mean7 = ttb::Vector::Zero(1);
  meas_mean7(0) = -0.856559;

  ttb::Matrix meas_cov7 = ttb::Matrix::Identity(1, 1);
  meas_cov7(0, 0) = 0.00030462;
  ttb::Measurement meas7(
      std::make_unique<ttb::GaussianDistribution>(meas_mean7, meas_cov7, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas7._ref_point_measured = true;
  LOG_DEB("Meas7: " + meas7.toString());
  measContainer._data.push_back(std::move(meas7));

  ttb::Vector meas_mean8 = ttb::Vector::Zero(1);
  meas_mean8(0) = -0.648161;
  ttb::Matrix meas_cov8 = ttb::Matrix::Identity(1, 1);
  meas_cov8(0, 0) = 0.00030462;
  ttb::Measurement meas8(
      std::make_unique<ttb::GaussianDistribution>(meas_mean8, meas_cov8, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas8._ref_point_measured = true;
  LOG_DEB("Meas8: " + meas8.toString());
  measContainer._data.push_back(std::move(meas8));

  ttb::Vector meas_mean9 = ttb::Vector::Zero(1);
  meas_mean9(0) = -2.20981;
  ttb::Matrix meas_cov9 = ttb::Matrix::Identity(1, 1);
  meas_cov9(0, 0) = 0.00030462;
  ttb::Measurement meas9(
      std::make_unique<ttb::GaussianDistribution>(meas_mean9, meas_cov9, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas9._ref_point_measured = true;
  LOG_DEB("Meas9: " + meas9.toString());
  measContainer._data.push_back(std::move(meas9));
  measContainer._sensorInfo._to_sensor_cs = sensor_pose1;

  auto egoDist2 = ttb::EgoMotionDistribution::zero();

  LOG_WARN(measContainer.toString());

  ttb::MeasurementContainer measContainer2{ ._id = measModelId2,
                                            ._egoMotion = std::move(egoDist2),
                                            ._time = ttb::Time(0s) };

  ttb::Vector meas_mean11 = ttb::Vector::Zero(1);
  meas_mean11(0) = -0.514243;

  ttb::Matrix meas_cov11 = ttb::Matrix::Identity(1, 1);
  meas_cov11(0, 0) = 0.00030462;
  ttb::Measurement meas11(
      std::make_unique<ttb::GaussianDistribution>(meas_mean11, meas_cov11, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas11._ref_point_measured = true;
  LOG_DEB("Meas11: " + meas11.toString());
  measContainer2._data.push_back(std::move(meas11));

  ttb::Vector meas_mean21 = ttb::Vector::Zero(1);
  meas_mean21(0) = -0.778418;

  ttb::Matrix meas_cov21 = ttb::Matrix::Identity(1, 1);
  meas_cov21(0, 0) = 0.00030462;
  ttb::Measurement meas21(
      std::make_unique<ttb::GaussianDistribution>(meas_mean21, meas_cov21, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas21._ref_point_measured = true;
  LOG_DEB("Meas21: " + meas21.toString());
  measContainer2._data.push_back(std::move(meas21));

  ttb::Vector meas_mean31 = ttb::Vector::Zero(1);
  meas_mean31(0) = -0.318594;

  ttb::Matrix meas_cov31 = ttb::Matrix::Identity(1, 1);
  meas_cov31(0, 0) = 0.00030462;
  ttb::Measurement meas31(
      std::make_unique<ttb::GaussianDistribution>(meas_mean31, meas_cov31, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas3._ref_point_measured = true;
  LOG_DEB("Meas31: " + meas31.toString());
  measContainer2._data.push_back(std::move(meas31));

  ttb::Vector meas_mean41 = ttb::Vector::Zero(1);
  meas_mean41(0) = -1.87417;

  ttb::Matrix meas_cov41 = ttb::Matrix::Identity(1, 1);
  meas_cov41(0, 0) = 0.00030462;
  ttb::Measurement meas41(
      std::make_unique<ttb::GaussianDistribution>(meas_mean41, meas_cov41, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas41._ref_point_measured = true;
  LOG_DEB("Meas41: " + meas41.toString());
  measContainer2._data.push_back(std::move(meas41));

  ttb::Vector meas_mean51 = ttb::Vector::Zero(1);
  meas_mean51(0) = -0.610508;

  ttb::Matrix meas_cov51 = ttb::Matrix::Identity(1, 1);
  meas_cov51(0, 0) = 0.00030462;
  ttb::Measurement meas51(
      std::make_unique<ttb::GaussianDistribution>(meas_mean51, meas_cov51, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas51._ref_point_measured = true;
  LOG_DEB("Meas51: " + meas51.toString());
  measContainer2._data.push_back(std::move(meas51));

  ttb::Vector meas_mean61 = ttb::Vector::Zero(1);
  meas_mean61(0) = -1.73718;

  ttb::Matrix meas_cov61 = ttb::Matrix::Identity(1, 1);
  meas_cov61(0, 0) = 0.00030462;
  ttb::Measurement meas61(
      std::make_unique<ttb::GaussianDistribution>(meas_mean61, meas_cov61, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas61._ref_point_measured = true;
  LOG_DEB("Meas61: " + meas61.toString());
  measContainer2._data.push_back(std::move(meas61));

  ttb::Vector meas_mean71 = ttb::Vector::Zero(1);
  meas_mean71(0) = -1.62158;

  ttb::Matrix meas_cov71 = ttb::Matrix::Identity(1, 1);
  meas_cov71(0, 0) = 0.00030462;
  ttb::Measurement meas71(
      std::make_unique<ttb::GaussianDistribution>(meas_mean71, meas_cov71, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas71._ref_point_measured = true;
  LOG_DEB("Meas71: " + meas71.toString());
  measContainer2._data.push_back(std::move(meas71));

  ttb::Vector meas_mean81 = ttb::Vector::Zero(1);
  meas_mean81(0) = 1.54007;

  ttb::Matrix meas_cov81 = ttb::Matrix::Identity(1, 1);
  meas_cov81(0, 0) = 0.00030462;
  ttb::Measurement meas81(
      std::make_unique<ttb::GaussianDistribution>(meas_mean81, meas_cov81, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas81._ref_point_measured = true;
  LOG_DEB("Meas81: " + meas81.toString());
  measContainer2._data.push_back(std::move(meas81));

  ttb::Vector meas_mean91 = ttb::Vector::Zero(1);
  meas_mean91(0) = 1.41364;

  ttb::Matrix meas_cov91 = ttb::Matrix::Identity(1, 1);
  meas_cov91(0, 0) = 0.00030462;
  ttb::Measurement meas91(
      std::make_unique<ttb::GaussianDistribution>(meas_mean91, meas_cov91, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas91._ref_point_measured = true;
  LOG_DEB("Meas91: " + meas91.toString());
  measContainer2._data.push_back(std::move(meas91));

  ttb::Vector meas_mean101 = ttb::Vector::Zero(1);
  meas_mean101(0) = -3.14197;

  ttb::Matrix meas_cov101 = ttb::Matrix::Identity(1, 1);
  meas_cov101(0, 0) = 0.00030462;
  ttb::Measurement meas101(
      std::make_unique<ttb::GaussianDistribution>(meas_mean101, meas_cov101, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas101._ref_point_measured = true;
  LOG_DEB("Meas101: " + meas101.toString());
  measContainer2._data.push_back(std::move(meas101));

  ttb::Vector meas_mean111 = ttb::Vector::Zero(1);
  meas_mean111(0) = 0.906631;

  ttb::Matrix meas_cov111 = ttb::Matrix::Identity(1, 1);
  meas_cov111(0, 0) = 0.00030462;
  ttb::Measurement meas111(
      std::make_unique<ttb::GaussianDistribution>(meas_mean111, meas_cov111, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas111._ref_point_measured = true;
  LOG_DEB("Meas111: " + meas111.toString());
  measContainer2._data.push_back(std::move(meas111));

  ttb::Vector meas_mean121 = ttb::Vector::Zero(1);
  meas_mean121(0) = -0.960077;

  ttb::Matrix meas_cov121 = ttb::Matrix::Identity(1, 1);
  meas_cov121(0, 0) = 0.00030462;
  ttb::Measurement meas121(
      std::make_unique<ttb::GaussianDistribution>(meas_mean121, meas_cov121, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::AZIMUTH }));
  meas121._ref_point_measured = true;
  LOG_DEB("Meas121: " + meas121.toString());
  measContainer2._data.push_back(std::move(meas121));
  measContainer2._sensorInfo._to_sensor_cs = sensor_pose2;

  LOG_WARN(measContainer2.toString());

  // create 6 tracks
  auto const& state_model = *manager.getStateModelMap().at(stateModelId);
  // track1
  ttb::State state1 = manager.createState();

  ttb::Vector x = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x(0) = -400;   // x
  x(1) = 400;    // y
  x(2) = 0.001;  // z
  x(3) = 0;      // vx
  x(4) = 0;      // vy

  ttb::Matrix P = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P(0, 0) = 225;    // x,x
  P(1, 1) = 225;    // y,y
  P(2, 2) = 0.001;  // z,z
  P(3, 3) = 25;     // vx, vx
  P(4, 4) = 25;     // vy, vy

  auto base_dist1 = std::make_unique<ttb::GaussianDistribution>(std::move(x), std::move(P));

  state1._state_dist.at(stateModelId)->merge(std::move(base_dist1));
  state1._label = ttb::Label{ 1 };
  state1._existenceProbability = 0.05;

  // track2
  ttb::State state2 = manager.createState();

  ttb::Vector x2 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x2(0) = -400;  // x
  x2(1) = 0;     // y
  x2(2) = 0;     // z
  x2(3) = 0;     // vx
  x2(4) = 0;     // vy

  ttb::Matrix P2 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P2(0, 0) = 225;    // x,x
  P2(1, 1) = 225;    // y,y
  P2(2, 2) = 0.001;  // z,z
  P2(3, 3) = 25;     // vx, vx
  P2(4, 4) = 25;     // vy, vy

  auto base_dist21 = std::make_unique<ttb::GaussianDistribution>(std::move(x2), std::move(P2));

  state2._state_dist.at(stateModelId)->merge(std::move(base_dist21));

  state2._label = ttb::Label{ 2 };
  state2._existenceProbability = 0.05;

  // track3
  ttb::State state3 = manager.createState();

  ttb::Vector x3 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x3(0) = -400.0;  // x
  x3(1) = -400.0;  // y
  x3(2) = 0;       // z
  x3(3) = 0;       // vx
  x3(4) = 0;       // vy

  ttb::Matrix P3 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P3(0, 0) = 225;    // x,x
  P3(1, 1) = 225;    // y,y
  P3(2, 2) = 0.001;  // z,z
  P3(3, 3) = 25;     // vx, vx
  P3(4, 4) = 25;     // vy, vy

  auto base_dist3 = std::make_unique<ttb::GaussianDistribution>(std::move(x3), std::move(P3));

  state3._state_dist.at(stateModelId)->merge(std::move(base_dist3));

  state3._existenceProbability = 0.05;
  state3._label = ttb::Label{ 3 };

  // track4
  ttb::State state4 = manager.createState();

  ttb::Vector x4 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x4(0) = 400.0;   // x
  x4(1) = -400.0;  // y
  x4(2) = 0;       // z
  x4(3) = 0;       // vx
  x4(4) = 0;       // vy

  ttb::Matrix P4 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P4(0, 0) = 225;    // x,x
  P4(1, 1) = 225;    // y,y
  P4(2, 2) = 0.001;  // z,z
  P4(3, 3) = 25;     // vx, vx
  P4(4, 4) = 25;     // vy, vy

  auto base_dist4 = std::make_unique<ttb::GaussianDistribution>(std::move(x4), std::move(P4));

  state4._state_dist.at(stateModelId)->merge(std::move(base_dist4));

  state4._existenceProbability = 0.05;
  state4._label = ttb::Label{ 4 };

  // track5
  ttb::State state5 = manager.createState();

  ttb::Vector x5 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x5(0) = 400.0;  // x
  x5(1) = 0;      // y
  x5(2) = 0;      // z
  x5(3) = 0;      // vx
  x5(4) = 0;      // vy

  ttb::Matrix P5 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P5(0, 0) = 225;    // x,x
  P5(1, 1) = 225;    // y,y
  P5(2, 2) = 0.001;  // z,z
  P5(3, 3) = 25;     // vx, vx
  P5(4, 4) = 25;     // vy, vy

  auto base_dist5 = std::make_unique<ttb::GaussianDistribution>(std::move(x5), std::move(P5));

  state5._state_dist.at(stateModelId)->merge(std::move(base_dist5));

  state5._existenceProbability = 0.05;
  state5._label = ttb::Label{ 5 };

  // track6
  ttb::State state6 = manager.createState();

  ttb::Vector x6 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x6(0) = 400.0;  // x
  x6(1) = 400;    // y
  x6(2) = 0;      // z
  x6(3) = 0;      // vx
  x6(4) = 0;      // vy

  ttb::Matrix P6 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P6(0, 0) = 225;    // x,x
  P6(1, 1) = 225;    // y,y
  P6(2, 2) = 0.001;  // z,z
  P6(3, 3) = 25;     // vx, vx
  P6(4, 4) = 25;     // vy, vy

  auto base_dist6 = std::make_unique<ttb::GaussianDistribution>(std::move(x6), std::move(P6));

  state6._state_dist.at(stateModelId)->merge(std::move(base_dist6));

  state6._existenceProbability = 0.05;
  state6._label = ttb::Label{ 6 };

  ttb::LMBDistribution lmb(&manager);
  lmb._tracks = { state1, state2, state3, state4, state5, state6 };

  LOG_DEB("####################################################");
  LOG_DEB("LMB: " << lmb.toString());
  LOG_DEB("####################################################");
  LOG_DEB("Measurements: " << measContainer.toString());
  LOG_DEB("####################################################");

  lmb.calcInnovation(measContainer);
  LOG_DEB("LMB after calcInnovation" << lmb.toString());
  LOG_DEB("####################################################");

  lmb.update(measContainer);
  LOG_ERR("LMB after first update" << lmb.toString());

  std::map<ttb::Label, double> label2gtExProb;
  label2gtExProb.emplace((ttb::Label)1, 0.26515158913996939);
  label2gtExProb.emplace((ttb::Label)2, 0.017071908949818925);
  label2gtExProb.emplace((ttb::Label)3, 0.017071908949818925);
  label2gtExProb.emplace((ttb::Label)4, 0.33637527963697128);
  label2gtExProb.emplace((ttb::Label)5, 0.017071908949818925);
  label2gtExProb.emplace((ttb::Label)6, 0.017071908949818925);

  for (const auto& track : lmb._tracks)
  {
    EXPECT_FLOAT_EQ(track._existenceProbability, label2gtExProb.at(track._label));
  }

  LOG_DEB("####################################################");

  lmb.calcInnovation(measContainer2);
  LOG_DEB("LMB after calcInnovation" << lmb.toString());
  LOG_DEB("####################################################");

  lmb.update(measContainer2);

  LOG_ERR("LMB after second update" << lmb.toString());
  LOG_DEB("####################################################");

  std::map<ttb::Label, double> label2gtExProb2;
  label2gtExProb2.emplace((ttb::Label)1, 0.58668579489046457);
  label2gtExProb2.emplace((ttb::Label)2, 0.09041256659485751);
  label2gtExProb2.emplace((ttb::Label)3, 0.0056989151651848039);
  label2gtExProb2.emplace((ttb::Label)4, 0.40349670714244701);
  label2gtExProb2.emplace((ttb::Label)5, 0.0056989151651848039);
  label2gtExProb2.emplace((ttb::Label)6, 0.0056989151651848039);

  for (const auto& track : lmb._tracks)
  {
    EXPECT_FLOAT_EQ(track._existenceProbability, label2gtExProb2.at(track._label));
  }
}
