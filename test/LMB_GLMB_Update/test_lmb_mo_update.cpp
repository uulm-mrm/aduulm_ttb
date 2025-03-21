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
std::string const params_string{ "thread_pool_size: 1\n"
                                 "version: 1.0.0\n"
                                 "meas_models:\n"
                                 "  gaussian_models:\n"
                                 "    -\n"
                                 "     id: 0\n"
                                 "     components: [POS_X, POS_Y]\n"
                                 "     clutter:\n"
                                 "       intensity: 1.093750000000000e-05\n"
                                 "     detection:\n"
                                 "       prob: 0.9\n"
                                 "       prob_min: 0.001\n"
                                 "     occlusion:\n"
                                 "       model: NO_OCCLUSION\n"
                                 "     gating_prob: 0.99\n"
                                 "    -\n"
                                 "     id: 1\n"
                                 "     components: [POS_X, POS_Y]\n"
                                 "     clutter:\n"
                                 "       intensity: 1e-9\n"
                                 "     detection:\n"
                                 "       prob: 0.9\n"
                                 "       prob_min: 0.001\n"
                                 "     occlusion:\n"
                                 "       model: NO_OCCLUSION\n"
                                 "     gating_prob: 0.99\n"
                                 "state:\n"
                                 "  multi_model:\n"
                                 "    use_state_models: [1]\n"
                                 "  classification:\n"
                                 "    use_meas_models: [0]\n"
                                 "    classes: [UNKNOWN, PEDESTRIAN, CAR, TRUCK, BICYCLE]\n"
                                 "    use_union_classes: false\n"
                                 "    discount_factor_prediction: 0.98\n"
                                 "  estimation:\n"
                                 "    transform_output_state: false\n"
                                 "    output_state_model: CTRV\n"
                                 "    type: AVERAGE\n"
                                 "state_models:\n"
                                 "    -\n"
                                 "      id: 1\n"
                                 "      type: CV\n"
                                 "      distribution:\n"
                                 "        type: GAUSSIAN\n"
                                 "        mixture: true\n"
                                 "        extraction_type: BEST_COMPONENT\n"
                                 "        post_process:\n"
                                 "          enable: false\n"
                                 "          max_components: 1000\n"
                                 "          merging_distance: 0\n"
                                 "          min_weight: 0.001\n"
                                 "          max_variance: 100\n"
                                 "      extent: NONE\n"
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
                                 "    persistence_prob: 0.98\n"
                                 "lmb_distribution:\n"
                                 "  update_method: GLMB\n"
                                 "  post_process_prediction:\n"
                                 "    enable: false\n"
                                 "    max_tracks: 1000000\n"
                                 "    pruning_threshold: 0\n"
                                 "    max_last_assoc_duration_ms: 50\n"
                                 "  post_process_update:\n"
                                 "    enable: false\n"
                                 "    max_tracks: 1000000000\n"
                                 "    pruning_threshold: 1e-8\n"
                                 "  extraction:\n"
                                 "    type: EXISTENCE_PROBABILITY  #Cardinality\n"
                                 "    threshold: 0.5\n"
                                 "glmb_distribution:\n"
                                 "  do_profiling: true\n"
                                 "  post_process_prediction:\n"
                                 "    enable: false\n"
                                 "    pruning_threshold: 1e-5\n"
                                 "    max_hypotheses: 300\n"
                                 "  post_process_update:\n"
                                 "    enable: false\n"
                                 "    pruning_threshold: 1e-5\n"
                                 "    max_hypotheses: 300\n"
                                 "  update:\n"
                                 "    joint_prediction_and_update: false\n"
                                 "    assignment_method: MURTY\n"
                                 "    max_total_number_update_hypotheses: 3000\n"
                                 "    num_update_hypotheses:\n"
                                 "      equal_allocation_share_ratio: 0\n"
                                 "      max_update_hypotheses: 3000\n"
                                 "    gibbs_sampling:\n"
                                 "      max_trials_factor: 3\n"
                                 "  extraction:\n"
                                 "    type: CARDINALITY\n"
                                 "    threshold: 0.5\n"
                                 "  lmb_2_glmb_conversion:\n"
                                 "    type: ALL  # K_BEST  # ALL\n"
                                 "    all:\n"
                                 "      num_track_limit: 10\n"
                                 "      fallback_type: SAMPLING\n"
                                 "    kBest:\n"
                                 "      max_number_hypotheses: 2000\n"
                                 "    sampling:\n"
                                 "      max_number_hypotheses: 2000\n"
                                 "      percentage_of_weight_hypotheses: 0.9\n"
                                 "      max_num_tries: 10000\n"
                                 "filter:\n"
                                 "  enable: true\n"
                                 "  type: LMB_IC\n" };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(LMBUpdate, ThreeTracksSevenMeas)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYaml<ttb::Params>(params_string);
  auto manager = ttb::TTBManager(params);
  ttb::MeasModelId measModelId{ "0" };
  ttb::MeasModelId measModelId2{ "1" };
  ttb::StateModelId stateModelId{ 1 };

  // Create Measurements for both sensors
  ttb::MeasurementContainer measContainer{ ._id = measModelId, ._time = ttb::Time(0s) };

  ttb::Vector meas_mean = ttb::Vector::Zero(2);
  meas_mean(0) = -355.0767797404343;
  meas_mean(1) = 59.754602959654940;

  ttb::Matrix meas_cov = ttb::Matrix::Identity(2, 2);
  meas_cov(0, 0) = 0.36;
  meas_cov(1, 1) = 0.36;
  ttb::Measurement meas1(
      std::make_unique<ttb::GaussianDistribution>(meas_mean, meas_cov, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas1._ref_point_measured = true;
  LOG_DEB("Meas: " + meas1.toString());
  measContainer._data.push_back(std::move(meas1));

  ttb::Vector meas_mean2 = ttb::Vector::Zero(2);
  meas_mean2(0) = -374.8856716042374;
  meas_mean2(1) = -363.3953851459543;

  ttb::Matrix meas_cov2 = ttb::Matrix::Identity(2, 2);
  meas_cov2(0, 0) = 0.36;
  meas_cov2(1, 1) = 0.36;
  ttb::Measurement meas2(
      std::make_unique<ttb::GaussianDistribution>(meas_mean2, meas_cov2, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas2._ref_point_measured = true;
  LOG_DEB("Meas2: " + meas2.toString());
  measContainer._data.push_back(std::move(meas2));

  ttb::Vector meas_mean3 = ttb::Vector::Zero(2);
  meas_mean3(0) = 369.8213293298233;
  meas_mean3(1) = 364.7892513727658;

  ttb::Matrix meas_cov3 = ttb::Matrix::Identity(2, 2);
  meas_cov3(0, 0) = 0.36;
  meas_cov3(1, 1) = 0.36;
  ttb::Measurement meas3(
      std::make_unique<ttb::GaussianDistribution>(meas_mean3, meas_cov3, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas3._ref_point_measured = true;
  LOG_DEB("Meas3: " + meas3.toString());
  measContainer._data.push_back(std::move(meas3));

  ttb::Vector meas_mean4 = ttb::Vector::Zero(2);
  meas_mean4(0) = 218.8506641642635;
  meas_mean4(1) = -377.6936775951114;

  ttb::Matrix meas_cov4 = ttb::Matrix::Identity(2, 2);
  meas_cov4(0, 0) = 0.36;
  meas_cov4(1, 1) = 0.36;
  ttb::Measurement meas4(
      std::make_unique<ttb::GaussianDistribution>(meas_mean4, meas_cov4, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas4._ref_point_measured = true;
  LOG_DEB("Meas4: " + meas4.toString());
  measContainer._data.push_back(std::move(meas4));

  ttb::Vector meas_mean5 = ttb::Vector::Zero(2);
  meas_mean5(0) = -223.0507033343160;
  meas_mean5(1) = -70.319093015156910;

  ttb::Matrix meas_cov5 = ttb::Matrix::Identity(2, 2);
  meas_cov5(0, 0) = 0.36;
  meas_cov5(1, 1) = 0.36;
  ttb::Measurement meas5(
      std::make_unique<ttb::GaussianDistribution>(meas_mean5, meas_cov5, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas5._ref_point_measured = true;
  LOG_DEB("Meas5: " + meas5.toString());
  measContainer._data.push_back(std::move(meas5));

  ttb::Vector meas_mean6 = ttb::Vector::Zero(2);
  meas_mean6(0) = 250.7703933647325;
  meas_mean6(1) = 330.6491491092934;

  ttb::Matrix meas_cov6 = ttb::Matrix::Identity(2, 2);
  meas_cov6(0, 0) = 0.36;
  meas_cov6(1, 1) = 0.36;
  ttb::Measurement meas6(
      std::make_unique<ttb::GaussianDistribution>(meas_mean6, meas_cov6, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas6._ref_point_measured = true;
  LOG_DEB("Meas6: " + meas6.toString());
  measContainer._data.push_back(std::move(meas6));

  ttb::Vector meas_mean7 = ttb::Vector::Zero(2);
  meas_mean7(0) = -495.1091774530706;
  meas_mean7(1) = 324.2468990386059;

  ttb::Matrix meas_cov7 = ttb::Matrix::Identity(2, 2);
  meas_cov7(0, 0) = 0.36;
  meas_cov7(1, 1) = 0.36;
  ttb::Measurement meas7(
      std::make_unique<ttb::GaussianDistribution>(meas_mean7, meas_cov7, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas7._ref_point_measured = true;
  LOG_DEB("Meas7: " + meas7.toString());
  measContainer._data.push_back(std::move(meas7));

  ttb::MeasurementContainer measContainer2{ ._id = measModelId,
                                            ._egoMotion = ttb::EgoMotionDistribution::zero(),
                                            ._time = ttb::Time(0s) };

  ttb::Vector meas_mean11 = ttb::Vector::Zero(2);
  meas_mean11(0) = -354.3802376945775;
  meas_mean11(1) = 59.223113382786950;

  ttb::Matrix meas_cov11 = ttb::Matrix::Identity(2, 2);
  meas_cov11(0, 0) = 0.36;
  meas_cov11(1, 1) = 0.36;
  ttb::Measurement meas11(
      std::make_unique<ttb::GaussianDistribution>(meas_mean11, meas_cov11, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas11._ref_point_measured = true;
  LOG_DEB("Meas11: " + meas11.toString());
  measContainer2._data.push_back(std::move(meas11));

  ttb::Vector meas_mean21 = ttb::Vector::Zero(2);
  meas_mean21(0) = -376.4643400815579;
  meas_mean21(1) = -364.5223246281104;

  ttb::Matrix meas_cov21 = ttb::Matrix::Identity(2, 2);
  meas_cov21(0, 0) = 0.36;
  meas_cov21(1, 1) = 0.36;
  ttb::Measurement meas21(
      std::make_unique<ttb::GaussianDistribution>(meas_mean21, meas_cov21, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas21._ref_point_measured = true;
  LOG_DEB("Meas21: " + meas21.toString());
  measContainer2._data.push_back(std::move(meas21));

  ttb::Vector meas_mean31 = ttb::Vector::Zero(2);
  meas_mean31(0) = -369.6716417509368;
  meas_mean31(1) = 364.2738470493076;

  ttb::Matrix meas_cov31 = ttb::Matrix::Identity(2, 2);
  meas_cov31(0, 0) = 0.36;
  meas_cov31(1, 1) = 0.36;
  ttb::Measurement meas31(
      std::make_unique<ttb::GaussianDistribution>(meas_mean31, meas_cov31, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas31._ref_point_measured = true;
  LOG_DEB("Meas31: " + meas31.toString());
  measContainer2._data.push_back(std::move(meas31));

  ttb::Vector meas_mean41 = ttb::Vector::Zero(2);
  meas_mean41(0) = 218.9385435521861;
  meas_mean41(1) = -376.4739591365427;

  ttb::Matrix meas_cov41 = ttb::Matrix::Identity(2, 2);
  meas_cov41(0, 0) = 0.36;
  meas_cov41(1, 1) = 0.36;
  ttb::Measurement meas41(
      std::make_unique<ttb::GaussianDistribution>(meas_mean41, meas_cov41, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas41._ref_point_measured = true;
  LOG_DEB("Meas41: " + meas41.toString());
  measContainer2._data.push_back(std::move(meas41));

  ttb::Vector meas_mean51 = ttb::Vector::Zero(2);
  meas_mean51(0) = -389.2415677580803;
  meas_mean51(1) = -256.7461047486872;

  ttb::Matrix meas_cov51 = ttb::Matrix::Identity(2, 2);
  meas_cov51(0, 0) = 0.36;
  meas_cov51(1, 1) = 0.36;
  ttb::Measurement meas51(
      std::make_unique<ttb::GaussianDistribution>(meas_mean51, meas_cov51, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas51._ref_point_measured = true;
  LOG_DEB("Meas51: " + meas51.toString());
  measContainer2._data.push_back(std::move(meas51));

  ttb::Vector meas_mean61 = ttb::Vector::Zero(2);
  meas_mean61(0) = -313.1757574289825;
  meas_mean61(1) = 161.9487826498748;

  ttb::Matrix meas_cov61 = ttb::Matrix::Identity(2, 2);
  meas_cov61(0, 0) = 0.36;
  meas_cov61(1, 1) = 0.36;
  ttb::Measurement meas61(
      std::make_unique<ttb::GaussianDistribution>(meas_mean61, meas_cov61, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas61._ref_point_measured = true;
  LOG_DEB("Meas61: " + meas61.toString());
  measContainer2._data.push_back(std::move(meas61));

  ttb::Vector meas_mean71 = ttb::Vector::Zero(2);
  meas_mean71(0) = -259.0108081058011;
  meas_mean71(1) = -427.9420119770139;

  ttb::Matrix meas_cov71 = ttb::Matrix::Identity(2, 2);
  meas_cov71(0, 0) = 0.36;
  meas_cov71(1, 1) = 0.36;
  ttb::Measurement meas71(
      std::make_unique<ttb::GaussianDistribution>(meas_mean71, meas_cov71, 1, ttb::REFERENCE_POINT::CENTER),
      ttb::Time(0s),
      ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }));
  meas71._ref_point_measured = true;
  LOG_DEB("Meas71: " + meas71.toString());
  measContainer2._data.push_back(std::move(meas71));

  // create 3 tracks
  auto const& state_model = *manager.getStateModelMap().at(stateModelId);
  // track1
  ttb::State state1 = manager.createState();

  ttb::Vector x = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x(0) = -376.0657882526539;  // x
  x(1) = -363.8193668827842;  // y
  x(2) = 0;                   // z
  x(3) = 3.969745495699935;   // vx
  x(4) = 5.970312638752488;   // vy

  ttb::Matrix P = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P(0, 0) = 0.403281171794258;            // x,x
  P(1, 1) = 0.403281171794258;            // y,y
  P(2, 2) = 0.001;                        // z,z
  P(3, 3) = 0.116241459088294;            // vx, vx
  P(4, 4) = 0.116241459088294;            // vy, vy
  P(0, 3) = P(3, 0) = 0.172543791141602;  // x,vx
  P(1, 4) = P(4, 1) = 0.172543791141602;  // y,vy

  auto base_dist1 = std::make_unique<ttb::GaussianDistribution>(std::move(x), std::move(P));

  state1._state_dist.at(stateModelId)->merge(std::move(base_dist1));
  state1._label = ttb::Label{ 58 };
  state1._existenceProbability = 0.979988592142434;

  // track2
  ttb::State state2 = manager.createState();

  ttb::Vector x2 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x2(0) = -400;  // x
  x2(1) = -400;  // y
  x2(2) = 0;     // z
  x2(3) = 0;     // vx
  x2(4) = 0;     // vy

  ttb::Matrix P2 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P2(0, 0) = 325.1;                          // x,x
  P2(1, 1) = 325.1;                          // y,y
  P2(2, 2) = 0.001;                          // z,z
  P2(3, 3) = 25.08;                          // vx, vx
  P2(4, 4) = 25.08;                          // vy, vy
  P2(0, 3) = P2(3, 0) = 50.080000000000005;  // x,vx
  P2(1, 4) = P2(4, 1) = 50.080000000000005;  // y,vy

  ttb::Vector x21 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x21(0) = -397.0548501344673;  // x
  x21(1) = -421.8671020878899;  // y
  x21(2) = 0;                   // z
  x21(3) = 0.267925861308324;   // vx
  x21(4) = -1.989291692684458;  // vy

  ttb::Matrix P21 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P21(0, 0) = 22.981132324160246;              // x,x
  P21(1, 1) = 22.981132324160246;              // y,y
  P21(2, 2) = 0.001;                           // z,z
  P21(3, 3) = 22.579698845708350;              // vx, vx
  P21(4, 4) = 22.579698845708350;              // vy, vy
  P21(0, 3) = P21(3, 0) = 22.595674401885210;  // x,vx
  P21(1, 4) = P21(4, 1) = 22.595674401885210;  // y,vy

  ttb::Vector x22 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x22(0) = -418.4900894148190;  // x
  x22(1) = -400;                // y
  x22(2) = 0;                   // z
  x22(3) = -1.682078453836929;  // vx
  x22(4) = -2.350588777203333;  // vy

  ttb::Matrix P22 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P22(0, 0) = 22.981132324160246;              // x,x
  P22(1, 1) = 22.981132324160246;              // y,y
  P22(2, 2) = 0.001;                           // z,z
  P22(3, 3) = 22.579698845708350;              // vx, vx
  P22(4, 4) = 22.579698845708350;              // vy, vy
  P22(0, 3) = P22(3, 0) = 22.595674401885210;  // x,vx
  P22(1, 4) = P22(4, 1) = 22.595674401885210;  // y,vy

  auto base_dist21 = std::make_unique<ttb::GaussianDistribution>(
      std::move(x2), std::move(P2), 0.003002215714468, ttb::REFERENCE_POINT::CENTER);
  auto base_dist22 = std::make_unique<ttb::GaussianDistribution>(
      std::move(x21), std::move(P21), 0.701594427037742, ttb::REFERENCE_POINT::CENTER);
  auto base_dist23 = std::make_unique<ttb::GaussianDistribution>(
      std::move(x22), std::move(P22), 0.295403357247791, ttb::REFERENCE_POINT::CENTER);

  state2._state_dist.at(stateModelId)->merge(std::move(base_dist21));
  state2._state_dist.at(stateModelId)->merge(std::move(base_dist22));
  state2._state_dist.at(stateModelId)->merge(std::move(base_dist23));

  state2._label = ttb::Label{ 88 };
  state2._existenceProbability = 0.143553041823563;

  // track3
  ttb::State state3 = manager.createState();

  ttb::Vector x3 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x3(0) = -400.0;  // x
  x3(1) = -400.0;  // y
  x3(2) = 0;       // z
  x3(3) = 0;       // vx
  x3(4) = 0;       // vy

  ttb::Matrix P3 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P3(0, 0) = 250.01;            // x,x
  P3(1, 1) = 250.01;            // y,y
  P3(2, 2) = 0.001;             // z,z
  P3(3, 3) = 25.04;             // vx, vx
  P3(4, 4) = 25.04;             // vy, vy
  P3(0, 3) = P3(3, 0) = 25.02;  // x,vx
  P3(1, 4) = P3(4, 1) = 25.02;  // y,vy

  auto base_dist3 = std::make_unique<ttb::GaussianDistribution>(std::move(x3), std::move(P3));

  state3._state_dist.at(stateModelId)->merge(std::move(base_dist3));

  state3._existenceProbability = 0.049;
  state3._label = ttb::Label{ 94 };

  ttb::LMBDistribution lmb(&manager);
  lmb._tracks = { state1, state2, state3 };

  // Save final result: This should be the result:
  //  double r1 = 0.999996521733549;
  std::vector<double> r_vec;
  std::vector<ttb::Label> label_vec;
  label_vec.push_back(ttb::Label{ 58 });
  label_vec.push_back(ttb::Label{ 88 });
  label_vec.push_back(ttb::Label{ 94 });

  double r1 = 0.99999654154702844;

  ttb::Vector mean_track1_final_true = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  mean_track1_final_true(0) = -3.760657882526539e+02;
  mean_track1_final_true(1) = -3.638193668827842e+02;
  mean_track1_final_true(2) = 0;
  mean_track1_final_true(3) = 3.969745495699935;
  mean_track1_final_true(4) = 5.970312638752488;

  ttb::Vector mean_track11_final_true = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  mean_track11_final_true(0) = -3.754422712023007e+02;
  mean_track11_final_true(1) = -3.638193668827842e+02;
  mean_track11_final_true(2) = 0;
  mean_track11_final_true(3) = 4.236517175693638;
  mean_track11_final_true(4) = 6.066155977944862;

  ttb::Matrix cov_track1_final_true =
      ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track1_final_true(0, 0) = 0.403281171794258;
  cov_track1_final_true(1, 1) = 0.403281171794258;
  cov_track1_final_true(2, 2) = 1;
  cov_track1_final_true(3, 3) = 0.116241459088294;
  cov_track1_final_true(4, 4) = 0.116241459088294;
  cov_track1_final_true(0, 3) = cov_track1_final_true(3, 0) = 0.172543791141602;  // x,vx
  cov_track1_final_true(1, 4) = cov_track1_final_true(4, 1) = 0.172543791141602;  // y,vy

  ttb::Matrix cov_track11_final_true =
      ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track11_final_true(0, 0) = 0.190206737976587;
  cov_track11_final_true(1, 1) = 0.190206737976587;
  cov_track11_final_true(2, 2) = 1;
  cov_track11_final_true(3, 3) = 0.077237012284591;
  cov_track11_final_true(4, 4) = 0.077237012284591;
  cov_track11_final_true(0, 3) = cov_track11_final_true(3, 0) = 0.081379925388386;  // x,vx
  cov_track11_final_true(1, 4) = cov_track11_final_true(4, 1) = 0.081379925388386;  // y,vy

  std::vector<double> weights;
  weights.push_back(1.703365062139331e-05);
  weights.push_back(0.999982966349379);

  //  double r2 = 0.016485166662246;
  double r2 = 0.016485163814530517;
  ttb::Vector mean_track2_final_true = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  mean_track2_final_true(0) = -400;
  mean_track2_final_true(1) = -400;
  mean_track2_final_true(2) = 0;
  mean_track2_final_true(3) = 0;
  mean_track2_final_true(4) = 0;

  ttb::Vector mean_track21_final_true = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  mean_track21_final_true(0) = -3.970548501344673e+02;
  mean_track21_final_true(1) = -4.218671020878899e+02;
  mean_track21_final_true(2) = 0;
  mean_track21_final_true(3) = 0.267925861308324;
  mean_track21_final_true(4) = -1.989291692684458;

  ttb::Vector mean_track22_final_true = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  mean_track22_final_true(0) = -4.184900894148190e+02;
  mean_track22_final_true(1) = -4.258386263546855e+02;
  mean_track22_final_true(2) = 0;
  mean_track22_final_true(3) = -1.682078453836929;
  mean_track22_final_true(4) = -2.350588777203333;

  ttb::Vector mean_track23_final_true = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  mean_track23_final_true(0) = -3.749134512337540e+02;
  mean_track23_final_true(1) = -3.634358744882620e+02;
  mean_track23_final_true(2) = 0;
  mean_track23_final_true(3) = 3.864455128310063;
  mean_track23_final_true(4) = 5.632517396579028;

  ttb::Matrix cov_track2_final_true =
      ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track2_final_true(0, 0) = 325.1;
  cov_track2_final_true(1, 1) = 325.1;
  cov_track2_final_true(2, 2) = 1;
  cov_track2_final_true(3, 3) = 25.08;
  cov_track2_final_true(4, 4) = 25.08;
  cov_track2_final_true(0, 3) = cov_track2_final_true(3, 0) = 50.08;  // x,vx
  cov_track2_final_true(1, 4) = cov_track2_final_true(4, 1) = 50.08;  // y,vy

  ttb::Matrix cov_track21_final_true =
      ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track21_final_true(0, 0) = 22.981132324160246;
  cov_track21_final_true(1, 1) = 22.981132324160246;
  cov_track21_final_true(2, 2) = 1;
  cov_track21_final_true(3, 3) = 22.579698845708350;
  cov_track21_final_true(4, 4) = 22.579698845708350;
  cov_track21_final_true(0, 3) = cov_track21_final_true(3, 0) = 22.595674401885210;  // x,vx
  cov_track21_final_true(1, 4) = cov_track21_final_true(4, 1) = 22.595674401885210;  // y,vy

  ttb::Matrix cov_track22_final_true =
      ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track22_final_true(0, 0) = 22.981132324160246;
  cov_track22_final_true(1, 1) = 22.981132324160246;
  cov_track22_final_true(2, 2) = 1;
  cov_track22_final_true(3, 3) = 22.579698845708350;
  cov_track22_final_true(4, 4) = 22.579698845708350;
  cov_track22_final_true(0, 3) = cov_track22_final_true(3, 0) = 22.595674401885210;  // x,vx
  cov_track22_final_true(1, 4) = cov_track22_final_true(4, 1) = 22.595674401885210;  // y,vy

  ttb::Matrix cov_track23_final_true =
      ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track23_final_true(0, 0) = 0.359601794383353;
  cov_track23_final_true(1, 1) = 0.359601794383353;
  cov_track23_final_true(2, 2) = 1;
  cov_track23_final_true(3, 3) = 17.373964235236280;
  cov_track23_final_true(4, 4) = 17.373964235236280;
  cov_track23_final_true(0, 3) = cov_track23_final_true(3, 0) = 0.055394825785046;  // x,vx
  cov_track23_final_true(1, 4) = cov_track23_final_true(4, 1) = 0.055394825785046;  // y,vy

  std::vector<double> weights2;
  weights2.push_back(0.003002212260640);
  weights2.push_back(0.701593619905125);
  weights2.push_back(0.295403017408026);
  weights2.push_back(1.150426207719032e-06);

  //  double r3 = 0.005127083808704;
  double r3 = 0.0051270640967412372;
  ttb::Vector mean_track3_final_true = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  mean_track3_final_true(0) = -400;
  mean_track3_final_true(1) = -400;
  mean_track3_final_true(2) = 0;
  mean_track3_final_true(3) = 0;
  mean_track3_final_true(4) = 0;

  ttb::Vector mean_track31_final_true = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  mean_track31_final_true(0) = -3.749217827925685e+02;
  mean_track31_final_true(1) = -3.634480178948757e+02;
  mean_track31_final_true(2) = 0;
  mean_track31_final_true(3) = 2.509727589016181;
  mean_track31_final_true(4) = 3.657976050038840;

  ttb::Matrix cov_track3_final_true =
      ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track3_final_true(0, 0) = 250.01;
  cov_track3_final_true(1, 1) = 250.01;
  cov_track3_final_true(2, 2) = 1;
  cov_track3_final_true(3, 3) = 25.04;
  cov_track3_final_true(4, 4) = 25.04;
  cov_track3_final_true(0, 3) = cov_track3_final_true(3, 0) = 25.02;  // x,vx
  cov_track3_final_true(1, 4) = cov_track3_final_true(4, 1) = 25.02;  // y,vy

  ttb::Matrix cov_track31_final_true =
      ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track31_final_true(0, 0) = 0.359482366098176;
  cov_track31_final_true(1, 1) = 0.359482366098176;
  cov_track31_final_true(2, 2) = 1;
  cov_track31_final_true(3, 3) = 22.539698845708350;
  cov_track31_final_true(4, 4) = 22.539698845708350;
  cov_track31_final_true(0, 3) = cov_track31_final_true(3, 0) = 0.035975556176858;  // x,vx
  cov_track31_final_true(1, 4) = cov_track31_final_true(4, 1) = 0.035975556176858;  // y,vy

  std::vector<double> weights3;
  weights3.push_back(0.999799130122145);
  weights3.push_back(2.008698778553331e-04);

  r_vec.push_back(r1);
  r_vec.push_back(r2);
  r_vec.push_back(r3);

  LOG_DEB("####################################################");
  LOG_DEB("LMB: " << lmb.toString());
  LOG_DEB("####################################################");
  LOG_DEB("Measurements: " << measContainer.toString());
  LOG_DEB("####################################################");

  lmb.calcInnovation(measContainer);
  LOG_DEB("LMB after calcInnovation" << lmb.toString());
  LOG_DEB("####################################################");
  lmb.update(measContainer);
  LOG_DEB("LMB after first update" << lmb.toString());
  LOG_DEB("####################################################");

  // compare results
  unsigned int counter = 0.;
  //  std::sort(lmb._tracks.begin(), lmb._tracks.end(), [](auto const& a, auto const& b) { return a._label < b._label;
  //  });
  std::stringstream rString;
  for (const auto& tmp : r_vec)
  {
    rString << tmp << " ";
  }
  rString << "\n";
  LOG_DEB("r_vec: " << rString.str());
  for (const auto& track : lmb._tracks)
  {
    LOG_DEB("track label: " << track._label << " exProb: " << track._existenceProbability);
    ASSERT_NEAR(r_vec[counter], track._existenceProbability, 1e-5);
    ASSERT_EQ(label_vec[counter], track._label);
    counter++;
  }
  ASSERT_EQ(3, lmb._tracks.size());

  // Todo: Check values of mean and cov of tracks!

  //  track1 = *lmb._tracks.begin();
  //  ASSERT_EQ(ttb::Label{58},track1._label);
  //  ASSERT_EQ(r1,track1._existenceProb);
  //  track2 = *std::next(lmb._tracks.begin());
  //  ASSERT_EQ(ttb::Label{88},track2._label);
  //  track3 = *std::next(std::next(lmb._tracks.begin()));
  //  ASSERT_EQ(ttb::Label{94},track3._label);
  //
  //  auto mean1 = track1._stateDist._model_dist.begin()->second->mean();
  ////  auto mean11 = std::next(track1._stateDist._model_dist.begin())->second->mean();
  //  auto cov1 = track1._stateDist._model_dist.begin()->second->covariance();
  ////  auto cov11 = std::next(track1._stateDist._model_dist.begin())->second->covariance();
  //  auto weights_tmp1 = track1._stateDist._model_dist.begin()->second->sumWeights();
  ////  auto weights_tmp11 = std::next(track1._stateDist._model_dist.begin())->second->sumWeights();
  //  for(unsigned int i=0; i<2; i++)
  //  {
  //    ASSERT_EQ(mean_track1_final_true(i),mean1(i));
  ////    ASSERT_EQ(mean_track11_final_true(i),mean11(i));
  //    ASSERT_EQ(cov_track1_final_true(i,i),cov1(i,i));
  ////    ASSERT_EQ(cov_track11_final_true(i,i),cov11(i,i));
  //  }
  //  ASSERT_EQ(cov_track1_final_true(3,0),cov1(3,0));
  //  ASSERT_EQ(cov_track1_final_true(4,1),cov1(4,1));
  ////  ASSERT_EQ(cov_track11_final_true(3,0),cov11(3,0));
  ////  ASSERT_EQ(cov_track11_final_true(4,1),cov11(4,1));
  //  ASSERT_EQ(weights[0],weights_tmp1);
  ////  ASSERT_EQ(weights[1],weights_tmp11);
  //
  //  track2 = *std::next(lmb._tracks.begin());
  //  ASSERT_EQ(ttb::Label{88},track2._label);
  //  ASSERT_EQ(r2,track2._existenceProb);
  //  auto mean2 = track2._stateDist._model_dist.begin()->second->mean();
  ////  auto mean21 = std::next(track2._stateDist._model_dist.begin())->second->mean();
  ////  auto mean22 = std::next(std::next(track2._stateDist._model_dist.begin()))->second->mean();
  ////  auto mean23 = std::next(std::next(std::next(track2._stateDist._model_dist.begin())))->second->mean();
  //  auto cov2 = track2._stateDist._model_dist.begin()->second->covariance();
  ////  auto cov21 = std::next(track2._stateDist._model_dist.begin())->second->covariance();
  ////  auto cov22 = std::next(std::next(track2._stateDist._model_dist.begin()))->second->covariance();
  ////  auto cov23 = std::next(std::next(std::next(track2._stateDist._model_dist.begin())))->second->covariance();
  //  auto weights_tmp2 = track2._stateDist._model_dist.begin()->second->sumWeights();
  ////  auto weights_tmp21 = std::next(track2._stateDist._model_dist.begin())->second->sumWeights();
  ////  auto weights_tmp22 = std::next(std::next(track2._stateDist._model_dist.begin()))->second->sumWeights();
  ////  auto weights_tmp23 =
  /// std::next(std::next(std::next(track2._stateDist._model_dist.begin())))->second->sumWeights();
  //  for(unsigned int i=0; i<2; i++)
  //  {
  //    ASSERT_EQ(mean_track2_final_true(i),mean2(i));
  ////    ASSERT_EQ(mean_track21_final_true(i),mean21(i));
  ////    ASSERT_EQ(mean_track22_final_true(i),mean22(i));
  ////    ASSERT_EQ(mean_track23_final_true(i),mean23(i));
  //    ASSERT_EQ(cov_track2_final_true(i,i),cov2(i,i));
  ////    ASSERT_EQ(cov_track21_final_true(i,i),cov21(i,i));
  ////    ASSERT_EQ(cov_track22_final_true(i,i),cov22(i,i));
  ////    ASSERT_EQ(cov_track23_final_true(i,i),cov23(i,i));
  //  }
  //  ASSERT_EQ(cov_track2_final_true(3,0),cov2(3,0));
  //  ASSERT_EQ(cov_track2_final_true(4,1),cov2(4,1));
  ////  ASSERT_EQ(cov_track21_final_true(3,0),cov21(3,0));
  ////  ASSERT_EQ(cov_track21_final_true(4,1),cov21(4,1));
  ////  ASSERT_EQ(cov_track22_final_true(3,0),cov22(3,0));
  ////  ASSERT_EQ(cov_track22_final_true(4,1),cov22(4,1));
  ////  ASSERT_EQ(cov_track23_final_true(3,0),cov23(3,0));
  ////  ASSERT_EQ(cov_track23_final_true(4,1),cov23(4,1));
  //
  //  ASSERT_EQ(weights2[0],weights_tmp2);
  ////  ASSERT_EQ(weights2[1],weights_tmp21);
  ////  ASSERT_EQ(weights2[2],weights_tmp22);
  ////  ASSERT_EQ(weights2[3],weights_tmp23);
  //
  //  track3 = *std::next(std::next(lmb._tracks.begin()));
  //  ASSERT_EQ(ttb::Label{94},track3._label);
  //  ASSERT_EQ(r3,track3._existenceProb);
  //  auto mean3 = track3._stateDist._model_dist.begin()->second->mean();
  ////  auto mean31 = std::next(track3._stateDist._model_dist.begin())->second->mean();
  //  auto cov3 = track3._stateDist._model_dist.begin()->second->covariance();
  ////  auto cov31 = std::next(track3._stateDist._model_dist.begin())->second->covariance();
  //  auto weights_tmp3 = track3._stateDist._model_dist.begin()->second->sumWeights();
  ////  auto weights_tmp31 = std::next(track3._stateDist._model_dist.begin())->second->sumWeights();
  //  for(unsigned int i=0; i<2; i++)
  //  {
  //    ASSERT_EQ(mean_track3_final_true(i),mean3(i));
  ////    ASSERT_EQ(mean_track31_final_true(i),mean31(i));
  //    ASSERT_EQ(cov_track3_final_true(i,i),cov3(i,i));
  ////    ASSERT_EQ(cov_track31_final_true(i,i),cov31(i,i));
  //  }
  //  ASSERT_EQ(cov_track3_final_true(3,0),cov3(3,0));
  //  ASSERT_EQ(cov_track3_final_true(4,1),cov3(4,1));
  ////  ASSERT_EQ(cov_track31_final_true(3,0),cov31(3,0));
  ////  ASSERT_EQ(cov_track31_final_true(4,1),cov31(4,1));
  //  ASSERT_EQ(weights3[0],weights_tmp3);
  ////  ASSERT_EQ(weights3[1],weights_tmp31);
}

TEST(LMBUpdate, NoMeasurementsGiven)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYaml<ttb::Params>(params_string);
  auto manager = ttb::TTBManager(params);
  ttb::MeasModelId measModelId{ "0" };
  ttb::MeasModelId measModelId2{ "1" };
  ttb::StateModelId stateModelId{ 1 };

  ttb::MeasurementContainer measContainer{ ._id = measModelId,
                                           ._egoMotion = ttb::EgoMotionDistribution::zero(),
                                           ._time = ttb::Time(0s) };

  ttb::MeasurementContainer measContainer2{ ._id = measModelId,
                                            ._egoMotion = ttb::EgoMotionDistribution::zero(),
                                            ._time = ttb::Time(0s) };

  // create 3 tracks
  auto const& state_model = *manager.getStateModelMap().at(stateModelId);
  // track1
  ttb::State state1 = manager.createState();

  ttb::Vector x = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x(0) = -376;  // x
  x(1) = -363;  // y
  x(2) = 0;     // z
  x(3) = 3.97;  // vx
  x(4) = 5.97;  // vy

  ttb::Matrix P = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P(0, 0) = 0.4;             // x,x
  P(1, 1) = 0.4;             // y,y
  P(2, 2) = 0.001;           // z,z
  P(3, 3) = 0.12;            // vx, vx
  P(4, 4) = 0.12;            // vy, vy
  P(0, 3) = P(3, 0) = 0.17;  // x,vx
  P(1, 4) = P(4, 1) = 0.17;  // y,vy

  auto base_dist1 = std::make_unique<ttb::GaussianDistribution>(x, P);

  state1._state_dist.at(stateModelId)->merge(std::move(base_dist1));
  state1._label = ttb::Label{ 58 };
  state1._existenceProbability = 0.98;

  // track2
  ttb::State state2 = manager.createState();

  ttb::Vector x2 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x2(0) = -400;  // x
  x2(1) = -400;  // y
  x2(2) = 0;     // z
  x2(3) = 0;     // vx
  x2(4) = 0;     // vy

  ttb::Matrix P2 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P2(0, 0) = 325.1;             // x,x
  P2(1, 1) = 325.1;             // y,y
  P2(2, 2) = 0.001;             // z,z
  P2(3, 3) = 25.08;             // vx, vx
  P2(4, 4) = 25.08;             // vy, vy
  P2(0, 3) = P2(3, 0) = 50.08;  // x,vx
  P2(1, 4) = P2(4, 1) = 50.08;  // y,vy

  ttb::Vector x21 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x21(0) = -397.05;  // x
  x21(1) = -421.87;  // y
  x21(2) = 0;        // z
  x21(3) = 0.27;     // vx
  x21(4) = -1.99;    // vy

  ttb::Matrix P21 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P21(0, 0) = 22.98;             // x,x
  P21(1, 1) = 22.98;             // y,y
  P21(2, 2) = 0.001;             // z,z
  P21(3, 3) = 22.58;             // vx, vx
  P21(4, 4) = 22.58;             // vy, vy
  P21(0, 3) = P21(3, 0) = 22.6;  // x,vx
  P21(1, 4) = P21(4, 1) = 22.6;  // y,vy

  ttb::Vector x22 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x22(0) = -418.49;  // x
  x22(1) = -400;     // y
  x22(2) = 0;        // z
  x22(3) = -1.68;    // vx
  x22(4) = -2.35;    // vy

  ttb::Matrix P22 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P22(0, 0) = 22.98;             // x,x
  P22(1, 1) = 22.98;             // y,y
  P22(2, 2) = 0.001;             // z,z
  P22(3, 3) = 22.58;             // vx, vx
  P22(4, 4) = 22.58;             // vy, vy
  P22(0, 3) = P22(3, 0) = 22.6;  // x,vx
  P22(1, 4) = P22(4, 1) = 22.6;  // y,vy

  auto base_dist21 = std::make_unique<ttb::GaussianDistribution>(x2, P2, 0.003, ttb::REFERENCE_POINT::CENTER);
  auto base_dist22 = std::make_unique<ttb::GaussianDistribution>(x21, P21, 0.7, ttb::REFERENCE_POINT::CENTER);
  auto base_dist23 =
      std::make_unique<ttb::GaussianDistribution>(x22, std::move(P22), 0.297, ttb::REFERENCE_POINT::CENTER);

  state2._state_dist.at(stateModelId)->merge(std::move(base_dist21));
  state2._state_dist.at(stateModelId)->merge(std::move(base_dist22));
  state2._state_dist.at(stateModelId)->merge(std::move(base_dist23));

  state2._label = ttb::Label{ 88 };
  state2._existenceProbability = 0.14;

  // track3
  ttb::State state3 = manager.createState();

  ttb::Vector x3 = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  x3(0) = -400.0;  // x
  x3(1) = -400.0;  // y
  x3(2) = 0;       // z
  x3(3) = 0;       // vx
  x3(4) = 0;       // vy

  ttb::Matrix P3 = ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P3(0, 0) = 250.01;            // x,x
  P3(1, 1) = 250.01;            // y,y
  P3(2, 2) = 0.001;             // z,z
  P3(3, 3) = 25.04;             // vx, vx
  P3(4, 4) = 25.04;             // vy, vy
  P3(0, 3) = P3(3, 0) = 25.02;  // x,vx
  P3(1, 4) = P3(4, 1) = 25.02;  // y,vy

  auto base_dist3 = std::make_unique<ttb::GaussianDistribution>(x3, P3);

  state3._state_dist.at(stateModelId)->merge(std::move(base_dist3));

  state3._existenceProbability = 0.049;
  state3._label = ttb::Label{ 94 };

  ttb::LMBDistribution lmb(&manager);
  lmb._tracks = { state1, state2, state3 };

  //  // Save final result: This should be the result:
  std::vector<double> r_vec;
  std::vector<ttb::Label> label_vec;
  label_vec.push_back(ttb::Label{ 58 });
  label_vec.push_back(ttb::Label{ 88 });
  label_vec.push_back(ttb::Label{ 94 });

  double r1 = 0.83050847457;

  ttb::Vector mean_track1_final_true = ttb::Vector::Zero(state_model.state_comps()._comps.size());
  mean_track1_final_true(0) = -376;  // x
  mean_track1_final_true(1) = -363;  // y
  mean_track1_final_true(2) = 0;     // z
  mean_track1_final_true(3) = 3.97;  // vx
  mean_track1_final_true(4) = 5.97;  // vy

  ttb::Matrix cov_track1_final_true =
      ttb::Matrix::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track1_final_true(0, 0) = 0.4;
  cov_track1_final_true(1, 1) = 0.4;
  cov_track1_final_true(2, 2) = 0.001;
  cov_track1_final_true(3, 3) = 0.12;
  cov_track1_final_true(4, 4) = 0.12;
  cov_track1_final_true(0, 3) = cov_track1_final_true(3, 0) = 0.17;  // x,vx
  cov_track1_final_true(1, 4) = cov_track1_final_true(4, 1) = 0.17;  // y,vy

  std::vector<double> weights;
  weights.push_back(1);

  double r2 = 0.01601830663;
  double r3 = 0.00512605921;

  r_vec.push_back(r1);
  r_vec.push_back(r2);
  r_vec.push_back(r3);

  LOG_DEB("####################################################");
  LOG_DEB("LMB: " << lmb.toString());
  LOG_DEB("####################################################");
  LOG_DEB("Measurements: " << measContainer.toString());
  LOG_DEB("####################################################");

  lmb.calcInnovation(measContainer);
  LOG_DEB("LMB after calcInnovation" << lmb.toString());
  LOG_DEB("####################################################");

  lmb.update(measContainer);
  LOG_DEB("LMB after first update" << lmb.toString());
  LOG_DEB("####################################################");

  // compare results
  unsigned int counter = 0.;
  //  std::sort(lmb._tracks.begin(), lmb._tracks.end(), [](auto const& a, auto const& b) { return a._label < b._label;
  //  });
  std::stringstream rString;
  for (const auto& tmp : r_vec)
  {
    rString << tmp << " ";
  }
  rString << "\n";
  LOG_DEB("r_vec: " << rString.str());
  for (const auto& track : lmb._tracks)
  {
    LOG_DEB("track label: " << track._label << " exProb: " << track._existenceProbability);
    EXPECT_NEAR(r_vec[counter], track._existenceProbability, 1e-4);
    EXPECT_EQ(label_vec[counter], track._label);
    counter++;
  }

  lmb.calcInnovation(measContainer2);
  LOG_DEB("LMB after calcInnovation" << lmb.toString());
  LOG_DEB("####################################################");

  lmb.update(measContainer2);

  LOG_DEB("LMB after second update" << lmb.toString());
  LOG_DEB("####################################################");

  // check results
  counter = 0.;
  for (const auto& track : lmb._tracks)
  {
    LOG_DEB("track label: " << track._label << " exProb: " << track._existenceProbability);
    ASSERT_EQ(label_vec[counter], track._label);
    if (track._label == ttb::Label{ 58 })
    {
      LOG_ERR(track.toString());
      EXPECT_EQ(track._state_dist.begin()->second->mean(), x);
      EXPECT_EQ(track._state_dist.begin()->second->covariance(), P);
    }
    if (track._label == ttb::Label{ 88 })
    {
      LOG_ERR(track.toString());
      EXPECT_EQ(track._state_dist.begin()->second->bestComponent().mean(), x21);
      EXPECT_EQ(track._state_dist.begin()->second->bestComponent().covariance(), P21);
    }
    if (track._label == ttb::Label{ 94 })
    {
      LOG_ERR(track.toString());
      EXPECT_EQ(track._state_dist.begin()->second->mean(), x3);
      EXPECT_EQ(track._state_dist.begin()->second->covariance(), P3);
    }
    counter++;
  }

  EXPECT_EQ(3, lmb._tracks.size());
}
