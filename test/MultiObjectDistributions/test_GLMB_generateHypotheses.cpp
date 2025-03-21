//
// Created by hermann on 3/1/24.
//
//
// Created by hermann on 2/9/24.
//
#include "gtest/gtest.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include <tracking_lib/Measurements/MeasurementContainer.h>
#include <tracking_lib/MeasurementModels/BaseMeasurementModel.h>
#include <tracking_lib/StateModels/BaseStateModel.h>
#include <tracking_lib/Distributions/GaussianDistribution.h>
#include <tracking_lib/States/Innovation.h>

#include <tracking_lib/TTBManager/TTBManager.h>

#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
#include "tracking_lib/MultiObjectStateDistributions/GLMBDistribution.h"

#include <aduulm_logger/aduulm_logger.hpp>
#include <filesystem>
#include <figcone/configreader.h>

using namespace std::literals;
std::string const params_string{ "thread_pool_size: 1\n"
                                 "version: 1.0.0\n"
                                 "meas_models:\n"
                                 "  gaussian_models:\n"
                                 "    -\n"
                                 "          id: 0\n"
                                 "          components: [POS_X, POS_Y]\n"
                                 "          can_init: true\n"
                                 "          clutter:\n"
                                 "            intensity: 1.093750000000000e-05\n"
                                 "          detection:\n"
                                 "            prob: 0.9\n"
                                 "            prob_min: 0.001\n"
                                 "          occlusion:\n"
                                 "            model: NO_OCCLUSION\n"
                                 "          default_values:\n"
                                 "            - mean:\n"
                                 "                WIDTH: 1\n"
                                 "                LENGTH: 1\n"
                                 "                HEIGHT: 1\n"
                                 "          gating_prob: 0.99\n"
                                 "    -\n"
                                 "          id: 1\n"
                                 "          components: [POS_X, POS_Y]\n"
                                 "          can_init: true\n"
                                 "          clutter:\n"
                                 "            intensity: 1e-9\n"
                                 "          detection:\n"
                                 "            prob: 0.9\n"
                                 "            prob_min: 0.001\n"
                                 "          occlusion:\n"
                                 "            model: NO_OCCLUSION\n"
                                 "          default_values:\n"
                                 "            - mean:\n"
                                 "                WIDTH: 1\n"
                                 "                LENGTH: 1\n"
                                 "                HEIGHT: 1\n"
                                 "          gating_prob: 0.99\n"
                                 "state:\n"
                                 "  multi_model:\n"
                                 "    use_state_models: [1]\n"
                                 "    birth_weights:    [1]\n"
                                 "    markov_transition:\n"
                                 "      - type: [ CAR_UNION, TRUCK_UNION, BIKE_UNION, PEDESTRIAN, UNKNOWN ]\n"
                                 "        transition_matrix: \"1\"\n"
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
                                 "  -\n"
                                 "    id: 1\n"
                                 "    type: CV\n"
                                 "    distribution:\n"
                                 "      type: GAUSSIAN\n"
                                 "      mixture: true\n"
                                 "      extraction_type: BEST_COMPONENT\n"
                                 "      post_process:\n"
                                 "        enable: true\n"
                                 "        max_components: 1000\n"
                                 "        merging_distance: 0\n"
                                 "        min_weight: 0.001\n"
                                 "        max_variance: 100\n"
                                 "    extent: NONE\n"
                                 "    model_noise_std_dev:\n"
                                 "      ACC_X: 0.2\n"
                                 "      ACC_Y: 0.2\n"
                                 "      VEL_Z: 0.001\n"
                                 "      WIDTH_CHANGE: 0.001\n"
                                 "      LENGTH_CHANGE: 0.001\n"
                                 "      HEIGHT_CHANGE: 0.001\n"
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
                                 "    type: ALL\n"
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
constexpr double TOL{ 1e-5 };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(generateHypotheses, generateHypothesesKBest)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYaml<ttb::Params>(params_string);
  auto manager = ttb::TTBManager(params);
  ttb::StateModelId stateModelId{ 1 };

  // Create Measurements for both sensors

  auto egoDist = ttb::EgoMotionDistribution::zero();

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

  state3._label = ttb::Label{ 94 };
  state3._existenceProbability = 0.049;

  ttb::LMBDistribution lmb(&manager);
  lmb._tracks = { state1, state2, state3 };

  // Save final result: This should be the result:
  // cardDist:
  ttb::Vector trueCardDist = ttb::Vector::Zero(lmb._tracks.size() + 1);
  trueCardDist(0) = 0.016298912628406919;
  trueCardDist(1) = 0.8017538776015567;
  trueCardDist(2) = 0.17505387294566899;
  trueCardDist(3) = 0.0068933368243674327;

  LOG_DEB("####################################################");
  LOG_DEB("LMB: " << lmb.toString());
  LOG_DEB("####################################################");
  manager.next_params().glmb_distribution.lmb_2_glmb_conversion.kBest.max_number_hypotheses = 8;
  manager.update_params();
  LOG_DEB("Calculate the best " << manager.params().glmb_distribution.lmb_2_glmb_conversion.kBest.max_number_hypotheses
                                << " hypotheses");
  ASSERT_EQ(3, lmb._tracks.size());
  ttb::GLMBDistribution glmbDist(&manager, lmb._tracks);
  glmbDist.generateHypotheses();
  ttb::Vector cardDist = glmbDist.cardinalityDistribution();
  LOG_DEB("GLMB dist: " << glmbDist.toString());
  ASSERT_EQ(glmbDist._hypotheses.size(), 8);
  ASSERT_EQ(cardDist.rows(), trueCardDist.rows());
  for (std::size_t counter = 0; counter < trueCardDist.rows(); counter++)
  {
    ASSERT_EQ(trueCardDist(counter), cardDist(counter));
  }

  LOG_DEB("####################################################");
  ttb::Vector trueCardDist2 = ttb::Vector::Zero(lmb._tracks.size() + 1);

  trueCardDist2(0) = 0.016359647742602047;
  trueCardDist2(1) = 0.80115643403655545;
  trueCardDist2(2) = 0.17556489455412186;
  trueCardDist2(3) = 0.0069190236667205701;
  manager.next_params().glmb_distribution.lmb_2_glmb_conversion.type = ttb::LMB_2_GLMB_CONVERISON_TYPE::K_BEST;
  manager.next_params().glmb_distribution.lmb_2_glmb_conversion.kBest.max_number_hypotheses = 5;
  manager.update_params();
  LOG_DEB("Calculate the best " << manager.params().glmb_distribution.lmb_2_glmb_conversion.kBest.max_number_hypotheses
                                << " hypotheses");
  ASSERT_EQ(3, lmb._tracks.size());
  ttb::GLMBDistribution glmbDist2(&manager, lmb._tracks);
  glmbDist2.generateHypotheses();
  cardDist = glmbDist2.cardinalityDistribution();
  LOG_DEB("GLMB dist: " << glmbDist2.toString());
  ASSERT_EQ(glmbDist2._hypotheses.size(), 5);

  ASSERT_EQ(cardDist.rows(), trueCardDist2.rows());
  for (std::size_t counter = 0; counter < trueCardDist2.rows(); counter++)
  {
    ASSERT_FLOAT_EQ(trueCardDist2(counter), cardDist(counter));
  }
}

TEST(generateHypotheses, generateHypothesesALL)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYaml<ttb::Params>(params_string);
  auto manager = ttb::TTBManager(params);
  ttb::StateModelId stateModelId{ 1 };

  // Create Measurements for both sensors

  auto egoDist = ttb::EgoMotionDistribution::zero();

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

  state3._label = ttb::Label{ 94 };
  state3._existenceProbability = 0.049;

  ttb::LMBDistribution lmb(&manager);
  lmb._tracks = { state1, state2, state3 };

  // Save final result: This should be the result:
  // cardDist:
  ttb::Vector trueCardDist = ttb::Vector::Zero(lmb._tracks.size() + 1);
  trueCardDist(0) = 0.016298912628406919;
  trueCardDist(1) = 0.8017538776015567;
  trueCardDist(2) = 0.17505387294566899;
  trueCardDist(3) = 0.0068933368243674327;

  LOG_DEB("####################################################");
  LOG_DEB("LMB: " << lmb.toString());
  LOG_DEB("####################################################");
  manager.next_params().glmb_distribution.lmb_2_glmb_conversion.type = ttb::LMB_2_GLMB_CONVERISON_TYPE::ALL;
  manager.update_params();
  ASSERT_EQ(3, lmb._tracks.size());
  ttb::GLMBDistribution glmbDist(&manager, std::move(lmb._tracks));
  glmbDist.generateHypotheses();
  ttb::Vector cardDist = glmbDist.cardinalityDistribution();
  LOG_DEB("GLMB dist: " << glmbDist.toString());
  ASSERT_EQ(glmbDist._hypotheses.size(), 8);
  ASSERT_EQ(cardDist.rows(), trueCardDist.rows());
  for (std::size_t counter = 0; counter < trueCardDist.rows(); counter++)
  {
    ASSERT_EQ(trueCardDist(counter), cardDist(counter));
  }
}
