#include "gtest/gtest.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
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
                                 "    output_state_model: 1\n"
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
                                 "  post_process_update:\n"
                                 "    enable: false\n"
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
                                 "filter:\n"
                                 "  enable: true\n"
                                 "  type: LMB_IC\n" };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(LMBUpdate, ThreeTracksSevenMeas)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYaml<ttb::Params>(params_string);
  auto manager = ttb::TTBManager<double>(params);
  ttb::SourceId measModelId{ "0" };
  ttb::SourceId measModelId2{ "1" };
  ttb::StateModelId stateModelId{ 1 };

  // Create Measurements for both sensors
  ttb::DataSourceContainer<double> measContainer{ ._source_id = measModelId, ._time = ttb::Time(0s) };

  ttb::Vector<double> meas_mean = ttb::Vector<double>::Zero(2);
  meas_mean(0) = -355.0767797404343;
  meas_mean(1) = 59.754602959654940;

  ttb::Matrix<double> meas_cov = ttb::Matrix<double>::Identity(2, 2);
  meas_cov(0, 0) = 0.36;
  meas_cov(1, 1) = 0.36;
  ttb::State<double> meas1 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean, meas_cov, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas: " + meas1.toString());
  measContainer._measurements.push_back(std::move(meas1));

  ttb::Vector<double> meas_mean2 = ttb::Vector<double>::Zero(2);
  meas_mean2(0) = -374.8856716042374;
  meas_mean2(1) = -363.3953851459543;

  ttb::Matrix<double> meas_cov2 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov2(0, 0) = 0.36;
  meas_cov2(1, 1) = 0.36;
  ttb::State<double> meas2 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean2, meas_cov2, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas2: " + meas2.toString());
  measContainer._measurements.push_back(std::move(meas2));

  ttb::Vector<double> meas_mean3 = ttb::Vector<double>::Zero(2);
  meas_mean3(0) = 369.8213293298233;
  meas_mean3(1) = 364.7892513727658;

  ttb::Matrix<double> meas_cov3 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov3(0, 0) = 0.36;
  meas_cov3(1, 1) = 0.36;
  ttb::State<double> meas3 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean3, meas_cov3, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas3: " + meas3.toString());
  measContainer._measurements.push_back(std::move(meas3));

  ttb::Vector<double> meas_mean4 = ttb::Vector<double>::Zero(2);
  meas_mean4(0) = 218.8506641642635;
  meas_mean4(1) = -377.6936775951114;

  ttb::Matrix<double> meas_cov4 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov4(0, 0) = 0.36;
  meas_cov4(1, 1) = 0.36;
  ttb::State<double> meas4 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean4, meas_cov4, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas4: " + meas4.toString());
  measContainer._measurements.push_back(std::move(meas4));

  ttb::Vector<double> meas_mean5 = ttb::Vector<double>::Zero(2);
  meas_mean5(0) = -223.0507033343160;
  meas_mean5(1) = -70.319093015156910;

  ttb::Matrix<double> meas_cov5 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov5(0, 0) = 0.36;
  meas_cov5(1, 1) = 0.36;
  ttb::State<double> meas5 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean5, meas_cov5, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas5: " + meas5.toString());
  measContainer._measurements.push_back(std::move(meas5));

  ttb::Vector<double> meas_mean6 = ttb::Vector<double>::Zero(2);
  meas_mean6(0) = 250.7703933647325;
  meas_mean6(1) = 330.6491491092934;

  ttb::Matrix<double> meas_cov6 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov6(0, 0) = 0.36;
  meas_cov6(1, 1) = 0.36;
  ttb::State<double> meas6 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean6, meas_cov6, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas6: " + meas6.toString());
  measContainer._measurements.push_back(std::move(meas6));

  ttb::Vector<double> meas_mean7 = ttb::Vector<double>::Zero(2);
  meas_mean7(0) = -495.1091774530706;
  meas_mean7(1) = 324.2468990386059;

  ttb::Matrix<double> meas_cov7 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov7(0, 0) = 0.36;
  meas_cov7(1, 1) = 0.36;
  ttb::State meas7 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean7, meas_cov7, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas7: " + meas7.toString());
  measContainer._measurements.push_back(std::move(meas7));

  ttb::DataSourceContainer<double> measContainer2{ ._source_id = measModelId, ._time = ttb::Time(0s) };

  ttb::Vector<double> meas_mean11 = ttb::Vector<double>::Zero(2);
  meas_mean11(0) = -354.3802376945775;
  meas_mean11(1) = 59.223113382786950;

  ttb::Matrix<double> meas_cov11 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov11(0, 0) = 0.36;
  meas_cov11(1, 1) = 0.36;
  ttb::State meas11 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean11, meas_cov11, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas11: " + meas11.toString());
  measContainer2._measurements.push_back(std::move(meas11));

  ttb::Vector<double> meas_mean21 = ttb::Vector<double>::Zero(2);
  meas_mean21(0) = -376.4643400815579;
  meas_mean21(1) = -364.5223246281104;

  ttb::Matrix<double> meas_cov21 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov21(0, 0) = 0.36;
  meas_cov21(1, 1) = 0.36;
  ttb::State meas21 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean21, meas_cov21, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas21: " + meas21.toString());
  measContainer2._measurements.push_back(std::move(meas21));

  ttb::Vector<double> meas_mean31 = ttb::Vector<double>::Zero(2);
  meas_mean31(0) = -369.6716417509368;
  meas_mean31(1) = 364.2738470493076;

  ttb::Matrix<double> meas_cov31 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov31(0, 0) = 0.36;
  meas_cov31(1, 1) = 0.36;
  ttb::State meas31 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean31, meas_cov31, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas31: " + meas31.toString());
  measContainer2._measurements.push_back(std::move(meas31));

  ttb::Vector<double> meas_mean41 = ttb::Vector<double>::Zero(2);
  meas_mean41(0) = 218.9385435521861;
  meas_mean41(1) = -376.4739591365427;

  ttb::Matrix<double> meas_cov41 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov41(0, 0) = 0.36;
  meas_cov41(1, 1) = 0.36;
  ttb::State meas41 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean41, meas_cov41, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas41: " + meas41.toString());
  measContainer2._measurements.push_back(std::move(meas41));

  ttb::Vector<double> meas_mean51 = ttb::Vector<double>::Zero(2);
  meas_mean51(0) = -389.2415677580803;
  meas_mean51(1) = -256.7461047486872;

  ttb::Matrix<double> meas_cov51 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov51(0, 0) = 0.36;
  meas_cov51(1, 1) = 0.36;
  ttb::State meas51 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean51, meas_cov51, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas51: " + meas51.toString());
  measContainer2._measurements.push_back(std::move(meas51));

  ttb::Vector<double> meas_mean61 = ttb::Vector<double>::Zero(2);
  meas_mean61(0) = -313.1757574289825;
  meas_mean61(1) = 161.9487826498748;

  ttb::Matrix<double> meas_cov61 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov61(0, 0) = 0.36;
  meas_cov61(1, 1) = 0.36;
  ttb::State meas61 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean61, meas_cov61, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas61: " + meas61.toString());
  measContainer2._measurements.push_back(std::move(meas61));

  ttb::Vector<double> meas_mean71 = ttb::Vector<double>::Zero(2);
  meas_mean71(0) = -259.0108081058011;
  meas_mean71(1) = -427.9420119770139;

  ttb::Matrix<double> meas_cov71 = ttb::Matrix<double>::Identity(2, 2);
  meas_cov71(0, 0) = 0.36;
  meas_cov71(1, 1) = 0.36;
  ttb::State meas71 =
      ttb::create_meas_state(&manager,
                             ttb::GaussianDistribution(meas_mean71, meas_cov71, ttb::REFERENCE_POINT::CENTER),
                             ttb::Time(0s),
                             ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }))
          .value();
  LOG_DEB("Meas71: " + meas71.toString());
  measContainer2._measurements.push_back(std::move(meas71));

  // create 3 tracks
  auto const& state_model = manager.getStateModel(stateModelId);
  // track1

  ttb::Vector<double> x = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x(0) = -376.0657882526539;  // x
  x(1) = -363.8193668827842;  // y
  x(2) = 0;                   // z
  x(3) = 3.969745495699935;   // vx
  x(4) = 5.970312638752488;   // vy

  ttb::Matrix<double> P =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P(0, 0) = 0.403281171794258;            // x,x
  P(1, 1) = 0.403281171794258;            // y,y
  P(2, 2) = 0.001;                        // z,z
  P(3, 3) = 0.116241459088294;            // vx, vx
  P(4, 4) = 0.116241459088294;            // vy, vy
  P(0, 3) = P(3, 0) = 0.172543791141602;  // x,vx
  P(1, 4) = P(4, 1) = 0.172543791141602;  // y,vy

  ttb::BernoulliState state1 = {
    ttb::State(&manager, { { stateModelId, { { 1.0, ttb::GaussianDistribution(x, P) } } } }), 0.979988592142434
  };
  state1.state._label = ttb::Label{ 58 };

  // track2

  ttb::Vector<double> x2 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x2(0) = -400;  // x
  x2(1) = -400;  // y
  x2(2) = 0;     // z
  x2(3) = 0;     // vx
  x2(4) = 0;     // vy

  ttb::Matrix<double> P2 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P2(0, 0) = 325.1;                          // x,x
  P2(1, 1) = 325.1;                          // y,y
  P2(2, 2) = 0.001;                          // z,z
  P2(3, 3) = 25.08;                          // vx, vx
  P2(4, 4) = 25.08;                          // vy, vy
  P2(0, 3) = P2(3, 0) = 50.080000000000005;  // x,vx
  P2(1, 4) = P2(4, 1) = 50.080000000000005;  // y,vy

  ttb::Vector<double> x21 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x21(0) = -397.0548501344673;  // x
  x21(1) = -421.8671020878899;  // y
  x21(2) = 0;                   // z
  x21(3) = 0.267925861308324;   // vx
  x21(4) = -1.989291692684458;  // vy

  ttb::Matrix<double> P21 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P21(0, 0) = 22.981132324160246;              // x,x
  P21(1, 1) = 22.981132324160246;              // y,y
  P21(2, 2) = 0.001;                           // z,z
  P21(3, 3) = 22.579698845708350;              // vx, vx
  P21(4, 4) = 22.579698845708350;              // vy, vy
  P21(0, 3) = P21(3, 0) = 22.595674401885210;  // x,vx
  P21(1, 4) = P21(4, 1) = 22.595674401885210;  // y,vy

  ttb::Vector<double> x22 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x22(0) = -418.4900894148190;  // x
  x22(1) = -400;                // y
  x22(2) = 0;                   // z
  x22(3) = -1.682078453836929;  // vx
  x22(4) = -2.350588777203333;  // vy

  ttb::Matrix<double> P22 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P22(0, 0) = 22.981132324160246;              // x,x
  P22(1, 1) = 22.981132324160246;              // y,y
  P22(2, 2) = 0.001;                           // z,z
  P22(3, 3) = 22.579698845708350;              // vx, vx
  P22(4, 4) = 22.579698845708350;              // vy, vy
  P22(0, 3) = P22(3, 0) = 22.595674401885210;  // x,vx
  P22(1, 4) = P22(4, 1) = 22.595674401885210;  // y,vy

  ttb::BernoulliState state2 = { ttb::State(&manager,
                                            { { stateModelId,
                                                { { 0.003002215714468, ttb::GaussianDistribution(x2, P2) },
                                                  { 0.701594427037742, ttb::GaussianDistribution(x21, P21) },
                                                  { 0.295403357247791, ttb::GaussianDistribution(x22, P22) } } } }),
                                 0.143553041823563 };

  state2.state._label = ttb::Label{ 88 };

  // track3

  ttb::Vector<double> x3 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x3(0) = -400.0;  // x
  x3(1) = -400.0;  // y
  x3(2) = 0;       // z
  x3(3) = 0;       // vx
  x3(4) = 0;       // vy

  ttb::Matrix<double> P3 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P3(0, 0) = 250.01;            // x,x
  P3(1, 1) = 250.01;            // y,y
  P3(2, 2) = 0.001;             // z,z
  P3(3, 3) = 25.04;             // vx, vx
  P3(4, 4) = 25.04;             // vy, vy
  P3(0, 3) = P3(3, 0) = 25.02;  // x,vx
  P3(1, 4) = P3(4, 1) = 25.02;  // y,vy

  ttb::BernoulliState state3 = {
    ttb::State(&manager, { { stateModelId, { { 1.0, ttb::GaussianDistribution(x3, P3) } } } }), 0.049
  };
  state3.state._label = ttb::Label{ 94 };

  ttb::LMBDistribution lmb(&manager);
  lmb.add_tracks({ state1, state2, state3 });
  assert(lmb._tracks.size() == 3);
  // Save final result: This should be the result:
  //  double r1 = 0.999996521733549;
  std::vector<double> r_vec;
  std::vector<ttb::Label> label_vec;
  label_vec.push_back(ttb::Label{ 58 });
  label_vec.push_back(ttb::Label{ 88 });
  label_vec.push_back(ttb::Label{ 94 });

  double r1 = 0.99999654154702844;

  ttb::Vector<double> mean_track1_final_true = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  mean_track1_final_true(0) = -3.760657882526539e+02;
  mean_track1_final_true(1) = -3.638193668827842e+02;
  mean_track1_final_true(2) = 0;
  mean_track1_final_true(3) = 3.969745495699935;
  mean_track1_final_true(4) = 5.970312638752488;

  ttb::Vector<double> mean_track11_final_true = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  mean_track11_final_true(0) = -3.754422712023007e+02;
  mean_track11_final_true(1) = -3.638193668827842e+02;
  mean_track11_final_true(2) = 0;
  mean_track11_final_true(3) = 4.236517175693638;
  mean_track11_final_true(4) = 6.066155977944862;

  ttb::Matrix<double> cov_track1_final_true =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track1_final_true(0, 0) = 0.403281171794258;
  cov_track1_final_true(1, 1) = 0.403281171794258;
  cov_track1_final_true(2, 2) = 1;
  cov_track1_final_true(3, 3) = 0.116241459088294;
  cov_track1_final_true(4, 4) = 0.116241459088294;
  cov_track1_final_true(0, 3) = cov_track1_final_true(3, 0) = 0.172543791141602;  // x,vx
  cov_track1_final_true(1, 4) = cov_track1_final_true(4, 1) = 0.172543791141602;  // y,vy

  ttb::Matrix<double> cov_track11_final_true =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
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
  ttb::Vector<double> mean_track2_final_true = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  mean_track2_final_true(0) = -400;
  mean_track2_final_true(1) = -400;
  mean_track2_final_true(2) = 0;
  mean_track2_final_true(3) = 0;
  mean_track2_final_true(4) = 0;

  ttb::Vector<double> mean_track21_final_true = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  mean_track21_final_true(0) = -3.970548501344673e+02;
  mean_track21_final_true(1) = -4.218671020878899e+02;
  mean_track21_final_true(2) = 0;
  mean_track21_final_true(3) = 0.267925861308324;
  mean_track21_final_true(4) = -1.989291692684458;

  ttb::Vector<double> mean_track22_final_true = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  mean_track22_final_true(0) = -4.184900894148190e+02;
  mean_track22_final_true(1) = -4.258386263546855e+02;
  mean_track22_final_true(2) = 0;
  mean_track22_final_true(3) = -1.682078453836929;
  mean_track22_final_true(4) = -2.350588777203333;

  ttb::Vector<double> mean_track23_final_true = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  mean_track23_final_true(0) = -3.749134512337540e+02;
  mean_track23_final_true(1) = -3.634358744882620e+02;
  mean_track23_final_true(2) = 0;
  mean_track23_final_true(3) = 3.864455128310063;
  mean_track23_final_true(4) = 5.632517396579028;

  ttb::Matrix<double> cov_track2_final_true =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track2_final_true(0, 0) = 325.1;
  cov_track2_final_true(1, 1) = 325.1;
  cov_track2_final_true(2, 2) = 1;
  cov_track2_final_true(3, 3) = 25.08;
  cov_track2_final_true(4, 4) = 25.08;
  cov_track2_final_true(0, 3) = cov_track2_final_true(3, 0) = 50.08;  // x,vx
  cov_track2_final_true(1, 4) = cov_track2_final_true(4, 1) = 50.08;  // y,vy

  ttb::Matrix<double> cov_track21_final_true =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track21_final_true(0, 0) = 22.981132324160246;
  cov_track21_final_true(1, 1) = 22.981132324160246;
  cov_track21_final_true(2, 2) = 1;
  cov_track21_final_true(3, 3) = 22.579698845708350;
  cov_track21_final_true(4, 4) = 22.579698845708350;
  cov_track21_final_true(0, 3) = cov_track21_final_true(3, 0) = 22.595674401885210;  // x,vx
  cov_track21_final_true(1, 4) = cov_track21_final_true(4, 1) = 22.595674401885210;  // y,vy

  ttb::Matrix<double> cov_track22_final_true =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track22_final_true(0, 0) = 22.981132324160246;
  cov_track22_final_true(1, 1) = 22.981132324160246;
  cov_track22_final_true(2, 2) = 1;
  cov_track22_final_true(3, 3) = 22.579698845708350;
  cov_track22_final_true(4, 4) = 22.579698845708350;
  cov_track22_final_true(0, 3) = cov_track22_final_true(3, 0) = 22.595674401885210;  // x,vx
  cov_track22_final_true(1, 4) = cov_track22_final_true(4, 1) = 22.595674401885210;  // y,vy

  ttb::Matrix<double> cov_track23_final_true =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
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

  double r3 = 0.0051270640967412372;
  ttb::Vector<double> mean_track3_final_true = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  mean_track3_final_true(0) = -400;
  mean_track3_final_true(1) = -400;
  mean_track3_final_true(2) = 0;
  mean_track3_final_true(3) = 0;
  mean_track3_final_true(4) = 0;

  ttb::Vector<double> mean_track31_final_true = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  mean_track31_final_true(0) = -3.749217827925685e+02;
  mean_track31_final_true(1) = -3.634480178948757e+02;
  mean_track31_final_true(2) = 0;
  mean_track31_final_true(3) = 2.509727589016181;
  mean_track31_final_true(4) = 3.657976050038840;

  ttb::Matrix<double> cov_track3_final_true =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  cov_track3_final_true(0, 0) = 250.01;
  cov_track3_final_true(1, 1) = 250.01;
  cov_track3_final_true(2, 2) = 1;
  cov_track3_final_true(3, 3) = 25.04;
  cov_track3_final_true(4, 4) = 25.04;
  cov_track3_final_true(0, 3) = cov_track3_final_true(3, 0) = 25.02;  // x,vx
  cov_track3_final_true(1, 4) = cov_track3_final_true(4, 1) = 25.02;  // y,vy

  ttb::Matrix<double> cov_track31_final_true =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
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
  LOG_DEB("LMB: " << lmb.to_string());
  LOG_DEB("####################################################");
  LOG_DEB("Measurements: " << measContainer.toString());
  LOG_DEB("####################################################");
  LOG_DEB("LMB after calcInnovation" << lmb.to_string());
  LOG_DEB("####################################################");
  assert(lmb._tracks.size() == 3);
  lmb.update(measContainer);
  assert(lmb._tracks.size() == 3);
  LOG_DEB("LMB after first update" << lmb.to_string());
  LOG_DEB("####################################################");

  // compare results
  unsigned int counter = 0.;
  std::stringstream rString;
  for (const auto& tmp : r_vec)
  {
    rString << tmp << " ";
  }
  rString << "\n";
  LOG_DEB("r_vec: " << rString.str());
  std::ranges::sort(lmb._tracks, {}, [&](ttb::StateId id) { return manager.get_state_existence_probability(id); });
  std::ranges::sort(r_vec);
  for (ttb::StateId id : lmb._tracks)
  {
    ASSERT_NEAR(r_vec.at(counter), manager.get_state_existence_probability(id), 1e-5);
    counter++;
  }
  ASSERT_EQ(3, lmb._tracks.size());
}

TEST(LMBUpdate, NoMeasurementsGiven)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);
  auto paramReader = figcone::ConfigReader{};
  auto params = paramReader.readYaml<ttb::Params>(params_string);
  auto manager = ttb::TTBManager<double>(params);
  ttb::SourceId measModelId{ "0" };
  ttb::SourceId measModelId2{ "1" };
  ttb::StateModelId stateModelId{ 1 };

  ttb::DataSourceContainer<double> measContainer{ ._source_id = measModelId, ._time = ttb::Time(0s) };

  ttb::DataSourceContainer<double> measContainer2{ ._source_id = measModelId, ._time = ttb::Time(0s) };

  // create 3 tracks
  auto const& state_model = manager.getStateModel(stateModelId);
  // track1

  ttb::Vector<double> x = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x(0) = -376;  // x
  x(1) = -363;  // y
  x(2) = 0;     // z
  x(3) = 3.97;  // vx
  x(4) = 5.97;  // vy

  ttb::Matrix<double> P =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P(0, 0) = 0.4;             // x,x
  P(1, 1) = 0.4;             // y,y
  P(2, 2) = 0.001;           // z,z
  P(3, 3) = 0.12;            // vx, vx
  P(4, 4) = 0.12;            // vy, vy
  P(0, 3) = P(3, 0) = 0.17;  // x,vx
  P(1, 4) = P(4, 1) = 0.17;  // y,vy

  ttb::BernoulliState state1 = {
    ttb::State(&manager, { { stateModelId, { { 1.0, ttb::GaussianDistribution(x, P) } } } }), 0.98
  };
  state1.state._label = ttb::Label{ 58 };

  // track2

  ttb::Vector<double> x2 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x2(0) = -400;  // x
  x2(1) = -400;  // y
  x2(2) = 0;     // z
  x2(3) = 0;     // vx
  x2(4) = 0;     // vy

  ttb::Matrix<double> P2 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P2(0, 0) = 325.1;             // x,x
  P2(1, 1) = 325.1;             // y,y
  P2(2, 2) = 0.001;             // z,z
  P2(3, 3) = 25.08;             // vx, vx
  P2(4, 4) = 25.08;             // vy, vy
  P2(0, 3) = P2(3, 0) = 50.08;  // x,vx
  P2(1, 4) = P2(4, 1) = 50.08;  // y,vy

  ttb::Vector<double> x21 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x21(0) = -397.05;  // x
  x21(1) = -421.87;  // y
  x21(2) = 0;        // z
  x21(3) = 0.27;     // vx
  x21(4) = -1.99;    // vy

  ttb::Matrix<double> P21 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P21(0, 0) = 22.98;             // x,x
  P21(1, 1) = 22.98;             // y,y
  P21(2, 2) = 0.001;             // z,z
  P21(3, 3) = 22.58;             // vx, vx
  P21(4, 4) = 22.58;             // vy, vy
  P21(0, 3) = P21(3, 0) = 22.6;  // x,vx
  P21(1, 4) = P21(4, 1) = 22.6;  // y,vy

  ttb::Vector<double> x22 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x22(0) = -418.49;  // x
  x22(1) = -400;     // y
  x22(2) = 0;        // z
  x22(3) = -1.68;    // vx
  x22(4) = -2.35;    // vy

  ttb::Matrix<double> P22 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P22(0, 0) = 22.98;             // x,x
  P22(1, 1) = 22.98;             // y,y
  P22(2, 2) = 0.001;             // z,z
  P22(3, 3) = 22.58;             // vx, vx
  P22(4, 4) = 22.58;             // vy, vy
  P22(0, 3) = P22(3, 0) = 22.6;  // x,vx
  P22(1, 4) = P22(4, 1) = 22.6;  // y,vy

  ttb::BernoulliState state2 = { ttb::State(&manager,
                                            { { stateModelId,
                                                { { 0.003, ttb::GaussianDistribution(x2, P2) },
                                                  { 0.7, ttb::GaussianDistribution(x21, P21) },
                                                  { 0.003, ttb::GaussianDistribution(x22, P22) } } } }),
                                 0.14 };
  state2.state._label = ttb::Label{ 88 };

  ttb::Vector<double> x3 = ttb::Vector<double>::Zero(state_model.state_comps()._comps.size());
  x3(0) = -400.0;  // x
  x3(1) = -400.0;  // y
  x3(2) = 0;       // z
  x3(3) = 0;       // vx
  x3(4) = 0;       // vy

  ttb::Matrix<double> P3 =
      ttb::Matrix<double>::Zero(state_model.state_comps()._comps.size(), state_model.state_comps()._comps.size());
  P3(0, 0) = 250.01;            // x,x
  P3(1, 1) = 250.01;            // y,y
  P3(2, 2) = 0.001;             // z,z
  P3(3, 3) = 25.04;             // vx, vx
  P3(4, 4) = 25.04;             // vy, vy
  P3(0, 3) = P3(3, 0) = 25.02;  // x,vx
  P3(1, 4) = P3(4, 1) = 25.02;  // y,vy

  ttb::BernoulliState state3 = {
    ttb::State(&manager, { { stateModelId, { { 1.0, ttb::GaussianDistribution(x3, P3) } } } }), 0.049
  };
  state3.state._label = ttb::Label{ 94 };

  ttb::LMBDistribution lmb(&manager);
  lmb.add_tracks({ state1, state2, state3 });
  std::vector<double> r_vec;
  double r1 = 0.83050847457;
  double r2 = 0.01601830663;
  double r3 = 0.00512605921;
  r_vec.push_back(r1);
  r_vec.push_back(r2);
  r_vec.push_back(r3);

  lmb.update(measContainer);

  // compare results
  unsigned int counter = 0.;
  std::ranges::sort(lmb._tracks, {}, [&](ttb::StateId id) { return manager.get_state_existence_probability(id); });
  std::ranges::sort(r_vec);
  for (ttb::StateId id : lmb._tracks)
  {
    ttb::State<double> const& track = *manager.database().states.find(id);

    ASSERT_NEAR(r_vec[counter], manager.get_state_existence_probability(id), 1e-4);
    counter++;
    // check results
    if (track._label == ttb::Label{ 58 })
    {
      LOG_ERR(track.toString());

      ASSERT_EQ(track.getEstimate().second.mean(), x);
      ASSERT_EQ(track.getEstimate().second.covariance(), P);
    }
    if (track._label == ttb::Label{ 88 })
    {
      LOG_ERR(track.toString());
      ASSERT_EQ(track.getEstimate().second.mean(), x21);
      ASSERT_EQ(track.getEstimate().second.covariance(), P21);
    }
    if (track._label == ttb::Label{ 94 })
    {
      LOG_ERR(track.toString());
      ASSERT_EQ(track.getEstimate().second.mean(), x3);
      ASSERT_EQ(track.getEstimate().second.covariance(), P3);
    }
  }

  LOG_DEB("LMB after calcInnovation" << lmb.to_string());
  LOG_DEB("####################################################");

  lmb.update(measContainer2);

  LOG_DEB("LMB after second update" << lmb.to_string());
  LOG_DEB("####################################################");

  // check results
  for (const auto& id : lmb._tracks)
  {
    ttb::State<double> const& track = *manager.database().states.find(id);

    if (track._label == ttb::Label{ 58 })
    {
      LOG_ERR(track.toString());

      ASSERT_EQ(track.getEstimate().second.mean(), x);
      ASSERT_EQ(track.getEstimate().second.covariance(), P);
    }
    if (track._label == ttb::Label{ 88 })
    {
      LOG_ERR(track.toString());
      ASSERT_EQ(track.getEstimate().second.mean(), x21);
      ASSERT_EQ(track.getEstimate().second.covariance(), P21);
    }
    if (track._label == ttb::Label{ 94 })
    {
      LOG_ERR(track.toString());
      ASSERT_EQ(track.getEstimate().second.mean(), x3);
      ASSERT_EQ(track.getEstimate().second.covariance(), P3);
    }
  }

  ASSERT_EQ(3, lmb._tracks.size());
}
