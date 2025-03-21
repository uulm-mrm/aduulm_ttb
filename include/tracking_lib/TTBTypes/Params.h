#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include <figcone/configreader.h>

/// ALL! parameters for the whole library are defined here.
/// The structure of the params must reflect the structure of the yaml where they are specified.
/// The top level struct is defined at the bottom and called Params ....

namespace ttb
{

auto constexpr param_version{ "1.0.0" };

struct ClutterEstimationParams
{
  bool enable = true;          ///< enable the clutter rate estimation
  bool use_in_filter = false;  ///< use the estimated value in the filter
  double prior_mean =
      std::numeric_limits<double>::quiet_NaN();  ///< prior mean of the gamma distribution describing the clutter rate
  double prior_var = std::numeric_limits<double>::quiet_NaN();  ///< prior variance of the gamma distribution describing
                                                                ///< the clutter rate
  using traits = figcone::FieldTraits<figcone::OptionalField<&ClutterEstimationParams::enable>,
                                      figcone::OptionalField<&ClutterEstimationParams::use_in_filter>,
                                      figcone::OptionalField<&ClutterEstimationParams::prior_mean>,
                                      figcone::OptionalField<&ClutterEstimationParams::prior_var>>;
};

struct ClutterParams
{
  // you can specify either the intensity OR the rate
  std::optional<double> intensity{};  ///< the clutter intensity
  std::optional<double> rate{};       ///< the clutter rate, i.e., intensity*area
  ClutterEstimationParams rate_estimation{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&ClutterParams::intensity>,
                                      figcone::OptionalField<&ClutterParams::rate>,
                                      figcone::OptionalField<&ClutterParams::rate_estimation>>;
};

struct ClutterParams_Validator
{
  void operator()(ClutterParams const& params)
  {
    if (params.intensity.has_value() + params.rate.has_value() != 1)
    {
      throw figcone::ValidationError("You must specify either the clutter rate OR the clutter intensity");
    }
  }
};

struct DetectionProbEstimationParams
{
  bool enable = true;          ///< enable the detection probability estimation
  bool use_in_filter = false;  ///< use the estimated detection probability in the filter
  double prior_mean = std::numeric_limits<double>::quiet_NaN();  ///< prior mean of the Beta distribution describing the
                                                                 ///< detection probability
  double prior_var = std::numeric_limits<double>::quiet_NaN();   ///< prior variance of the Beta distribution describing
                                                                 ///< the detection probability
  using traits = figcone::FieldTraits<figcone::OptionalField<&DetectionProbEstimationParams::enable>,
                                      figcone::OptionalField<&DetectionProbEstimationParams::use_in_filter>,
                                      figcone::OptionalField<&DetectionProbEstimationParams::prior_mean>,
                                      figcone::OptionalField<&DetectionProbEstimationParams::prior_var>>;
};

struct DetectionParams
{
  double prob = 0.6;       ///< the (default) detection prob (inside the fov)
  double prob_min = 0.01;  ///< the detection prob used when the (default) prob does not apply
  DetectionProbEstimationParams prob_estimation{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&DetectionParams::prob>,
                                      figcone::OptionalField<&DetectionParams::prob_min>,
                                      figcone::OptionalField<&DetectionParams::prob_estimation>>;
};

struct OcclusionModelParams
{
  OCCLUSION_MODEL_TYPE model = OCCLUSION_MODEL_TYPE::NO_OCCLUSION;  ///< the type of occlusion model
  using traits = figcone::FieldTraits<figcone::OptionalField<&OcclusionModelParams::model>>;
};

struct FixOrientationParamsWeightsValidator
{
  void operator()(std::vector<double> const& weights)
  {
    if (weights.size() != 2 or std::abs(weights.at(0) + weights.at(1) - 1) > 1e-5)
    {
      std::string out = "fix_orientation weights must has size 2 and sum to 1 but has size " +
                        std::to_string(weights.size()) + " and sum " + std::to_string(weights.at(0) + weights.at(1)) +
                        "\nWeights:";
      for (auto const& val : weights)
      {
        out += std::to_string(val) + " ";
      }
      throw figcone::ValidationError(out);
    }
  }
};

struct FixOrientationParams
{
  bool enable = false;  ///< Enable the fix_orientation step: This tries to correct measurement which measures the
                        ///< orientation of an object that are 180° flipped. Therefore it creates a measurement mixture
                        ///< where one component is the flipped of the other
  std::vector<double> weights;  ///< the weights of the measurement mixture components. The first entry is
                                ///< the original measurement from the sensor, the second the flipped one
  using traits = figcone::FieldTraits<
      figcone::OptionalField<&FixOrientationParams::enable>,
      figcone::ValidatedField<&FixOrientationParams::weights, FixOrientationParamsWeightsValidator>>;
};

struct TypeConstraints  ///< state constraints for objects (currently only applied at the birth)
{
  CLASS type;
  std::map<std::string, double> max_vals;
};

struct DefaultValuesParams
{
  std::vector<CLASS> type;             ///< apply to measurements of this classes
  std::map<std::string, double> mean;  ///< the default mean values for this class
  std::map<std::string, double> var;   ///< If you need a variance for some component you can not derive, use
                                       ///< these values (string must match a value in COMPONENT_2_STRING)
  using traits =
      figcone::FieldTraits<figcone::OptionalField<&DefaultValuesParams::type>,  ///< use all classes specified in the
                                                                                ///< state as default
                           figcone::OptionalField<&DefaultValuesParams::mean>,
                           figcone::OptionalField<&DefaultValuesParams::var>>;
};

/// The Parameters used in the GaussianMeasurementModel
struct GaussianMeasurementModelParams
{
  MeasModelId id;                     ///< the unique ID of this Model
  bool enable = true;                 ///< use input form this measurement model or not?
  std::vector<COMPONENT> components;  ///< the components this measurement Model considers
  bool filter_outside_fov = true;     ///< filter measurements which are outside the fov of this sensor

  DetectionParams detection{};  ///< parameters describing the detection of objects

  OcclusionModelParams occlusion{};  ///< the occlusion model, NO_OCCLUSION is transparent

  ClutterParams clutter{};  ///< parameters describing the clutter process

  bool can_init = true;  ///< can this model initiate new tracks ?

  bool overwrite_meas_vars =
      false;  ///< If true, overwrite the variances of a measurement by the default_var specified below

  std::vector<DefaultValuesParams> default_values = {};

  double gating_prob = 0.99;  ///< confidence probability for the measurement inside the gating

  FixOrientationParams fix_orientation{};  ///< treat a measurement as mixture where one component has orientation
                                           ///< flipped
  bool force_estimate_rp =
      false;  ///< ignore reference point of measurement and infer based on position/orientation of track
  bool consider_inverse_rp = false;
  bool mult_hyp_rp_estimation = false;  ///< try different (nearest 3) reference points
  bool check_constraints = false;       ///< check constraints in the initialization for new tracks
  std::vector<TypeConstraints> constraints{};
  bool estimate_edge_rps = false;
  bool check_matching_classification_init2 =
      false;  ///< Check matching classification between old and new Measurement inside init2
  bool check_matching_reference_point_init2 =
      true;  ///< Check matching reference point between old and new Measurement inside init2

  using traits = figcone::FieldTraits<
      figcone::OptionalField<&GaussianMeasurementModelParams::fix_orientation>,
      figcone::OptionalField<&GaussianMeasurementModelParams::force_estimate_rp>,
      figcone::OptionalField<&GaussianMeasurementModelParams::consider_inverse_rp>,
      figcone::OptionalField<&GaussianMeasurementModelParams::mult_hyp_rp_estimation>,
      figcone::OptionalField<&GaussianMeasurementModelParams::check_constraints>,
      figcone::OptionalField<&GaussianMeasurementModelParams::constraints>,
      figcone::OptionalField<&GaussianMeasurementModelParams::estimate_edge_rps>,
      figcone::OptionalField<&GaussianMeasurementModelParams::check_matching_classification_init2>,
      figcone::OptionalField<&GaussianMeasurementModelParams::check_matching_reference_point_init2>,
      figcone::OptionalField<&GaussianMeasurementModelParams::enable>,
      figcone::OptionalField<&GaussianMeasurementModelParams::occlusion>,
      figcone::OptionalField<&GaussianMeasurementModelParams::detection>,
      figcone::OptionalField<&GaussianMeasurementModelParams::clutter>,
      figcone::OptionalField<&GaussianMeasurementModelParams::can_init>,
      figcone::OptionalField<&GaussianMeasurementModelParams::overwrite_meas_vars>,
      figcone::OptionalField<&GaussianMeasurementModelParams::default_values>,
      figcone::CopyNodeListField<&GaussianMeasurementModelParams::default_values>,  ///< copy unspecified entries of the
                                                                                    ///< second and following elements
                                                                                    ///< from previous first entry
      figcone::OptionalField<&GaussianMeasurementModelParams::filter_outside_fov>,
      figcone::OptionalField<&GaussianMeasurementModelParams::gating_prob>,
      figcone::ValidatedField<&GaussianMeasurementModelParams::clutter, ClutterParams_Validator>>;
};

struct GaussianMeasurementModelParams_Validator
{
  void operator()(std::vector<GaussianMeasurementModelParams> const& params)
  {
    for (GaussianMeasurementModelParams const& param : params)
    {
      if (param.components.empty())
      {
        throw figcone::ValidationError("No components for measurement model " + param.id.value_ + " specified");
      }
    }
  }
};

struct MeasurementModelsParams
{
  std::vector<GaussianMeasurementModelParams> gaussian_models{};
  using traits = figcone::FieldTraits<
      figcone::OptionalField<&MeasurementModelsParams::gaussian_models>,
      figcone::ValidatedField<&MeasurementModelsParams::gaussian_models, GaussianMeasurementModelParams_Validator>>;
};

struct ClassificationParams
{
  std::vector<MeasModelId> use_meas_models{};  ///< Use only this Measurement Models for classification
  std::vector<CLASS> classes{};    ///< Use this Classes, you can disable state classification by setting the classes to
                                   ///< NOT_CLASSIFIED
  bool use_union_classes = false;  ///< unify the measured classes of some input
  double discount_factor_prediction = 0.98;
};

struct EstimationParams
{
  bool perform_prediction = false;  ///< predict the estimation to the current cycle time
  bool transform_output_state;      ///< Transform the estimated tracks to this state Model
  StateModelId output_state_model;  ///< Must be one of the State Models used by the State Distribution
  STATE_DISTRIBUTION_EXTRACTION_TYPE type =
      STATE_DISTRIBUTION_EXTRACTION_TYPE::BEST_STATE_MODEL;  ///< How to extract from possible multiple State Models?
  using traits = figcone::FieldTraits<figcone::OptionalField<&EstimationParams::perform_prediction>>;
};

struct MarkovTransitionParams
{
  std::vector<CLASS> type;
  Matrix transition_matrix;
};

struct MultiModelParams
{
  std::vector<StateModelId> use_state_models;
  std::vector<double> birth_weights{};
  bool enable_markov_transition = false;
  std::vector<MarkovTransitionParams> markov_transition = {};
  using traits = figcone::FieldTraits<figcone::OptionalField<&MultiModelParams::enable_markov_transition>,
                                      figcone::OptionalField<&MultiModelParams::markov_transition>,
                                      figcone::OptionalField<&MultiModelParams::birth_weights>>;
};

struct MultiModelParamsValidator
{
  void operator()(MultiModelParams const& params)
  {
    if (std::abs(std::accumulate(params.birth_weights.begin(), params.birth_weights.end(), 0.0) - 1) > 1e-7)
    {
      throw figcone::ValidationError("Birth weights does not sum up to 1.");
    }
  }
};
struct TrackHistoryThreshold
{  ///< The track history logic uses the recent history of assignments or misses
  ///< to determines the stage of the state/track
  std::size_t M;
  std::size_t N;
};

struct StageManagementParams
{
  bool use_history_based_stage_logic;
  TrackHistoryThreshold first_stage_confirmation_history_threshold;  ///< if in M out of the last N updates a
  ///< measurement is assigned to the tentative
  ///< track, a track is moved from tentative to
  ///< preliminary, else it is deleted
  TrackHistoryThreshold second_stage_confirmation_history_threshold;  ///< if in M out of the last N updates a
  ///< measurement is assigned to the preliminary
  ///< track, a track is confirmed, else it is
  ///< deleted
  TrackHistoryThreshold deletion_history_threshold;  ///< if in M out of the last N updates a measurement is assigned to
  ///< the track, a track is deleted
  bool use_score_based_stage_logic;  ///< determines whether score-based track logic (as described in Blackman, Popoli,
  ///< Design and Analysis of Modern Tracking Systems, Chapter 6.2) is used
  double false_track_confirmation_probability =
      1e-3;  ///< used to calculate thresholds for score-based track management (see Blackman, Popoli,
  ///< Design and Analysis of Modern Tracking Systems, Chapter 6.2)
  double true_track_deletion_probability =
      1e-2;  ///< used to calculate thresholds for score-based track management (see Blackman, Popoli,
  ///< Design and Analysis of Modern Tracking Systems, Chapter 6.2)
};

struct StateParams
{
  MultiModelParams multi_model;
  ClassificationParams classification;
  EstimationParams estimation;
  StageManagementParams stage_management;
  using traits = figcone::FieldTraits<figcone::OptionalField<&StateParams::multi_model>,
                                      figcone::OptionalField<&StateParams::classification>,
                                      figcone::OptionalField<&StateParams::stage_management>>;
};

struct DistributionPostProcessingParams
{
  bool enable = true;
  std::size_t max_components = 10;  ///< keep only the biggest max_components
  double merging_distance =
      0.1;                   ///< merge Components with mahalanobis distance^2 < merging_distance (set to 0 to disable)
  double min_weight = 0.01;  ///< prune Components with weight < min_weight (set to 0 to disable)
  double max_variance =
      100;  ///< prune Components with max(diag(|Var|)) > threshold (set to a big value to effectively disable)
  using traits = figcone::FieldTraits<figcone::OptionalField<&DistributionPostProcessingParams::enable>,
                                      figcone::OptionalField<&DistributionPostProcessingParams::max_components>,
                                      figcone::OptionalField<&DistributionPostProcessingParams::merging_distance>,
                                      figcone::OptionalField<&DistributionPostProcessingParams::min_weight>,
                                      figcone::OptionalField<&DistributionPostProcessingParams::max_variance>>;
};

struct DistributionParams
{
  DISTRIBUTION_TYPE type = DISTRIBUTION_TYPE::GAUSSIAN;  ///< Type (Gaussian|...) of the Distribution
  bool mixture = true;                                   ///< Is this a Mixture
  DISTRIBUTION_EXTRACTION extraction_type =
      DISTRIBUTION_EXTRACTION::BEST_COMPONENT;  ///< How to extract a estimation out of this ?
  DistributionPostProcessingParams post_process{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&DistributionParams::type>,
                                      figcone::OptionalField<&DistributionParams::mixture>,
                                      figcone::OptionalField<&DistributionParams::extraction_type>,
                                      figcone::OptionalField<&DistributionParams::post_process>>;
};

struct StateModelParams
{
  StateModelId id;
  STATE_MODEL_TYPE type;
  DistributionParams distribution{};
  EXTENT extent = EXTENT::RECTANGULAR;
  bool assume_orient_in_velocity_direction = true;
  std::map<std::string, double> model_noise_std_dev;  ///< the noise of the state model, i.e., for the CV the noise in
                                                      ///< the acceleration
  std::map<std::string, double> default_mean;  ///< mainly for the multi model and track to track fusion, if this state
                                               ///< model needs to be converted to some other state model and this is
                                               ///< not possible, use this default mean values, i.e., when using a CP
                                               ///< and CTRV model
  std::map<std::string, double> default_var;   ///< mainly for the multi model and track to track fusion, if this state
                                              ///< model needs to be converted to some other state model and this is not
                                              ///< possible, use this default variance values
  std::map<std::string, double> misc_params;

  using traits = figcone::FieldTraits<figcone::OptionalField<&StateModelParams::distribution>,
                                      figcone::OptionalField<&StateModelParams::extent>,
                                      figcone::OptionalField<&StateModelParams::assume_orient_in_velocity_direction>,
                                      figcone::OptionalField<&StateModelParams::misc_params>,
                                      figcone::OptionalField<&StateModelParams::default_mean>,
                                      figcone::OptionalField<&StateModelParams::default_var>>;
};

struct DynamicBirthModelParams
{
  double mean_num_birth = 10;         ///< mean number of birth tracks per time step
  Probability birth_threshold = 0.1;  ///< Dynamic Model: If associationProb of Measurement < this threshold -> initiate
                                      ///< new birth
  bool force_one_step_birth =
      true;  ///< force a one-step birth even if the transformation form measurement to state space fails
  bool use_default_values = false;  ///< use the class dependent variances below for birth tracks (and not from the
                                    ///< measurement/measurement model)
  std::vector<DefaultValuesParams> default_values = {};
  using traits = figcone::FieldTraits<figcone::OptionalField<&DynamicBirthModelParams::mean_num_birth>,
                                      figcone::OptionalField<&DynamicBirthModelParams::birth_threshold>,
                                      figcone::OptionalField<&DynamicBirthModelParams::force_one_step_birth>,
                                      figcone::OptionalField<&DynamicBirthModelParams::use_default_values>,
                                      figcone::OptionalField<&DynamicBirthModelParams::default_values>>;
};

struct StaticBirthLocation
{
  std::map<std::string, double> mean;  /// < string must match a value in COMPONENT_2_STRING >
  std::map<std::string, double> var;   /// < string must match a value in COMPONENT_2_STRING >
};

struct StaticBirthModelParams
{
  std::vector<StaticBirthLocation> locations{};  ///< the birth locations
  using traits = figcone::FieldTraits<figcone::CopyNodeListField<&StaticBirthModelParams::locations>>;
};

struct BirthModelParams
{
  BIRTH_MODEL_TYPE type = BIRTH_MODEL_TYPE::DYNAMIC;
  bool allow_overlapping = true;
  double min_mhd_4_overlapping_wo_extent = 1;  ///< If overlapping birth Tracks are not allowed but no extent is
  ///< estimated, ensure mhd distance > this param
  Probability default_birth_existence_prob = 0.2;  ///< Default existence Probability of newly initiated tracks
  DynamicBirthModelParams dynamic_model{};
  StaticBirthModelParams static_model{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&BirthModelParams::type>,
                                      figcone::OptionalField<&BirthModelParams::allow_overlapping>,
                                      figcone::OptionalField<&BirthModelParams::default_birth_existence_prob>,
                                      figcone::OptionalField<&BirthModelParams::dynamic_model>,
                                      figcone::OptionalField<&BirthModelParams::static_model>>;
};

struct ConstantPersistenceModelParams
{
  Probability persistence_prob = 0.9;  ///< the existence probability of a track decreases per second
  using traits = figcone::FieldTraits<figcone::OptionalField<&ConstantPersistenceModelParams::persistence_prob>>;
};

struct PersistenceModelParams
{
  PERSISTENCE_MODEL_TYPE type = PERSISTENCE_MODEL_TYPE::CONSTANT;
  ConstantPersistenceModelParams constant{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&PersistenceModelParams::type>,
                                      figcone::OptionalField<&PersistenceModelParams::constant>>;
};

struct LMBDistributionPostProcessPrediction
{
  bool enable = true;
  std::size_t max_tracks = 1000;    ///< keep only the best (regarding existence probability) max_tracks
  double pruning_threshold = 1e-2;  ///< keep only tracks with existence probability >= pruning_threshold
  std::size_t max_last_assoc_duration_ms =
      2000;  ///< delete tracks after last measurement is associated over max_last_assoc_duration_ms
  bool enable_state_dist_post_processing = true;  ///< enable post-processing of the state distribution

  using traits = figcone::FieldTraits<
      figcone::OptionalField<&LMBDistributionPostProcessPrediction::enable>,
      figcone::OptionalField<&LMBDistributionPostProcessPrediction::max_tracks>,
      figcone::OptionalField<&LMBDistributionPostProcessPrediction::pruning_threshold>,
      figcone::OptionalField<&LMBDistributionPostProcessPrediction::max_last_assoc_duration_ms>,
      figcone::OptionalField<&LMBDistributionPostProcessPrediction::enable_state_dist_post_processing>>;
};

struct LMBDistributionPostProcessUpdate
{
  bool enable = true;
  std::size_t max_tracks = 1000;                  ///< keep only the best (regarding existence probability) max_tracks
  double pruning_threshold = 1e-2;                ///< keep only tracks with existence probability >= pruning_threshold
  bool enable_state_dist_post_processing = true;  ///< enable post-processing of the state distribution
  using traits = figcone::FieldTraits<
      figcone::OptionalField<&LMBDistributionPostProcessUpdate::enable>,
      figcone::OptionalField<&LMBDistributionPostProcessUpdate::max_tracks>,
      figcone::OptionalField<&LMBDistributionPostProcessUpdate::pruning_threshold>,
      figcone::OptionalField<&LMBDistributionPostProcessUpdate::enable_state_dist_post_processing>>;
};

struct LMBDistributionExtraction
{
  MO_DISTRIBUTION_EXTRACTION_TYPE type = MO_DISTRIBUTION_EXTRACTION_TYPE::EXISTENCE_PROBABILITY;
  double threshold = 0.5;  ///< extract tracks with existence probability >= threshold
  using traits = figcone::FieldTraits<figcone::OptionalField<&LMBDistributionExtraction::type>,
                                      figcone::OptionalField<&LMBDistributionExtraction::threshold>>;
};

struct LoopyBeliefProapagationParams
{
  std::size_t max_iterations = 20;  ///< the maximum number of loopy belief iterations
  double tol = 1e-5;                ///< stop iterations after converging with this tolerance
  using traits = figcone::FieldTraits<figcone::OptionalField<&LoopyBeliefProapagationParams::max_iterations>,
                                      figcone::OptionalField<&LoopyBeliefProapagationParams::tol>>;
};

struct LMBDistributionParams
{
  LMB_UPDATE_METHOD update_method = LMB_UPDATE_METHOD::LBP;
  bool use_grouping = false;
  bool calculate_single_sensor_group_updates_parallel =
      true;  ///< if thread_pool_size>0 and this is true, lmb groups are calculated parallel
  LMBDistributionPostProcessPrediction post_process_prediction{};
  LMBDistributionPostProcessUpdate post_process_update{};
  LMBDistributionExtraction extraction{};
  LoopyBeliefProapagationParams loopy_belief_propagation{};
  using traits = figcone::FieldTraits<
      figcone::OptionalField<&LMBDistributionParams::update_method>,
      figcone::OptionalField<&LMBDistributionParams::use_grouping>,
      figcone::OptionalField<&LMBDistributionParams::calculate_single_sensor_group_updates_parallel>,
      figcone::OptionalField<&LMBDistributionParams::post_process_prediction>,
      figcone::OptionalField<&LMBDistributionParams::post_process_update>,
      figcone::OptionalField<&LMBDistributionParams::extraction>,
      figcone::OptionalField<&LMBDistributionParams::loopy_belief_propagation>>;
};

struct GLMBDistributionPostProcessPrediction
{
  bool enable = true;                ///< enable post processing after a prediction step
  double pruning_threshold = 1e-5;   ///< delete every hypothesis with weight < pruning_threshold
  std::size_t max_hypotheses = 1e4;  ///< keep at most the best max_hypotheses, delete the smaller ones
  using traits = figcone::FieldTraits<figcone::OptionalField<&GLMBDistributionPostProcessPrediction::pruning_threshold>,
                                      figcone::OptionalField<&GLMBDistributionPostProcessPrediction::max_hypotheses>>;
};

struct GLMBDistributionPostProcessUpdate
{
  bool enable = true;                ///< enable post processing after a update step
  double pruning_threshold = 1e-5;   ///< delete every hypothesis with weight < pruning_threshold
  std::size_t max_hypotheses = 1e4;  ///< keep at most the best max_hypotheses, delete the smaller ones
  using traits = figcone::FieldTraits<figcone::OptionalField<&GLMBDistributionPostProcessUpdate::pruning_threshold>,
                                      figcone::OptionalField<&GLMBDistributionPostProcessUpdate::max_hypotheses>>;
};

struct GLMBDistributionExtraction
{
  MO_DISTRIBUTION_EXTRACTION_TYPE type;
  double threshold;
};

struct AllParams
{
  std::size_t num_track_limit;
  LMB_2_GLMB_CONVERISON_TYPE fallback_type;
};

struct KBestParams
{
  std::size_t max_number_hypotheses;
};

struct SamplingParams
{
  std::size_t max_number_hypotheses;
  double percentage_of_weight_hypotheses;
  std::size_t max_num_tries;
};

struct LMB2GLMBConversion
{
  LMB_2_GLMB_CONVERISON_TYPE type;
  AllParams all;
  KBestParams kBest;
  SamplingParams sampling;
};

struct GibbsSamplingParams
{
  double max_trials_factor;  ///< if you demand n different solutions from the Gibbs sampler, try at most (n * *this
                             ///< factor) random trials
  std::size_t abort_after_ntimes_no_new_sol = 20;  ///< if after this number of tries no new solutions is found -> break
  using traits = figcone::FieldTraits<figcone::OptionalField<&GibbsSamplingParams::abort_after_ntimes_no_new_sol>>;
};

struct UpdateHypsParams
{
  double equal_allocation_share_ratio;  ///< Todo: what is this?
  std::size_t max_update_hypotheses;    ///< Every existing hypotheses can maximal generate *this number of update
                                        ///< hypotheses (used if joint_prediction and update==true)
};

struct GLMBUpdateParams
{
  bool joint_prediction_and_update;  ///< jointly estimate the created hypotheses with the update step (with Murty or
                                     ///< Gibbs), i.e., the hypotheses are estimated simultaneously with the assignment
  GLMB_ASSIGNMENT_METHOD assignment_method;        ///< how to perform the GLMB update
  std::size_t max_total_number_update_hypotheses;  ///< generate at most *this number of updated hypotheses
  GibbsSamplingParams gibbs_sampling;
  UpdateHypsParams num_update_hypotheses;
};

struct GLMBDistributionParams
{
  bool do_profiling = false;
  GLMBDistributionPostProcessPrediction post_process_prediction;
  GLMBDistributionPostProcessUpdate post_process_update;
  GLMBUpdateParams update;
  GLMBDistributionExtraction extraction;
  LMB2GLMBConversion lmb_2_glmb_conversion;
  using traits = figcone::FieldTraits<figcone::OptionalField<&GLMBDistributionParams::do_profiling>,
                                      figcone::OptionalField<&GLMBDistributionParams::post_process_prediction>,
                                      figcone::OptionalField<&GLMBDistributionParams::post_process_update>,
                                      figcone::OptionalField<&GLMBDistributionParams::extraction>,
                                      figcone::OptionalField<&GLMBDistributionParams::lmb_2_glmb_conversion>>;
};

struct PHDDistributionPostProcessPredictionParams
{
  bool enable = true;
  double pruning_threshold = 1e-3;
  double merge_distance = 1;
  using traits =
      figcone::FieldTraits<figcone::OptionalField<&PHDDistributionPostProcessPredictionParams::enable>,
                           figcone::OptionalField<&PHDDistributionPostProcessPredictionParams::pruning_threshold>,
                           figcone::OptionalField<&PHDDistributionPostProcessPredictionParams::merge_distance>>;
};

struct PHDDistributionPredictionParams
{
  PHDDistributionPostProcessPredictionParams post_process{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&PHDDistributionPredictionParams::post_process>>;
};

struct PHDDistributionPostProcessUpdateParams
{
  bool enable = true;
  double pruning_threshold = 1e-3;
  double merge_distance = 1;
  using traits =
      figcone::FieldTraits<figcone::OptionalField<&PHDDistributionPostProcessUpdateParams::enable>,
                           figcone::OptionalField<&PHDDistributionPostProcessUpdateParams::pruning_threshold>,
                           figcone::OptionalField<&PHDDistributionPostProcessUpdateParams::merge_distance>>;
};

struct PHDDistributionUpdateParams
{
  PHDDistributionPostProcessUpdateParams post_process{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&PHDDistributionUpdateParams::post_process>>;
};

struct PHDDistributionParams
{
  PHDDistributionPredictionParams prediction{};
  PHDDistributionUpdateParams update{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&PHDDistributionParams::prediction>,
                                      figcone::OptionalField<&PHDDistributionParams::update>>;
};

struct BufferParams
{
  std::size_t max_wait_duration_ms = 500;      ///< maximal duration we wait for an expected measurement
  double max_wait_duration_quantile = 0.9999;  ///< duration we wait for an expected measurement based on an estimation
  std::size_t max_delta_ms = 50;               ///< max time difference of measurements in one batch
  using traits = figcone::FieldTraits<figcone::OptionalField<&BufferParams::max_wait_duration_ms>,
                                      figcone::OptionalField<&BufferParams::max_wait_duration_quantile>,
                                      figcone::OptionalField<&BufferParams::max_delta_ms>>;
};

struct ClutterRateEstimation
{
  double static_discount = 0.99;
  double dynamic_discount_alpha = 0.5;
  double dynamic_discount_min = 0.3;
  using traits = figcone::FieldTraits<figcone::OptionalField<&ClutterRateEstimation::static_discount>,
                                      figcone::OptionalField<&ClutterRateEstimation::dynamic_discount_alpha>,
                                      figcone::OptionalField<&ClutterRateEstimation::dynamic_discount_min>>;
};
struct DetectionRateEstimation
{
  double static_discount = 0.99;
  double dynamic_discount_alpha = 4;
  double dynamic_discount_min = 0.3;
  using traits = figcone::FieldTraits<figcone::OptionalField<&DetectionRateEstimation::static_discount>,
                                      figcone::OptionalField<&DetectionRateEstimation::dynamic_discount_alpha>,
                                      figcone::OptionalField<&DetectionRateEstimation::dynamic_discount_min>>;
};

struct LMBICFilterParams
{
};

struct SelfAssessmentParams
{
  ClutterRateEstimation clutter_rate_estimation{};
  DetectionRateEstimation detection_prob_estimation{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&SelfAssessmentParams::clutter_rate_estimation>,
                                      figcone::OptionalField<&SelfAssessmentParams::detection_prob_estimation>>;
};

/// The Parameters used in the GaussianMeasurementModel
struct TTTUncorrelatedTracksParams
{
  TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY strategy;
  double gating_prob;
  double pseudo_birth_prob;
  double pseudo_birth_threshold;
  /// flip measurement orientation to fit the orientation of the associated track -> Birth decides over orientation of
  /// track!
  bool fix_orientation = false;
  bool overwrite_var = false;
  std::map<std::string, double> var;

  using traits = figcone::FieldTraits<figcone::OptionalField<&TTTUncorrelatedTracksParams::fix_orientation>,
                                      figcone::OptionalField<&TTTUncorrelatedTracksParams::overwrite_var>,
                                      figcone::OptionalField<&TTTUncorrelatedTracksParams::var>>;
};

struct LMBFPMFilterParams
{
  bool do_profiling = false;
  std::size_t sensor_number;
  std::size_t pmNumBestComponents_k;
  bool calculate_true_k_best = true;  // if true -> generalizedKBest algorithm is used, else kBestSelection is used
  bool calculate_poisson_k_best =
      false;  // if true -> use poisson distribution for calculation of k, if false only scale by weight
  double equal_allocation_share_ratio = 0.0;
  bool calculate_fpm_fusion_tracks_parallel = true;
  MULTI_SENSOR_UPDATE_METHOD multi_sensor_update_method = MULTI_SENSOR_UPDATE_METHOD::FPM;
  TTTUncorrelatedTracksParams dynamicBirth;
  using traits = figcone::FieldTraits<figcone::OptionalField<&LMBFPMFilterParams::do_profiling>,
                                      figcone::OptionalField<&LMBFPMFilterParams::calculate_true_k_best>,
                                      figcone::OptionalField<&LMBFPMFilterParams::calculate_poisson_k_best>,
                                      figcone::OptionalField<&LMBFPMFilterParams::equal_allocation_share_ratio>,
                                      figcone::OptionalField<&LMBFPMFilterParams::calculate_fpm_fusion_tracks_parallel>,
                                      figcone::OptionalField<&LMBFPMFilterParams::multi_sensor_update_method>>;
};

struct NoFilterParams
{
  bool do_profiling;
  // TODO:
};

struct NNFilterParams
{
  bool do_profiling;
  // TODO:
};

struct GNNTrackManagementParams
{
  bool do_post_process_after_update;
  bool use_existence_prob_from_birth_model_for_score_initialisation;
  bool output_only_confirmed;  ///< if a track management is used that distinguishes between confirmed and unconfirmed
                               ///< (preliminary or tentative) tracks, getEstimate outputs only the confirmed tracks
};

struct GNNCostMatrixParams
{
  bool use_external_sources_density;  ///< Determines whether the CostMatrix for the measurement to track assignment
                                      ///< logic is extended with a diagonal Matrix considering the False Target and new
                                      ///< Track Density . This represents the Cost for assigning this measurement to a
                                      ///< new Track instead of an already existing one. See
                                      ///< https://users.metu.edu.tr/umut/ee793/files/METULecture5.pdf for more if the
                                      ///< parameter is set to false, an assignment to a new track is set to a value
                                      ///< greater than the Maximum cost of any assignment

  bool calculate_birth_track_density;  ///< Try to calculate the birth_track_density
  double birth_track_density;          ///< Sets a default value for birth_track_density
};

struct GNNFilterParams
{
  bool do_profiling;  // TODO: profiling
  bool use_grouping;
  GNNCostMatrixParams costMatrixParams;
  GNNTrackManagementParams trackManagementParams;
};

struct UshiftTrackerParams
{
};

struct FilterParams
{
  bool enable = false;
  FILTER_TYPE type;
  LMBICFilterParams lmb_ic;
  LMBFPMFilterParams lmb_fpm;
  NoFilterParams no;
  NNFilterParams nn;
  GNNFilterParams gnn;
  UshiftTrackerParams ushift;
  using traits = figcone::FieldTraits<figcone::OptionalField<&FilterParams::enable>,
                                      figcone::OptionalField<&FilterParams::lmb_ic>,
                                      figcone::OptionalField<&FilterParams::lmb_fpm>,
                                      figcone::OptionalField<&FilterParams::no>,
                                      figcone::OptionalField<&FilterParams::nn>,
                                      figcone::OptionalField<&FilterParams::gnn>,
                                      figcone::OptionalField<&FilterParams::ushift>>;
};

struct NoTTTFilterParams
{
  bool do_profiling;
};

struct TransTTTFilterParams
{
  std::size_t max_prediction_duration_ms;  ///< how long a track will be predicted if no new stateContainers arrive,
                                           ///< also how long a stateContainer get maximal predicted to align them in
};

struct EvalTTTFilterParams
{
  std::string reference;  ///< compare the tracks from different sources to, possible multiple, references from sources
                          ///< containing this name
  std::size_t max_reference_prediction_duration_ms;  ///< the reference will be maximal this time predicted to some
                                                     ///< possible fitting tracks
  double max_distance;                               ///< consider only tracks with maximal this distance for evaluation
};

struct TTTFilterParams
{
  bool enable = false;
  TTT_FILTER_TYPE type;
  NoTTTFilterParams no;
  TransTTTFilterParams trans;
  EvalTTTFilterParams eval;
  using traits = figcone::FieldTraits<figcone::OptionalField<&TTTFilterParams::enable>,
                                      figcone::OptionalField<&TTTFilterParams::no>,
                                      figcone::OptionalField<&TTTFilterParams::trans>,
                                      figcone::OptionalField<&TTTFilterParams::eval>>;
};

struct StaticTrackParams
{
  std::map<std::string, double> mean;  ///< mean values of the track
};

struct AddStaticTracksParams
{
  bool enable = false;
  MeasModelId meas_model_id;
  std::vector<StaticTrackParams> tracks;
};

struct Params
{
  std::string version;
  std::size_t thread_pool_size = 0;  ///< the number of worker threads, 0 means no parallelization
  bool show_gui = false;             ///< show the gui at startup
  std::string name = "Tracking";
  MeasurementModelsParams meas_models{};
  StateParams state{};
  std::vector<StateModelParams> state_models{};
  std::optional<BirthModelParams> birth_model{};
  PersistenceModelParams persistence_model{};
  LMBDistributionParams lmb_distribution{};
  GLMBDistributionParams glmb_distribution{};
  PHDDistributionParams phd_distribution{};
  BufferParams buffer{};
  SelfAssessmentParams self_assessment{};
  FilterParams filter{};
  TTTFilterParams ttt_filter{};
  AddStaticTracksParams static_tracks{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&Params::show_gui>,
                                      figcone::OptionalField<&Params::name>,
                                      figcone::OptionalField<&Params::meas_models>,
                                      figcone::OptionalField<&Params::state>,
                                      figcone::OptionalField<&Params::birth_model>,
                                      figcone::OptionalField<&Params::persistence_model>,
                                      figcone::OptionalField<&Params::lmb_distribution>,
                                      figcone::OptionalField<&Params::glmb_distribution>,
                                      figcone::OptionalField<&Params::phd_distribution>,
                                      figcone::OptionalField<&Params::buffer>,
                                      figcone::OptionalField<&Params::self_assessment>,
                                      figcone::OptionalField<&Params::filter>,
                                      figcone::OptionalField<&Params::ttt_filter>,
                                      figcone::OptionalField<&Params::static_tracks>>;
};

}  // namespace ttb

/// Provide converter to read the yaml parameter string into the defined type
/// new parameter types must be added here
namespace figcone
{

template <>
inline void PostProcessor<ttb::Params>::operator()(ttb::Params& params)
{
  if (not params.filter.enable and not params.ttt_filter.enable)
  {
    throw ValidationError("Both Filter and TTT Filter are disabled. This is not what you want");
  }

  for (ttb::GaussianMeasurementModelParams& model_params : params.meas_models.gaussian_models)
  {
    // set mean of clutter/detection estimation to value used in filter
    if (std::isnan(model_params.clutter.rate_estimation.prior_mean))
    {
      model_params.clutter.rate_estimation.prior_mean = model_params.clutter.rate.value_or(1);
    }
    if (std::isnan(model_params.clutter.rate_estimation.prior_var))
    {
      model_params.clutter.rate_estimation.prior_var = model_params.clutter.rate_estimation.prior_mean;
    }
    if (std::isnan(model_params.detection.prob_estimation.prior_mean))
    {
      model_params.detection.prob_estimation.prior_mean = model_params.detection.prob;
    }
    if (std::isnan(model_params.detection.prob_estimation.prior_var))
    {
      model_params.detection.prob_estimation.prior_var = 0.05;
    }
    // default values exactly defined once for all classes
    std::vector<ttb::CLASS> state_classes_used = params.state.classification.classes;
    std::vector<ttb::CLASS> classes_defined;
    for (auto& [type, mean, var] : model_params.default_values)
    {
      if (type.empty())
      {
        type = state_classes_used;
      }
      for (ttb::CLASS clazz : type)
      {
        if (std::ranges::find(classes_defined, clazz) != classes_defined.end())
        {
          throw ValidationError("multiple definitions for class " + ttb::to_string(clazz));
        }
        classes_defined.push_back(clazz);
      }
    }
    std::ranges::sort(state_classes_used);
    std::ranges::sort(classes_defined);
    if (not model_params.default_values.empty() and state_classes_used != classes_defined)
    {
      throw ValidationError("not all classes in state have default_values defined in meas model " +
                            model_params.id.value_);
    }
    // default mean = 0 and var = 1 if not defined otherwise
    for (ttb::COMPONENT comp : ttb::ALL_COMPONENTS)
    {
      for (auto& [type, mean, var] : model_params.default_values)
      {
        if (not mean.contains(ttb::to_string(comp)))
        {
          mean.emplace(ttb::to_string(comp), 0);
        }
        if (not var.contains(ttb::to_string(comp)))
        {
          var.emplace(ttb::to_string(comp), 1);
        }
      }
    }
  }
  // check markov transition classes
  if (params.state.multi_model.enable_markov_transition)
  {
    std::vector<ttb::CLASS> state_classes_used = params.state.classification.classes;
    std::vector<ttb::CLASS> classes_defined;
    for (ttb::MarkovTransitionParams& markov_params : params.state.multi_model.markov_transition)
    {
      if (markov_params.type.empty())
      {
        markov_params.type = state_classes_used;
      }
      for (ttb::CLASS clazz : markov_params.type)
      {
        if (std::ranges::find(classes_defined, clazz) != classes_defined.end())
        {
          throw ValidationError("multiple definitions for class " + ttb::to_string(clazz) + " in Markov Transition");
        }
        classes_defined.push_back(clazz);
      }
    }
    std::ranges::sort(state_classes_used);
    std::ranges::sort(classes_defined);
    if (state_classes_used != classes_defined)
    {
      throw ValidationError("not all classes in state have default_values defined in Markov Transition");
    }
  }
  // default values for dynamic birth
  if (params.birth_model.has_value())
  {
    if (params.birth_model.value().dynamic_model.use_default_values)
    {
      std::vector<ttb::CLASS> state_classes_used = params.state.classification.classes;
      std::vector<ttb::CLASS> classes_defined;
      for (ttb::DefaultValuesParams& default_values : params.birth_model.value().dynamic_model.default_values)
      {
        if (default_values.type.empty())
        {
          default_values.type = state_classes_used;
        }
        for (ttb::CLASS clazz : default_values.type)
        {
          if (std::ranges::find(classes_defined, clazz) != classes_defined.end())
          {
            throw ValidationError("multiple definitions for class " + ttb::to_string(clazz) +
                                  " in Dynamic Birth Model default params");
          }
          classes_defined.push_back(clazz);
        }
      }
      std::ranges::sort(state_classes_used);
      std::ranges::sort(classes_defined);
      if (state_classes_used != classes_defined)
      {
        throw ValidationError("not all classes in default_values in dynamic birth model have default_values defined");
      }
    }
  }
}

template <>
struct StringConverter<bool>
{
  static std::optional<bool> fromString(const std::string& data)
  {
    if (data == "true")
    {
      return true;
    }
    if (data == "false")
    {
      return false;
    }
    return {};
  }
};

template <>
struct StringConverter<ttb::LMB_UPDATE_METHOD>
{
  static std::optional<ttb::LMB_UPDATE_METHOD> fromString(const std::string& data)
  {
    return ttb::to_LMB_UPDATE_METHOD(data);
  }
};

template <>
struct StringConverter<ttb::EXTENT>
{
  static std::optional<ttb::EXTENT> fromString(const std::string& data)
  {
    return ttb::to_EXTENT(data);
  }
};

template <>
struct StringConverter<ttb::OCCLUSION_MODEL_TYPE>
{
  static std::optional<ttb::OCCLUSION_MODEL_TYPE> fromString(const std::string& data)
  {
    return ttb::to_OCCLUSION_MODEL(data);
  }
};

template <>
struct StringConverter<ttb::COMPONENT>
{
  static std::optional<ttb::COMPONENT> fromString(const std::string& data)
  {
    return ttb::to_COMPONENT(data);
  }
};

template <>
struct StringConverter<ttb::BUILD_MODE>
{
  static std::optional<ttb::BUILD_MODE> fromString(const std::string& data)
  {
    return ttb::to_BUILD_MODE(data);
  }
};

template <>
struct StringConverter<ttb::Matrix>
{
  static std::optional<ttb::Matrix> fromString(const std::string& data)
  {
    std::stringstream ss(data);
    std::string row_str;
    double val;
    std::vector<std::vector<double>> vals;
    while (std::getline(ss, row_str, ';'))
    {
      std::stringstream row_ss(row_str);
      std::vector<double> row_vals;
      while (row_ss >> val)
      {
        row_vals.push_back(val);
      }
      vals.push_back(std::move(row_vals));
    }
    if (vals.empty())
    {
      return {};
    }
    ttb::Matrix matrix(vals.size(), vals.front().size());
    for (std::size_t i = 0; i < matrix.rows(); ++i)
    {
      for (std::size_t j = 0; j < matrix.cols(); ++j)
      {
        matrix(i, j) = vals.at(i).at(j);
      }
    }
    return matrix;
  }
};

template <>
struct StringConverter<ttb::DISTRIBUTION_EXTRACTION>
{
  static std::optional<ttb::DISTRIBUTION_EXTRACTION> fromString(const std::string& data)
  {
    return ttb::to_DISTRIBUTION_EXTRACTION(data);
  }
};

template <>
struct StringConverter<ttb::STATE_DISTRIBUTION_EXTRACTION_TYPE>
{
  static std::optional<ttb::STATE_DISTRIBUTION_EXTRACTION_TYPE> fromString(const std::string& data)
  {
    return ttb::to_STATE_DISTRIBUTION_EXTRACTION_TYPE(data);
  }
};

template <>
struct StringConverter<ttb::LMB_2_GLMB_CONVERISON_TYPE>
{
  static std::optional<ttb::LMB_2_GLMB_CONVERISON_TYPE> fromString(const std::string& data)
  {
    return ttb::to_LMB_2_GLMB_CONVERISON_TYPE(data);
  }
};

template <>
struct StringConverter<ttb::MO_DISTRIBUTION_EXTRACTION_TYPE>
{
  static std::optional<ttb::MO_DISTRIBUTION_EXTRACTION_TYPE> fromString(const std::string& data)
  {
    return ttb::to_MO_DISTRIBUTION_EXTRACTION_TYPE(data);
  }
};

template <>
struct StringConverter<ttb::PERSISTENCE_MODEL_TYPE>
{
  static std::optional<ttb::PERSISTENCE_MODEL_TYPE> fromString(const std::string& data)
  {
    return ttb::to_PERSISTENCE_MODEL(data);
  }
};

template <>
struct StringConverter<ttb::BIRTH_MODEL_TYPE>
{
  static std::optional<ttb::BIRTH_MODEL_TYPE> fromString(const std::string& data)
  {
    return ttb::to_BIRTH_MODEL(data);
  }
};

template <>
struct StringConverter<ttb::TRANSITION_MODEL_TYPE>
{
  static std::optional<ttb::TRANSITION_MODEL_TYPE> fromString(const std::string& data)
  {
    return ttb::to_TRANSITION_MODEL(data);
  }
};

template <>
struct StringConverter<ttb::TRANSITION_TYPE>
{
  static std::optional<ttb::TRANSITION_TYPE> fromString(const std::string& data)
  {
    return ttb::to_TRANSITION_TYPE(data);
  }
};

template <>
struct StringConverter<ttb::STATE_MODEL_TYPE>
{
  static std::optional<ttb::STATE_MODEL_TYPE> fromString(const std::string& data)
  {
    return ttb::to_STATE_MODEL_TYPE(data);
  }
};

template <>
struct StringConverter<ttb::CLASS>
{
  static std::optional<ttb::CLASS> fromString(const std::string& data)
  {
    return ttb::to_CLASS(data);
  }
};

template <>
struct StringConverter<ttb::CLASSIFICATION_TYPE>
{
  static std::optional<ttb::CLASSIFICATION_TYPE> fromString(const std::string& data)
  {
    return ttb::to_CLASSIFICATION_TYPE(data);
  }
};

template <>
struct StringConverter<ttb::DISTRIBUTION_TYPE>
{
  static std::optional<ttb::DISTRIBUTION_TYPE> fromString(const std::string& data)
  {
    return ttb::to_DISTRIBUTION_TYPE(data);
  }
};

template <>
struct StringConverter<ttb::MEASUREMENT_MODEL_TYPE>
{
  static std::optional<ttb::MEASUREMENT_MODEL_TYPE> fromString(const std::string& data)
  {
    return ttb::to_MEASUREMENT_MODEL(data);
  }
};

template <>
struct StringConverter<ttb::FILTER_TYPE>
{
  static std::optional<ttb::FILTER_TYPE> fromString(const std::string& data)
  {
    return ttb::to_FILTER_TYPE(data);
  }
};

template <>
struct StringConverter<ttb::TTT_FILTER_TYPE>
{
  static std::optional<ttb::TTT_FILTER_TYPE> fromString(const std::string& data)
  {
    return ttb::to_TTT_FILTER_TYPE(data);
  }
};

template <>
struct StringConverter<ttb::TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY>
{
  static std::optional<ttb::TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY> fromString(const std::string& data)
  {
    return ttb::to_TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY(data);
  }
};

template <>
struct StringConverter<ttb::StateModelId>
{
  static std::optional<ttb::StateModelId> fromString(const std::string& data)
  {
    std::stringstream ss(data);
    ttb::StateModelId::type val;
    ss >> val;
    return ttb::StateModelId{ val };
  }
};

template <>
struct StringConverter<ttb::MeasModelId>
{
  static std::optional<ttb::MeasModelId> fromString(const std::string& data)
  {
    std::stringstream ss(data);
    ttb::MeasModelId::type val;
    ss >> val;
    return ttb::MeasModelId{ val };
  }
};

template <>
struct StringConverter<ttb::DistributionId>
{
  static std::optional<ttb::DistributionId> fromString(const std::string& data)
  {
    std::stringstream ss(data);
    ttb::DistributionId::type val;
    ss >> val;
    return ttb::DistributionId{ val };
  }
};

template <>
struct StringConverter<ttb::SELECTION_STRATEGY>
{
  static std::optional<ttb::SELECTION_STRATEGY> fromString(const std::string& data)
  {
    return ttb::to_SELECTION_STRATEGY(data);
  }
};

template <>
struct StringConverter<ttb::GLMB_ASSIGNMENT_METHOD>
{
  static std::optional<ttb::GLMB_ASSIGNMENT_METHOD> fromString(const std::string& data)
  {
    return ttb::to_GLMB_UPDATE_METHOD(data);
  }
};

template <>
struct StringConverter<ttb::MULTI_SENSOR_UPDATE_METHOD>
{
  static std::optional<ttb::MULTI_SENSOR_UPDATE_METHOD> fromString(const std::string& data)
  {
    return ttb::to_MULTI_SENSOR_UPDATE_METHOD(data);
  }
};

}  // namespace figcone
