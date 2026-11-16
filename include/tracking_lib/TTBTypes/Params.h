#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include <figcone/configreader.h>
/// All parameters for the whole library are defined and should be DOCUMENTED here
///
/// The structure of the params must reflect the structure of the yaml where they are specified.
/// The top level struct is defined at the bottom and called Params ....
/// The configuration file is automatically read in using figcone and must EXACTLY match the definitions here

namespace ttb
{

auto constexpr param_version{ "1.0.0.0" };

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
  bool learn_model = false;
  using traits = figcone::FieldTraits<figcone::OptionalField<&DetectionParams::prob>,
                                      figcone::OptionalField<&DetectionParams::prob_min>,
                                      figcone::OptionalField<&DetectionParams::prob_estimation>,
                                      figcone::OptionalField<&DetectionParams::learn_model>>;
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

struct DefaultVal
{
  std::optional<double> mean;
  std::optional<double> var;
};

inline DefaultVal default_val(DefaultValuesParams const& params, COMPONENT comp)
{
  auto const mean = [&] -> std::optional<double> {
    auto const mean_it = params.mean.find(to_string(comp));
    if (mean_it == params.mean.end())
    {
      return std::nullopt;
    }
    return mean_it->second;
  }();
  auto const var = [&] -> std::optional<double> {
    auto const var_it = params.var.find(to_string(comp));
    if (var_it == params.var.end())
    {
      return std::nullopt;
    }
    return var_it->second;
  }();
  return DefaultVal{ .mean = mean, .var = var };
}

inline DefaultVal default_val(std::vector<DefaultValuesParams> const& params, COMPONENT comp, CLASS clazz)
{
  const auto it = std::ranges::find_if(params, [&](DefaultValuesParams const& class_vars) {
    return std::ranges::find(class_vars.type, clazz) != class_vars.type.end();
  });
  if (it == params.end())
  {
    return {};
  }
  return default_val(*it, comp);
}

/// The Parameters used in the GaussianMeasurementModel
struct GaussianMeasurementModelParams
{
  SourceId id;                        ///< the unique ID of this Model
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
  std::vector<SourceId> use_meas_models{};  ///< Use only this Measurement Models for classification
  bool use_union_classes = false;           ///< unify the measured classes of some input
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic push
  std::vector<CLASS> classes{ CLASS::NOT_CLASSIFIED };  ///< Use this Classes, you can disable state classification by
#pragma GCC diagnostic pop
  ///< setting the classes to NOT_CLASSIFIED
  double discount_factor_prediction = 0.98;
  using traits = figcone::FieldTraits<figcone::OptionalField<&ClassificationParams::use_meas_models>,
                                      figcone::OptionalField<&ClassificationParams::use_union_classes>,
                                      figcone::OptionalField<&ClassificationParams::discount_factor_prediction>>;
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
  Matrix<double> transition_matrix;
};

struct MultiModelParams
{
  std::vector<StateModelId> use_state_models;
  std::vector<double> birth_weights{ { 1.0 } };
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
  MultiModelParams multi_model{};
  ClassificationParams classification{};
  EstimationParams estimation;
  StageManagementParams stage_management{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&StateParams::multi_model>,
                                      figcone::OptionalField<&StateParams::classification>,
                                      figcone::OptionalField<&StateParams::stage_management>>;
};

struct StateParamsValidator
{
  void operator()(StateParams const& params)
  {
    if (std::ranges::find(params.multi_model.use_state_models, params.estimation.output_state_model) ==
        params.multi_model.use_state_models.end())
    {
      throw figcone::ValidationError(std::format("output_state_model {} is not in use_state_models {}. The "
                                                 "output_state_model must be one "
                                                 "of the use_state_models",
                                                 params.estimation.output_state_model,
                                                 params.multi_model.use_state_models));
    }
  }
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

struct BirthModelExistenceParams
{
  std::string source_id;
  double existence_prob;
};

struct DynamicBirthModelParams
{
  double mean_num_birth = 10;                 ///< mean number of birth tracks per time step
  Probability<double> birth_threshold = 0.1;  ///< Dynamic Model: If associationProb of Measurement < this threshold ->
                                              ///< initiate new birth
  bool force_one_step_birth =
      true;  ///< force a one-step birth even if the transformation form measurement to state space fails
  bool use_default_values = false;  ///< use the class dependent variances below for birth tracks (and not from the
                                    ///< measurement/measurement model)
  std::vector<BirthModelExistenceParams> meas_model_specific_existence_probs{};  ///< overwrite the
                                                                                 ///< default_birth_existence_prob for
                                                                                 ///< the given sources with the given
                                                                                 ///< values
  std::vector<DefaultValuesParams> default_values = {};
  using traits =
      figcone::FieldTraits<figcone::OptionalField<&DynamicBirthModelParams::mean_num_birth>,
                           figcone::OptionalField<&DynamicBirthModelParams::birth_threshold>,
                           figcone::OptionalField<&DynamicBirthModelParams::force_one_step_birth>,
                           figcone::OptionalField<&DynamicBirthModelParams::use_default_values>,
                           figcone::OptionalField<&DynamicBirthModelParams::meas_model_specific_existence_probs>,
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

struct ROIBirthModelParams
{
  bool use_sensors_fov_instead = false;
  double cov_scaling_factor = 1.0;
  double confidence_prob = 0.99;
  using traits = figcone::FieldTraits<figcone::OptionalField<&ROIBirthModelParams::use_sensors_fov_instead>,
                                      figcone::OptionalField<&ROIBirthModelParams::cov_scaling_factor>,
                                      figcone::OptionalField<&ROIBirthModelParams::confidence_prob>>;
};

struct BirthModelParams
{
  BIRTH_MODEL_TYPE type = BIRTH_MODEL_TYPE::DYNAMIC;
  bool allow_overlapping = true;
  double min_mhd_4_overlapping_wo_extent = 1;  ///< If overlapping birth Tracks are not allowed but no extent is
  ///< estimated, ensure mhd distance > this param
  Probability<double> default_birth_existence_prob = 0.2;  ///< Default existence Probability of newly initiated tracks
  DynamicBirthModelParams dynamic_model{};
  StaticBirthModelParams static_model{};
  ROIBirthModelParams roi_model{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&BirthModelParams::type>,
                                      figcone::OptionalField<&BirthModelParams::allow_overlapping>,
                                      figcone::OptionalField<&BirthModelParams::default_birth_existence_prob>,
                                      figcone::OptionalField<&BirthModelParams::dynamic_model>,
                                      figcone::OptionalField<&BirthModelParams::static_model>,
                                      figcone::OptionalField<&BirthModelParams::roi_model>>;
};

struct ConstantPersistenceModelParams
{
  Probability<double> persistence_prob = 0.9;  ///< the existence probability of a track decreases per second
  using traits = figcone::FieldTraits<figcone::OptionalField<&ConstantPersistenceModelParams::persistence_prob>>;
};

struct ROIPersistenceModelParams
{
  Probability<double> inside_persistence_prob = 0.99;  ///< the existence probability of a track decreases per second
  Probability<double> outside_persistence_prob = 0;    ///< the existence probability of a track decreases per second
  using traits = figcone::FieldTraits<figcone::OptionalField<&ROIPersistenceModelParams::inside_persistence_prob>,
                                      figcone::OptionalField<&ROIPersistenceModelParams::outside_persistence_prob>>;
};

struct PersistenceModelParams
{
  PERSISTENCE_MODEL_TYPE type = PERSISTENCE_MODEL_TYPE::ROI;
  ConstantPersistenceModelParams constant{};
  ROIPersistenceModelParams roi{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&PersistenceModelParams::type>,
                                      figcone::OptionalField<&PersistenceModelParams::constant>,
                                      figcone::OptionalField<&PersistenceModelParams::roi>>;
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

struct PMFusionPoissonParams
{
  double equal_allocation_share_ratio = 0.0;

  using traits = figcone::FieldTraits<figcone::OptionalField<&PMFusionPoissonParams::equal_allocation_share_ratio>>;
};

struct FPMFusionParams
{
  std::size_t num_components = 10;  // maximum number of fused state components calculated by fpm fusion
  PM_MULTI_SENSOR_UPDATE_METHOD method =
      PM_MULTI_SENSOR_UPDATE_METHOD::GENERALIZED_K_BEST;  // if true -> generalizedKBest algorithm is used, else
                                                          // kBestSelection is used
  PMFusionPoissonParams poisson_method{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&FPMFusionParams::num_components>,
                                      figcone::OptionalField<&FPMFusionParams::method>,
                                      figcone::OptionalField<&FPMFusionParams::poisson_method>>;
};

struct StandardPostProcessingParams
{
  double pruning_threshold = 1e-5;  ///< delete every hypothesis with weight < pruning_threshold
  double prune_labels_existence_probability_threshold =
      1e-3;                               ///< delete every label with existence probability < pruning_threshold
  std::size_t max_hypotheses = 1000;      ///< keep at most the best max_hypotheses, delete the smaller ones
  std::size_t max_states_per_label = 10;  ///< keep at most the max_states_per_label best states per label
  bool limit_states_per_label = true;
  using traits = figcone::FieldTraits<
      figcone::OptionalField<&StandardPostProcessingParams::pruning_threshold>,
      figcone::OptionalField<&StandardPostProcessingParams::prune_labels_existence_probability_threshold>,
      figcone::OptionalField<&StandardPostProcessingParams::max_hypotheses>,
      figcone::OptionalField<&StandardPostProcessingParams::max_states_per_label>,
      figcone::OptionalField<&StandardPostProcessingParams::limit_states_per_label>>;
};

struct DissHerrmannPostProcessingParams
{
  std::size_t pruning_max_comps_retain = 300;
  std::size_t pruning_min_comps_per_card = 10;
  double pruning_min_comps_per_card_thresh = 1e-12;
  double pruning_max_weight_ratio = 1e18;
  bool pruning_alloc_min_comps_per_card_by_poisson = false;

  using traits = figcone::FieldTraits<
      figcone::OptionalField<&DissHerrmannPostProcessingParams::pruning_max_comps_retain>,
      figcone::OptionalField<&DissHerrmannPostProcessingParams::pruning_min_comps_per_card>,
      figcone::OptionalField<&DissHerrmannPostProcessingParams::pruning_min_comps_per_card_thresh>,
      figcone::OptionalField<&DissHerrmannPostProcessingParams::pruning_max_weight_ratio>,
      figcone::OptionalField<&DissHerrmannPostProcessingParams::pruning_alloc_min_comps_per_card_by_poisson>>;
};

struct GLMBDistributionPostProcessUpdate
{
  bool enable = true;  ///< enable post processing after a update step
  GLMB_POST_PROCESSING_METHOD method =
      GLMB_POST_PROCESSING_METHOD::STANDARD;  ///< method which should be used for post processing>
  StandardPostProcessingParams standard_pp{};
  DissHerrmannPostProcessingParams diss_herrmann_pp{};

  using traits = figcone::FieldTraits<figcone::OptionalField<&GLMBDistributionPostProcessUpdate::method>,
                                      figcone::OptionalField<&GLMBDistributionPostProcessUpdate::standard_pp>,
                                      figcone::OptionalField<&GLMBDistributionPostProcessUpdate::diss_herrmann_pp>>;
};

struct PMFusionParams
{
  std::size_t num_components = 1000;  // maximum overall calculated fused hypotheses number
  PM_MULTI_SENSOR_UPDATE_METHOD method =
      PM_MULTI_SENSOR_UPDATE_METHOD::GENERALIZED_K_BEST;  // if true -> generalizedKBest algorithm is used, else
                                                          // kBestSelection is used
  PMFusionPoissonParams poisson_method{};
  std::size_t max_num_state_components = 10;  // max number of fused state components in pm-lmb fusion!
  GLMBDistributionPostProcessUpdate post_processing{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&PMFusionParams::num_components>,
                                      figcone::OptionalField<&PMFusionParams::method>,
                                      figcone::OptionalField<&PMFusionParams::poisson_method>,
                                      figcone::OptionalField<&PMFusionParams::max_num_state_components>,
                                      figcone::OptionalField<&PMFusionParams::post_processing>>;
};

struct PUFusionParams
{
  std::size_t num_components = 10;  // maximum number of fused state components calculated by fpm fusion
  bool use_best_component =
      true;  // if true -> only best component is used, otherwise all components are merged together
  using traits = figcone::FieldTraits<figcone::OptionalField<&PUFusionParams::num_components>,
                                      figcone::OptionalField<&PUFusionParams::use_best_component>>;
};

struct LMBMultiSensorParams
{
  LMB_MULTI_SENSOR_UPDATE_METHOD method = LMB_MULTI_SENSOR_UPDATE_METHOD::FPM;
  FPMFusionParams fpm;
  PUFusionParams pu;
  using traits = figcone::FieldTraits<figcone::OptionalField<&LMBMultiSensorParams::method>,
                                      figcone::OptionalField<&LMBMultiSensorParams::fpm>,
                                      figcone::OptionalField<&LMBMultiSensorParams::pu>>;
};

struct LMBDistributionParams
{
  LMB_UPDATE_METHOD update_method = LMB_UPDATE_METHOD::LBP;
  bool calculate_single_sensor_group_updates_parallel =
      true;  ///< if thread_pool_size>0 and this is true, lmb groups are calculated parallel
  LMBDistributionPostProcessPrediction post_process_prediction{};
  LMBDistributionPostProcessUpdate post_process_update{};
  LMBMultiSensorParams multi_sensor{};
  LMBDistributionExtraction extraction{};
  LoopyBeliefProapagationParams loopy_belief_propagation{};
  using traits = figcone::FieldTraits<
      figcone::OptionalField<&LMBDistributionParams::update_method>,
      figcone::OptionalField<&LMBDistributionParams::calculate_single_sensor_group_updates_parallel>,
      figcone::OptionalField<&LMBDistributionParams::post_process_prediction>,
      figcone::OptionalField<&LMBDistributionParams::post_process_update>,
      figcone::OptionalField<&LMBDistributionParams::multi_sensor>,
      figcone::OptionalField<&LMBDistributionParams::extraction>,
      figcone::OptionalField<&LMBDistributionParams::loopy_belief_propagation>>;
};

struct GLMBDistributionPostProcessPrediction
{
  bool enable = true;               ///< enable post processing after a prediction step
  double pruning_threshold = 1e-5;  ///< delete every hypothesis with weight < pruning_threshold
  double prune_labels_existence_probability_threshold =
      1e-3;                           ///< delete every label with existence probability < pruning_threshold
  std::size_t max_hypotheses = 1000;  ///< keep at most the best max_hypotheses, delete the smaller ones
  using traits = figcone::FieldTraits<
      figcone::OptionalField<&GLMBDistributionPostProcessPrediction::pruning_threshold>,
      figcone::OptionalField<&GLMBDistributionPostProcessPrediction::prune_labels_existence_probability_threshold>,
      figcone::OptionalField<&GLMBDistributionPostProcessPrediction::max_hypotheses>>;
};

struct GLMBDistributionExtraction
{
  MO_DISTRIBUTION_EXTRACTION_TYPE type;
  double threshold;
};

struct GLMBDistributionAddTracksParams
{
  std::size_t max_number_hypotheses = 500;
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
  double equal_allocation_share_ratio;  ///< do not allocate this proportion of new hypothesis based on the weight but
                                        ///< uniformly across all old hypotheses
  std::size_t max_update_hypotheses;    ///< Every existing hypotheses can maximal generate *this number of update
                                        ///< hypotheses (used if joint_prediction and update==true)
                                        ///< anymore?!-> remove??
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

struct GLMBMultiSensorParams
{
  GLMB_MULTI_SENSOR_UPDATE_METHOD method = GLMB_MULTI_SENSOR_UPDATE_METHOD::MS;
  PMFusionParams pm;
  using traits = figcone::FieldTraits<figcone::OptionalField<&GLMBMultiSensorParams::method>,
                                      figcone::OptionalField<&GLMBMultiSensorParams::pm>>;
};

struct SocialForceInteractionPrediction
{
  bool enable = false;
  std::size_t horizon_ms = 1000;  /// check interaction at this prediction horizon
  double amplifier = 1;           // [N/kg]
  double sensitivity = 1;         // [m]
  using traits = figcone::FieldTraits<figcone::OptionalField<&SocialForceInteractionPrediction::enable>,
                                      figcone::OptionalField<&SocialForceInteractionPrediction::horizon_ms>,
                                      figcone::OptionalField<&SocialForceInteractionPrediction::amplifier>,
                                      figcone::OptionalField<&SocialForceInteractionPrediction::sensitivity>>;
};

struct NonOverlappingParams
{
  bool enabled = false;
  std::size_t horizon_ms = 1000;    // T: look-ahead horizon [s]
  double object_radius_m = 1.0;     // r: interaction radius [m]
  double separation_sigma_m = 0.5;  // sigma_d: softening width [m]
};

struct GLMBDistributionParams
{
  bool do_profiling = false;
  bool use_grouping = true;
  GLMBDistributionPostProcessPrediction post_process_prediction;
  GLMBDistributionPostProcessUpdate post_process_update;
  NonOverlappingParams non_overlapping_update;
  SocialForceInteractionPrediction social_force_interaction_prediction;
  GLMBMultiSensorParams multi_sensor;
  GLMBUpdateParams update;
  GLMBDistributionExtraction extraction;
  GLMBDistributionAddTracksParams add_tracks;
  using traits =
      figcone::FieldTraits<figcone::OptionalField<&GLMBDistributionParams::do_profiling>,
                           figcone::OptionalField<&GLMBDistributionParams::use_grouping>,
                           figcone::OptionalField<&GLMBDistributionParams::post_process_prediction>,
                           figcone::OptionalField<&GLMBDistributionParams::post_process_update>,
                           figcone::OptionalField<&GLMBDistributionParams::non_overlapping_update>,
                           figcone::OptionalField<&GLMBDistributionParams::social_force_interaction_prediction>,
                           figcone::OptionalField<&GLMBDistributionParams::multi_sensor>,
                           figcone::OptionalField<&GLMBDistributionParams::extraction>,
                           figcone::OptionalField<&GLMBDistributionParams::add_tracks>>;
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

struct PPPDistributionParams
{
  double pruning_threshold = 1e-5;
  double birth_weight = 0.005;
  using traits = figcone::FieldTraits<figcone::OptionalField<&PPPDistributionParams::pruning_threshold>,
                                      figcone::OptionalField<&PPPDistributionParams::birth_weight>>;
};

struct MBMDistributionParams
{
  double pruning_threshold_bernoulli = 1e-5;
  double recycle_threshold_bernoulli = 1e-2;
  bool enable_recycling = true;
  double pruning_threshold_hypotheses = 1e-5;
  std::size_t max_number_hypotheses = 10;
  GLMB_ASSIGNMENT_METHOD assignment_method;  ///< how to perform the MBM update
  bool enable_projection = false;
  using traits = figcone::FieldTraits<figcone::OptionalField<&MBMDistributionParams::max_number_hypotheses>,
                                      figcone::OptionalField<&MBMDistributionParams::pruning_threshold_bernoulli>,
                                      figcone::OptionalField<&MBMDistributionParams::recycle_threshold_bernoulli>,
                                      figcone::OptionalField<&MBMDistributionParams::enable_recycling>,
                                      figcone::OptionalField<&MBMDistributionParams::pruning_threshold_hypotheses>,
                                      figcone::OptionalField<&MBMDistributionParams::assignment_method>,
                                      figcone::OptionalField<&MBMDistributionParams::enable_projection>>;
};

struct PMBMDistributionParams
{
  double estimation_threshold = 0.5;
  PPPDistributionParams ppp{};
  MBMDistributionParams mbm{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&PMBMDistributionParams::estimation_threshold>,
                                      figcone::OptionalField<&PMBMDistributionParams::mbm>,
                                      figcone::OptionalField<&PMBMDistributionParams::ppp>>;
};

struct BufferParams
{
  std::size_t max_wait_duration_ms = 500;   ///< maximal duration we wait for an expected measurement
  double max_wait_duration_quantile = 0.9;  ///< duration we wait for an expected measurement based on an estimation
  std::size_t max_delta_ms = 10;            ///< max time difference of measurements in one batch
  using traits = figcone::FieldTraits<figcone::OptionalField<&BufferParams::max_wait_duration_ms>,
                                      figcone::OptionalField<&BufferParams::max_wait_duration_quantile>,
                                      figcone::OptionalField<&BufferParams::max_delta_ms>>;
};

struct ClutterRateEstimation
{
  bool enable = true;
  std::size_t short_term_window_length_ms = 550;  ///< short term estimate of lambda
  std::size_t long_term_window_length_ms = 3000;  ///< lorm term estimate of lambda
  double diff_threshold = 5;  ///< if the mean of the short term and long term estimate differ more than this threshold,
                              ///< take the short term estimate
  double quantile = 0.95;

  using traits = figcone::FieldTraits<figcone::OptionalField<&ClutterRateEstimation::short_term_window_length_ms>,
                                      figcone::OptionalField<&ClutterRateEstimation::enable>,
                                      figcone::OptionalField<&ClutterRateEstimation::long_term_window_length_ms>,
                                      figcone::OptionalField<&ClutterRateEstimation::diff_threshold>,
                                      figcone::OptionalField<&ClutterRateEstimation::quantile>>;
};
struct DetectionRateEstimation
{
  bool enable = true;
  std::size_t short_term_window_length_ms = 550;  ///< short term estimate of lambda
  std::size_t long_term_window_length_ms = 3000;  ///< lorm term estimate of lambda
  double diff_threshold = 5;  ///< if the mean of the short term and long term estimate differ more than this threshold,
                              ///< take the short term estimate
  double quantile = 0.95;
  using traits = figcone::FieldTraits<figcone::OptionalField<&DetectionRateEstimation::short_term_window_length_ms>,
                                      figcone::OptionalField<&DetectionRateEstimation::enable>,
                                      figcone::OptionalField<&DetectionRateEstimation::long_term_window_length_ms>,
                                      figcone::OptionalField<&DetectionRateEstimation::diff_threshold>,

                                      figcone::OptionalField<&DetectionRateEstimation::quantile>>;
};

struct PersistenceProbEstimation
{
  bool enable = true;
  double prior_mean = 0.95;
  double prior_var = 0.001;
  std::size_t window_length_ms = 300;  ///< how many updates should influence the prior
  using traits = figcone::FieldTraits<figcone::OptionalField<&PersistenceProbEstimation::prior_mean>,
                                      figcone::OptionalField<&PersistenceProbEstimation::enable>,
                                      figcone::OptionalField<&PersistenceProbEstimation::prior_var>,
                                      figcone::OptionalField<&PersistenceProbEstimation::window_length_ms>>;
};

struct LMBICFilterParams
{
};

struct BayesFactorParams
{
  struct ComparisonClutterRate
  {
    double mean = 20;
    double var = 300;
    using traits = figcone::FieldTraits<figcone::OptionalField<&ComparisonClutterRate::mean>,
                                        figcone::OptionalField<&ComparisonClutterRate::var>>;
  };
  struct Clutter
  {
    bool enable = true;
    ComparisonClutterRate comparison{};
    using traits =
        figcone::FieldTraits<figcone::OptionalField<&Clutter::enable>, figcone::OptionalField<&Clutter::comparison>>;
  };
  Clutter clutter{};
  struct ComparisonDetectionProb
  {
    double mean = 0.7;
    double var = 0.02;
    using traits = figcone::FieldTraits<figcone::OptionalField<&ComparisonDetectionProb::mean>,
                                        figcone::OptionalField<&ComparisonDetectionProb::var>>;
  };
  struct Detection
  {
    bool enable = true;
    std::vector<ComparisonDetectionProb> comparison{ { .mean = 0.5, .var = 0.083 },     // Beta(1, 1)
                                                     { .mean = 0.5, .var = 0.019 },     // Beta(6, 6)
                                                     { .mean = 0.8, .var = 0.026 },     // Beta(4, 1)
                                                     { .mean = 0.2, .var = 0.026 },     // Beta(1, 4)
                                                     { .mean = 0.75, .var = 0.014 },    // Beta(9, 3)
                                                     { .mean = 0.25, .var = 0.014 } };  // Beta(3, 9)
    using traits = figcone::FieldTraits<figcone::OptionalField<&Detection::enable>,
                                        figcone::OptionalField<&Detection::comparison>>;
  };
  Detection detection;

  std::size_t num_updates = 1;  ///< collect the last num_updates data together and evaluate combined
  using traits = figcone::FieldTraits<figcone::OptionalField<&BayesFactorParams::clutter>,
                                      figcone::OptionalField<&BayesFactorParams::detection>>;
};

struct TrackAssessment
{
  std::size_t window_length_ms = 150;
};

struct SelfAssessmentParams
{
  bool enable = false;
  ClutterRateEstimation clutter_rate_estimation{};
  DetectionRateEstimation detection_prob_estimation{};
  PersistenceProbEstimation persistence_prob_estimation{};
  BayesFactorParams bayes_factor{};
  TrackAssessment track_assessment{};
  std::size_t opinion_filter_size = 7;
  using traits = figcone::FieldTraits<figcone::OptionalField<&SelfAssessmentParams::enable>,
                                      figcone::OptionalField<&SelfAssessmentParams::clutter_rate_estimation>,
                                      figcone::OptionalField<&SelfAssessmentParams::detection_prob_estimation>,
                                      figcone::OptionalField<&SelfAssessmentParams::persistence_prob_estimation>,
                                      figcone::OptionalField<&SelfAssessmentParams::bayes_factor>,
                                      figcone::OptionalField<&SelfAssessmentParams::track_assessment>,
                                      figcone::OptionalField<&SelfAssessmentParams::opinion_filter_size>>;
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

struct NoFilterParams
{
  bool do_profiling;
};

struct NNFilterParams
{
  bool do_profiling;
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
  bool do_profiling;
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
  NoFilterParams no;
  NNFilterParams nn;
  GNNFilterParams gnn;
  UshiftTrackerParams ushift;
  using traits = figcone::FieldTraits<figcone::OptionalField<&FilterParams::enable>,
                                      figcone::OptionalField<&FilterParams::lmb_ic>,
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
  bool reuse_source_ids = false;  ///< set the id of a track to the id given by its source, does not perform any checks
  std::size_t max_prediction_duration_ms =
      200;  ///< how long a track will be predicted if no new stateContainers arrive,
            ///< also how long a stateContainer get maximal predicted to align them in
  TRACK_TO_TRACK_FUSION_STRATEGY track_to_track_fusion = TRACK_TO_TRACK_FUSION_STRATEGY::TAKE_BEST;
  using traits = figcone::FieldTraits<figcone::OptionalField<&TransTTTFilterParams::reuse_source_ids>,
                                      figcone::OptionalField<&TransTTTFilterParams::max_prediction_duration_ms>,
                                      figcone::OptionalField<&TransTTTFilterParams::track_to_track_fusion>>;
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
  std::vector<StaticTrackParams> tracks;
};

struct OracleMeasModelParams
{
  SourceId id;
  bool detection_model = false;
  bool clutter_model = false;
  bool meas_noise = false;
};

struct OracleParameters
{
  bool persistence_prob = false;
  bool birth_model = false;
  bool state_model_noise = false;
  std::vector<OracleMeasModelParams> meas_models;
};

struct OspaParams
{
  bool evaluate = true;
  double p = 2;   /// 2 means euclidean distance
  double c = 10;  /// cutoff distance, high value means focus on cardinality error, low value means focus on
                  /// localization error (see https://ieeexplore.ieee.org/document/4632413)
  using traits = figcone::FieldTraits<figcone::OptionalField<&OspaParams::evaluate>,
                                      figcone::OptionalField<&OspaParams::p>,
                                      figcone::OptionalField<&OspaParams::c>>;
};

struct Ospa2Params
{
  bool evaluate = false;
  double p = 2;   /// 2 means euclidean distance
  double c = 10;  /// cutoff distance, high value means focus on cardinality error, low value means focus on
                  /// localization error (see https://ieeexplore.ieee.org/document/4632413)
  std::size_t sliding_window_length_ms = 1000;
  using traits = figcone::FieldTraits<figcone::OptionalField<&Ospa2Params::evaluate>,
                                      figcone::OptionalField<&Ospa2Params::p>,
                                      figcone::OptionalField<&Ospa2Params::c>,
                                      figcone::OptionalField<&Ospa2Params::sliding_window_length_ms>>;
};

struct GospaParams
{
  bool evaluate = true;
  double p = 2;   /// 2 means euclidean distance
  double c = 10;  /// cutoff distance, high value means focus on cardinality error, low value means focus on
                  /// localization error (see https://ieeexplore.ieee.org/document/4632413)
  using traits = figcone::FieldTraits<figcone::OptionalField<&GospaParams::evaluate>,
                                      figcone::OptionalField<&GospaParams::p>,
                                      figcone::OptionalField<&GospaParams::c>>;
};

struct MetricParams
{
  OspaParams ospa{};
  Ospa2Params ospa2{};
  GospaParams gospa{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&MetricParams::ospa>,
                                      figcone::OptionalField<&MetricParams::ospa2>,
                                      figcone::OptionalField<&MetricParams::gospa>>;
};

struct TrajectoryParams
{
  std::size_t sliding_window_length_ms = 1000;
  using traits = figcone::FieldTraits<figcone::OptionalField<&TrajectoryParams::sliding_window_length_ms>>;
};

struct Params
{
  std::string version;
  std::size_t thread_pool_size = 0;  ///< the number of worker threads, 0 means no parallelization
  bool show_gui = false;             ///< show the gui at startup
  std::string name = "Tracking";
  bool save_all_data = false;
  bool sort_measurement_containers = false;
  OracleParameters oracle_model_parameters = {};
  MetricParams ground_truth_metrics = {};
  TrajectoryParams trajectory = {};
  MeasurementModelsParams meas_models{};
  StateParams state{};
  std::vector<StateModelParams> state_models{};
  std::optional<BirthModelParams> birth_model{};
  PersistenceModelParams persistence_model{};
  LMBDistributionParams lmb_distribution{};
  GLMBDistributionParams glmb_distribution{};
  PMBMDistributionParams pmbm_distribution{};
  PHDDistributionParams phd_distribution{};
  BufferParams buffer{};
  SelfAssessmentParams self_assessment{};
  FilterParams filter{};
  TTTFilterParams ttt_filter{};
  AddStaticTracksParams static_tracks{};
  using traits = figcone::FieldTraits<figcone::OptionalField<&Params::show_gui>,
                                      figcone::OptionalField<&Params::oracle_model_parameters>,
                                      figcone::OptionalField<&Params::ground_truth_metrics>,
                                      figcone::OptionalField<&Params::trajectory>,
                                      figcone::OptionalField<&Params::name>,
                                      figcone::OptionalField<&Params::save_all_data>,
                                      figcone::OptionalField<&Params::sort_measurement_containers>,
                                      figcone::OptionalField<&Params::meas_models>,
                                      figcone::OptionalField<&Params::state>,
                                      figcone::OptionalField<&Params::birth_model>,
                                      figcone::OptionalField<&Params::persistence_model>,
                                      figcone::OptionalField<&Params::lmb_distribution>,
                                      figcone::OptionalField<&Params::glmb_distribution>,
                                      figcone::OptionalField<&Params::pmbm_distribution>,
                                      figcone::OptionalField<&Params::phd_distribution>,
                                      figcone::OptionalField<&Params::buffer>,
                                      figcone::OptionalField<&Params::self_assessment>,
                                      figcone::OptionalField<&Params::filter>,
                                      figcone::OptionalField<&Params::ttt_filter>,
                                      figcone::OptionalField<&Params::static_tracks>>;
  StateModelParams& state_model_params(StateModelId id)
  {
    assert([&] {
      if (std::ranges::find(state_models, id, &StateModelParams::id) == state_models.end())
      {
        LOG_FATAL("state model params requested for state model that does not exist.");
        LOG_FATAL(std::stacktrace::current());
        return false;
      }
      return true;
    }());
    return *std::ranges::find(state_models, id, &StateModelParams::id);
  }
  [[nodiscard]] StateModelParams const& state_model_params(StateModelId id) const
  {
    assert([&] {
      if (std::ranges::find(state_models, id, &StateModelParams::id) == state_models.end())
      {
        LOG_FATAL("state model params requested for state model that does not exist.");
        LOG_FATAL(std::stacktrace::current());
        return false;
      }
      return true;
    }());
    return *std::ranges::find(state_models, id, &StateModelParams::id);
  }
  [[nodiscard]] GaussianMeasurementModelParams const& meas_model_params(SourceId const& id) const
  {
    assert([&] {
      int found = 0;
      for (GaussianMeasurementModelParams const& model_params : meas_models.gaussian_models)
      {
        if (model_params.id == id or
            (model_params.id.value_.back() == '*' and model_params.id.value_.starts_with(id.value_)))
        {
          found++;
        }
      }
      if (found != 1)
      {
        LOG_FATAL("No or multiple measurement models found that match the Source Id " << id);
        for (GaussianMeasurementModelParams const& model_params : meas_models.gaussian_models)
        {
          if (model_params.id == id or
              (model_params.id.value_.back() == '*' and model_params.id.value_.starts_with(id.value_)))
          {
            LOG_FATAL("Match Model: " << model_params.id);
          }
          else
          {
            LOG_FATAL("NO match Model: " << model_params.id);
          }
        }
        return false;
      }
      return true;
    }());
    for (GaussianMeasurementModelParams const& model_params : meas_models.gaussian_models)
    {
      if (model_params.id == id or
          (model_params.id.value_.back() == '*' and model_params.id.value_.starts_with(id.value_)))
      {
        return model_params;
      }
    }
    assert(false);
    std::unreachable();
  }
  [[nodiscard]] GaussianMeasurementModelParams& meas_model_params(SourceId const& id)
  {
    assert([&] {
      int found = 0;
      for (GaussianMeasurementModelParams const& model_params : meas_models.gaussian_models)
      {
        if (model_params.id == id or
            (model_params.id.value_.back() == '*' and model_params.id.value_.starts_with(id.value_)))
        {
          found++;
        }
      }
      if (found != 1)
      {
        LOG_FATAL("No or multiple measurement models found that match the Source Id " << id);
        for (GaussianMeasurementModelParams const& model_params : meas_models.gaussian_models)
        {
          if (model_params.id == id or
              (model_params.id.value_.back() == '*' and model_params.id.value_.starts_with(id.value_)))
          {
            LOG_FATAL("Match Model: " << model_params.id);
          }
          else
          {
            LOG_FATAL("NO match Model: " << model_params.id);
          }
        }
        return false;
      }
      return true;
    }());
    for (GaussianMeasurementModelParams& model_params : meas_models.gaussian_models)
    {
      if (model_params.id == id or
          (model_params.id.value_.back() == '*' and model_params.id.value_.starts_with(id.value_)))
      {
        return model_params;
      }
    }
    assert(false);
    std::unreachable();
  }
  [[nodiscard]] bool have_measurement_model(SourceId const& id) const
  {
    int found = 0;
    for (GaussianMeasurementModelParams const& model_params : meas_models.gaussian_models)
    {
      if (model_params.id == id or
          (model_params.id.value_.back() == '*' and model_params.id.value_.starts_with(id.value_)))
      {
        found++;
      }
    }
    if (found > 1)
    {
      LOG_FATAL("Multiple measurement models found that match the Source Id " << id);
      for (GaussianMeasurementModelParams const& model_params : meas_models.gaussian_models)
      {
        if (model_params.id == id or
            (model_params.id.value_.back() == '*' and model_params.id.value_.starts_with(id.value_)))
        {
          LOG_FATAL("Match Model: " << model_params.id);
        }
        else
        {
          LOG_FATAL("NO match Model: " << model_params.id);
        }
      }
      return false;
    }
    return true;
  }
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
  if (std::ranges::find(params.state.multi_model.use_state_models, params.state.estimation.output_state_model) ==
      params.state.multi_model.use_state_models.end())
  {
    LOG_FATAL(std::format("output state model {} was not in the use state models {}. I inserted it.",
                          params.state.estimation.output_state_model,
                          params.state.multi_model.use_state_models));
    params.state.multi_model.use_state_models.push_back(params.state.estimation.output_state_model);
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
struct StringConverter<ttb::GLMB_POST_PROCESSING_METHOD>
{
  static std::optional<ttb::GLMB_POST_PROCESSING_METHOD> fromString(const std::string& data)
  {
    return ttb::to_GLMB_POST_PROCESSING_METHOD(data);
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
struct StringConverter<ttb::TRACK_TO_TRACK_FUSION_STRATEGY>
{
  static std::optional<ttb::TRACK_TO_TRACK_FUSION_STRATEGY> fromString(const std::string& data)
  {
    return ttb::to_TRACK_TO_TRACK_FUSION_STRATEGY(data);
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
struct StringConverter<ttb::Matrix<double>>
{
  static std::optional<ttb::Matrix<double>> fromString(const std::string& data)
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
    ttb::Matrix<double> matrix(vals.size(), vals.front().size());
    for (ttb::Index i = 0; i < matrix.rows(); ++i)
    {
      for (ttb::Index j = 0; j < matrix.cols(); ++j)
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
struct StringConverter<ttb::SourceId>
{
  static std::optional<ttb::SourceId> fromString(const std::string& data)
  {
    std::stringstream ss(data);
    ttb::SourceId::type val;
    ss >> val;
    return ttb::SourceId{ val };
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
struct StringConverter<ttb::LMB_MULTI_SENSOR_UPDATE_METHOD>
{
  static std::optional<ttb::LMB_MULTI_SENSOR_UPDATE_METHOD> fromString(const std::string& data)
  {
    return ttb::to_LMB_MULTI_SENSOR_UPDATE_METHOD(data);
  }
};

template <>
struct StringConverter<ttb::PM_MULTI_SENSOR_UPDATE_METHOD>
{
  static std::optional<ttb::PM_MULTI_SENSOR_UPDATE_METHOD> fromString(const std::string& data)
  {
    return ttb::to_PM_MULTI_SENSOR_UPDATE_METHOD(data);
  }
};

template <>
struct StringConverter<ttb::GLMB_MULTI_SENSOR_UPDATE_METHOD>
{
  static std::optional<ttb::GLMB_MULTI_SENSOR_UPDATE_METHOD> fromString(const std::string& data)
  {
    return ttb::to_GLMB_MULTI_SENSOR_UPDATE_METHOD(data);
  }
};

}  // namespace figcone
