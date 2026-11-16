#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Classification/StateClassification.h"
#include "tracking_lib/Classification/MeasClassification.h"
#include "tracking_lib/DataContainer/SlidingWindow.h"
#include "tracking_lib/Distributions/Distribution.h"
#include <any>

namespace ttb
{

class BaseDistribution;
class EgoMotionDistribution;

/// Represents a State, i.e., a single object
/// Supports multi-model kinematic, i.e., the same object is described by different state models with
/// different kinematic components.
/// The weights of the distribution must always sum to 1, the weight of the different state models represents their
/// multi-model weight
const Opinion default_opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.0 } },
                               Opinion::BeliefType{ std::array{ 0.5, 0.5 } } };

template <class UDT>
class State
{
public:
  State(TTBManager<UDT> const* manager,
        std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> dist);
  [[nodiscard]] std::map<StateModelId, std::pair<UDT, Distribution<UDT>>> build_consistent_state_dist(
      std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> dist) const;
  /// is this valid?
  [[nodiscard]] bool isValid() const;
  /// are all stateModel Distributions empty?
  [[nodiscard]] bool isEmpty() const;
  /// string representation
  [[nodiscard]] std::string toString(std::string_view prefix = "") const;
  /// Predict the State Distribution
  void predict(Duration deltaT, EgoMotionDistribution const& egoMotion);
  /// compensate only the egoMotion for the state
  void compensateEgoMotion(Duration dt, EgoMotionDistribution const& egoMotion);
  /// calculate the innovation for this object
  [[nodiscard]] std::
      tuple<StateDetection<UDT>, std::vector<std::tuple<StateId, StateId, State, UDT>>, std::vector<MeasClutter<UDT>>>
      innovate(DataSourceContainer<UDT> const& measContainer) const;
  /// Merge, Prune, Truncate and Normalize all underlying Distributions
  void postProcess();
  /// Merge all components to one component
  void approximateToOneComponent();
  /// Return the BaseDistribution corresponding to the StateModel with biggest weight
  [[nodiscard]] std::pair<StateModelId, Distribution<UDT> const&> bestState() const;
  /// return an Estimate of the State with the specified method, BaseDistribution
  [[nodiscard]] std::pair<StateModelId, GaussianDistribution<UDT>> getEstimate() const;

private:
  /// Get the sum of weights of all underlying Distributions
  [[nodiscard]] UDT sumWeights() const;
  /// Multiply the weight of all underlying Distributions with the factor
  void multiplyWeights(UDT factor);
  /// Perform the Markov State Transition Step
  void markovTransition();

public:
  void performStageUpdate(std::size_t num_sensors);
  void performHistoryStageLogic(Index num_sensors);
  /// transform the state into the given coordinate system
  [[nodiscard]] State transform(SE3Trafo const& trafo) const;

  TTBManager<UDT> const* _manager;
  /// unique identifier for this state
  StateId _id{ _idGenerator.getID() };
  /// last relevant prior id, used to check possible fusion and merging steps
  StateId _prior_id{ NO_PRIOR };
  /// previous identifiers for this state, not necessarily in order
  std::vector<StateId> _all_prior_ids{ NO_PRIOR };
  /// identifier for one object, consistent over time
  Label _label{ _labelGenerator.getID() };  ///< internal object id, consistent over time

  std::optional<ObjectId> _objectId;  ///< external object id, should be consistent over time, same semantic as the
                                      ///< label, but comes from detector
  STATE_TYPE _type{ STATE_TYPE::PRIOR };
  std::map<StateModelId, std::pair<UDT, Distribution<UDT>>> _state_dist;
  classification::StateClassification<UDT> _classification;
  std::optional<classification::MeasClassification> _meas_classification;

  STAGE _stage = STAGE::TENTATIVE;
  UDT _score{ UDT{ std::numeric_limits<double>::quiet_NaN() } };  ///< Track Existence Score according to
                                                                  ///< Blackman/Popoli Design and Analysis of modern
                                                                  ///< Tracking Systems Chapter 6.2
  Time _time{ Time(Duration::zero()) };

  struct MetaData
  {
    std::size_t _numPredictions{ 0 };
    std::size_t _numUpdates{ 0 };
    Time _timeOfLastAssociation{ Time(Duration::zero()) };
    Duration _durationSinceLastAssociation{ Duration::zero() };
    /// meas likelihood of the last update
    std::map<SourceId, UDT> _last_meas_log_likelihood{};
    /// how much contributed this measurement to the update
    std::map<SourceId, std::vector<std::pair<MeasurementId, UDT>>> _meas_update_weight{};
    std::map<SourceId, bool> _detectable;  ///< flag indicating if this track was detectable during the
                                           ///< update
    std::map<SourceId, std::map<StateModelId, std::vector<std::tuple<UDT, std::map<COMPONENT, UDT>>>>>
        _nis;  ///< Weighted
    ///< element-wise NIS of
    ///< the last update.
    ///< The weights
    ///< correspond to the
    ///< weights of the
    ///< mixture components
    std::vector<bool> _detection_hist;  /// list of recent hits and misses (cycles with and without measurement
                                        /// associations for the track) more recent ones are in the back of the list
    /// track assessment score
    Opinion _assessment{ Opinion::DogmaticTrust() };
    std::map<std::string, Opinion> _assessment_debug_info;
    std::vector<std::pair<double, Opinion>> _input_opinions_debug_info;
  };
  MetaData _meta_data;
  static IDGenerator<Label> _labelGenerator;
  static IDGenerator<StateId> _idGenerator;
  std::map<std::string, std::any> _misc{};
  UDT _log_weight{ std::numeric_limits<double>::quiet_NaN() };
  UDT _get_weight() const;
};

template <class UDT>
struct BernoulliState
{
  State<UDT> state;
  UDT existence_probability;
};

template <class UDT>
[[nodiscard]] State<UDT> merge(TTBManager<UDT> const& manager, std::vector<std::pair<UDT, StateId>> const& states);

/// fuse the posteriors. All posteriors have to descend from the same prior.
/// If you use this function a lot, consider changing the representation of the Gaussian Distributions from covariance
/// to precision!.
template <class UDT>
[[nodiscard]] std::optional<State<UDT>>
fuse_posteriors(State<UDT> const& prior,
                std::vector<State<UDT>> const& posteriors,
                std::size_t max_components,
                PM_MULTI_SENSOR_UPDATE_METHOD method = PM_MULTI_SENSOR_UPDATE_METHOD::GENERALIZED_K_BEST,
                double equal_allocation_share_ratio = 0.0,
                bool is_pm_glmb = false);

/// fuse the posteriors using an approximation if given states are gaussian mixtures. All posteriors have to descend
/// from the same prior. If you use this function a lot, consider changing the representation of the Gaussian
/// Distributions from covariance to precision!
/// see for original implementation:
/// https://github.com/scjrobertson/multisensor-lmb-filters/blob/main/multisensorLmb/puLmbTrackMerging.m
template <class UDT>
[[nodiscard]] std::optional<State<UDT>> fuse_posteriors_pu(State<UDT> const& prior,
                                                           std::vector<State<UDT>>& posteriors,
                                                           bool use_best_component,
                                                           std::size_t num_components);

/// the (approximate) KL divergence if the State is viewed as Bernoulli RFS
template <class UDT>
[[nodiscard]] double kl_divergence(State<UDT> const& first, State<UDT> const& second);

/// create measurement state
template <class UDT>
[[nodiscard]] std::optional<State<UDT>>
create_meas_state(TTBManager<UDT>* manager,
                  std::vector<std::pair<UDT, GaussianDistribution<UDT>>> dist,
                  Time time,
                  Components const& meas_comps);
template <class UDT>
[[nodiscard]] std::optional<State<UDT>>
create_meas_state(TTBManager<UDT>* manager, Distribution<UDT> dist, Time time, Components const& meas_comps);

template <class UDT>
[[nodiscard]] std::optional<State<UDT>> create_meas_state(TTBManager<UDT>* manager,
                                                          GaussianDistribution<UDT> const& gauss_dist,
                                                          Time time,
                                                          Components const& meas_comps);

}  // namespace ttb

template <class UDT>
struct std::formatter<ttb::State<UDT>>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::State<UDT> const& state, std::format_context& ctx) const
  {
    return std::format_to(ctx.out(), "\n{}", state.toString());
  }
};