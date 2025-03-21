#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Classification/StateClassification.h"
#include "tracking_lib/Distributions/BaseDistribution.h"

#include <any>

namespace ttb
{

class TTBManager;
class MeasurementContainer;
class BaseDistribution;
class EgoMotionDistribution;

/// Represents a State, i.e., a single object
/// Supports multi-model kinematic, i.e., the same object is described by different state models with
/// different kinematic components.
/// The weights of the distribution must always sum to 1, the weight of the different state models represents their
/// multi-model weight
class State final
{
public:
  State(TTBManager* manager, std::map<StateModelId, std::unique_ptr<BaseDistribution>> dist);

  State(State const& other);
  State(State&& other) noexcept;
  State& operator=(State const& other);
  State& operator=(State&& other) noexcept;
  ~State();
  /// is this valid?
  [[nodiscard]] bool isValid() const;
  /// are all stateModel Distributions empty?
  [[nodiscard]] bool isEmpty() const;
  /// string representation
  [[nodiscard]] std::string toString(std::string prefix = "") const;
  /// Predict the State Distribution
  void predict(Duration deltaT, EgoMotionDistribution const& egoMotion);
  /// compensate only the egoMotion for the state
  void compensateEgoMotion(Duration dt, EgoMotionDistribution const& egoMotion);
  /// calculate the innovation for this object
  void innovate(MeasurementContainer const& measContainer);
  /// Merge, Prune, Truncate and Normalize all underlying Distributions
  void postProcess();
  /// reset the prior_ids of gaussian distributions (needed for fpm_lmb and pm_lmb)
  void resetPriorId();
  /// Return the BaseDistribution corresponding to the StateModel with biggest weight
  [[nodiscard]] std::pair<StateModelId, BaseDistribution const&> bestState() const;
  /// return an Estimate of the State with the specified method, BaseDistribution
  [[nodiscard]] std::pair<StateModelId, std::unique_ptr<BaseDistribution>> getEstimate() const;
  /// Get the sum of weights of all underlying Distributions
  [[nodiscard]] double sumWeights() const;
  /// Multiply the weight of all underlying Distributions with the factor
  void multiplyWeights(double factor);
  /// sum the NIS weights
  [[nodiscard]] double sumNisWeights() const;
  /// multiply the NIS weights
  void multiplyNisWeights(double factor);
  /// Merge the other StateDistribution into this, all underlying Distributions are merged
  void merge(State other, bool enablePostProcess = true);
  /// Merge all other StateDistributions into this, all underlying Distributions are merged
  void merge(std::vector<State> others);
  /// Perform the Markov State Transition Step
  void markovTransition();
  void performStageUpdate(std::size_t num_sensors);
  void performHistoryStageLogic(std::size_t num_sensors);

  TTBManager* _manager;
  StateId _id{ _idGenerator.getID() };
  Label _label{ _labelGenerator.getID() };
  std::map<StateModelId, std::unique_ptr<BaseDistribution>> _state_dist;
  std::map<MeasModelId, Innovation> _innovation;
  classification::StateClassification _classification;
  Probability _existenceProbability{
    std::numeric_limits<double>::quiet_NaN()
  };  ///< the existence probability of this track. Not all filters compute this value.
  Probability _survival_probability{ std::numeric_limits<double>::quiet_NaN() };  ///< the survival probability of the
                                                                                  ///< track of the last prediction
  STAGE _stage = STAGE::TENTATIVE;
  double _score{ std::numeric_limits<double>::quiet_NaN() };  ///< Track Existence Score according to Blackman/Popoli
                                                              ///< Design and Analysis of modern Tracking Systems
                                                              ///< Chapter 6.2
  Time _time{ Time(Duration::zero()) };

  struct MetaData
  {
    std::size_t _numPredictions{ 0 };
    std::size_t _numUpdates{ 0 };
    Time _timeOfLastAssociation{ Time(Duration::zero()) };
    Duration _durationSinceLastAssociation{ Duration::zero() };
    MeasurementId _lastAssociatedMeasurement{ NOT_DETECTED };
    std::vector<bool> _detection_hist;  /// list of recent hits and misses (cycles with and without measurement
                                        /// associations for the track) more recent ones are in the back of the list
  };
  MetaData _meta_data;

  double _weight_mis_detection{ 0 };  ///< how much attributed the mis detection in the last update
  std::map<StateModelId, std::vector<std::tuple<Probability, std::map<COMPONENT, Nis>>>> _nis;  ///< Weighted
                                                                                                ///< element-wise NIS of
                                                                                                ///< the last update.
                                                                                                ///< The weights
                                                                                                ///< correspond to the
                                                                                                ///< weights of the
                                                                                                ///< mixture components
  bool _detectable = false;  ///< flag indicating if this track was inside the fov of the last update
  static IDGenerator<Label> _labelGenerator;
  static IDGenerator<StateId> _idGenerator;
  std::map<std::string, std::any> _misc{};

private:
  mutable std::optional<std::pair<StateModelId, std::unique_ptr<BaseDistribution>>> _estimationCache;
};

}  // namespace ttb