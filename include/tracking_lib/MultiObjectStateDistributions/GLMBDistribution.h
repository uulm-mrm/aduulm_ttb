#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Concepts.h"
#include "tracking_lib/MultiObjectStateDistributions/Hypothesis.h"
#include "tracking_lib/DataContainer/Database.h"
#include "tracking_lib/Misc/Profiler.h"

#include "tracking_lib/DataContainer/DataSourceContainer.h"
#include "tracking_lib/Misc/GibbsSampler.h"

#include <optional>
#include <unordered_map>

namespace ttb
{

/// Represents a generalized labeled multi-Bernoulli distribution
template <class UDT>
class GLMBDistribution
{
public:
  explicit GLMBDistribution(TTBManager<UDT>* manager);
  /// checks whether the tracks and hypotheses are consistent + valid
  [[nodiscard]] bool is_valid() const;
  /// Predict the distribution
  void predict(Duration deltaT, PredictionInformation const& prediction_info);
  /// Perform the GLMB Update as described in:
  /// B. -T. Vo and B. -N. Vo, "Labeled Random Finite Sets and Multi-Object Conjugate Priors," in IEEE Transactions on
  /// Signal Processing, vol. 61, no. 13, pp. 3460-3475, July1, 2013, doi: 10.1109/TSP.2013.2259822.
  /// B. -N. Vo, B. -T. Vo and H. G. Hoang, "An Efficient Implementation of the Generalized Labeled Multi-Bernoulli
  /// Filter," in IEEE Transactions on Signal Processing, vol. 65, no. 8, pp. 1975-1987, 15 April15, 2017,
  /// doi: 10.1109/TSP.2016.2641392.
  void update(DataSourceContainer<UDT> const& measurements, bool do_innovate = true, bool do_split_and_merge = true);

private:
  [[nodiscard]] std::vector<Hypothesis<UDT>>
  update_single_hypothesis(Hypothesis<UDT> const& prior_hypothesis,
                           std::size_t num_update_hypothesis,
                           DataSourceContainer<UDT> const& data_source_container) const;

public:
  /// multi-sensor glmb update: either pm or so_ms_gibbs
  void update(std::vector<DataSourceContainer<UDT>> const& data_source_containers,
              bool is_pmlmb = false,
              bool do_innovate = true);
  /// number of unique labels
  [[nodiscard]] std::size_t number_of_labels() const;
  /// distribution of the number of objects
  [[nodiscard]] Vector<UDT> const& cardinality_distribution() const;
  /// number of hypotheses per cardinality
  [[nodiscard]] Vector<std::size_t> num_hypotheses_per_cardinality() const;
  /// expectation value of the number of objects
  [[nodiscard]] UDT estimated_number_of_tracks() const;
  /// add the given states to the current birth tracks
  void add_tracks(std::vector<StateId> const& states);
  void add_tracks(std::vector<BernoulliState<UDT>> states);
  /// add the birth_tracks to the hypotheses. Birth tracks ara a priori independent ant therefore every birth tracks
  /// makes a new separate group
  void add_birth_tracks();
  /// return an estimation
  std::vector<BernoulliState<UDT>> get_estimate() const;
  /// return all tracks
  std::vector<StateId> all_tracks() const;
  /// string representation
  [[nodiscard]] std::string to_string(std::string_view prefix = "",
                                      bool print_tracks = false,
                                      bool print_hyps = false) const;
  /// Compute a map that express the probability of a measurement is used to update
  /// This is used for the adaptive birth model to generate birth candidates accordingly.
  /// Furthermore, the existence probs for each track label and for each track id is calculated
  [[nodiscard]] std::map<MeasurementId, UDT> const&
  meas_assignment_prob(DataSourceContainer<UDT> const& data_source_container) const;
  /// all tracks of label
  [[nodiscard]] std::unordered_map<Label, std::vector<StateId>> const& label_2_tracks() const;
  /// the existence Prob of a label
  [[nodiscard]] std::unordered_map<Label, UDT> const& label_2_ex_prob() const;
  /// the existence probability of a track
  [[nodiscard]] std::unordered_map<StateId, UDT> const& track_2_ex_prob() const;

  [[nodiscard]] Vector<UDT> compute_clutter_distribution(DataSourceContainer<UDT> const& data_source_container) const;
  /// return the (categorical) distribution of the number of Clutter c, i.e, P(c=k) = return.at(k)
  [[nodiscard]] Vector<UDT> const& clutter_distribution(DataSourceContainer<UDT> const& data_source_container) const;

  [[nodiscard]] std::vector<Matrix<UDT>>
  compute_detection_distribution(DataSourceContainer<UDT> const& data_source_container) const;
  /// return the probability of the #tracks / #detections
  [[nodiscard]] std::vector<Matrix<UDT>> const&
  detection_distribution(DataSourceContainer<UDT> const& data_source_container) const;
  /// number of surviving / death tracks, matrices in the vector are independent
  std::pair<std::vector<Matrix<UDT>>, Duration> surviving_distribution();
  /// number of detected tracks and weight of all other terms
  std::map<std::pair<UDT, UDT>, UDT> bayes_factor_statistics_measurement_likelihood_without_detection(
      DataSourceContainer<UDT> const& data_source_container) const
  {
    return {};
  }
  /// number of missed tracks and weight of all other terms
  std::map<UDT, UDT> bayes_factor_statistics_measurement_likelihood_without_clutter(
      DataSourceContainer<UDT> const& data_source_container) const
  {
    return {};
  }
  UDT bayes_factor_statistics_measurement_likelihood(DataSourceContainer<UDT> const& data_source_container) const
  {
    return 1;
  }

private:
  struct MergedHypotheses
  {
    Hypotheses<UDT> hypotheses;
    std::vector<DataSourceContainer<UDT>> data_source_containers;
  };
  enum class SplitAndMergeHypothesesType
  {
    COMMON_MEASUREMENT,
    INTERACTING_STATE,
  };
  /// split and merge hypotheses if necessary
  /// M. Beard, B. T. Vo and B. -N. Vo, "A Solution for Large-Scale Multi-Object Tracking," in IEEE Transactions on
  /// Signal Processing, vol. 68, pp. 2754-2769, 2020, doi: 10.1109/TSP.2020.2986136.
  /// Split a hypotheses, if tracks are no longer connected through hypotheses
  /// Merge hypotheses of they have common measurements
  [[nodiscard]] std::vector<MergedHypotheses>
  split_and_merge_hypotheses(std::vector<Hypotheses<UDT>> const& hypotheseses,
                             std::span<DataSourceContainer<UDT> const> data_source_containers,
                             SplitAndMergeHypothesesType type) const;
  /// split into different labels
  [[nodiscard]] std::vector<Hypotheses<UDT>> split_hypotheses(std::vector<Hypotheses<UDT>> const& hypotheseses) const;

  /// split the hypotheses into smaller ones where in one hypotheses group is only one label
  std::vector<Hypotheses<UDT>> split_hypotheses(Hypotheses<UDT> const& original_hypotheses) const;

  /// merge hypotheses with interacting states, i.e. state that are within the prediction horizon nearer than the
  /// min_distance
  std::vector<Hypotheses<UDT>> merge_hypotheses_interacting_states(std::vector<Hypotheses<UDT>> const& hypotheseses,
                                                                   Duration prediction_horizon,
                                                                   double min_distance) const;
  /// Product-Multi Sensor GLMB Update
  /// M. Herrmann, C. Hermann and M. Buchholz,
  /// "Distributed Implementation of the Centralized Generalized Labeled Multi-Bernoulli Filter," in IEEE Transactions
  /// on Signal Processing, vol. 69, pp. 5159-5174, 2021, doi: 10.1109/TSP.2021.3107632.
  /// M. Herrmann, T. Luchterhand, C. Hermann, T. Wodtko, J. Strohbeck
  /// and M. Buchholz, "Notes on the Product Multi-Sensor Generalized Labeled Multi-Bernoulli Filter and its
  /// Implementation," 2022 25th International Conference on Information Fusion (FUSION), Linköping, Sweden, 2022, pp.
  /// 1-8, doi: 10.23919/FUSION49751.2022.9841275.
  void pm_update(std::vector<DataSourceContainer<UDT>> const& data_source_containers, bool is_pmlmb = false);
  /// fuse the posterior GLMBs that derived from a common prior
  void pm_fusion(std::vector<GLMBDistribution> posteriors, bool is_pmlmb = false);
  /// Suboptimal-Gibbs Sampling Multi-Sensor update
  /// B. -N. Vo, B. -T. Vo and M. Beard, "Multi-Sensor Multi-Object Tracking With the Generalized Labeled
  /// Multi-Bernoulli Filter," in IEEE Transactions on Signal Processing, vol. 67, no. 23, pp. 5952-5967, 1 Dec.1, 2019,
  /// doi: 10.1109/TSP.2019.2946023.
  void so_ms_gibbs_update(std::vector<DataSourceContainer<UDT>> const& data_source_containers);
  /// Suboptimal-Gibbs Sampling Multi-Sensor update for a single hypothesis
  [[nodiscard]] std::pair<std::vector<Hypothesis<UDT>>, std::vector<State<UDT>>>
  so_ms_gibbs_update_single_hypothesis(Hypothesis<UDT> const& prior_hypothesis,
                                       std::size_t num_update_hypothesis,
                                       std::vector<DataSourceContainer<UDT>> const& data_source_containers) const;
  /// How many new hypotheses should be created per old hypothesis
  [[nodiscard]] std::unordered_map<HypothesisId, std::size_t> allocation(Hypotheses<UDT> const& hypotheses) const;
  /// reset all cache variable
  void reset_caches() const;
  /// post process the prediction
  std::size_t post_process_prediction();
  /// do the post process like in dissertation of Martin Herrmann
  /// https://oparu.uni-ulm.de/items/7d363d70-eee7-46e0-a999-678813e4c37e! In really complex scenarios relevant!
  std::size_t post_process_update_diss_herrmann();
  /// do the post process in a standard way (remove hyps with to small weights, limit states per label)
  std::size_t post_process_update_standard();
  /// post process the update
  std::size_t post_process_update();
  /// delete hypotheses with weight smaller than the threshold
  std::pair<double, std::size_t> prune_hypotheses_weight_threshold(double threshold);
  /// delete labels with existence probability smaller than the threshold
  std::pair<double, std::size_t> prune_labels_existence_probability_threshold(double threshold);
  /// keep at most max_hypotheses, delete the ones with the smallest weight
  std::pair<double, std::size_t> prune_max_hypotheses(std::size_t max_hypotheses);

  /// Build track to measurement association cost matrix
  [[nodiscard]] std::pair<Cost, Cost>
  cost_matrix(std::vector<StateId> const& tracks,
              std::span<DataSourceContainer<UDT> const> data_source_containers) const;

  [[nodiscard]] std::pair<Cost, Cost> cost_matrix(std::vector<StateId> const& tracks,
                                                  DataSourceContainer<UDT> const& data_source_container) const;

public:
  /// access to the TTBManager
  TTBManager<UDT>* _manager;
  /// current Hypotheses
  /// the individual elements of the vector are independent groups and (mathematically) GLMB distributions on their own
  std::vector<Hypotheses<UDT>> _hypotheses;

private:
  /// the current birth tracks. These are added before the next prediction step
  std::vector<StateId> birth_tracks;
  /// tracks before the n-th prediction step -> needed for estimation of persistence_prob
  SlidingWindow<std::unordered_map<Label, UDT>> _previous_label_ex_probs;
  /// CACHE VARIABLES to save some computations
  /// saves existence probability of each track label for later use in convertUpdatedGLMB2LMB() in LMBDistribution
  mutable std::optional<std::unordered_map<Label, UDT>> _label2exProb;
  mutable std::optional<std::unordered_map<StateId, UDT>> _trackId2exProb;
  /// --> cardinalityDist()
  mutable std::optional<Vector<UDT>> _cacheCardDist;
  /// log of cardinality distribution
  mutable std::optional<Vector<UDT>> _cacheCardDistLog;
  /// contains number of hypotheses per card
  mutable std::optional<Indices> _numHypsPerCard;
  /// Map holding a list of tracks for each track label
  mutable std::optional<std::unordered_map<Label, std::vector<StateId>>> _label2Tracks;
  mutable std::optional<std::unordered_map<Label, std::vector<HypothesisId>>> _label2Hypotheses;
  /// how likely was this measurement assigned to a track
  mutable std::map<DataSourceContainerId, std::map<MeasurementId, UDT>> _meas_assignment_prob;
  /// i=#clutter -> prob
  std::map<DataSourceContainerId, Vector<UDT>> _clutter_dist;
  /// i=#tracks j=#detections -> prob
  std::map<DataSourceContainerId, std::vector<Matrix<UDT>>> _detection_probs;
  /// i=#tracks j=#detections -> weight * meas_likelihood
  std::map<DataSourceContainerId, Matrix<UDT>> _prior_weight_detection_distribution;
  std::map<DataSourceContainerId, Vector<UDT>> _prior_weight_clutter_distribution;
  // mutable std::map<DataSourceContainerId, double> _meas_update_likelihood;

  /// getProportionalAllocation Cache
  mutable std::optional<std::unordered_map<HypothesisId, std::size_t>> _kBestCache;
  /// ID of this GLMB distribution
  MODistributionId _id{ _idGenerator.getID() };
  static IDGenerator<MODistributionId> _idGenerator;
};

static_assert(CPTCommonMODist<GLMBDistribution<double>, double>);
static_assert(CPTICUpdate<GLMBDistribution<double>, double>);
static_assert(CPTMSUpdate<GLMBDistribution<double>, double>);
static_assert(CPTSelfAssessableMODist<GLMBDistribution<double>, double>);

}  // namespace ttb
