#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/MultiObjectStateDistributions/GLMBDistribution.h"
#include "tracking_lib/DataContainer/DataSourceContainer.h"
#include "tracking_lib/TTBTypes/Concepts.h"

namespace ttb
{

/// Represents a labeled multi-Bernoulli distribution
template <class UDT>
class LMBDistribution
{
public:
  explicit LMBDistribution(TTBManager<UDT>* manager);
  LMBDistribution(TTBManager<UDT>* manager, std::vector<State<UDT>> tracks);
  LMBDistribution(TTBManager<UDT>* manager, std::vector<StateId> tracks);
  /// Convert/Approximate the given GLMBDistribution into this LMBDistribution
  void approx_from(GLMBDistribution<UDT> glmb);
  /// string representation
  [[nodiscard]] std::string to_string(std::string const& prefix = "") const;
  /// check if all tracks are valid
  [[nodiscard]] bool is_valid() const;
  /// return all tracks of the filter
  [[nodiscard]] std::vector<StateId> all_tracks() const;
  /// Return an estimate of this distribution.
  [[nodiscard]] std::vector<BernoulliState<UDT>> get_estimate() const;
  /// Predict this distribution deltaT time into the future.
  void predict(Duration deltaT, PredictionInformation const& egoDist);
  /// Update this LMB distribution with the given MeasurementContainer.
  /// The update method is selected depending on the parameters.
  void update(DataSourceContainer<UDT> const& measurementContainer, bool do_innovate = true);
  /// multi-sensor lmb update -> this uses the multi-sensor GLMB update
  void update(std::vector<DataSourceContainer<UDT>> const& data_source_containers);
  /// add a track to this LMBDistribution
  void add_tracks(std::vector<BernoulliState<UDT>> tracks);
  void add_tracks(std::vector<StateId> const& tracks);
  /// the clutter distribution of the last update for this data_container
  [[nodiscard]] Vector<UDT> clutter_distribution(DataSourceContainer<UDT> const& data_source_container) const;
  /// the detection distribution of the last update for this data_container
  [[nodiscard]] std::vector<Matrix<UDT>>
  detection_distribution(DataSourceContainer<UDT> const& data_source_container) const;
  /// number of detected,missed tracks and weight of all other terms
  [[nodiscard]] std::map<std::pair<UDT, UDT>, UDT> bayes_factor_statistics_measurement_likelihood_without_detection(
      DataSourceContainer<UDT> const& data_source_container) const;
  /// number of missed tracks and weight of all other terms
  [[nodiscard]] std::map<UDT, UDT> bayes_factor_statistics_measurement_likelihood_without_clutter(
      DataSourceContainer<UDT> const& data_source_container) const;
  /// measurement likelihood
  UDT bayes_factor_statistics_measurement_likelihood(DataSourceContainer<UDT> const& data_source_container) const;
  /// the probability/weight a measurement contributed to the update of any track in the last update
  [[nodiscard]] std::map<MeasurementId, UDT> const&
  meas_assignment_prob(DataSourceContainer<UDT> const& data_container) const;
  /// #tracks survived / #tracks died
  [[nodiscard]] std::pair<std::vector<Matrix<UDT>>, Duration> surviving_distribution();

private:
  void disable_state_distribution_post_processing();
  void reset_state_distribution_post_processing();
  void innovate(std::span<DataSourceContainer<UDT> const> data_source_containers);

  /// clear the last update meta_data
  void clear_update_metadata();
  /// Calculate the cardinality Distribution
  [[nodiscard]] Vector<UDT> cardinality_distribution() const;
  /// The expected value of the number of tracks.
  [[nodiscard]] UDT estimated_number_of_tracks() const;
  /// Post-processing of the prediction step.
  void post_process_prediction();
  /// Post-processing of the update step.
  void post_process_update();
  /// Delete all tracks for which the pred is true.
  template <class Policy>
  std::pair<UDT, std::size_t> prune_if(Policy pred);
  /// Keep at most maxNTracks, delete the tracks with the smallest existence probability.
  std::pair<UDT, std::size_t> truncate(std::size_t maxNTracks);
  /// Merge another LMB distribution into this one
  void merge(LMBDistribution otherDist);

  TTBManager<UDT>* _manager;
  /// ID of this LMB distribution, normally you should NEVER set this manually
  MODistributionId _id{ _idGenerator.getID() };
  static IDGenerator<MODistributionId> _idGenerator;

public:
  /// tracks before the n-th prediction step -> needed for estimation of persistence_prob
  SlidingWindow<std::map<Label, UDT>> _previous_label_ex_probs;
  /// all tracks
  std::vector<StateId> _tracks;

private:
  /// meas_assignment_prob cache
  std::map<DataSourceContainerId, std::map<MeasurementId, UDT>> _meas_assignment_prob;  ///< P(Measurement i
                                                                                        ///< assigned to some
  using Groups = std::vector<std::pair<LMBDistribution, DataSourceContainer<UDT>>>;     ///< track)
  /// Update this LMB distribution with a loopy belief propagation update, see
  /// T. Kropfreiter, F. Meyer and F. Hlawatsch, "A Fast Labeled Multi-Bernoulli Filter Using Belief Propagation," in
  /// IEEE Transactions on Aerospace and Electronic Systems, vol. 56, no. 3, pp. 2478-2488, June 2020,
  /// doi: 10.1109/TAES.2019.2941104.
  /// This should generally be the fastest and preferred method.
  void single_group_lbp_update(DataSourceContainer<UDT> const& measurementContainer);
  /// Update this LMB distribution by converting it to an GLMBDistribution, updating this, and converting it back, see
  /// S. Reuter, B. -T. Vo, B. -N. Vo and K. Dietmayer, "The Labeled Multi-Bernoulli Filter," in IEEE Transactions on
  /// Signal Processing, vol. 62, no. 12, pp. 3246-3260, June15, 2014, doi: 10.1109/TSP.2014.2323064.
  /// and
  /// S. Reuter, A. Danzer, M. Stübler, A. Scheel and K. Granström, "A fast implementation of the Labeled
  /// Multi-Bernoulli filter using gibbs sampling," 2017 IEEE Intelligent Vehicles Symposium (IV), Los Angeles, CA, USA,
  /// 2017, pp. 765-772, doi: 10.1109/IVS.2017.7995809.
  void single_group_glmb_update(DataSourceContainer<UDT> const& measurementContainer);
  /// Multi-Sensor LMB update based on the multi-sensor GLMB update
  void single_group_glmb_update(std::vector<DataSourceContainer<UDT>> const& data_source_containers);
  /// multi-sensor update based on
  /// C. Hermann, M. Herrmann, T. Griebel, M. Buchholz and K. Dietmayer, "The Fast Product Multi-Sensor Labeled
  /// Multi-Bernoulli Filter," 2023 26th International Conference on Information Fusion (FUSION), Charleston, SC, USA,
  /// 2023, pp. 1-8, doi: 10.23919/FUSION52260.2023.10224189. https://ieeexplore.ieee.org/document/10224189
  /// Use LMB Distribution update for every DataSourceContainer on the common prior, and merge the LMB posteriors
  void single_group_fpm_update(std::vector<DataSourceContainer<UDT>> const& data_source_containers);
  /// multi-sensor update based on
  /// S.C.J. Robertson, C.E. van Daalen, J.A. du Preez, "Efficient approximations of the multi-sensor labelled
  /// multi-Bernoulli filter", Signal Processing, Volume 199, 2022, 108633, ISSN 0165-1684,
  /// https://doi.org/10.1016/j.sigpro.2022.108633. Use LMB Distribution update for every DataSourceContainer on the
  /// common prior, and merge the LMB posteriors,
  void single_group_pu_update(std::vector<DataSourceContainer<UDT>> const& data_source_containers);
  /// performs FPM-LMB fusion mentioned in
  /// C. Hermann, M. Herrmann, T. Griebel, M. Buchholz and K. Dietmayer, "The Fast Product Multi-Sensor Labeled
  /// Multi-Bernoulli Filter," 2023 26th International Conference on Information Fusion (FUSION), Charleston, SC, USA,
  /// 2023, pp. 1-8, doi: 10.23919/FUSION52260.2023.10224189. https://ieeexplore.ieee.org/document/10224189
  void fpm_fusion(std::vector<LMBDistribution> updated_lmbs);
  /// FPM fusion on track area using a graph or graphs. Returns fused track
  [[nodiscard]] std::optional<std::pair<State<UDT>, UDT>> fuse_tracks_fpm(State<UDT> priorTrack,
                                                                          std::vector<State<UDT>> posteriors) const;
  /// performs fusion mentioned in
  /// S.C.J. Robertson, C.E. van Daalen, J.A. du Preez, "Efficient approximations of the multi-sensor labelled
  /// multi-Bernoulli filter", Signal Processing, Volume 199, 2022, 108633, ISSN 0165-1684,
  /// https://doi.org/10.1016/j.sigpro.2022.108633.
  void pu_fusion(std::vector<LMBDistribution> updated_lmbs);
  /// PU fusion on track area using a graph or graphs. Returns fused track
  [[nodiscard]] std::optional<std::pair<State<UDT>, UDT>> fuse_tracks_pu(State<UDT> priorTrack,
                                                                         std::vector<State<UDT>> posteriors) const;
};

static_assert(CPTCommonMODist<LMBDistribution<double>, double>);
static_assert(CPTICUpdate<LMBDistribution<double>, double>);
static_assert(CPTMSUpdate<LMBDistribution<double>, double>);
static_assert(CPTSelfAssessableMODist<LMBDistribution<double>, double>);
static_assert(CPTCommonMODist<LMBDistribution<AD>, AD>);
static_assert(CPTICUpdate<LMBDistribution<AD>, AD>);
static_assert(CPTMSUpdate<LMBDistribution<AD>, AD>);
static_assert(CPTSelfAssessableMODist<LMBDistribution<AD>, AD>);

}  // namespace ttb
