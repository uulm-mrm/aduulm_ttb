#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/MultiObjectStateDistributions/GLMBDistribution.h"
#include "tracking_lib/Misc/Profiler.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"

namespace ttb
{
class TTBManager;
class MarkovTransition;

/// Profiler Data for the GLMB->LMB Conversion
struct GLMB2LMBConversionProfilerData
{
  Duration _duration;
  std::size_t _numGlmbHypotheses;
  std::size_t _numTracks;
};
std::string to_string(GLMB2LMBConversionProfilerData const& data);
std::string to_stringStatistics(std::vector<GLMB2LMBConversionProfilerData> const& datas);

/// Profiler Data for the calcInnovation
struct LMBCalcInnovationProfilerData
{
  Duration _duration;
  std::size_t _numTracks;
  std::size_t _numUpdatedTracks;
  std::size_t _numMeasurements;
  MeasModelId _id;
};
std::string to_string(LMBCalcInnovationProfilerData const& data);
std::string to_stringStatistics(std::vector<LMBCalcInnovationProfilerData> const& datas);

/// Profiler Data for the Prediction
struct LMBPredictProfilerData
{
  Duration _duration;
  std::size_t _numTracks;
};
std::string to_string(LMBPredictProfilerData const& data);
std::string to_stringStatistics(std::vector<LMBPredictProfilerData> const& datas);

/// Profiler Data for Grouping
struct LMBGroupingProfilerData
{
  Duration _duration;
  std::size_t _numTracks;
  std::size_t _numMesurements;
  std::vector<std::pair<std::size_t, std::size_t>> _groups_num_tracks_num_meas;
  std::size_t _numNonAssocMeasurements;
};
std::string to_string(LMBGroupingProfilerData const& data);
std::string to_stringStatistics(std::vector<LMBGroupingProfilerData> const& datas);

/// Represents a labeled multi-Bernoulli distribution
class LMBDistribution final
{
public:
  explicit LMBDistribution(TTBManager* manager);
  LMBDistribution(TTBManager* manager, std::vector<State> tracks);
  /// Convert/Approximate the given GLMBDistribution into this LMBDistribution
  void convertGLMB2LMB(GLMBDistribution glmb);
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
  /// check if all tracks are valid
  [[nodiscard]] bool isValid() const;
  /// Calculate the cardinality Distribution
  [[nodiscard]] Vector cardinalityDistribution() const;
  using Groups = std::vector<std::pair<LMBDistribution, MeasurementContainer>>;
  /// The expected value of the number of tracks.
  [[nodiscard]] double estimatedNumberOfTracks() const;
  /// Post-processing of the prediction step.
  void postProcessPrediction();
  /// Post-processing of the update step.
  void postProcessUpdate();
  /// Delete all tracks for which the pred is true.
  template <class Policy>
  std::size_t prune_if(Policy pred);
  /// Keep at most maxNTracks, delete the tracks with the smallest existence probability.
  std::size_t truncate(std::size_t maxNTracks);
  /// Reset the prior id of all tracks
  void resetPriorId();
  /// Return an estimate of this distribution.
  [[nodiscard]] std::vector<State> getEstimate() const;
  /// Predict this distribution deltaT time into the future.
  void predict(Duration deltaT, EgoMotionDistribution const& egoDist);
  /// Update this LMB distribution with the given MeasurementContainer.
  /// The update method is selected depending on the parameters.
  void update(MeasurementContainer const& measurementContainer);
  /// performs FPM-LMB fusion mentioned in
  /// C. Hermann, M. Herrmann, T. Griebel, M. Buchholz and K. Dietmayer, "The Fast Product Multi-Sensor Labeled
  /// Multi-Bernoulli Filter," 2023 26th International Conference on Information Fusion (FUSION), Charleston, SC, USA,
  /// 2023, pp. 1-8, doi: 10.23919/FUSION52260.2023.10224189. https://ieeexplore.ieee.org/document/10224189
  void fpm_fusion(std::vector<LMBDistribution>&& updated_lmbs, bool isDynamicBirth);
  /// add a track to this LMBDistribution
  void addTrack(State track);
  /// delete a track
  bool eraseTrack(Label label);
  /// Merge another LMB distribution into this one
  void merge(LMBDistribution otherDist);
  /// calculate the state Innovation for all Tracks for all Measurements in the MeasurementContainer
  void calcInnovation(MeasurementContainer const& Z);
  /// the clutter distribution of the last update
  Vector const& clutter_distribution() const;
  /// the detection distribution of the last update
  Matrix const& detection_distribution() const;
  /// the probability/weight a measurement contributed to the update of any track in the last update
  std::map<MeasurementId, Probability> const& meas_assignment_prob() const;
  TTBManager* _manager;
  /// ID of this LMB distribution, normally you should NEVER set this manually
  MODistributionId _id{ _idGenerator.getID() };
  static IDGenerator<MODistributionId> _idGenerator;
  /// all tracks
  std::vector<State> _tracks;

private:
  /// clutter_distribution cache
  Vector _clutter_dist;  ///< i=#clutter -> prob
  /// detection_distribution cache
  Matrix _detection_dist;  ///< i=#tracks, j=#detections -> prob
  /// meas_assignment_prob cache
  std::map<MeasurementId, Probability> _meas_assignment_prob;  ///< P(Measurement i assigned to some track)
  /// update multiple Groups
  void update(Groups groups);
  /// Update this LMB distribution without performing the grouping step.
  void single_group_update(MeasurementContainer const& measurementContainer);
  /// Update this LMB distribution with a loopy belief propagation update, see
  /// T. Kropfreiter, F. Meyer and F. Hlawatsch, "A Fast Labeled Multi-Bernoulli Filter Using Belief Propagation," in
  /// IEEE Transactions on Aerospace and Electronic Systems, vol. 56, no. 3, pp. 2478-2488, June 2020,
  /// doi: 10.1109/TAES.2019.2941104.
  /// This should generally be the fastest and preferred method.
  void single_group_lbp_update(MeasurementContainer const& measurementContainer);
  /// Update this LMB distribution by converting it to an GLMBDistribution, updating this, and converting it back, see
  /// S. Reuter, B. -T. Vo, B. -N. Vo and K. Dietmayer, "The Labeled Multi-Bernoulli Filter," in IEEE Transactions on
  /// Signal Processing, vol. 62, no. 12, pp. 3246-3260, June15, 2014, doi: 10.1109/TSP.2014.2323064.
  /// and
  /// S. Reuter, A. Danzer, M. Stübler, A. Scheel and K. Granström, "A fast implementation of the Labeled
  /// Multi-Bernoulli filter using gibbs sampling," 2017 IEEE Intelligent Vehicles Symposium (IV), Los Angeles, CA, USA,
  /// 2017, pp. 765-772, doi: 10.1109/IVS.2017.7995809.
  void single_group_glmb_update(MeasurementContainer const& measurementContainer);
  /// needed information's for the FPM-LMB fusion
  using Edge_FPM = std::pair<std::size_t, DistributionId>;
  struct FPMFusionPart
  {
    /// used prior components and belonging edge_ids
    std::map<DistributionId, BaseDistribution*> _usedPriorComps;
    /// save information which local component corresponds to which edgeID of the graph
    std::vector<std::map<DistributionId, const BaseDistribution*>> _edgeId2localComp;
  };
  struct FPMFusionInfos
  {
    /// graph describing the k shortest path problem
    graph::DiGraph<std::size_t, Edge_FPM, double> _graph;
    /// contains prior dists and info which edge_id corresponds to which localComponent
    FPMFusionPart _partInfos;
    /// start node and end node
    std::size_t _startNode;
    std::size_t _endNode;
    /// how many solutions shall be calculated at the kBestSelection/generalizedKBestSelection algorithm
    std::size_t _k;
  };
  /// only passes the tracks with same label to fuse_tracks_fpm method.
  /// @param label2PriorTracksMap map containing prior labels
  /// @param label2UpdatedTracksMap map containing posterior tracks ordered by label
  void fuse_distributions_fpm(std::map<Label, std::vector<State>>&& label2PriorTracksMap,
                              std::map<Label, std::vector<State>>&& label2UpdatedTracksMap);

  /// FPM fusion on track area using a graph or graphs. Returns fused track
  /// @tparam TrackIt Type of track iterator. Must point to Track (not pointer to Track)
  /// @param priorTrack pointer to prior track
  /// @param localTracksBegin iterator to first sensor updated track
  /// @param localTracksEnd iterator to last sensor updated track
  template <typename TrackIt>
  [[nodiscard]] State fuse_tracks_fpm(State&& priorTrack, TrackIt localTracksBegin, TrackIt localTracksEnd) const;

  /// Calculates product of existence probs and (1-existence probs)
  /// @tparam TrackIt Type of track iterator. Must point to Track (not pointer to Track)
  /// @param localTracksBegin iterator to first sensor updated track
  /// @param localTracksEnd iterator to last sensor updated track
  /// @param r_posterior_product here the product of exProbs is saved (r1*r2*..*rV)
  /// @param r_inverse_posterior_product here the product of 1-exProbs is saved ((1-r1)*(1-r2)*...*(1-rV))
  template <typename TrackIt>
  void calculate_product_r_factors(TrackIt localTracksBegin,
                                   TrackIt localTracksEnd,
                                   double& r_posterior_product,
                                   double& r_inverse_posterior_product) const;

  /// creates graph, which describes the given k shortest path problem. The graph can be solved by the
  /// generalizedKShortestPath algorithm
  template <typename TrackIt>
  [[nodiscard]] FPMFusionInfos create_graph_fpm(TrackIt localTracksBegin,
                                                TrackIt localTracksEnd,
                                                const std::vector<BaseDistribution*>& priorMixtureComponents,
                                                const StateModelId model_id,
                                                std::size_t& numSensors) const;

  /// calculates k, needed for kShortestPath algorithm in fpm lmb, based on a scaling by the prior component weight
  [[nodiscard]] std::size_t scaled_by_prior_weight(const double weight) const;

  /// calculates k, needed for kShortestPath algorithm in fpm lmb, based on a scaling using a poisson distribution
  [[nodiscard]] std::size_t scaled_by_prior_weight_poisson(std::map<DistributionId, Index>& componentID2Alloc,
                                                           DistributionId prior_id) const;

  /// calculates number of solutions to be calculated, dependent on the total number of solutions which shall be
  /// calculated at the fusion
  [[nodiscard]] std::size_t get_k(const double weight,
                                  std::map<DistributionId, Index>& componentID2Alloc,
                                  DistributionId prior_id) const;

  /// create one graph per prior component id. Each graph describes the given k shortest path problem, which can be
  /// solved by the kShortestPath algorithm
  template <typename TrackIt>
  [[nodiscard]] std::vector<FPMFusionInfos>
  create_graphs_fpm(TrackIt localTracksBegin,
                    TrackIt localTracksEnd,
                    const std::vector<BaseDistribution*>& priorMixtureComponents,
                    const StateModelId model_id,
                    std::size_t& numSensors) const;

  /// calculate the fused state distribution of the mixture components based on the given solution of the
  /// generalizedKBestAlgorithm
  [[nodiscard]] std::vector<std::unique_ptr<BaseDistribution>>
  fuse_mixture_components_generalized_k_best_fpm(const FPMFusionInfos& fusionInfos,
                                                 std::vector<std::pair<std::vector<Edge_FPM>, double>>& solutions,
                                                 const std::size_t numSensors,
                                                 double& weightSumMMF,
                                                 const BaseStateModel& stateModel) const;

  /// calculate the fused state distribution of the mixture components based on the given solution of the
  /// kBestAlgorithm
  [[nodiscard]] std::vector<std::unique_ptr<BaseDistribution>> fuse_mixture_components_k_best_fpm(
      const std::vector<std::pair<FPMFusionPart, std::vector<std::pair<std::vector<Edge_FPM>, double>>>>& solutionsVec,
      const std::size_t numSensors,
      double& weightSumMMF,
      const BaseStateModel& stateModel) const;

  /// calculates the fused existence probability
  [[nodiscard]] double calculate_fused_existence_probability(const double r_prior,
                                                             const double r_posterior_product,
                                                             const double r_inverse_posterior_product,
                                                             const double weightSumMMF,
                                                             const std::size_t numSensors) const;
};

template <class Policy>
std::size_t LMBDistribution::prune_if(Policy pred)
{
  std::size_t const before = _tracks.size();
  _tracks.erase(std::remove_if(_tracks.begin(), _tracks.end(), pred), _tracks.end());
  return before - _tracks.size();
}

}  // namespace ttb