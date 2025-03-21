#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/MultiObjectStateDistributions/Hypothesis.h"
#include "tracking_lib/Misc/Profiler.h"
#include "tracking_lib/TTBTypes/Params.h"

#include "tracking_lib/Graph/Graph.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"

#include <optional>
#include <unordered_map>
#include <execution>

namespace ttb
{

class TTBManager;
class MarkovTransition;

struct GLMBUpdateProfilerData
{
  Duration _duration;
  std::size_t _numTracks;
  /// #Hypotheses before the Update
  std::size_t _numExistingHyps;
  /// #Hypotheses after the Update
  std::size_t _numUpdateHyps;
};
std::string to_string(GLMBUpdateProfilerData const& data);
std::string to_stringStatistics(std::vector<GLMBUpdateProfilerData> const& datas);

struct GLMBCreateHypothesesProfilerData
{
  Duration _duration;
  std::size_t _numTracks;
  std::size_t _numHypotheses;
};
std::string to_string(GLMBCreateHypothesesProfilerData const& data);
std::string to_stringStatistics(std::vector<GLMBCreateHypothesesProfilerData> const& datas);

/// Represents a generalized labeled multi-Bernoulli distribution
class GLMBDistribution final
{
public:
  explicit GLMBDistribution(TTBManager* manager);
  /// Creates an empty GLMBDistribution with the given tracks but does NOT create hypotheses
  GLMBDistribution(TTBManager* manager, std::vector<State> tracks);
  /// checks whether the tracks and hypotheses are consistent + valid
  [[nodiscard]] bool isValid() const;
  /// Predict the distribution
  void predict(Duration deltaT, EgoMotionDistribution const& egoDist);
  /// Perform the GLMB Update as described in:
  /// B. -T. Vo and B. -N. Vo, "Labeled Random Finite Sets and Multi-Object Conjugate Priors," in IEEE Transactions on
  /// Signal Processing, vol. 61, no. 13, pp. 3460-3475, July1, 2013, doi: 10.1109/TSP.2013.2259822.
  /// B. -N. Vo, B. -T. Vo and H. G. Hoang, "An Efficient Implementation of the Generalized Labeled Multi-Bernoulli
  /// Filter," in IEEE Transactions on Signal Processing, vol. 65, no. 8, pp. 1975-1987, 15 April15, 2017,
  /// doi: 10.1109/TSP.2016.2641392.
  void update(MeasurementContainer const& measurements);
  /// performs PM-GLMB fusion published in
  /// M. Herrmann, C. Hermann and M. Buchholz, "Distributed Implementation of the Centralized Generalized Labeled
  /// Multi-Bernoulli Filter," in IEEE Transactions on Signal Processing, vol. 69, pp. 5159-5174, 2021,
  /// doi: 10.1109/TSP.2021.3107632. https://ieeexplore.ieee.org/document/9524466 and M. Herrmann, T. Luchterhand, C.
  /// Hermann, T. Wodtko, J. Strohbeck and M. Buchholz, "Notes on the Product Multi-Sensor Generalized Labeled
  /// Multi-Bernoulli Filter and its Implementation," 2022 25th International Conference on Information Fusion (FUSION),
  /// Linköping, Sweden, 2022, pp. 1-8, doi: 10.23919/FUSION49751.2022.9841275.
  /// https://ieeexplore.ieee.org/document/9841275/citations?tabFilter=papers#citations and also used in M. Herrmann, T.
  /// Luchterhand, C. Hermann and M. Buchholz, "The Product Multi-Sensor Labeled Multi-Bernoulli Filter," 2023 26th
  /// International Conference on Information Fusion (FUSION), Charleston, SC, USA, 2023, pp. 1-8,
  /// doi: 10.23919/FUSION52260.2023.10224121. https://ieeexplore.ieee.org/document/10224121
  void pm_fusion(std::vector<GLMBDistribution>&& updated_glmbs);
  /// number of unique labels
  [[nodiscard]] std::size_t numberOfLabels() const;
  /// distribution of the number of objects
  [[nodiscard]] Vector const& cardinalityDistribution() const;
  /// expectation value of the number of objects
  [[nodiscard]] double estimatedNumberOfTracks() const;
  /// add the given state to the distribution, i.e., update the tracks + hypotheses
  void add_track(State state);
  /// add the given states to the GLMB distribution
  void add_tracks(std::vector<State> states);

private:
  /// Compute the distribution of the update hypotheses that should be considered
  [[nodiscard]] std::unordered_map<HypothesisId, std::size_t> getProportionalAllocation() const;
  /// erase all tracks that are not referenced by any hypothesis
  std::size_t erase_unreferenced_tracks();
  /// reset all cache variable
  void reset_caches() const;
  /// needed information's for the FPM-LMB fusion
  using Edge_PM = std::pair<std::size_t, HypothesisId>;
  struct PMFusionPart
  {
    /// used prior components and belonging edge_ids
    HypothesisId _priorHyp;
    /// save information which local component corresponds to which edgeID of the graph
    std::vector<std::map<HypothesisId, const Hypothesis*>> _edgeId2localHyp;
  };
  struct PMFusionInfos
  {
    /// graph describing the k shortest path problem
    graph::DiGraph<std::size_t, Edge_PM, double> _graph;
    /// contains prior dists and info which edge_id corresponds to which localComponent
    PMFusionPart _partInfos;
    /// start node and end node
    std::size_t _startNode;
    std::size_t _endNode;
    /// how many solutions shall be calculated at the kBestSelection/generalizedKBestSelection algorithm
    std::size_t _k;
  };
  /// calculates k, needed for kShortestPath algorithm in pm glmb, based on a scaling by the prior component weight
  [[nodiscard]] std::size_t scaled_by_prior_weight(const double weight) const;

  /// calculates k, needed for kShortestPath algorithm in pm glmb, based on a scaling using a poisson distribution
  [[nodiscard]] std::size_t scaled_by_prior_weight_poisson(std::map<HypothesisId, Index>& hypID2Alloc,
                                                           HypothesisId prior_id) const;

  /// calculates number of solutions to be calculated, dependent on the total number of solutions which shall be
  /// calculated at the fusion
  [[nodiscard]] std::size_t get_k(const double weight,
                                  std::map<HypothesisId, Index>& hypID2Alloc,
                                  HypothesisId prior_id) const;
  /// create one graph per prior hypothesis id. Each graph describes the given k shortest path problem, which can be
  /// solved by the kShortestPath algorithm
  [[nodiscard]] std::vector<PMFusionInfos> create_graphs_pm(std::vector<GLMBDistribution>& updated_glmbs,
                                                            std::size_t num_sensors) const;

  /// calculate the fused hypotheses based on the given solution of the kBestAlgorithm
  void fuse_hypotheses_pm(
      const std::vector<std::pair<PMFusionPart, std::vector<std::pair<std::vector<Edge_PM>, double>>>>& solutionsVec,
      std::map<Label, std::map<StateId, State>>& updated_tracks_all_sensors,
      std::map<StateId, Label>& updated_track_id_2_label,
      const std::size_t num_sensors,
      std::unordered_map<HypothesisId, Hypothesis>& fused_hypotheses,
      std::unordered_map<StateId, State>& fused_tracks,
      std::unordered_map<std::vector<StateId>, HypothesisId>& fused_state_2_hypotheses);

  /// fuses the given states based on the bayes parallel rule given in
  /// Luchterhand, C. Hermann and M. Buchholz, "The Product Multi-Sensor Labeled Multi-Bernoulli Filter," 2023 26th
  /// International Conference on Information Fusion (FUSION), Charleston, SC, USA, 2023, pp. 1-8,
  /// doi: 10.23919/FUSION52260.2023.10224121. https://ieeexplore.ieee.org/document/10224121
  /// returns vector of fused states and belonging eta in misc entry
  [[nodiscard]] StateId fuse_tracks_pm(StateId prior_track_id,
                                       std::vector<State*>& updated_tracks,
                                       std::unordered_map<StateId, State>& fused_tracks,
                                       std::unordered_map<std::vector<StateId>, StateId>& updated_tracks_2_fused_track,
                                       std::size_t num_sensors) const;

public:
  /// How many update hypotheses should be created for the given predicted hypothesis?
  [[nodiscard]] std::size_t maxNumUpdateHyps(HypothesisId predicted_hyp) const;
  /// post process the prediction
  std::size_t postProcessPrediction();
  /// post process the update
  std::size_t postProcessUpdate();

private:
  /// delete hypotheses with weight smaller than the threshold
  std::size_t prune_threshold(double threshold);
  /// keep at most max_hypotheses, delete the ones with smallest weight
  std::size_t prune_max_hypotheses(std::size_t max_hypotheses);

public:
  /// return an estimation
  std::vector<State> getEstimate() const;
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
  /// Compute a map that express the probability of a measurement is used to update
  /// This is used for the adaptive birth model to generate birth candidates accordingly.
  /// Furthermore, the existence probs for each track label and for each track id is calculated
  [[nodiscard]] std::map<MeasurementId, Probability> probOfAssigment(MeasurementContainer const& measContainer) const;
  /// generate hypotheses for all tracks based on the existence probability of them using the method specified in the
  /// params.
  void generateHypotheses();

private:
  /// Generate GLMB Hypotheses for the LMB to GLMB Conversion
  struct GenerateHypothesesInfos
  {
    /// graph needed for the generalizedKBestSelection algorithm
    graph::DiGraph<std::size_t, StateId, double> _graph;
    /// start and end node
    std::size_t _startNode{};
    std::size_t _endNode{};
  };
  /// create the graph that models the LMB -> GLMB Conversion as shortest path problem
  [[nodiscard]] GenerateHypothesesInfos createGraph() const;
  /// generate all possible hypotheses
  void generateHypothesesAll();
  /// generate the k-best hypotheses
  void generateHypothesesKBest();
  /// generate the hypotheses with sampling
  void generateHypothesesSampling();

public:
  /// normalize hypotheses weights to sum up to 1
  void normalizeHypothesesWeights();
  /// sum up hypotheses weights
  [[nodiscard]] double sumHypothesesWeights() const;
  /// multiply all hypotheses weights with fac
  void multiplyHypothesesWeights(double fac);
  /// calculate the state update for all tracks
  void calcInnovation(MeasurementContainer const& Z);

private:
  /// Build track to measurement association cost matrix
  [[nodiscard]] Matrix buildCostMatrix(std::vector<StateId> const& tracks,
                                       MeasurementContainer const& measurements) const;

public:
  /// fix the tracks and hypotheses and make the GLMB distribution valid, i.e., remove invalid states and adapt
  /// hypotheses referring to them
  void make_valid();
  /// prints the cardinality distribution and also how many hypotheses per cardinality are given
  void printCardandHypothesesNumbersPerCard(aduulm_logger::LoggerLevel printLogLevel) const;
  /// all tracks of label
  [[nodiscard]] std::unordered_map<Label, std::vector<StateId>> const& label2Tracks() const;
  /// the existence Prob of a label
  [[nodiscard]] std::unordered_map<Label, Probability> const& label2ExProb() const;
  /// the existence probability of a track
  [[nodiscard]] std::unordered_map<StateId, Probability> const& track2ExProb() const;
  /// creates map of _priorId2updatedHypId
  [[nodiscard]] std::unordered_map<HypothesisId, std::vector<HypothesisId>> const& priorId2UpdatedHypId() const;
  /// number of hypotheses with a given number of tracks
  [[nodiscard]] Indices numHypsPerCard() const;
  /// return the (categorical) distribution of the number of Clutter c, i.e, P(c=k) = return.at(k)
  [[nodiscard]] Vector const& clutter_distribution(Index num_measurements) const;
  /// return the probability of the #tracks / #detections
  [[nodiscard]] Matrix const& detection_distribution() const;
  /// access to the TTBManager
  TTBManager* _manager;
  /// current tracks
  std::unordered_map<StateId, State> _tracks;
  /// current Hypotheses -> modify only through add_hyp
  std::unordered_map<HypothesisId, Hypothesis> _hypotheses;
  /// states to hypotheses -> automagically modified by add_hyp
  std::unordered_map<std::vector<StateId>, HypothesisId> _state_2_hypotheses;

private:
  /// CACHE VARIABLES to save some computations
  /// saves existence probability of each track label for later use in convertUpdatedGLMB2LMB() in LMBDistribution
  mutable std::optional<std::unordered_map<Label, Probability>> _label2exProb;
  mutable std::optional<std::unordered_map<StateId, Probability>> _trackId2exProb;
  /// --> cardinalityDist()
  mutable std::optional<Vector> _cacheCardDist;
  /// log of cardinality distribution
  mutable std::optional<Vector> _cacheCardDistLog;
  /// contains number of hypotheses per card
  mutable std::optional<Indices> _numHypsPerCard;
  /// Map holding a list of tracks for each track label
  mutable std::optional<std::unordered_map<Label, std::vector<StateId>>> _label2Tracks;
  mutable std::optional<std::unordered_map<Label, std::vector<HypothesisId>>> _label2Hypotheses;
  /// Map holding a list of updated hypothesis ids for each prior hypothesis id
  mutable std::optional<std::unordered_map<HypothesisId, std::vector<HypothesisId>>> _priorId2updatedHypId;
  /// i=#clutter -> prob
  mutable std::optional<Vector> _clutter_dist;
  /// i=#tracks j=#detections -> prob
  mutable std::optional<Matrix> _detection_probs;
  /// getProportionalAllocation Cache
  mutable std::optional<std::unordered_map<HypothesisId, std::size_t>> _kBestCache;

public:
  /// ID of this GLMB distribution
  MODistributionId _id{ _idGenerator.getID() };
  static IDGenerator<MODistributionId> _idGenerator;
};

/// add the hypotheses to hypotheses and state_2_hypotheses
void add_hyp(Hypothesis hyp,
             std::unordered_map<HypothesisId, Hypothesis>& hypotheses,
             std::unordered_map<std::vector<StateId>, HypothesisId>& state_2_hypotheses);

}  // namespace ttb
