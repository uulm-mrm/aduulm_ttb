#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>

namespace bmi = boost::multi_index;
namespace ttb
{

/// Represents one (1) Hypothesis, i.e., one specific combination of tracks
template <class UDT>
class Hypothesis
{
public:
  Hypothesis(std::vector<StateId> tracks,
             UDT weight_log,
             HypothesisId origin = NO_HYPOTHESIS_ID_HISTORY,
             UDT origin_weight_log = UDT{ std::numeric_limits<double>::quiet_NaN() });
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
  /// weight of the hypothesis
  [[nodiscard]] UDT getWeight() const;
  HypothesisId _id{ _idGenerator.getID() };
  /// The hypothesis id of which this hypothesis originated
  HypothesisId _prior_id = NO_HYPOTHESIS_ID_HISTORY;
  /// Weight of the Hypothesis (log-space)
  UDT _weightLog{ UDT{ std::numeric_limits<double>::quiet_NaN() } };
  /// Weight of this Hypothesis' origin (log-space)
  UDT _prior_weight_log{ UDT{ std::numeric_limits<double>::quiet_NaN() } };
  /// List with tracks
  std::vector<StateId> _tracks;
  /// weight of this hypothesis without detection probability contribution
  std::map<SourceId, UDT> _soft_detection_weight{};
  /// weight of this hypothesis without clutter contribution
  std::map<SourceId, UDT> _soft_clutter_weight{};
  /// weight of this hyp for the point comparison
  std::map<SourceId, UDT> _point_weight{};
  std::map<SourceId, std::size_t> _num_clutter{};
  std::map<SourceId, std::size_t> _num_detected{};
  static IDGenerator<HypothesisId> _idGenerator;
};

class StateIds
{
};
class PriorHypothesisId
{
};
/// Represents a group, i.e., a set of alternative, of possible hypothesis
template <class UDT>
class Hypotheses
{
public:
  using HypothesesDataStructure = bmi::multi_index_container<
      Hypothesis<UDT>,
      bmi::indexed_by<
          bmi::hashed_unique<bmi::tag<HypothesisId>, bmi::member<Hypothesis<UDT>, HypothesisId, &Hypothesis<UDT>::_id>>,
          bmi::hashed_non_unique<bmi::tag<PriorHypothesisId>,
                                 bmi::member<Hypothesis<UDT>, HypothesisId, &Hypothesis<UDT>::_prior_id>>,
          bmi::hashed_unique<bmi::tag<StateIds>,
                             bmi::member<Hypothesis<UDT>, std::vector<StateId>, &Hypothesis<UDT>::_tracks>>>>;
  [[nodiscard]] std::size_t max_tracks() const;
  [[nodiscard]] std::vector<StateId> all_tracks() const;
  [[nodiscard]] std::unordered_map<Label, std::vector<StateId>> label_2_tracks(TTBManager<UDT> const& manager) const;
  /// add the hypotheses to hypotheses and state_2_hypotheses
  void add(Hypothesis<UDT> hyp);
  /// normalize hypotheses weights to sum up to 1
  void normalize_hypotheses_weight();
  /// sum up hypotheses weights
  [[nodiscard]] UDT sum_hypotheses_weight() const;
  /// multiply all hypotheses weights with fac
  void multiply_hypotheses_weight(UDT fac);

  Vector<UDT> cardinality_distribution() const;
  /// delete hypotheses like in dissertation of Martin Herrmann
  /// https://oparu.uni-ulm.de/items/7d363d70-eee7-46e0-a999-678813e4c37e
  std::pair<double, std::size_t> prune_hypotheses_diss_herrmann(std::size_t max_comps_retain,
                                                                std::size_t min_comps_per_card,
                                                                double min_comps_per_card_thresh,
                                                                double max_weight_ratio,
                                                                bool alloc_min_comps_per_card_by_poisson);
  /// delete hypotheses with weight smaller than the threshold
  std::pair<double, std::size_t> prune_hypotheses_weight_threshold(double threshold);
  /// delete labels with existence probability smaller than the threshold
  std::pair<double, std::size_t> prune_labels_existence_threshold(TTBManager<UDT> const& manager, double threshold);
  /// keep at most the max_hypotheses with the biggest weight
  std::pair<double, std::size_t> prune_max_hypotheses(std::size_t max_hypotheses);

  /// existence probability of a track
  [[nodiscard]] std::unordered_map<StateId, UDT> track_existence_prob() const;
  [[nodiscard]] std::unordered_map<Label, UDT> label_existence_prob(TTBManager<UDT> const& manager) const;
  /// limit the number of different states per label.
  /// Hypotheses that contain non-used states get assigned the more likely states
  void limit_states_per_label(TTBManager<UDT> const& manager, std::size_t max_states);
  /// split the Hypotheses into groups defined by the given vector of states.
  /// States not mentioned in any split_ids are collected in the last Hypotheses, i.e., there are in total
  /// split_ids.size()+1 split groups
  [[nodiscard]] std::vector<Hypotheses> split(std::vector<std::vector<StateId>> const& split_ids) const;
  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] bool is_valid() const;
  HypothesesDataStructure _hyps{};
};

/// merge the independent list of hypotheses into one big merged hypotheses, uses k-shortest path algo for this
template <class UDT>
[[nodiscard]] Hypotheses<UDT> merge(std::vector<Hypotheses<UDT>> const& independent_hypotheses,
                                    std::size_t max_merged_size);

}  // namespace ttb
