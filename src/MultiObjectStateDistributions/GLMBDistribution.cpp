#include "tracking_lib/MultiObjectStateDistributions/GLMBDistribution.h"

#include "tracking_lib/MultiObjectStateDistributions/Utils.h"
#include "tracking_lib/Misc/MurtyAlgorithm.h"

#include "tracking_lib/Misc/ProportionalAllocation.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Misc/Numeric.h"
#include "tracking_lib/Misc/SeparatingAxisTest.h"
#include "tracking_lib/SelfAssessment/SelfAssessment.h"

#include <utility>
#include <ranges>
#include <unordered_set>
#include <set>
#include <stacktrace>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions.hpp>
#include <tracy/tracy/Tracy.hpp>
#include "tracking_lib/Transformations/Transformation.h"

namespace ttb
{
constexpr auto tracy_color = tracy::Color::RebeccaPurple;

template class GLMBDistribution<double>;
template class GLMBDistribution<AD>;

template <class UDT>
IDGenerator<MODistributionId> GLMBDistribution<UDT>::_idGenerator{};

template <class UDT>
GLMBDistribution<UDT>::GLMBDistribution(TTBManager<UDT>* manager)
  : _manager{ manager }
  , _previous_label_ex_probs{ std::chrono::milliseconds(
        _manager->params()->self_assessment.persistence_prob_estimation.window_length_ms) }
{
  _hypotheses.emplace_back(Hypotheses<UDT>{});
  _hypotheses.back().add(Hypothesis<UDT>({}, UDT{ 0 }));
  assert(is_valid());
}

template <class UDT>
std::unordered_map<Label, std::vector<StateId>> const& GLMBDistribution<UDT>::label_2_tracks() const
{
  if (_label2Tracks.has_value())
  {
    return _label2Tracks.value();
  }
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution before label2Tracks.");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  std::unordered_map<Label, std::vector<StateId>> out;
  for (StateId const id : all_tracks())
  {
    if (std::vector<StateId>& tmp = out[_manager->get_state(id)._label]; std::ranges::find(tmp, id) == tmp.end())
    {
      tmp.push_back(id);
    }
  }
  _label2Tracks = std::move(out);
  return _label2Tracks.value();
}

template <class UDT>
std::unordered_map<Label, UDT> const& GLMBDistribution<UDT>::label_2_ex_prob() const
{
  if (_label2exProb.has_value())
  {
    return _label2exProb.value();
  }
  std::unordered_map<Label, UDT> out;
  for (auto const& [trackId, prob] : track_2_ex_prob())
  {
    assert(std::isfinite(prob));
    out[_manager->get_state(trackId)._label] += prob;
  }
  for (auto& [label, prob] : out)
  {
    if (prob > 1 + 1e-7)
    {
      LOG_WARN("Invalid Probability " + std::to_string(prob) + " for Label " + std::to_string(label.value_) +
               ". Closely monitor. Clamping for now....");
    }
    prob = std::min(prob, UDT{ 1.0 });
  }
  _label2exProb = std::move(out);
  return _label2exProb.value();
}

template <class UDT>
std::unordered_map<StateId, UDT> const& GLMBDistribution<UDT>::track_2_ex_prob() const
{
  if (_trackId2exProb.has_value())
  {
    return _trackId2exProb.value();
  }
  LOG_DEB("track to ex prob");
  std::unordered_map<StateId, std::vector<UDT>> log_weights;
  for (Hypotheses<UDT> const& hypotheses : _hypotheses)
  {
    for (Hypothesis<UDT> const& hyp : hypotheses._hyps)
    {
      for (StateId const track_id : hyp._tracks)
      {
        assert([&] {  // NOLINT
          if (not std::isfinite(hyp._weightLog) and hyp._weightLog != -std::numeric_limits<double>::infinity())
          {
            LOG_FATAL(hyp._weightLog);
            LOG_FATAL(to_string("", true, true));
            return false;
          }
          return true;
        }());
        log_weights[track_id].push_back(hyp._weightLog);
      }
    }
  }
  std::unordered_map<StateId, UDT> out;
  for (auto& [track, log_w] : log_weights)
  {
    UDT prob = exp(numeric::logsumexp(log_w));
    if (prob > 1 + 1e-7)
    {
      LOG_WARN("Invalid Probability " + std::to_string(prob) + " for Track " + std::to_string(track.value_) +
               ". Closely monitor. Clamping for now....");
    }
    prob = std::min(prob, UDT{ 1.0 });
    assert([&] {  // NOLINT
      if (not std::isfinite(prob))
      {
        for (UDT d : log_w)
        {
          LOG_FATAL(d);
        }
        return false;
      }
      return true;
    }());
    out[track] = prob;
  }
  assert(out.size() == all_tracks().size());
  _trackId2exProb = std::move(out);
  LOG_DEB("track to ex prob");
  return _trackId2exProb.value();
}

template <class UDT>
Vector<UDT>
GLMBDistribution<UDT>::compute_clutter_distribution(DataSourceContainer<UDT> const& data_source_container) const
{
  // ZoneScopedN("GLMBDistribution::compute_clutter_distribution");
  assert(is_valid());
  Vector<UDT> dist = Vector<UDT>::Zero(std::ssize(data_source_container._measurements) + 1);
  std::vector<Matrix<UDT>> const& detection_dists = compute_detection_distribution(data_source_container);
  Matrix<UDT> detection_dist = detection_dists.front();
  for (std::size_t i = 1; i < detection_dists.size(); ++i)
  {
    detection_dist = numeric::fold<UDT>(detection_dist, detection_dists.at(i));
  }
  for (Index tracks = 0; tracks < detection_dist.rows(); ++tracks)
  {
    for (Index detected = 0; detected < detection_dist.cols(); ++detected)
    {
      if (detection_dist(tracks, detected) > 0)
      {
        dist(std::max(std::ssize(data_source_container._measurements) - detected, 0L)) +=
            detection_dist(tracks, detected);
      }
    }
  }
  assert([&] {  // NOLINT
    if (std::abs(dist.sum() - 1) > 1e-5 or not dist.array().isFinite().all())
    {
      LOG_FATAL("clutter dist does not sum up to 1: " << dist);
      LOG_FATAL(to_string());
      return false;
    }
    for (Index num_clutter = 0; num_clutter < dist.rows(); ++num_clutter)
    {
      if (dist(num_clutter) > 0 and num_clutter > static_cast<Index>(data_source_container._measurements.size()))
      {
        LOG_FATAL("invalid clutter dist " << dist);
        LOG_FATAL(std::format(
            "num clutter {}\nnum measurements: {}", num_clutter, data_source_container._measurements.size()));
        LOG_FATAL(std::stacktrace::current());
        return false;
      }
    }
    return true;
  }());
  return dist;
}

template <class UDT>
Vector<UDT> const&
GLMBDistribution<UDT>::clutter_distribution(DataSourceContainer<UDT> const& data_source_container) const
{
  LOG_DEB("GLMBDistribution::clutter_distribution");
  assert(_clutter_dist.contains(data_source_container._id));
  return _clutter_dist.at(data_source_container._id);
}

template <class UDT>
std::vector<Matrix<UDT>>
GLMBDistribution<UDT>::compute_detection_distribution(DataSourceContainer<UDT> const& data_source_container) const
{
  // ZoneScopedN("GLMBDistribution::compute_detection_distribution");
  std::vector<Matrix<UDT>> all_detection_probs;
  for (Hypotheses<UDT> const& hypotheses : _hypotheses)
  {
    Matrix<UDT> detection_probs = Matrix<UDT>::Zero(static_cast<Index>(hypotheses.max_tracks() + 1),
                                                    static_cast<Index>(hypotheses.max_tracks() + 1));
    for (Hypothesis<UDT> const& hyp : hypotheses._hyps)
    {
      Index detections = 0;
      Index tracks = 0;
      for (StateId trackId : hyp._tracks)
      {
        State<UDT> const& state = _manager->get_state(trackId);
        assert([&] {  // NOLINT
          if (not state._meta_data._detectable.contains(data_source_container._source_id))
          {
            LOG_FATAL("track does not contain detectable flag");
            LOG_FATAL(state.toString());
            LOG_FATAL(std::stacktrace::current());
            return false;
          }
          return true;
        }());
        if (not state._meta_data._detectable.at(data_source_container._source_id) or state._meta_data._numUpdates <= 1)
        {
          continue;
        }
        tracks++;
        assert(state._meta_data._meas_update_weight.contains(data_source_container._source_id));
        assert(state._meta_data._meas_update_weight.at(data_source_container._source_id).size() == 1);
        if (state._meta_data._meas_update_weight.at(data_source_container._source_id).front().first != NOT_DETECTED)
        {
          detections++;
        }
      }
      detection_probs(tracks, detections) += hyp.getWeight();
    }
    all_detection_probs.push_back(std::move(detection_probs));
  }
  return all_detection_probs;
}

template <class UDT>
std::vector<Matrix<UDT>> const&
GLMBDistribution<UDT>::detection_distribution(DataSourceContainer<UDT> const& data_source_container) const
{
  LOG_DEB("GLMBDistribution::detection_distribution");
  assert(_detection_probs.contains(data_source_container._id));
  return _detection_probs.at(data_source_container._id);
}

template <class UDT>
std::vector<Hypotheses<UDT>> GLMBDistribution<UDT>::split_hypotheses(Hypotheses<UDT> const& original_hypotheses) const
{
  ZoneScopedN("GLMBDistribution::split_hypotheses");
  using NodeId = StateId;
  std::vector<NodeId> nodes;
  for (StateId state_id : original_hypotheses.all_tracks())
  {
    nodes.emplace_back(state_id);
  }

  using Edge = graph::Edge<std::size_t, NodeId, double>;
  std::vector<Edge> edges;
  /// or have the same label
  for (auto const& [label, tracks] : original_hypotheses.label_2_tracks(*_manager))
  {
    if (tracks.size() > 1)
    {
      for (std::size_t i = 0; i + 1 < tracks.size(); ++i)
      {
        edges.emplace_back(NOT_DETECTED.value_, tracks.at(i), tracks.at(i + 1), 1);
        edges.emplace_back(NOT_DETECTED.value_, tracks.at(i + 1), tracks.at(i), 1);
      }
    }
  }
  graph::DiGraph const graph(std::move(nodes), std::move(edges));
  std::vector<std::vector<NodeId>> const comps = graph.components();
  std::vector<Hypotheses<UDT>> res = original_hypotheses.split(comps);
  assert([&] {
    for (Hypotheses<UDT> const& hyps : res)
    {
      if (hyps.label_2_tracks(*_manager).size() != 1 and hyps.label_2_tracks(*_manager).size() != 0)
      {
        LOG_FATAL(hyps.to_string());
        return false;
      }
    }
    return true;
  }());
  return res;
}

template <class UDT>
std::vector<Hypotheses<UDT>>
GLMBDistribution<UDT>::merge_hypotheses_interacting_states(std::vector<Hypotheses<UDT>> const& hypotheseses,
                                                           Duration prediction_horizon,
                                                           double min_distance) const
{
  ZoneScopedN("GLMBDistribution::merge_hypotheses_interacting_states");
  if (hypotheseses.size() == 1)
  {
    return hypotheseses;
  }
  using NodeId = std::size_t;
  std::vector<NodeId> nodes;

  using Edge = graph::Edge<std::size_t, NodeId, double>;
  std::vector<Edge> edges;

  std::unordered_map<StateId, std::size_t> track_to_hypotheses_group;
  std::vector<StateId> all_tracks;
  for (auto const& [ind, hypotheses] : std::views::enumerate(hypotheseses))
  {
    for (StateId state : hypotheses.all_tracks())
    {
      track_to_hypotheses_group.emplace(state, ind);
      all_tracks.push_back(state);
    }
  }

  auto const overlapping_states = overlapping_tracks(*_manager, all_tracks, prediction_horizon, min_distance);
  /// the tracks are connected, if they interact
  for (auto const& [ind, hypotheses] : std::views::enumerate(hypotheseses))
  {
    nodes.push_back(ind);
  }
  for (auto const& [first, second, prob] : overlapping_states)
  {
    edges.emplace_back(
        NOT_DETECTED.value_, track_to_hypotheses_group.at(first), track_to_hypotheses_group.at(second), 1);
    edges.emplace_back(
        NOT_DETECTED.value_, track_to_hypotheses_group.at(second), track_to_hypotheses_group.at(first), 1);
  }
  graph::DiGraph const graph(std::move(nodes), std::move(edges));
  std::vector<std::vector<NodeId>> const comps = graph.components();
  std::vector<Hypotheses<UDT>> merged_hyps;
  for (std::vector<NodeId> const& comp : comps)
  {
    std::set<std::size_t> hyp_group;
    std::vector<Hypotheses<UDT>> for_merging;
    for (NodeId const& node : comp)
    {
      if (not hyp_group.contains(node))
      {
        for_merging.push_back(hypotheseses.at(node));
        hyp_group.insert(node);
      }
    }
    merged_hyps.push_back(
        merge(for_merging, _manager->params()->glmb_distribution.update.max_total_number_update_hypotheses));
  }
  return merged_hyps;
}

template <class UDT>
std::vector<Hypotheses<UDT>>
GLMBDistribution<UDT>::split_hypotheses(std::vector<Hypotheses<UDT>> const& hypotheseses) const
{
  std::vector<Hypotheses<UDT>> split;
  for (Hypotheses<UDT> const& ori : hypotheseses)
  {
    for (Hypotheses<UDT>& s : split_hypotheses(ori))
    {
      split.push_back(std::move(s));
    }
  }
  return split;
}

template <class UDT>
std::vector<typename GLMBDistribution<UDT>::MergedHypotheses>
GLMBDistribution<UDT>::split_and_merge_hypotheses(std::vector<Hypotheses<UDT>> const& hypotheseses,
                                                  std::span<DataSourceContainer<UDT> const> data_source_containers,
                                                  SplitAndMergeHypothesesType type) const
{
  ZoneScopedN("GLMBDistribution::split_and_merge_hypotheses");

  std::vector<Hypotheses<UDT>> splitted = [&] {
    // ZoneScopedN("GLMBDistribution::split_and_merge_hypotheses::split");
    if (_manager->params()->glmb_distribution.use_grouping)
    {
      std::vector<Hypotheses<UDT>> split;
      for (Hypotheses<UDT> const& ori : hypotheseses)
      {
        for (Hypotheses<UDT>& s : split_hypotheses(ori))
        {
          split.push_back(std::move(s));
        }
      }
      return split;
    }
    return hypotheseses;
  }();
  if (type == SplitAndMergeHypothesesType::COMMON_MEASUREMENT)
  {
    LOG_DEB("GLMBDistribution::merge_hypotheses");
    ZoneScopedN("GLMBDistribution::merge_hypotheses::COMMON_MEASUREMENT");
    if (splitted.size() == 1)
    {
      return { MergedHypotheses{ .hypotheses = std::move(splitted.front()), .data_source_containers = [&] {
                                  std::vector<DataSourceContainer<UDT>> containers;
                                  for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
                                  {
                                    containers.push_back(data_source_container);
                                  }
                                  return containers;
                                }() } };
    }
    std::unordered_set<StateId> relevant_meas_ids;
    for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
    {
      for (State<UDT> const& meas : data_source_container._measurements)
      {
        relevant_meas_ids.insert(meas._id);
      }
    }
    std::unordered_map<MeasurementId, std::vector<std::pair<std::size_t, StateId>>> meas_to_tracks;
    // this meas is used to update all this tracks
    {
      for (auto const& [ind, hypotheses] : std::views::enumerate(splitted))
      {
        for (StateId state_id : hypotheses.all_tracks())
        {
          auto const [first, second] = _manager->database().state_updates.equal_range(std::tuple{ state_id });
          for (auto it = first; it != second; ++it)
          {
            if (relevant_meas_ids.contains(it->meas_id))
            {
              meas_to_tracks[it->meas_id].emplace_back(ind, state_id);
            }
          }
        }
      }
    }

    using NodeId = std::pair<std::size_t, StateId>;
    std::vector<NodeId> nodes;

    using Edge = graph::Edge<std::size_t, NodeId, double>;
    std::vector<Edge> edges;
    {
      // the tracks are connected, if they were in the same hypotheses group before
      for (auto const& [ind, hypotheses] : std::views::enumerate(splitted))
      {
        std::vector<StateId> all_tracks = hypotheses.all_tracks();
        for (StateId state_id : all_tracks)
        {
          nodes.emplace_back(ind, state_id);
        }
        if (all_tracks.size() > 1)
        {
          for (std::size_t i = 0; i + 1 < all_tracks.size(); ++i)
          {
            edges.emplace_back(
                NOT_DETECTED.value_, NodeId{ ind, all_tracks.at(i) }, NodeId{ ind, all_tracks.at(i + 1) }, 1);
            edges.emplace_back(
                NOT_DETECTED.value_, NodeId{ ind, all_tracks.at(i + 1) }, NodeId{ ind, all_tracks.at(i) }, 1);
          }
        }
      }
    }
    {
      /// the tracks are connected, if they have a common measurement
      for (auto const& [meas_id, meas_updates] : meas_to_tracks)
      {
        if (meas_updates.size() > 1)
        {
          for (std::size_t i = 0; i + 1 < meas_updates.size(); ++i)
          {
            edges.emplace_back(meas_id.value_, meas_updates.at(i), meas_updates.at(i + 1), 1);
            edges.emplace_back(meas_id.value_, meas_updates.at(i + 1), meas_updates.at(i), 1);
          }
        }
      }
    }
    graph::DiGraph const graph(std::move(nodes), std::move(edges));
    std::vector<std::vector<NodeId>> const comps = graph.components();
    std::vector<MergedHypotheses> merged_hyps;
    std::unordered_set<StateId> used_measurements;
    for (std::vector<NodeId> const& group : comps)
    {
      std::set<std::size_t> hyp_group;
      std::vector<Hypotheses<UDT>> for_merging;
      std::vector<DataSourceContainer<UDT>> reduced_data_container;
      std::set<StateId> relevant_meas;
      for (NodeId const& node : group)
      {
        if (not hyp_group.contains(node.first))
        {
          for_merging.push_back(splitted.at(node.first));
          hyp_group.insert(node.first);
        }
        for (std::size_t edge_id : graph.edges_from(node))
        {
          relevant_meas.insert(StateId{ edge_id });
        }
        assert([&] {
          for (std::size_t edge_id : graph.edges_to(node))
          {
            if (not relevant_meas.contains(StateId{ edge_id }))
            {
              LOG_FATAL("graph has different edges from and to node but is not directed graph");
              for (std::size_t to : graph.edges_to(node))
              {
                LOG_FATAL(to);
              }
              for (std::size_t from : graph.edges_from(node))
              {
                LOG_FATAL(from);
              }
              return false;
            }
          }
          return true;
        }());
      }
      for (DataSourceContainer<UDT> const& full_data_container : data_source_containers)
      {
        DataSourceContainer reduced = full_data_container;
        reduced._measurements.clear();
        for (State<UDT> const& meas : full_data_container._measurements)
        {
          if (relevant_meas.contains(meas._id))
          {
            used_measurements.insert(meas._id);
            reduced._measurements.push_back(meas);
          }
        }
        reduced_data_container.push_back(std::move(reduced));
      }
      assert(reduced_data_container.size() == data_source_containers.size());
      merged_hyps.emplace_back(
          merge(for_merging, _manager->params()->glmb_distribution.update.max_total_number_update_hypotheses),
          std::move(reduced_data_container));
    }
    MergedHypotheses clutter_measurements{ .hypotheses = Hypotheses<UDT>{}, .data_source_containers = {} };
    Hypothesis<UDT> no_tracks({}, UDT{ 0 });
    clutter_measurements.hypotheses.add(std::move(no_tracks));
    for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
    {
      DataSourceContainer reduced = data_source_container;
      reduced._measurements.clear();
      for (State<UDT> const& state : data_source_container._measurements)
      {
        if (not used_measurements.contains(state._id))
        {
          reduced._measurements.push_back(state);
        }
      }
      clutter_measurements.data_source_containers.push_back(std::move(reduced));
    }
    merged_hyps.push_back(std::move(clutter_measurements));
    assert([&] {  // NOLINT
      std::vector<std::size_t> measurements_before;
      for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
      {
        measurements_before.push_back(data_source_container._measurements.size());
      }
      std::vector<std::size_t> measurements_after(data_source_containers.size(), 0);
      for (MergedHypotheses const& merged : merged_hyps)
      {
        if (merged.data_source_containers.size() != data_source_containers.size())
        {
          LOG_FATAL("different sizes of data containers");
          return false;
        }
        for (auto const& [ind, data_source_container] : std::views::enumerate(merged.data_source_containers))
        {
          measurements_after.at(ind) += data_source_container._measurements.size();
        }
      }
      if (measurements_before.size() != measurements_after.size())
      {
        LOG_FATAL("different sizes");
        return false;
      }
      for (auto const& [first, second] : std::views::zip(measurements_before, measurements_after))
      {
        if (first != second)
        {
          LOG_FATAL(std::format("different number of measurements for container {} {}", first, second));
          return false;
        }
      }
      return true;
    }());
    return merged_hyps;
  }
  else if (type == SplitAndMergeHypothesesType::INTERACTING_STATE)
  {
    ZoneScopedN("GLMBDistribution::merge_hypotheses::INTERACTING_STATE");
    std::vector<Hypotheses<UDT>> merged = merge_hypotheses_interacting_states(
        splitted,
        std::chrono::milliseconds(_manager->params()->glmb_distribution.non_overlapping_update.horizon_ms),
        3 * _manager->params()->glmb_distribution.non_overlapping_update.object_radius_m);

    std::vector<MergedHypotheses> merged_type;
    for (Hypotheses<UDT>& hyp : merged)
    {
      merged_type.push_back(MergedHypotheses{ .hypotheses = std::move(hyp), .data_source_containers = {} });
    }
    return merged_type;
  }
  assert(false);
  std::unreachable();
}

template <class UDT>
std::size_t GLMBDistribution<UDT>::number_of_labels() const
{
  std::vector<Label> labels;
  for (StateId id : all_tracks())
  {
    State<UDT> const& state = _manager->get_state(id);
    if (std::ranges::find(labels, state._label) == labels.end())
    {
      labels.push_back(state._label);
    }
  }
  return labels.size();
}

template <class UDT>
Vector<UDT> const& GLMBDistribution<UDT>::cardinality_distribution() const
{
  if (_cacheCardDist.has_value())
  {
    return _cacheCardDist.value();
  }
  assert(is_valid());
  std::vector<Vector<UDT>> all;
  for (Hypotheses<UDT> const& hypotheses : _hypotheses)
  {
    all.push_back(hypotheses.cardinality_distribution());
  }
  Vector<UDT> cardDist = std::move(all.front());
  for (std::size_t i = 1; i < all.size(); ++i)
  {
    cardDist = numeric::fold(cardDist, all.at(i));
  }
  _cacheCardDist = std::move(cardDist);
  return _cacheCardDist.value();
}

template <class UDT>
Vector<std::size_t> GLMBDistribution<UDT>::num_hypotheses_per_cardinality() const
{
  Vector<std::size_t> num_hyps = Vector<std::size_t>::Zero(static_cast<Index>(1 + number_of_labels()));
  for (Hypotheses<UDT> const& hypotheses : _hypotheses)
  {
    for (Hypothesis<UDT> const& hyp : hypotheses._hyps)
    {
      num_hyps(static_cast<Index>(hyp._tracks.size())) += 1;
    }
  }
  return num_hyps;
}

template <class UDT>
UDT GLMBDistribution<UDT>::estimated_number_of_tracks() const
{
  Vector<UDT> cardDist = cardinality_distribution();
  UDT val = 0;
  for (Index n_elem = 0; n_elem != cardDist.rows(); n_elem++)
  {
    val += static_cast<double>(n_elem) * cardDist(n_elem);
  }
  return val;
}

template <class UDT>
std::unordered_map<HypothesisId, std::size_t> GLMBDistribution<UDT>::allocation(Hypotheses<UDT> const& hypotheses) const
{
  LOG_DEB("Proportional Allocation");
  Vector<UDT> weights(hypotheses._hyps.size());
  // save weight values
  for (const auto& [idx, hyp] : std::views::enumerate(hypotheses._hyps))
  {
    weights(idx) = hyp.getWeight();
  }
  std::unordered_map<HypothesisId, std::size_t> numNewHyps;

  // calc for each hypothesis of this cardinality the value k for the murty algorithm
  Indices allocs =
      propAllocation(_manager->params()->glmb_distribution.update.max_total_number_update_hypotheses,
                     weights,
                     _manager->params()->glmb_distribution.update.num_update_hypotheses.equal_allocation_share_ratio);
  for (const auto& [idx, hyp] : std::views::enumerate(hypotheses._hyps))
  {
    numNewHyps[hyp._id] = allocs(idx);
  }

  assert([&] {  // NOLINT
    std::size_t allocated_new_hyps = 0;
    for (auto const& [hyp, alloc] : numNewHyps)
    {
      allocated_new_hyps += alloc;
    }
    if (_manager->params()->glmb_distribution.update.max_total_number_update_hypotheses != allocated_new_hyps)
    {
      LOG_FATAL(
          "requested to create " << _manager->params()->glmb_distribution.update.max_total_number_update_hypotheses
                                 << " new hypotheses but allocated " << allocated_new_hyps);
      return false;
    }
    return true;
  }());

  return numNewHyps;
}

template <class UDT>
std::pair<std::vector<Matrix<UDT>>, Duration> GLMBDistribution<UDT>::surviving_distribution()
{
  // ZoneScopedN("GLMBDistribution::surviving_distribution");
  // assert (Called after update)
  Matrix<UDT> cum_surv = Matrix<UDT>::Ones(1, 1);
  if (_previous_label_ex_probs.data_buffer.empty())
  {
    return { { cum_surv.array().log() }, 1s };
  }
  std::vector<Matrix<UDT>> independent_data;
  std::unordered_map<Label, UDT> const& current_ex_probs = label_2_ex_prob();
  for (auto const& [label, ex_prob_before] : _previous_label_ex_probs.data_buffer.front().data)
  {
    UDT const rb = ex_prob_before;
    UDT const ra = [&] -> UDT {
      if (not current_ex_probs.contains(label))
      {
        return UDT{ 0.0 };
      }
      return current_ex_probs.at(label);
    }();
    Matrix<UDT> const mult{ { 1 - rb, rb * (1 - ra) }, { rb * ra, 0 } };
    independent_data.push_back(mult.array().log());
  }
  return { independent_data, _manager->filter_time() - _previous_label_ex_probs.data_buffer.front().time };
}

template <class UDT>
void GLMBDistribution<UDT>::add_tracks(std::vector<BernoulliState<UDT>> states)
{
  assert(is_valid());
  std::vector<StateId> state_ids;
  for (BernoulliState<UDT>& birth_track : states)
  {
    StateId const birth_id = birth_track.state._id;
    state_ids.push_back(birth_id);
    _manager->database().states.insert(std::move(birth_track.state));
    _manager->database().state_existences.emplace(birth_id, birth_track.existence_probability);
  }
  add_tracks(state_ids);
}

template <class UDT>
void GLMBDistribution<UDT>::add_tracks(std::vector<StateId> const& states)
{
  LOG_DEB("GLMBDistribution::add_tracks");
  assert(is_valid());
  for (StateId state_id : states)
  {
    birth_tracks.push_back(state_id);
    assert(_manager->try_get_state(state_id).has_value());
  }
}

template <class UDT>
void GLMBDistribution<UDT>::add_birth_tracks()
{
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution before add_birth_tracks().");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  for (StateId state_id : birth_tracks)
  {
    if (_manager->params()->glmb_distribution.update.joint_prediction_and_update)
    {
      _hypotheses.emplace_back(Hypotheses<UDT>{});
      _hypotheses.back().add(Hypothesis<UDT>({ state_id }, UDT{ 0 }));
      _manager->database().state_persistences.emplace(state_id, _manager->get_state_existence_probability(state_id));
    }
    else
    {
      _hypotheses.emplace_back(Hypotheses<UDT>{});
      _hypotheses.back().add(Hypothesis<UDT>({ state_id }, log(_manager->get_state_existence_probability(state_id))));
      _hypotheses.back().add(Hypothesis<UDT>({}, log(1 - _manager->get_state_existence_probability(state_id))));
    }
  }
  reset_caches();
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution after add_birth_tracks().");
      LOG_FATAL(std::format("Birth Tracks {}", birth_tracks));
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  birth_tracks.clear();
  LOG_DEB("done");
}

template <class UDT>
void GLMBDistribution<UDT>::predict(Duration deltaT, PredictionInformation const& prediction_info)
{
  TracyFiberEnter("GLMBDistribution::predict");
  TracyCZone(ctx, 1);
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution before predict.");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  _previous_label_ex_probs.insert(_manager->filter_time(), label_2_ex_prob());
  add_birth_tracks();
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution after added birth tracks.");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  std::unordered_map<StateId, StateId> const old_2_predicted_id =
      utils::predict_db(_manager, all_tracks(), deltaT, prediction_info._egoMotion);
  for (auto const& [old, predicted] : old_2_predicted_id)
  {
    if (predicted != EMPTY_STATE)
    {
      _manager->database().state_persistences.emplace(predicted, [&] -> UDT {
        UDT const surv_prob = _manager->getPersistenceModel().getPersistenceProbability(
            _manager->get_state(old), deltaT, prediction_info._roi);
        if (_manager->get_state(old)._type == STATE_TYPE::BIRTH and
            _manager->params()->glmb_distribution.update.joint_prediction_and_update)
        {
          return surv_prob * _manager->get_state_existence_probability(old);
        }
        return surv_prob;
      }());
    }
  }
  {
    std::vector<Hypotheses<UDT>> all_predicted_hyps;
    for (Hypotheses<UDT> const& hypotheses : _hypotheses)
    {
      Hypotheses<UDT> predicted_hyps;
      if (_manager->params()->glmb_distribution.update.joint_prediction_and_update or
          _manager->params()->glmb_distribution.multi_sensor.method == GLMB_MULTI_SENSOR_UPDATE_METHOD::MS)
      {
        // just take the old hypotheses with updated state ids
        for (Hypothesis<UDT> const& hyp : hypotheses._hyps)
        {
          Hypothesis new_hyp({}, hyp._weightLog, hyp._prior_id, hyp._prior_weight_log);
          for (StateId id : hyp._tracks)
          {
            if (StateId predicted_id = old_2_predicted_id.at(id); predicted_id != EMPTY_STATE)
            {
              new_hyp._tracks.push_back(predicted_id);
            }
          }
          std::ranges::sort(new_hyp._tracks);
          predicted_hyps.add(std::move(new_hyp));
        }
      }
      else  // no joint prediction and update
      {
        // use the k-shortest path algorithm to come up with the most likely hypotheses
        using Edge = std::pair<StateId, HypothesisId>;
        using Node = std::size_t;  // nodes have no meaning
        Node start_node{ 0 };
        Node end_node{ 1 };
        std::vector<Node> nodes{ start_node, end_node };  // start = 0
        std::size_t const largest_num_tracks = hypotheses.max_tracks();
        nodes.reserve(2 + hypotheses._hyps.size() * largest_num_tracks);
        std::vector<graph::Edge<Edge, Node, double>> edges;
        edges.reserve(hypotheses._hyps.size() * (2 + largest_num_tracks * 2));
        std::size_t next_node = 2;
        for (Hypothesis<UDT> const& hyp : hypotheses._hyps)
        {
          if (largest_num_tracks > 0)
          {
            edges.emplace_back(std::pair{ StateId{ 0 }, hyp._id }, start_node, next_node, to_val(hyp._weightLog));
            for (std::size_t i = 0; i < largest_num_tracks; ++i)
            {
              if (i < hyp._tracks.size() and old_2_predicted_id.at(hyp._tracks.at(i)) != EMPTY_STATE)
              {
                State<UDT> const& new_state = _manager->get_state(old_2_predicted_id.at(hyp._tracks.at(i)));
                UDT surv_prob = [&] {
                  assert(_manager->database().state_persistences.contains(new_state._id));
                  return _manager->database().state_persistences.find(new_state._id)->persistence_probability;
                }();
                assert([&] {  // NOLINT
                  if (not std::isfinite(surv_prob))
                  {
                    LOG_FATAL("can not predict this");
                    LOG_FATAL(new_state.toString());
                    return false;
                  }
                  return true;
                }());
                edges.emplace_back(std::pair{ hyp._tracks.at(i), HypothesisId{ 0 } },
                                   next_node,
                                   next_node + 1,
                                   to_val(log(surv_prob)));
                edges.emplace_back(
                    std::pair{ StateId{ 0 }, HypothesisId{ 0 } }, next_node, next_node + 1, to_val(log(1 - surv_prob)));
                nodes.push_back(next_node);
                ++next_node;
              }
              else  // add dummy edge so all sub-graphs have the same size, this is needed for the k_shortest_path
                    // algorithm
              {
                edges.emplace_back(std::pair{ StateId{ 0 }, HypothesisId{ 0 } }, next_node, next_node + 1, 0);
                nodes.push_back(next_node);
                ++next_node;
              }
            }
            edges.emplace_back(std::pair{ StateId{ 0 }, HypothesisId{ 0 } }, next_node, end_node, 0);
            nodes.push_back(next_node);
            ++next_node;
          }
          else
          {
            edges.emplace_back(std::pair{ StateId{ 0 }, hyp._id }, start_node, end_node, to_val(hyp._weightLog));
          }
        }
        graph::DiGraph graph(std::move(nodes), std::move(edges));
        auto best_paths = graph.k_longest_paths(
            start_node, end_node, _manager->params()->glmb_distribution.post_process_prediction.max_hypotheses);
        for (auto const& [path, weight_log] : best_paths)
        {
          assert(path.front().second != HypothesisId{ 0 });
          assert(hypotheses._hyps.contains(path.front().second));
          assert([&] {  // NOLINT
            if (path.size() - 2 != largest_num_tracks and
                not hypotheses._hyps.find(path.front().second)->_tracks.empty())
            {
              LOG_FATAL("path size: " << path.size());
              LOG_FATAL("#track size: " << hypotheses._hyps.find(path.front().second)->_tracks.size());
              return false;
            }
            return true;
          }());
          assert(path.front().first == StateId{ 0 });
          std::vector<StateId> hyp_tracks = {};
          for (auto const& [state_id, _] : path)
          {
            if (state_id != StateId{ 0 })
            {
              hyp_tracks.push_back(old_2_predicted_id.at(state_id));
            }
          }
          assert(std::isfinite(std::exp(weight_log)));
          Hypothesis<UDT> hyp(std::move(hyp_tracks), UDT{ weight_log });
          predicted_hyps.add(std::move(hyp));
        }
      }
      predicted_hyps.normalize_hypotheses_weight();
      all_predicted_hyps.push_back(std::move(predicted_hyps));
    }
    _hypotheses = std::move(all_predicted_hyps);
  }
  post_process_prediction();
  reset_caches();
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution after predict.");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  post_process_prediction();
  reset_caches();
  TracyFiberEnter("GLMBDistribution::predict");
  TracyCZoneEnd(ctx);
}

template <class UDT>
std::pair<Cost, Cost> GLMBDistribution<UDT>::cost_matrix(std::vector<StateId> const& tracks,
                                                         DataSourceContainer<UDT> const& data_source_container) const
{
  return cost_matrix(tracks, std::span{ &data_source_container, 1 });
}

template <class UDT>
std::pair<Cost, Cost>
GLMBDistribution<UDT>::cost_matrix(std::vector<StateId> const& tracks,
                                   std::span<DataSourceContainer<UDT> const> data_source_containers) const
{
  LOG_DEB("GLMBDistribution::cost_matrix");
  ZoneScopedN("GLMBDistribution::cost_matrix");
  Cost cost;
  cost.reserve(tracks.size());
  Cost soft_cost_able;
  soft_cost_able.reserve(tracks.size());
  for (StateId track_id : tracks)
  {
    StateLikelihood state_cost{ .id = track_id };
    state_cost.alive.reserve(data_source_containers.size());
    StateLikelihood state_soft_cost{ .id = track_id };
    state_soft_cost.alive.reserve(data_source_containers.size());
    for (auto const& [n, data_source_container] : std::views::enumerate(data_source_containers))
    {
      // Detection and Misdetection probability
      Probability<double> const detection_prob =
          to_val(std::as_const(*_manager)
                     .database()
                     .state_detection_probabilities.find(std::tuple{ track_id, data_source_container._id })
                     ->detection_probability);
      double const logPD = std::log(detection_prob);
      double const logPND = std::log(1 - detection_prob);
      double const predictedExProb = [&] {
        if (n > 0 or
            not _manager->params()->glmb_distribution.update.joint_prediction_and_update)  /// apply existence cost only
                                                                                           /// in case of joint update
                                                                                           /// and only for the first
                                                                                           /// sensor
        {
          return 1.0;
        }
        // GLMB filter -> survival probability plays role of existence probability in joint prediction and update
        return [&] {
          assert([&] {
            if (not _manager->database().state_persistences.contains(track_id))  /// birth tracks may not have a
                                                                                 /// persistence probability
            {
              LOG_FATAL(std::stacktrace::current());
              LOG_FATAL(_manager->get_state(track_id).toString());
              LOG_FATAL(_manager->get_state_existence_probability(track_id));
              return false;
            }
            return true;
          }());
          auto const it = _manager->database().state_persistences.find(track_id);
          if (it != _manager->database().state_persistences.end())
          {
            return to_val(it->persistence_probability);
          }
          return 1.0;
        }();
      }();
      assert([&] {  // NOLINT
        if (not std::isfinite(predictedExProb))
        {
          return false;
        }
        return true;
      }());
      double const log_predictedExProb = std::log(predictedExProb);

      state_cost.alive.emplace_back(data_source_container._id, std::vector<std::pair<MeasurementId, double>>{});
      state_cost.alive.back().second.reserve(data_source_container._measurements.size());
      state_soft_cost.alive.emplace_back(data_source_container._id, std::vector<std::pair<MeasurementId, double>>{});
      state_soft_cost.alive.back().second.reserve(data_source_container._measurements.size());
      std::vector<StateId> const& relevant_meas =
          std::as_const(*_manager)
              .database()
              .relevant_meas.find(std::tuple{ track_id, data_source_container._id })
              ->relevant_meas_ids;
      for (StateId const meas_id : relevant_meas)
      {
        if (meas_id == NOT_DETECTED)
        {
          continue;
        }
        double const clutter_intensity = [&] {
          return to_val(std::as_const(*_manager).database().meas_clutter_intensity.find(meas_id)->clutter_intensity);
        }();
        double const log_likelihood = to_val(std::as_const(*_manager)
                                                 .database()
                                                 .state_meas_log_likelihood.find(std::tuple{ track_id, meas_id })
                                                 ->meas_log_likelihood);
        // track exists and is detected
        if (_manager->params()->glmb_distribution.update.joint_prediction_and_update)
        {
          state_cost.alive.back().second.emplace_back(
              meas_id, std::exp(log_predictedExProb + logPD + log_likelihood - std::log(clutter_intensity)));
          state_soft_cost.alive.back().second.emplace_back(
              meas_id,
              std::exp(
                  std::log(to_val(_manager->database().state_persistences.find(track_id)->persistence_probability)) +
                  log_likelihood));
        }
        else
        {
          state_cost.alive.back().second.emplace_back(meas_id,
                                                      std::exp(logPD + log_likelihood - std::log(clutter_intensity)));
          state_soft_cost.alive.back().second.emplace_back(meas_id, std::exp(log_likelihood));
        }
      }

      // track exists but mis detected
      if (_manager->params()->glmb_distribution.update.joint_prediction_and_update)
      {
        state_cost.alive.back().second.emplace_back(NOT_DETECTED, std::exp(log_predictedExProb + logPND));
        state_soft_cost.alive.back().second.emplace_back(
            NOT_DETECTED, to_val(_manager->database().state_persistences.find(track_id)->persistence_probability));
      }
      else
      {
        state_cost.alive.back().second.emplace_back(NOT_DETECTED, std::exp(logPND));
        state_soft_cost.alive.back().second.emplace_back(NOT_DETECTED, std::numeric_limits<double>::quiet_NaN());
      }
      // track does not exist, i.e., has died
      if (n == 0 and _manager->params()->glmb_distribution.update.joint_prediction_and_update)
      {
        state_cost.surv_likelihood = predictedExProb;
        state_soft_cost.surv_likelihood =
            to_val(_manager->database().state_persistences.find(track_id)->persistence_probability);
      }
    }
    cost.push_back(std::move(state_cost));
    soft_cost_able.push_back(std::move(state_soft_cost));
  }
  return { std::move(cost), std::move(soft_cost_able) };
}

template <class UDT>
std::vector<Hypothesis<UDT>>
GLMBDistribution<UDT>::update_single_hypothesis(Hypothesis<UDT> const& prior_hypothesis,
                                                std::size_t num_update_hypothesis,
                                                DataSourceContainer<UDT> const& data_source_container) const
{
  ZoneScopedN("GLMBDistribution::update_single_hypothesis");
  LOG_DEB("Update hypothesis: " << prior_hypothesis.toString());
  std::vector<Hypothesis<UDT>> updated_hypotheses;
  if (num_update_hypothesis == 0)
  {
    return updated_hypotheses;
  }
  if (prior_hypothesis._tracks.empty())
  {
    Hypothesis<UDT> updated_hyp({}, prior_hypothesis._weightLog, prior_hypothesis._id, prior_hypothesis._weightLog);
    if (_manager->params()->self_assessment.enable)
    {
      SoftCosts<UDT> soft = soft_cost_assignment(*_manager, std::span{ &data_source_container, 1 }, {}, {});
      updated_hyp._soft_detection_weight = std::move(soft.soft_detection_likelihood);
      updated_hyp._soft_clutter_weight = std::move(soft.soft_clutter_likelihood);
      updated_hyp._point_weight = std::move(soft.point_likelihood);
    }
    LOG_DEB(std::format("updated hyp: {}", updated_hyp.toString()));
    updated_hypotheses.push_back(std::move(updated_hyp));
    return updated_hypotheses;
  }

  auto const [cost, soft_cost] = cost_matrix(prior_hypothesis._tracks, data_source_container);
  std::vector<Assignment> assignments{ [&] -> std::vector<Assignment> {
    switch (_manager->params()->glmb_distribution.update.assignment_method)
    {
      case GLMB_ASSIGNMENT_METHOD::MURTY:
        return murty::murty(cost, num_update_hypothesis);
      case GLMB_ASSIGNMENT_METHOD::GIBBS_SAMPLING:
      {
        Assignment const init_assignment = [&] {
          Assignment init;
          for (StateId state : prior_hypothesis._tracks)
          {
            init.emplace_back(state, true, std::vector{ std::pair{ data_source_container._id, NOT_DETECTED } });
          }
          return init;
        }();
        return gibbs::sample(*_manager,
                             cost,
                             num_update_hypothesis,
                             init_assignment,
                             num_update_hypothesis *
                                 _manager->params()->glmb_distribution.update.gibbs_sampling.max_trials_factor,
                             _manager->params()->glmb_distribution.update.gibbs_sampling.abort_after_ntimes_no_new_sol);
      }
    }
    assert(false);
    std::unreachable();
  }() };
  for (Assignment const& assignment : assignments)
  {
    std::vector<StateId> updatedTrackList;
    for (auto const& [state_id, alive, meas_assignments] : assignment)
    {
      if (not alive)
      {
        continue;
      }
      assert(meas_assignments.size() == 1);
      updatedTrackList.push_back(
          std::as_const(*_manager)
              .database()
              .state_updates
              .find(std::tuple{ state_id, meas_assignments.front().first, meas_assignments.front().second })
              ->updated_state_id);
    }
    UDT const weight = prior_hypothesis._weightLog + log(cost_assignment<UDT>(cost, assignment));
    Hypothesis updated_hyp(std::move(updatedTrackList), weight, prior_hypothesis._id, prior_hypothesis._weightLog);
    if (_manager->params()->self_assessment.enable)
    {
      LOG_DEB("enable self assessment");
      SoftCosts<UDT> soft =
          soft_cost_assignment(*_manager, std::span{ &data_source_container, 1 }, soft_cost, assignment);
      updated_hyp._soft_detection_weight = std::move(soft.soft_detection_likelihood);
      updated_hyp._soft_clutter_weight = std::move(soft.soft_clutter_likelihood);
      updated_hyp._point_weight = std::move(soft.point_likelihood);
    }
    updated_hypotheses.push_back(std::move(updated_hyp));
  }
  return updated_hypotheses;
}

template <class UDT>
void GLMBDistribution<UDT>::update(DataSourceContainer<UDT> const& measurements,
                                   bool do_innovate,
                                   bool do_split_and_merge)
{
  TracyFiberEnter("GLMBDistribution::update");
  TracyCZone(ctx, 1);
  LOG_DEB("update");
  add_birth_tracks();
  reset_caches();
  if (do_innovate)
  {
    utils::innovate_db(_manager, all_tracks(), measurements);
  }

  std::vector<MergedHypotheses> merged_hypotheseses = [&] {
    if (do_split_and_merge)
    {
      return split_and_merge_hypotheses(
          _hypotheses, std::span{ &measurements, 1 }, SplitAndMergeHypothesesType::COMMON_MEASUREMENT);
    }
    std::vector<MergedHypotheses> merged;
    for (Hypotheses<UDT> const& hyps : _hypotheses)
    {
      merged.emplace_back(hyps, std::vector{ measurements });
    }
    return merged;
  }();
  LOG_DEB(std::format("Num groups {}", merged_hypotheseses.size()));
  LOG_DEB("GLMBDistribution::update");
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution before update.");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  LOG_DEB("Updated tracks moved to _updated_tracks");
  LOG_DEB("GLMB density before update: " << to_string());
  {
    struct WorkItem
    {
      std::size_t hypotheses_ind;
      HypothesisId hypothesis_ind;
      std::vector<Hypothesis<UDT>> updates;
    };
    std::vector<std::unordered_map<HypothesisId, std::size_t>> number_update_hyps;

    std::vector<WorkItem> work_items;
    for (auto const& [hypotheses_ind, merged_hypotheses] : std::views::enumerate(merged_hypotheseses))
    {
      number_update_hyps.emplace_back(allocation(merged_hypotheses.hypotheses));
      for (Hypothesis<UDT> const& hyp : merged_hypotheses.hypotheses._hyps)
      {
        work_items.emplace_back(hypotheses_ind, hyp._id, std::vector<Hypothesis<UDT>>{});
      }
    }
    std::vector<Hypotheses<UDT>> updated_hyps(merged_hypotheseses.size());
    hpx::for_each(hpx::execution::par, work_items.begin(), work_items.end(), [&](WorkItem& update) {
      update.updates = update_single_hypothesis(
          *merged_hypotheseses.at(update.hypotheses_ind).hypotheses._hyps.find(update.hypothesis_ind),
          number_update_hyps.at(update.hypotheses_ind).at(update.hypothesis_ind),
          merged_hypotheseses.at(update.hypotheses_ind).data_source_containers.front());
    });
    for (WorkItem& update : work_items)
    {
      for (Hypothesis<UDT>& hyp : update.updates)
      {
        updated_hyps.at(update.hypotheses_ind).add(std::move(hyp));
      }
    }
    for (Hypotheses<UDT>& hyp : updated_hyps)
    {
      hyp.normalize_hypotheses_weight();
    }
    _hypotheses = std::move(updated_hyps);
  }
  post_process_update();
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("invalid after update");
      return false;
    }
    return true;
  }());
  if (_manager->params()->self_assessment.enable)
  {
    _clutter_dist.emplace(measurements._id, compute_clutter_distribution(measurements));
    _detection_probs.emplace(measurements._id, compute_detection_distribution(measurements));
  }
  if (_manager->params()->show_gui)
  {
    double assoc = 0;
    for (auto const& [meas_id, assoc_prob] : meas_assignment_prob(measurements))
    {
      assoc += to_val(assoc_prob);
    }
    {
      _manager->vizu()._meas_model_data.wlock().data[measurements._source_id]._num_assoc_measurements.emplace_back(
          measurements._time, assoc);
      _manager->vizu()._meas_model_data.wlock().data[measurements._source_id]._cardinality_distribution.emplace_back(
          measurements._time, to_val(cardinality_distribution()));
      _manager->vizu()
          ._meas_model_data.wlock()
          .data[measurements._source_id]
          ._num_hypotheses_per_cardinality.emplace_back(measurements._time,
                                                        num_hypotheses_per_cardinality().template cast<double>());
      _manager->vizu()._meas_model_data.wlock().data[measurements._source_id]._num_hypotheses.emplace_back(
          measurements._time, _hypotheses.size());
    }
    if (_manager->params()->show_gui)
    {
      std::vector<std::tuple<std::vector<State<UDT>>, std::size_t, Hypotheses<UDT>>> groups;
      for (Hypotheses<UDT> const& hyps : _hypotheses)
      {
        std::vector<State<UDT>> group;
        for (StateId id : hyps.all_tracks())
        {
          group.push_back(_manager->get_state(id));
        }
        groups.emplace_back(std::move(group), hyps._hyps.size(), hyps);
      }
      _manager->vizu()._meas_model_data.wlock().data[measurements._source_id]._track_groups.emplace_back(
          measurements._time, std::move(groups));
    }
  }
  LOG_DEB("GLMBDistribution::update done");
  TracyFiberEnter("GLMBDistribution::update");
  TracyCZoneEnd(ctx);
}

template <class UDT>
void GLMBDistribution<UDT>::update(std::vector<DataSourceContainer<UDT>> const& data_source_containers,
                                   bool is_pmlmb,
                                   bool do_innovate)
{
  LOG_DEB("GLMBDistribution::multi-sensor update");
  add_birth_tracks();
  reset_caches();
  if (data_source_containers.empty())
  {
    return;
  }
  assert([&] {  // NOLINT
    Time const first = data_source_containers.front()._time;
    if (not std::ranges::all_of(data_source_containers, [&](DataSourceContainer<UDT> const& data_container) {
          if (std::chrono::abs(first - data_container._time) > 10ms)
          {
            return false;
          }
          return true;
        }))
    {
      LOG_FATAL("DataContainers not from the same time point.");
      return false;
    }
    return true;
  }());
  if (do_innovate)
  {
    for (DataSourceContainer<UDT> const& data_container : data_source_containers)
    {
      utils::innovate_db(_manager, all_tracks(), data_container);
    }
  }
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution before update.");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  switch (_manager->params()->glmb_distribution.multi_sensor.method)
  {
    case GLMB_MULTI_SENSOR_UPDATE_METHOD::MS:
    {
      so_ms_gibbs_update(data_source_containers);
      break;
    }
    case GLMB_MULTI_SENSOR_UPDATE_METHOD::PM:
    {
      pm_update(data_source_containers, is_pmlmb);
      break;
    }
  }
  if (_manager->params()->show_gui)
  {
    std::vector<std::tuple<std::vector<State<UDT>>, std::size_t, Hypotheses<UDT>>> groups;
    for (Hypotheses<UDT> const& hyps : _hypotheses)
    {
      std::vector<State<UDT>> group;
      for (StateId id : hyps.all_tracks())
      {
        group.push_back(_manager->get_state(id));
      }
      groups.emplace_back(std::move(group), hyps._hyps.size(), hyps);
    }
    for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
    {
      _manager->vizu()._meas_model_data.wlock().data[data_source_container._source_id]._track_groups.emplace_back(
          data_source_container._time, std::move(groups));
    }
  }
}

template <class UDT>
std::pair<std::vector<Hypothesis<UDT>>, std::vector<State<UDT>>>
GLMBDistribution<UDT>::so_ms_gibbs_update_single_hypothesis(
    Hypothesis<UDT> const& prior_hypothesis,
    std::size_t num_update_hypothesis,
    std::vector<DataSourceContainer<UDT>> const& data_source_containers) const
{
  // ScopedN("GLMBDistribution::so_ms_gibbs_update_single_hypothesis");
  LOG_DEB("Update hypothesis: " << prior_hypothesis.toString());
  std::vector<Hypothesis<UDT>> updated_hypothesis;
  std::vector<State<UDT>> updated_states;
  if (num_update_hypothesis == 0)
  {
    return { std::move(updated_hypothesis), std::move(updated_states) };
  }
  if (prior_hypothesis._tracks.empty())
  {
    Hypothesis updated_hyp({}, prior_hypothesis._weightLog, prior_hypothesis._id, prior_hypothesis._weightLog);
    if (_manager->params()->self_assessment.enable)
    {
      SoftCosts soft = soft_cost_assignment<UDT>(*_manager, data_source_containers, Cost{}, {});
      updated_hyp._soft_detection_weight = std::move(soft.soft_detection_likelihood);
      updated_hyp._soft_clutter_weight = std::move(soft.soft_clutter_likelihood);
      updated_hyp._point_weight = std::move(soft.point_likelihood);
      updated_hyp._num_clutter = std::move(soft.num_clutter);
      updated_hyp._num_detected = std::move(soft.num_detected);
    }
    updated_hypothesis.push_back(std::move(updated_hyp));
    return { std::move(updated_hypothesis), std::move(updated_states) };
  }
  auto const [cost, soft_costs] = cost_matrix(prior_hypothesis._tracks, data_source_containers);
  std::vector<Assignment> assignments{ [&] {
    Assignment const init_assignment = [&] {
      Assignment init;
      for (StateId state : prior_hypothesis._tracks)
      {
        init.emplace_back(state, true, [&] {
          std::vector<std::pair<DataSourceContainerId, MeasurementId>> out;
          for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
          {
            out.emplace_back(data_source_container._id, NOT_DETECTED);
          }
          return out;
        }());
      }
      return init;
    }();
    return gibbs::sample(*_manager,
                         cost,
                         num_update_hypothesis,
                         init_assignment,
                         num_update_hypothesis *
                             _manager->params()->glmb_distribution.update.gibbs_sampling.max_trials_factor,
                         _manager->params()->glmb_distribution.update.gibbs_sampling.abort_after_ntimes_no_new_sol);
  }() };
  LOG_DEB(std::format("Found {} new hypotheses", assignments.size()));
  for (Assignment const& assignment : assignments)
  {
    std::vector<StateId> updatedTrackList;
    bool fused_state_is_empty = false;
    for (auto const& [state_id, alive, meas_assignments] : assignment)
    {
      if (not alive)
      {
        continue;
      }
      assert(meas_assignments.size() == data_source_containers.size());
      std::vector<State<UDT>> posteriors;
      for (auto const& [data_container_id, meas_id] : meas_assignments)
      {
        assert(std::as_const(*_manager).database().state_updates.contains(
            std::tuple{ state_id, data_container_id, meas_id }));
        posteriors.push_back(
            _manager->get_state(std::as_const(*_manager)
                                    .database()
                                    .state_updates.find(std::tuple{ state_id, data_container_id, meas_id })
                                    ->updated_state_id));
      }
      std::optional<State<UDT>> updated_state = fuse_posteriors(
          _manager->get_state(state_id), posteriors, _manager->params()->state.multi_model.use_state_models.size());
      if (updated_state.has_value())
      {
        StateId const id = updated_state.value()._id;
        updated_states.push_back(std::move(updated_state.value()));
        updatedTrackList.push_back(id);
      }
      else
      {
        LOG_DEB("check how thi scan happen");
        LOG_DEB(_manager->get_state(state_id).toString());
        for (State<UDT> const& post : posteriors)
        {
          LOG_DEB(post.toString());
        }
        assert((_manager->params()->filter.type != FILTER_TYPE::LMB_IC ||
                _manager->params()->filter.type != FILTER_TYPE::LMB_MS) &&
               "Fused mixture is empty due to pruning in posteriors. Is probably only ok for mixture based densities");
        LOG_WARN("Fused mixture is empty due to pruning in posteriors. Is only ok for LMB based Trackers!");
        // This case happens when not all posteriors have a value for one prior component! (Fused mixture is empty due
        // to pruning in local updates) This happens in very complex scenarios like bearing only! Todo: Make sure there
        // is at least one common component! (Must be probably fixed in innovation part...) this fused component does
        // not exist which is worse....
        fused_state_is_empty = true;
      }
    }
    if (fused_state_is_empty)
    {
      LOG_WARN("Fused hypotheses is not created, because one state is empty...");
      continue;
    }
    UDT const weight = prior_hypothesis._weightLog + log(cost_assignment<UDT>(cost, assignment));
    Hypothesis update_hyp(std::move(updatedTrackList), weight, prior_hypothesis._id, prior_hypothesis._weightLog);
    if (_manager->params()->self_assessment.enable)
    {
      SoftCosts soft = soft_cost_assignment<UDT>(*_manager, data_source_containers, soft_costs, assignment);
      update_hyp._soft_detection_weight = std::move(soft.soft_detection_likelihood);
      update_hyp._soft_clutter_weight = std::move(soft.soft_clutter_likelihood);
      update_hyp._point_weight = std::move(soft.point_likelihood);
      update_hyp._num_clutter = std::move(soft.num_clutter);
      update_hyp._num_detected = std::move(soft.num_detected);
    }
    updated_hypothesis.push_back(std::move(update_hyp));
  }
  return { std::move(updated_hypothesis), std::move(updated_states) };
}

template <class UDT>
void GLMBDistribution<UDT>::so_ms_gibbs_update(std::vector<DataSourceContainer<UDT>> const& data_source_containers)
{
  // ZoneScopedN("GLMBDistribution::so_ms_gibbs_update");

  LOG_DEB("Updated tracks moved to _updated_tracks");
  LOG_DEB("GLMB density before update: " << to_string());
  {
    struct WorkItem
    {
      std::size_t hypotheses_ind;
      HypothesisId hypothesis_ind;
      std::pair<std::vector<Hypothesis<UDT>>, std::vector<State<UDT>>> updates;
    };
    std::vector<MergedHypotheses> merged_hypotheseses = split_and_merge_hypotheses(
        _hypotheses, data_source_containers, SplitAndMergeHypothesesType::COMMON_MEASUREMENT);
    assert(not merged_hypotheseses.empty());
    std::vector<std::vector<State<UDT>>> all_updated_states(merged_hypotheseses.size());
    std::vector<Hypotheses<UDT>> all_hypotheses(merged_hypotheseses.size());
    std::atomic<std::size_t> atomic_index = 0;
    hpx::for_each(
        hpx::execution::par_unseq,
        merged_hypotheseses.begin(),
        merged_hypotheseses.end(),
        [&](MergedHypotheses const& merged_hypotheses) {
          std::unordered_map<HypothesisId, std::size_t> const number_updates = allocation(merged_hypotheses.hypotheses);
          std::vector<WorkItem> work_items;
          for (Hypothesis<UDT> const& hyp : merged_hypotheses.hypotheses._hyps)
          {
            work_items.emplace_back(-1, hyp._id, std::pair<std::vector<Hypothesis<UDT>>, std::vector<State<UDT>>>{});
          }
          hpx::for_each(hpx::execution::par_unseq, work_items.begin(), work_items.end(), [&](WorkItem& to_update) {
            to_update.updates =
                so_ms_gibbs_update_single_hypothesis(*merged_hypotheses.hypotheses._hyps.find(to_update.hypothesis_ind),
                                                     number_updates.at(to_update.hypothesis_ind),
                                                     merged_hypotheses.data_source_containers);
          });
          std::vector<State<UDT>> updated_states;
          Hypotheses<UDT> updated_hypotheses;
          for (WorkItem& update : work_items)
          {
            for (Hypothesis<UDT>& hyp : update.updates.first)
            {
              updated_hypotheses.add(std::move(hyp));
            }
            updated_states.insert(updated_states.end(),
                                  std::make_move_iterator(update.updates.second.begin()),
                                  std::make_move_iterator(update.updates.second.end()));
          }
          if (updated_hypotheses.sum_hypotheses_weight() > 0)
          {
            updated_hypotheses.normalize_hypotheses_weight();
          }
          std::size_t const loc_index = atomic_index++;
          all_hypotheses.at(loc_index) = std::move(updated_hypotheses);
          all_updated_states.at(loc_index) = std::move(updated_states);
        });

    std::vector<Hypotheses<UDT>> relevant_hyps;
    relevant_hyps.reserve(all_hypotheses.size());
    for (Hypotheses<UDT>& hyp : all_hypotheses)
    {
      if (hyp.sum_hypotheses_weight() > 0)
      {
        relevant_hyps.push_back(std::move(hyp));
      }
    }
    _hypotheses = std::move(relevant_hyps);
    {
      // ZoneScopedN("GLMBDistribution::so_ms_gibbs_update::update_state_database");
      for (std::vector<State<UDT>>& states : all_updated_states)
      {
        for (State<UDT>& state : states)
        {
          _manager->database().states.insert(std::move(state));
        }
      }
    }
  }

  {
    if (_manager->params()->self_assessment.enable)
    {
      for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
      {
        _prior_weight_detection_distribution[data_source_container._id];
        _prior_weight_clutter_distribution[data_source_container._id];
        _clutter_dist[data_source_container._id];
        _detection_probs[data_source_container._id];
      }
      hpx::for_each(
          hpx::execution::par_unseq,
          data_source_containers.begin(),
          data_source_containers.end(),
          [&](DataSourceContainer<UDT> const& data_source_container) {
            std::vector tmp_todo_inds{ 1, 2, 3, 4 };
            hpx::for_each(hpx::execution::par_unseq, tmp_todo_inds.begin(), tmp_todo_inds.end(), [&](int ind) {
              if (ind == 3)
              {
                _clutter_dist.at(data_source_container._id) = compute_clutter_distribution(data_source_container);
              }
              else if (ind == 4)
              {
                _detection_probs.at(data_source_container._id) = compute_detection_distribution(data_source_container);
              }
            });
          });
    }
  }
  post_process_update();
  assert([&] {  // NOLINT
    if (not std::ranges::all_of(all_tracks(), [&](StateId id) {
          return _manager->get_state(id)._meta_data._last_meas_log_likelihood.size() ==
                     data_source_containers.size() and
                 _manager->get_state(id)._meta_data._meas_update_weight.size() == data_source_containers.size() and
                 _manager->get_state(id)._meta_data._detectable.size() == data_source_containers.size();
        }))
    {
      LOG_FATAL("meta data not consistent after update");
      LOG_FATAL(to_string("", true));
      return false;
    }
    if (_manager->params()->self_assessment.enable)
    {
      for (Hypotheses<UDT> const& hypotheses : _hypotheses)
      {
        for (Hypothesis<UDT> const& hyp : hypotheses._hyps)
        {
          for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
          {
            if (not hyp._soft_detection_weight.contains(data_source_container._source_id) or
                not hyp._soft_clutter_weight.contains(data_source_container._source_id) or
                not hyp._point_weight.contains(data_source_container._source_id))
            {
              LOG_FATAL("Hypothesis: " << hyp.toString() << " does not contain the soft likelihoods");
              return false;
            }
          }
        }
      }
    }
    if (not is_valid())
    {
      LOG_FATAL("invalid after update");
      return false;
    }
    return true;
  }());
}

template <class UDT>
void GLMBDistribution<UDT>::pm_update(std::vector<DataSourceContainer<UDT>> const& data_source_containers,
                                      bool is_pmlmb)
{
  std::vector<GLMBDistribution> posteriors;
  std::vector<MergedHypotheses> merged_hypotheseses =
      split_and_merge_hypotheses(_hypotheses, data_source_containers, SplitAndMergeHypothesesType::COMMON_MEASUREMENT);
  _hypotheses.clear();
  for (MergedHypotheses& merged : merged_hypotheseses)
  {
    _hypotheses.push_back(std::move(merged.hypotheses));
  }
  reset_caches();
  Params next_params = *_manager->params();
  next_params.glmb_distribution.post_process_update.standard_pp.limit_states_per_label = false;
  _manager->update_params(next_params);
  reset_caches();

  for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
  {
    GLMBDistribution post = *this;
    post.update(data_source_container, false, false);
    _meas_assignment_prob[data_source_container._id] = post.meas_assignment_prob(data_source_container);
    if (_manager->params()->self_assessment.enable)
    {
      _clutter_dist[data_source_container._id] = post.compute_clutter_distribution(data_source_container);
      _detection_probs[data_source_container._id] = post.compute_detection_distribution(data_source_container);
    }
    posteriors.push_back(std::move(post));
  }
  next_params.glmb_distribution.post_process_update.standard_pp.limit_states_per_label =
      _manager->original_params().glmb_distribution.post_process_update.standard_pp.limit_states_per_label;
  for (auto const& [model_id, _] : _manager->getStateModelMap())
  {
    next_params.state_model_params(model_id).distribution.post_process.enable =
        _manager->original_params().state_model_params(model_id).distribution.post_process.enable;
  }
  _manager->update_params(std::move(next_params));
  pm_fusion(std::move(posteriors), is_pmlmb);
}

/// needed information's for the State posterior fusion
using EdgeId = std::tuple<std::size_t, HypothesisId>;
struct PMFusionInfos
{
  /// graph describing the k shortest path problem
  graph::DiGraph<std::size_t, EdgeId, double> _graph;
  /// start node and end node
  std::size_t _startNode;
  std::size_t _endNode;
  /// how many solutions shall be calculated at the kBestSelection/generalizedKBestSelection algorithm
  std::size_t _k;
};

template <class UDT>
PMFusionInfos create_graph_fuse_posteriors_GKBest(GLMBDistribution<UDT> const& prior,
                                                  std::vector<GLMBDistribution<UDT>> const& posteriors,
                                                  std::size_t hypotheses_ind)
{
  assert([&] {
    for (GLMBDistribution<UDT> const& post : posteriors)
    {
      if (post._hypotheses.size() != prior._hypotheses.size())
      {
        LOG_FATAL("Different number of glmb groups");
        return false;
      }
      for (std::size_t i = 0; i < prior._hypotheses.size(); ++i)
      {
        for (Hypothesis<UDT> const& post_hyp : post._hypotheses.at(i)._hyps)
        {
          if (not prior._hypotheses.at(i)._hyps.contains(post_hyp._prior_id) and not post_hyp._tracks.empty())
          {
            LOG_FATAL(std::format("Posterior Hypotheses contains hyp {}, which prior_id is not in the prior "
                                  "hypotheses",
                                  post_hyp.toString()));
            return false;
          }
        }
      }
    }
    return true;
  }());
  // edge.first ==0. prior distribution id,
  // else
  // posteriors(edge.first-1) distribution id
  std::vector<graph::Edge<EdgeId, std::size_t, double>> edges;
  constexpr std::size_t start_node = 0;
  constexpr std::size_t end_node = 1;
  std::vector nodes{ start_node, end_node };  // nodes have no meaning
  std::size_t next_node = 2;
  for (Hypothesis<UDT> const& prior_hyp : prior._hypotheses.at(hypotheses_ind)._hyps)
  {
    nodes.push_back(next_node);
    next_node++;
    edges.emplace_back(std::tuple{ std::size_t{ 0 }, prior_hyp._id },
                       start_node,
                       next_node - 1,
                       to_val(-prior_hyp._weightLog * (posteriors.size() - 1.0)));
    assert(not posteriors.empty());
    for (auto const& [post_ind, post] : std::views::enumerate(posteriors))
    {
      std::size_t const edge_end_node = [&] {
        if (post_ind + 1 < static_cast<Index>(posteriors.size()))
        {
          return next_node;
        }
        return end_node;
      }();
      bool added_edge = false;
      for (Hypothesis<UDT> const& post_hyp : post._hypotheses.at(hypotheses_ind)._hyps)
      {
        if (post_hyp._prior_id == prior_hyp._id)
        {
          edges.emplace_back(std::tuple{ static_cast<std::size_t>(post_ind + 1), post_hyp._id },
                             next_node - 1,
                             edge_end_node,
                             to_val(post_hyp._weightLog));
          assert(std::isfinite(post_hyp.getWeight()));
          added_edge = true;
        }
      }
      if (not added_edge)
      {
        LOG_DEB("No Posterior Hypothesis found for prior hypothesis " << prior_hyp.toString());
        edges.emplace_back(std::tuple{ static_cast<std::size_t>(post_ind + 1), NO_HYPOTHESIS_ID_HISTORY },
                           next_node - 1,
                           edge_end_node,
                           -std::numeric_limits<double>::infinity());
      }
      nodes.push_back(next_node);
      next_node++;
    }
  }
  auto graph = graph::DiGraph(std::move(nodes), std::move(edges));
  return { ._graph = std::move(graph), ._startNode = start_node, ._endNode = end_node, ._k = 0 };
}

template <class UDT>
std::vector<PMFusionInfos> create_graph_fuse_posteriors_KBest(GLMBDistribution<UDT> const& prior,
                                                              std::vector<GLMBDistribution<UDT>> const& posteriors,
                                                              std::size_t hypotheses_ind,
                                                              std::size_t num_components,
                                                              PM_MULTI_SENSOR_UPDATE_METHOD method,
                                                              std::unordered_map<HypothesisId, std::size_t> alloc)
{
  assert([&] {
    for (GLMBDistribution<UDT> const& post : posteriors)
    {
      if (post._hypotheses.size() != prior._hypotheses.size())
      {
        LOG_FATAL("Different number of glmb groups");
        return false;
      }
      for (std::size_t i = 0; i < prior._hypotheses.size(); ++i)
      {
        for (Hypothesis<UDT> const& post_hyp : post._hypotheses.at(i)._hyps)
        {
          if (not prior._hypotheses.at(i)._hyps.contains(post_hyp._prior_id) and not post_hyp._tracks.empty())
          {
            LOG_FATAL(std::format("Posterior Hypotheses contains hyp {}, which prior_id is not in the prior "
                                  "hypotheses",
                                  post_hyp.toString()));
            return false;
          }
        }
      }
    }
    return true;
  }());

  std::vector<PMFusionInfos> graphs;
  for (Hypothesis<UDT> const& prior_hyp : prior._hypotheses.at(hypotheses_ind)._hyps)
  {
    // edge.first ==0. prior distribution id,
    // else
    // posteriors(edge.first-1) distribution id

    std::vector<graph::Edge<EdgeId, std::size_t, double>> edges;
    constexpr std::size_t start_node = 0;
    constexpr std::size_t end_node = 1;
    std::vector nodes{ start_node, end_node };  // nodes have no meaning
    std::size_t next_node = 2;
    const std::size_t k = [&] {
      switch (method)
      {
        case PM_MULTI_SENSOR_UPDATE_METHOD::K_BEST_POISSON:
          return alloc.at(prior_hyp._id);
          break;
        case PM_MULTI_SENSOR_UPDATE_METHOD::K_BEST:
          return static_cast<std::size_t>(prior_hyp.getWeight() * num_components);
          break;
        default:
          LOG_WARN("PM_MULTI_SENSOR_UPDATE_METHOD not known. Use K_BEST");
          return static_cast<std::size_t>(prior_hyp.getWeight() * num_components);
      }
    }();
    if (k == 0)
    {
      // calculation of graph is not necessary...
      PMFusionInfos infos;
      infos._k = 0;
      graphs.emplace_back(std::move(infos));
      LOG_DEB("continue");
      continue;
    }
    nodes.push_back(next_node);
    next_node++;
    edges.emplace_back(std::tuple{ std::size_t{ 0 }, prior_hyp._id },
                       start_node,
                       next_node - 1,
                       to_val(-prior_hyp._weightLog * (posteriors.size() - 1.0)));
    assert(not posteriors.empty());
    for (auto const& [post_ind, post] : std::views::enumerate(posteriors))
    {
      std::size_t const edge_end_node = [&] {
        if (post_ind + 1 < static_cast<Index>(posteriors.size()))
        {
          return next_node;
        }
        return end_node;
      }();
      bool added_edge = false;
      for (Hypothesis<UDT> const& post_hyp : post._hypotheses.at(hypotheses_ind)._hyps)
      {
        if (post_hyp._prior_id == prior_hyp._id)
        {
          edges.emplace_back(std::tuple{ static_cast<std::size_t>(post_ind + 1), post_hyp._id },
                             next_node - 1,
                             edge_end_node,
                             to_val(post_hyp._weightLog));
          assert(std::isfinite(post_hyp.getWeight()));
          added_edge = true;
        }
      }
      if (not added_edge)
      {
        LOG_DEB("No Posterior Hypothesis found for prior hypothesis " << prior_hyp.toString());
        edges.emplace_back(std::tuple{ static_cast<std::size_t>(post_ind + 1), NO_HYPOTHESIS_ID_HISTORY },
                           next_node - 1,
                           edge_end_node,
                           -std::numeric_limits<double>::infinity());
      }
      nodes.push_back(next_node);
      next_node++;
    }
    auto single_graph = graph::DiGraph(std::move(nodes), std::move(edges));
    PMFusionInfos infos;
    infos._startNode = start_node;
    infos._endNode = end_node;
    infos._graph = std::move(single_graph);
    infos._k = k;
    graphs.emplace_back(std::move(infos));
  }
  return graphs;
}

template <class UDT>
void GLMBDistribution<UDT>::pm_fusion(std::vector<GLMBDistribution<UDT>> posteriors, bool is_pmlmb)
{
  LOG_DEB("pm fusion");
  if (posteriors.size() == 1)
  {
    *this = std::move(posteriors.front());
    return;
  }

  std::unordered_map<StateId, State<UDT>> fused_tracks;
  std::vector<Hypotheses<UDT>> all_fused_hypotheses;
  std::unordered_map<std::vector<StateId>, StateId> posterior_tracks_2_fused_track;  // prevent multiple calculation of
                                                                                     // same fused track...
  std::unordered_map<std::vector<StateId>, bool> posterior_tracks_2_fused_result_given;

  if (_manager->params()->glmb_distribution.multi_sensor.pm.method == PM_MULTI_SENSOR_UPDATE_METHOD::GENERALIZED_K_BEST)
  {
    // Use the GeneralizedKBest algo for each GLMB group!
    for (auto const& [hyp_ind, hypotheses] : std::views::enumerate(_hypotheses))
    {
      Hypotheses<UDT> fused_hypotheses;
      PMFusionInfos fusion_infos = create_graph_fuse_posteriors_GKBest(*this, posteriors, hyp_ind);
      for (auto const& [edges, weight_log] :
           fusion_infos._graph.k_longest_paths(fusion_infos._startNode,
                                               fusion_infos._endNode,
                                               _manager->params()->glmb_distribution.multi_sensor.pm.num_components))
      {
        if (weight_log == -std::numeric_limits<double>::infinity())
        {
          LOG_DEB("no solution found");
          continue;
        }
        std::map<StateId, std::vector<State<UDT>>> prior_post;
        std::map<StateId, std::vector<StateId>> prior_postId;
        type_safe::deferred_construction<HypothesisId> prior_hyp_id;
        type_safe::deferred_construction<UDT> prior_hyp_weight;
        type_safe::deferred_construction<UDT> fused_weight_log_pmlmb;
        for (auto const& [ind, hyp] : edges)
        {
          if (ind == 0)  // prior
          {
            prior_hyp_id = hyp;
            assert(hypotheses._hyps.contains(hyp));
            prior_hyp_weight = hypotheses._hyps.find(hyp)->_weightLog;
            if (is_pmlmb)
            {
              fused_weight_log_pmlmb = hypotheses._hyps.find(hyp)->_weightLog;
            }
            for (StateId state_id : hypotheses._hyps.find(hyp)->_tracks)
            {
              prior_post[state_id];
              prior_postId[state_id];
            }
          }
          else
          {
            assert([&] {  // NOLINT
              if (not posteriors.at(ind - 1)._hypotheses.at(hyp_ind)._hyps.contains(hyp))
              {
                LOG_FATAL(hyp_ind);
                LOG_FATAL(hypotheses.to_string());
                LOG_FATAL("posterior does not contain hyp: " << hyp);
                LOG_FATAL("post:" << posteriors.at(ind - 1)._hypotheses.at(hyp_ind).to_string());
                return false;
              }
              return true;
            }());
            for (StateId state_id : posteriors.at(ind - 1)._hypotheses.at(hyp_ind)._hyps.find(hyp)->_tracks)
            {
              State<UDT> const& post = _manager->get_state(state_id);
              assert(prior_post.contains(post._prior_id));
              assert(prior_post.contains(posteriors.at(ind - 1)._manager->get_state(state_id)._prior_id));
              assert(prior_postId.contains(post._prior_id));
              assert(prior_postId.contains(posteriors.at(ind - 1)._manager->get_state(state_id)._prior_id));
              prior_post.at(posteriors.at(ind - 1)._manager->get_state(state_id)._prior_id).push_back(post);
              prior_postId.at(posteriors.at(ind - 1)._manager->get_state(state_id)._prior_id).push_back(post._id);
            }
          }
        }
        std::vector<StateId> fused_state_ids;
        bool fused_state_is_empty = false;
        for (auto const& [prior_id, post] : prior_post)
        {
          assert([&] {
            if (not _manager->params()->glmb_distribution.update.joint_prediction_and_update)
            {
              if (post.size() != posteriors.size())
              {
                return false;
              }
            }
            return true;
          }());
          if (post.size() != posteriors.size())
          {
            LOG_WARN("Can only happen in joint update");
            LOG_WARN("track disappeared in one of the updates. I am not sure what theoretically must be done in this "
                     "case. "
                     "I decided to delete this this track, since this represent a situation which is invalid (Track "
                     "died "
                     "for one sensor while stayed alive for the other ");
            LOG_WARN("Prior: State Id: " << prior_id);
            auto const [first, second] = _manager->database().states.template get<PriorId>().equal_range(prior_id);
            for (auto it = first; it != second; ++it)
            {
              LOG_WARN("Post updated " << it->toString());
            }
            for (State<UDT> const& p : post)
            {
              LOG_WARN("post used in pm_fusion: " << p.toString());
            }
            LOG_WARN(_manager->get_state(prior_id).toString());
            continue;
          }

          const auto posterior_state_ids = prior_postId.find(prior_id);
          assert(posterior_state_ids != prior_postId.end() && "prior id is not given! This  should not be possible -> "
                                                              "Bug!");
          std::ranges::sort(posterior_state_ids->second);
          if (auto const fusedStateResultGiven =
                  posterior_tracks_2_fused_result_given.find(posterior_state_ids->second);
              fusedStateResultGiven != posterior_tracks_2_fused_result_given.end())
          {
            if (auto const fusedTrackIt = posterior_tracks_2_fused_track.find(posterior_state_ids->second);
                fusedTrackIt != posterior_tracks_2_fused_track.end())
            {
              if (fusedStateResultGiven->second)
              {
                // the fusion result for this state is already calculated!
                fused_state_ids.push_back(fusedTrackIt->second);
                if (is_pmlmb)
                {
                  fused_weight_log_pmlmb.value() += std::log(std::any_cast<UDT>(
                      _manager->database()
                          .states.find(fusedTrackIt->second)
                          ->_misc.at("eta")));  // see eq (11) and (15) in https://ieeexplore.ieee.org/document/10224121
                }
              }
              else
              {
                // fuse posteriors does not deliver fused state since no gaussian components with common prior component
                // id are given
                fused_state_is_empty = true;
              }
            }
          }
          else
          {
            // calculate fused state!
            std::optional<State<UDT>> fused =
                fuse_posteriors(_manager->get_state(prior_id),
                                post,
                                _manager->params()->glmb_distribution.multi_sensor.pm.max_num_state_components,
                                PM_MULTI_SENSOR_UPDATE_METHOD::GENERALIZED_K_BEST,
                                !is_pmlmb);
            if (fused.has_value())
            {
              posterior_tracks_2_fused_result_given.emplace(posterior_state_ids->second, true);
              posterior_tracks_2_fused_track.emplace(std::move(posterior_state_ids->second), fused.value()._id);
              fused_state_ids.push_back(fused.value()._id);
              if (is_pmlmb)
              {
                fused_weight_log_pmlmb.value() += std::log(std::any_cast<UDT>(fused.value()._misc.at(
                    "eta")));  // see eq (11) and (15) in https://ieeexplore.ieee.org/document/10224121
              }
              _manager->database().states.insert(std::move(fused.value()));
            }
            else
            {
              assert(is_pmlmb && "fused state is empty, which sould only possible in the case of using Mixtures caused "
                                 "by using an LMB Tracker");

              LOG_WARN("Fused mixture is empty due to pruning in posteriors. Is only ok for PM-LMB!");
              // This case happens when not all posteriors have a value for one prior component! (Fused mixture is empty
              // due to pruning in local updates) This happens in very complex scenarios like bearing only! Todo: Make
              // sure there is at least one common component! (Must be probably fixed in innovation part...) this fused
              // component does not exist which is worse....
              posterior_tracks_2_fused_result_given.emplace(std::move(posterior_state_ids->second), false);
              fused_state_is_empty = true;
            }
          }
        }
        if (fused_state_is_empty)
        {
          LOG_WARN("Fused hypotheses is not created, because one state is empty...");
          continue;
        }
        UDT weight_log_final = is_pmlmb ? fused_weight_log_pmlmb.value() : weight_log;
        Hypothesis fused_hyp{
          std::move(fused_state_ids), weight_log_final, prior_hyp_id.value(), prior_hyp_weight.value()
        };
        fused_hypotheses.add(std::move(fused_hyp));
      }
      if (fused_hypotheses._hyps.empty())
      {
        fused_hypotheses.add(Hypothesis<UDT>({}, UDT{ 0 }));
      }
      fused_hypotheses.normalize_hypotheses_weight();
      all_fused_hypotheses.push_back(std::move(fused_hypotheses));
    }
  }
  else
  {
    for (auto const& [hyp_ind, hypotheses] : std::views::enumerate(_hypotheses))
    {
      // create graphs for every glmb group
      Hypotheses<UDT> fused_hypotheses;
      std::unordered_map<HypothesisId, std::size_t> alloc;
      if (_manager->params()->glmb_distribution.multi_sensor.pm.method == PM_MULTI_SENSOR_UPDATE_METHOD::K_BEST_POISSON)
      {
        alloc = allocation(hypotheses);
      }
      std::vector<PMFusionInfos> fusion_infos_vec =
          create_graph_fuse_posteriors_KBest(*this,
                                             posteriors,
                                             hyp_ind,
                                             _manager->params()->glmb_distribution.multi_sensor.pm.num_components,
                                             _manager->params()->glmb_distribution.multi_sensor.pm.method,
                                             alloc);
      if (fusion_infos_vec.empty())
      {
        LOG_WARN("No Fusion result for glmb group with hyp_ind " << hyp_ind
                                                                 << " since no prior mixture component is given by all "
                                                                    "sensors! This is a case with very low "
                                                                    "possibility! Check if there is really no Bug!");
        break;
      }
      LOG_DEB("calculated " << fusion_infos_vec.size() << " graphs!");
      for (auto const& fusion_infos : fusion_infos_vec)
      {
        LOG_DEB("calculate " << fusion_infos._k << " solutions");
        if (fusion_infos._k == 0)
        {
          continue;
        }

        for (auto const& [edges, weight_log] :
             fusion_infos._graph.k_longest_paths(fusion_infos._startNode, fusion_infos._endNode, fusion_infos._k))
        {
          if (weight_log == -std::numeric_limits<double>::infinity())
          {
            LOG_DEB("no solution found");
            continue;
          }
          std::map<StateId, std::vector<State<UDT>>> prior_post;
          std::map<StateId, std::vector<StateId>> prior_postId;
          type_safe::deferred_construction<HypothesisId> prior_hyp_id;
          type_safe::deferred_construction<UDT> prior_hyp_weight;
          type_safe::deferred_construction<UDT> fused_weight_log_pmlmb;
          for (auto const& [ind, hyp] : edges)
          {
            if (ind == 0)  // prior
            {
              prior_hyp_id = hyp;
              assert(hypotheses._hyps.contains(hyp));
              prior_hyp_weight = hypotheses._hyps.find(hyp)->_weightLog;
              if (is_pmlmb)
              {
                fused_weight_log_pmlmb = prior_hyp_weight.value();
              }
              for (StateId state_id : hypotheses._hyps.find(hyp)->_tracks)
              {
                prior_post[state_id];
                prior_postId[state_id];
              }
            }
            else
            {
              assert([&] {  // NOLINT
                if (not posteriors.at(ind - 1)._hypotheses.at(hyp_ind)._hyps.contains(hyp))
                {
                  LOG_FATAL(hyp_ind);
                  LOG_FATAL(hypotheses.to_string());
                  LOG_FATAL("posterior does not contain hyp: " << hyp);
                  LOG_FATAL("post:" << posteriors.at(ind - 1)._hypotheses.at(hyp_ind).to_string());
                  return false;
                }
                return true;
              }());
              for (StateId state_id : posteriors.at(ind - 1)._hypotheses.at(hyp_ind)._hyps.find(hyp)->_tracks)
              {
                State<UDT> const& post = _manager->get_state(state_id);
                assert(prior_post.contains(post._prior_id));
                assert(prior_post.contains(posteriors.at(ind - 1)._manager->get_state(state_id)._prior_id));
                assert(prior_postId.contains(post._prior_id));
                assert(prior_postId.contains(posteriors.at(ind - 1)._manager->get_state(state_id)._prior_id));
                prior_post.at(posteriors.at(ind - 1)._manager->get_state(state_id)._prior_id).push_back(post);
                prior_postId.at(posteriors.at(ind - 1)._manager->get_state(state_id)._prior_id).push_back(post._id);
              }
            }
          }
          std::vector<StateId> fused_state_ids;
          bool fused_state_is_empty = false;
          for (auto const& [prior_id, post] : prior_post)
          {
            assert([&] {
              if (not _manager->params()->glmb_distribution.update.joint_prediction_and_update)
              {
                if (post.size() != posteriors.size())
                {
                  return false;
                }
              }
              return true;
            }());
            if (post.size() != posteriors.size())
            {
              LOG_WARN("Can only happen in joint update");
              LOG_WARN("track disappeared in one of the updates. I am not sure what theoretically must be done in this "
                       "case. "
                       "I decided to delete this this track, since this represent a situation which is invalid (Track "
                       "died "
                       "for one sensor while stayed alive for the other ");
              LOG_WARN("Prior: State Id: " << prior_id);
              auto const [first, second] = _manager->database().states.template get<PriorId>().equal_range(prior_id);
              for (auto it = first; it != second; ++it)
              {
                LOG_WARN("Post updated " << it->toString());
              }
              for (State<UDT> const& p : post)
              {
                LOG_WARN("post used in pm_fusion: " << p.toString());
              }
              LOG_WARN(_manager->get_state(prior_id).toString());
              continue;
            }

            const auto posterior_state_ids = prior_postId.find(prior_id);
            assert(posterior_state_ids != prior_postId.end() && "prior id is not given! This  should not be possible "
                                                                "-> Bug!");
            std::ranges::sort(posterior_state_ids->second);
            if (auto const fusedStateResultGiven =
                    posterior_tracks_2_fused_result_given.find(posterior_state_ids->second);
                fusedStateResultGiven != posterior_tracks_2_fused_result_given.end())
            {
              if (auto const fusedTrackIt = posterior_tracks_2_fused_track.find(posterior_state_ids->second);
                  fusedTrackIt != posterior_tracks_2_fused_track.end())
              {
                if (fusedStateResultGiven->second)
                {
                  // the fusion result for this combination is already calculated!
                  fused_state_ids.push_back(fusedTrackIt->second);
                  if (is_pmlmb)
                  {
                    fused_weight_log_pmlmb.value() += std::log(
                        std::any_cast<double>(_manager->database()
                                                  .states.find(fusedTrackIt->second)
                                                  ->_misc.at("eta")));  // see eq (11) and (15) in
                                                                        // https://ieeexplore.ieee.org/document/10224121
                  }
                }
                else
                {
                  // fuse posteriors does not deliver fused state since no gaussian components with common prior
                  // component id are given
                  fused_state_is_empty = true;
                }
              }
            }
            else
            {
              // calculate fused state!
              std::optional<State<UDT>> fused = fuse_posteriors(
                  _manager->get_state(prior_id), post, 1, PM_MULTI_SENSOR_UPDATE_METHOD::GENERALIZED_K_BEST, !is_pmlmb);
              if (fused.has_value())
              {
                posterior_tracks_2_fused_result_given.emplace(posterior_state_ids->second, true);
                posterior_tracks_2_fused_track.emplace(std::move(posterior_state_ids->second), fused.value()._id);
                fused_state_ids.push_back(fused.value()._id);
                if (is_pmlmb)
                {
                  fused_weight_log_pmlmb.value() += std::log(std::any_cast<double>(fused.value()._misc.at(
                      "eta")));  // see eq (11) and (15) in https://ieeexplore.ieee.org/document/10224121
                }
                _manager->database().states.insert(std::move(fused.value()));
              }
              else
              {
                assert(is_pmlmb && "fused state is empty, which sould only possible in the case of using Mixtures "
                                   "caused by using an LMB Tracker");
                LOG_WARN("Fused mixture is empty due to pruning in posteriors. Is only ok for PM-LMB!");
                // This case happens when not all posteriors have a value for one prior component! (Fused mixture is
                // empty due to pruning in local updates) This happens in very complex scenarios like bearing only!
                // Todo: Make sure there is at least one common component! (Must be probably fixed in innovation
                // part...) this fused component does not exist which is worse....
                posterior_tracks_2_fused_result_given.emplace(std::move(posterior_state_ids->second), false);
                fused_state_is_empty = true;
              }
            }
          }
          if (fused_state_is_empty)
          {
            LOG_WARN("Fused hypotheses is not created, because one state is empty...");
            continue;
          }
          Hypothesis fused_hyp{ std::move(fused_state_ids),
                                [&] {
                                  if (is_pmlmb)
                                  {
                                    return fused_weight_log_pmlmb.value();
                                  }
                                  return UDT{ weight_log };
                                }(),
                                prior_hyp_id.value(),
                                prior_hyp_weight.value() };
          fused_hypotheses.add(std::move(fused_hyp));
        }
        if (fused_hypotheses._hyps.empty())
        {
          fused_hypotheses.add(Hypothesis<UDT>({}, UDT{ 0 }));
        }
      }
      fused_hypotheses.normalize_hypotheses_weight();
      all_fused_hypotheses.push_back(std::move(fused_hypotheses));
    }
  }
  assert(not all_fused_hypotheses.empty());
  _hypotheses = std::move(all_fused_hypotheses);
  Params next_params = *_manager->params();
  next_params.glmb_distribution.post_process_update.diss_herrmann_pp.pruning_max_comps_retain =
      _manager->params()->glmb_distribution.multi_sensor.pm.post_processing.diss_herrmann_pp.pruning_max_comps_retain;
  next_params.glmb_distribution.post_process_update.diss_herrmann_pp.pruning_min_comps_per_card =
      _manager->params()->glmb_distribution.multi_sensor.pm.post_processing.diss_herrmann_pp.pruning_min_comps_per_card;
  next_params.glmb_distribution.post_process_update.diss_herrmann_pp.pruning_min_comps_per_card_thresh =
      _manager->params()
          ->glmb_distribution.multi_sensor.pm.post_processing.diss_herrmann_pp.pruning_min_comps_per_card_thresh;
  next_params.glmb_distribution.post_process_update.diss_herrmann_pp.pruning_max_weight_ratio =
      _manager->params()->glmb_distribution.multi_sensor.pm.post_processing.diss_herrmann_pp.pruning_max_weight_ratio;
  next_params.glmb_distribution.post_process_update.diss_herrmann_pp.pruning_alloc_min_comps_per_card_by_poisson =
      _manager->params()
          ->glmb_distribution.multi_sensor.pm.post_processing.diss_herrmann_pp
          .pruning_alloc_min_comps_per_card_by_poisson;
  _manager->update_params(next_params);
  post_process_update();
  next_params.glmb_distribution.post_process_update.diss_herrmann_pp.pruning_max_comps_retain =
      _manager->original_params()
          .glmb_distribution.multi_sensor.pm.post_processing.diss_herrmann_pp.pruning_max_comps_retain;
  next_params.glmb_distribution.post_process_update.diss_herrmann_pp.pruning_min_comps_per_card =
      _manager->original_params()
          .glmb_distribution.multi_sensor.pm.post_processing.diss_herrmann_pp.pruning_min_comps_per_card;
  next_params.glmb_distribution.post_process_update.diss_herrmann_pp.pruning_min_comps_per_card_thresh =
      _manager->original_params()
          .glmb_distribution.multi_sensor.pm.post_processing.diss_herrmann_pp.pruning_min_comps_per_card_thresh;
  next_params.glmb_distribution.post_process_update.diss_herrmann_pp.pruning_max_weight_ratio =
      _manager->original_params()
          .glmb_distribution.multi_sensor.pm.post_processing.diss_herrmann_pp.pruning_max_weight_ratio;
  next_params.glmb_distribution.post_process_update.diss_herrmann_pp.pruning_alloc_min_comps_per_card_by_poisson =
      _manager->original_params()
          .glmb_distribution.multi_sensor.pm.post_processing.diss_herrmann_pp
          .pruning_alloc_min_comps_per_card_by_poisson;
  _manager->update_params(next_params);
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution after pm_fusion.");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  LOG_DEB("pm fusion doene");
}

template <class UDT>
std::vector<BernoulliState<UDT>> GLMBDistribution<UDT>::get_estimate() const
{
  LOG_DEB("Get Estimation");
  Vector<UDT> const& cardDist = cardinality_distribution();
  if (cardDist.rows() == 0)
  {
    return {};
  }
  switch (_manager->params()->glmb_distribution.extraction.type)
  {
    case MO_DISTRIBUTION_EXTRACTION_TYPE::EXISTENCE_PROBABILITY:
    {
      std::vector<BernoulliState<UDT>> estimated_tracks;
      // for each label with existence prob > existence threshold find the track with the highest existence
      // probability
      std::unordered_map<Label, std::vector<StateId>> const& label_to_tracks = label_2_tracks();
      for (auto const& [label, exProb] : label_2_ex_prob())
      {
        if (exProb > _manager->params()->glmb_distribution.extraction.threshold)
        {
          // find best track of label
          std::vector<StateId> const& tracks = label_to_tracks.at(label);
          auto bestTrack = std::ranges::max_element(
              tracks, [&](StateId a, StateId b) { return track_2_ex_prob().at(a) < track_2_ex_prob().at(b); });
          estimated_tracks.emplace_back(_manager->get_state(*bestTrack), exProb);
        }
      }
      return estimated_tracks;
    }
    case MO_DISTRIBUTION_EXTRACTION_TYPE::CARDINALITY:
    {
      std::vector<BernoulliState<UDT>> estimated_tracks;
      assert(std::ranges::max_element(cardDist) != cardDist.end());
      std::size_t const maxCard = std::distance(cardDist.begin(), std::ranges::max_element(cardDist));
      // copy values for sorting in descending order
      std::vector<std::pair<Label, UDT>> label_probs;
      for (auto const& it : label_2_ex_prob())
      {
        label_probs.emplace_back(it);
      }
      std::ranges::sort(label_probs, std::greater{}, &std::pair<Label, UDT>::second);
      std::unordered_map<Label, std::vector<StateId>> const& label_to_tracks = label_2_tracks();
      std::unordered_map<StateId, UDT> const& track_to_ex_prob = track_2_ex_prob();
      std::unordered_map<Label, UDT> const& label_to_ex_prob = label_2_ex_prob();
      for (auto const& [label, ex_prob] : label_probs | std::views::take(maxCard))
      {
        // find best track of label
        assert(label_to_tracks.contains(label));
        if (auto const& tracks = label_to_tracks.at(label); not tracks.empty())
        {
          auto bestTrack = std::ranges::max_element(tracks, [&](StateId a, StateId b) {
            assert(track_to_ex_prob.contains(a));
            assert(track_to_ex_prob.contains(b));
            return track_to_ex_prob.at(a) < track_to_ex_prob.at(b);
          });
          State<UDT> out = _manager->get_state(*bestTrack);
          assert(label_to_ex_prob.contains(out._label));
          assert(not out.isEmpty());
          estimated_tracks.emplace_back(std::move(out), label_to_ex_prob.at(out._label));
        }
      }

      return estimated_tracks;
    }
    case MO_DISTRIBUTION_EXTRACTION_TYPE::BEST_HYPOTHESIS:
    {
      std::vector<BernoulliState<UDT>> estimated_tracks;
      std::unordered_map<Label, UDT> const& label_to_ex_prob = label_2_ex_prob();
      for (Hypotheses<UDT> const& hypotheses : _hypotheses)
      {
        if (auto bestHyp = std::ranges::max_element(
                hypotheses._hyps,
                [&](Hypothesis<UDT> const& a, Hypothesis<UDT> const& b) { return a._weightLog < b._weightLog; });
            bestHyp != hypotheses._hyps.end())
        {
          for (StateId track_id : bestHyp->_tracks)
          {
            if (_manager->get_state(track_id)._type != STATE_TYPE::BIRTH)
            {
              estimated_tracks.emplace_back(_manager->get_state(track_id),
                                            label_to_ex_prob.at(_manager->get_state(track_id)._label));
            }
          }
        }
      }
      return estimated_tracks;
    }
  }
  assert(false);
  std::unreachable();
}

template <class UDT>
std::vector<StateId> GLMBDistribution<UDT>::all_tracks() const
{
  std::vector<StateId> tracks;
  for (Hypotheses<UDT> const& hypotheses : _hypotheses)
  {
    for (StateId const& id : hypotheses.all_tracks())
    {
      tracks.push_back(id);
    }
  }
  std::ranges::sort(tracks);
  return tracks;
}

template <class UDT>
std::map<MeasurementId, UDT> const&
GLMBDistribution<UDT>::meas_assignment_prob(DataSourceContainer<UDT> const& measContainer) const
{
  LOG_DEB("GLMBDistribution::meas_assignment_prob");
  assert(is_valid());
  if (auto const it = _meas_assignment_prob.find(measContainer._id); it != _meas_assignment_prob.end())
  {
    return it->second;
  }
  std::map<MeasurementId, UDT> rzMap;
  for (State<UDT> const& meas : measContainer._measurements)
  {
    rzMap.emplace(meas._id, 0);
  }
  for (Hypotheses<UDT> const& hypotheses : _hypotheses)
  {
    for (Hypothesis<UDT> const& hyp : hypotheses._hyps)
    {
      for (StateId updated_track_id : hyp._tracks)
      {
        State<UDT> const& track = _manager->get_state(updated_track_id);
        assert(track._meta_data._meas_update_weight.contains(measContainer._source_id));
        if (auto const meas_it = track._meta_data._meas_update_weight.find(measContainer._source_id);
            meas_it != track._meta_data._meas_update_weight.end() and meas_it->second.front().first != NOT_DETECTED)
        {
          assert(meas_it->second.size() == 1);
          assert([&] {  // NOLINT
            if (not rzMap.contains(meas_it->second.front().first))
            {
              LOG_FATAL(track.toString());
              LOG_FATAL(measContainer.toString());
              LOG_FATAL(std::stacktrace::current());
              return false;
            }
            return true;
          }());
          rzMap.at(meas_it->second.front().first) += hyp.getWeight();
        }
      }
    }
  }
  auto const [it, inserted] = _meas_assignment_prob.emplace(measContainer._id, std::move(rzMap));
  assert(inserted);
  LOG_DEB("computation done new GLMBDistribution::meas_assignment_prob");
  return it->second;
}

template <class UDT>
bool GLMBDistribution<UDT>::is_valid() const
{
  if (bool const tracks_valid =
          std::ranges::all_of(all_tracks(),
                              [&](StateId id) {
                                auto const& track = _manager->try_get_state(id);
                                if (not track.has_value())
                                {
                                  return true;
                                }
                                if (not track.value().get().isValid())
                                {
                                  LOG_FATAL("Track: " + track.value().get().toString() + " not valid");
                                  return false;
                                }
                                return true;
                              });
      not tracks_valid)
  {
    LOG_FATAL("Tracks are not valid");
    LOG_FATAL(std::stacktrace::current());
    return false;
  }
  for (StateId id : all_tracks())
  {
    if (not _manager->try_get_state(id).has_value())
    {
      LOG_FATAL("State with id " << id << "is not available");
      return false;
    }
    State<UDT> const& track = _manager->get_state(id);
    for (auto const& weights : track._meta_data._meas_update_weight | std::views::values)
    {
      if (not weights.empty() and (weights.size() != 1 or weights.front().second != 1))
      {
        LOG_FATAL("GLMB Track is updated with more than one measurement");
        LOG_FATAL(track.toString());
        LOG_FATAL(std::stacktrace::current());
        return false;
      }
    }
  }
  if (_hypotheses.empty())
  {
    LOG_FATAL("No Hypotheses after the update");
    return false;
  }
  for (Hypotheses<UDT> const& hypotheses : _hypotheses)
  {
    if (double const hyp_sum = to_val(hypotheses.sum_hypotheses_weight());
        std::abs(hyp_sum - 1) > 1e-5 or not std::isfinite(hyp_sum))
    {
      LOG_FATAL("Hyp sum not == 1: " << hyp_sum);
      return false;
    }
  }
  for (StateId state_id : all_tracks())
  {
    bool found = false;
    for (Hypotheses<UDT> const& hypotheses : _hypotheses)
    {
      for (Hypothesis<UDT> const& hyp : hypotheses._hyps)
      {
        if (std::ranges::find(hyp._tracks, state_id) != hyp._tracks.end())
        {
          found = true;
          break;
        }
      }
    }
    if (not found)
    {
      LOG_FATAL("State: " << state_id << " is not in any hypothesis");
      return false;
    }
  }
  std::vector<std::set<StateId>> hypotheses_states(_hypotheses.size());
  std::vector<std::set<Label>> hypotheses_labels(_hypotheses.size());
  for (auto const& [ind, hypotheses] : std::views::enumerate(_hypotheses))
  {
    std::vector<StateId> all_tracks = hypotheses.all_tracks();
    hypotheses_states.at(ind).insert(all_tracks.begin(), all_tracks.end());
    for (StateId state_id : all_tracks)
    {
      hypotheses_labels.at(ind).insert(_manager->get_state(state_id)._label);
    }
    for (Hypothesis<UDT> const& hyp : hypotheses._hyps)
    {
      if (not std::ranges::is_sorted(hyp._tracks))
      {
        LOG_FATAL("Tracks of hypothesis are not sorted");
        LOG_FATAL(hyp.toString());
        return false;
      }
      if (not std::isfinite(hyp.getWeight()))
      {
        LOG_FATAL("Hypothesis illegal");
        LOG_FATAL(hyp.toString());
        return false;
      }
      if (hyp.getWeight() < 0 or hyp.getWeight() > 1)
      {
        LOG_FATAL("Hypothesis illegal");
        LOG_FATAL(hyp.toString());
        return false;
      }
    }
    if (hypotheses._hyps.empty())
    {
      LOG_FATAL("Empty hypotheses");
      return false;
    }
    if (std::abs(hypotheses.sum_hypotheses_weight() - 1) > 1e-7)
    {
      LOG_FATAL("Hypotheses weight is not normalized");
      return false;
    }
  }
  for (std::size_t ind_1 = 0; ind_1 < hypotheses_states.size(); ++ind_1)
  {
    for (std::size_t ind_2 = 0; ind_2 < hypotheses_states.size(); ++ind_2)
    {
      if (ind_1 == ind_2)
      {
        continue;
      }
      std::vector<StateId> intersection;
      std::ranges::set_intersection(hypotheses_states.at(ind_1), hypotheses_states.at(ind_2), intersection.end());
      if (not intersection.empty())
      {
        LOG_FATAL("Same track id in different hypotheses");
      }
      std::vector<Label> label_intersection;
      std::ranges::set_intersection(hypotheses_labels.at(ind_1), hypotheses_labels.at(ind_2), label_intersection.end());
      if (not label_intersection.empty())
      {
        LOG_FATAL("Same Label in different hypotheses");
      }
    }
  }
  return true;
}

template <class UDT>
std::string GLMBDistribution<UDT>::to_string(std::string_view prefix, bool print_tracks, bool print_hyps) const
{
  std::string out = std::format("{} GLMB Distribution\n"
                                "{}|\tId: {}\n"
                                "{}|\t#Hypotheses Groups: {}\n"
                                "{}|\t#All Tracks: {}\n",
                                prefix,
                                prefix,
                                _id,
                                prefix,
                                _hypotheses.size(),
                                prefix,
                                all_tracks().size());
  if (print_hyps)
  {
    for (Hypotheses<UDT> const& hypotheses : _hypotheses)
    {
      out += std::string(prefix) + "|\tNEW GROUP ----------------\n";
      for (Hypothesis<UDT> const& hyp : hypotheses._hyps)
      {
        out += hyp.toString(std::string(prefix) + "|\t");
      }
    }
  }
  if (print_tracks)
  {
    for (StateId state_id : all_tracks())
    {
      out += _manager->get_state(state_id).toString(std::string(prefix) + "|\t");
    }
  }
  return out;
}

template <class UDT>
void GLMBDistribution<UDT>::reset_caches() const
{
  _label2exProb.reset();
  _trackId2exProb.reset();
  _cacheCardDist.reset();
  _cacheCardDistLog.reset();
  _numHypsPerCard.reset();
  _label2Tracks.reset();
  _label2Hypotheses.reset();
  _kBestCache.reset();
}

template <class UDT>
std::pair<double, std::size_t> GLMBDistribution<UDT>::prune_hypotheses_weight_threshold(double threshold)
{
  reset_caches();
  std::size_t tot_size = 0;
  std::size_t num_erased = 0;
  double wsum = 0;
  double psum = 0;
  for (Hypotheses<UDT>& hypotheses : _hypotheses)
  {
    wsum += to_val(hypotheses.sum_hypotheses_weight());
    tot_size += hypotheses._hyps.size();
    auto const [p, n] = hypotheses.prune_hypotheses_weight_threshold(threshold);
    psum += p;
    num_erased += n;
  }
  if (num_erased / static_cast<double>(tot_size) > 0.9)
  {
    LOG_INF(std::format("pruned over 90% of all hypotheses. Threshold: {}", threshold));
    LOG_INF("Hypotheses before: " << tot_size);
    LOG_INF("Hypotheses now: " << to_string());
  }
  return { psum / wsum, num_erased };
}

template <class UDT>
std::pair<double, std::size_t> GLMBDistribution<UDT>::prune_labels_existence_probability_threshold(double threshold)
{
  // ZoneScopedN("GLMBDistribution::prune_threshold");
  reset_caches();
  std::size_t num_erased = 0;
  double wsum = 0;
  double psum = 0;
  for (Hypotheses<UDT>& hypotheses : _hypotheses)
  {
    wsum += to_val(hypotheses.sum_hypotheses_weight());
    auto const [p, n] = hypotheses.prune_labels_existence_threshold(*_manager, threshold);
    psum += p;
    num_erased += n;
  }
  return { psum / wsum, num_erased };
}

template <class UDT>
std::pair<double, std::size_t> GLMBDistribution<UDT>::prune_max_hypotheses(std::size_t max_hypotheses)
{
  reset_caches();
  double wsum = 0;
  double psum = 0;
  std::size_t num_pruned = 0;
  for (Hypotheses<UDT>& hypotheses : _hypotheses)
  {
    wsum += to_val(hypotheses.sum_hypotheses_weight());
    auto const [p, n] = hypotheses.prune_max_hypotheses(max_hypotheses);
    psum += p;
    num_pruned += n;
  }
  return { psum / wsum, num_pruned };
}

template <class UDT>
std::size_t GLMBDistribution<UDT>::post_process_prediction()
{
  // ZoneScopedN("GLMBDistribution::post_process_prediction");
  LOG_DEB("GLMBDistribution::postProcessPrediction");
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution before postProcessPrediction.");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  if (not _manager->params()->glmb_distribution.post_process_prediction.enable)
  {
    return 0;
  }
  // auto const [w0, n0] = prune_labels_existence_probability_threshold(
  //     _manager->params().glmb_distribution.post_process_prediction.prune_labels_existence_probability_threshold);
  auto const [w0, n0] = std::pair{ 0, 0 };
  auto const [w1, n1] = prune_hypotheses_weight_threshold(
      _manager->params()->glmb_distribution.post_process_prediction.pruning_threshold);

  LOG_DEB("#pruned threshold: " << n1);
  auto const [w2, n2] =
      prune_max_hypotheses(_manager->params()->glmb_distribution.post_process_prediction.max_hypotheses);
  double const b = std::max(0.0, 1 - w0 - w1 - w2);
  Opinion op{ Opinion::BeliefType{ std::array{ b, 1 - b } }, Opinion::BeliefType{ std::array{ 0.5, 0.5 } } };
  auto [it, _] = _manager->assessment().implementation_opinions.wlock().data.emplace("glmb_post_process_prediction",
                                                                                     SlidingWindow<Opinion>(1s));
  it->second.insert(_manager->filter_time(), std::move(op));
  std::vector<Hypotheses<UDT>> processed_hyps;
  for (Hypotheses<UDT>& hypotheses : _hypotheses)
  {
    if (hypotheses.sum_hypotheses_weight() > 0)
    {
      hypotheses.normalize_hypotheses_weight();
      processed_hyps.push_back(std::move(hypotheses));
    }
  }
  _hypotheses = std::move(processed_hyps);
  reset_caches();
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution after postProcessPrediction.");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  return n0 + n1 + n2;
}

template <class UDT>
std::size_t GLMBDistribution<UDT>::post_process_update_standard()
{
  auto const [w0, n0] = std::pair{ 0, 0 };

  auto const [w1, n1] = prune_hypotheses_weight_threshold(
      _manager->params()->glmb_distribution.post_process_update.standard_pp.pruning_threshold);

  auto const [w2, n2] =
      prune_max_hypotheses(_manager->params()->glmb_distribution.post_process_update.standard_pp.max_hypotheses);

  if (_manager->params()->glmb_distribution.post_process_update.standard_pp.limit_states_per_label)
  {
    for (Hypotheses<UDT>& hypotheses : _hypotheses)
    {
      hypotheses.limit_states_per_label(
          *_manager, _manager->params()->glmb_distribution.post_process_update.standard_pp.max_states_per_label);
    }
  }

  double const b = std::max(1.0, 1 - w0 - w1 - w2);
  Opinion op{ Opinion::BeliefType{ std::array{ b, 1 - b } }, Opinion::BeliefType{ std::array{ 0.5, 0.5 } } };

  auto [it, inserted] = _manager->assessment().implementation_opinions.wlock().data.emplace("glmb_post_process_update",
                                                                                            SlidingWindow<Opinion>(1s));
  it->second.insert(_manager->filter_time(), std::move(op));
  std::vector<Hypotheses<UDT>> processed_hyps;
  for (Hypotheses<UDT>& hypotheses : _hypotheses)
  {
    if (hypotheses.sum_hypotheses_weight() > 0)
    {
      hypotheses.normalize_hypotheses_weight();
      processed_hyps.push_back(std::move(hypotheses));
    }
  }
  _hypotheses = std::move(processed_hyps);
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution after postProcessUpdate.");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  reset_caches();
  LOG_DEB("GLMBDistribution::postProcessUpdate done");
  return n0 + n1 + n2;
}

template <class UDT>
std::size_t GLMBDistribution<UDT>::post_process_update_diss_herrmann()
{
  reset_caches();
  std::size_t tot_size = 0;
  std::size_t num_erased = 0;
  double wsum = 0;
  double psum = 0;

  std::size_t pruning_max_comps_retain =
      _manager->params()->glmb_distribution.post_process_update.diss_herrmann_pp.pruning_max_comps_retain;
  std::size_t pruning_min_comps_per_card =
      _manager->params()->glmb_distribution.post_process_update.diss_herrmann_pp.pruning_min_comps_per_card;
  double pruning_min_comps_per_card_thresh =
      _manager->params()->glmb_distribution.post_process_update.diss_herrmann_pp.pruning_min_comps_per_card_thresh;
  double pruning_max_weight_ratio =
      _manager->params()->glmb_distribution.post_process_update.diss_herrmann_pp.pruning_max_weight_ratio;
  bool pruning_alloc_min_comps_per_card_by_poisson =
      _manager->params()
          ->glmb_distribution.post_process_update.diss_herrmann_pp.pruning_alloc_min_comps_per_card_by_poisson;

  for (Hypotheses<UDT>& hypotheses : _hypotheses)
  {
    wsum += to_val(hypotheses.sum_hypotheses_weight());
    tot_size += hypotheses._hyps.size();
    auto const [p, n] = hypotheses.prune_hypotheses_diss_herrmann(pruning_max_comps_retain,
                                                                  pruning_min_comps_per_card,
                                                                  pruning_min_comps_per_card_thresh,
                                                                  pruning_max_weight_ratio,
                                                                  pruning_alloc_min_comps_per_card_by_poisson);
    psum += p;
    num_erased += n;
  }

  double w0 = psum / wsum;
  double const b = std::max(1.0, 1 - w0);
  Opinion op{ Opinion::BeliefType{ std::array{ b, 1 - b } }, Opinion::BeliefType{ std::array{ 0.5, 0.5 } } };

  auto [it, inserted] = _manager->assessment().implementation_opinions.wlock().data.emplace("glmb_post_process_update",
                                                                                            SlidingWindow<Opinion>(1s));
  it->second.insert(_manager->filter_time(), std::move(op));
  std::vector<Hypotheses<UDT>> processed_hyps;
  for (Hypotheses<UDT>& hypotheses : _hypotheses)
  {
    if (hypotheses.sum_hypotheses_weight() > 0)
    {
      hypotheses.normalize_hypotheses_weight();
      processed_hyps.push_back(std::move(hypotheses));
    }
  }
  _hypotheses = std::move(processed_hyps);
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution after postProcessUpdate.");
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  reset_caches();
  LOG_DEB("GLMBDistribution::postProcessUpdate done");
  return num_erased;
}

template <class UDT>
std::size_t GLMBDistribution<UDT>::post_process_update()
{
  LOG_DEB("GLMBDistribution::postProcessUpdate");
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL("Invalid GLMB distribution before postProcessUpdate.");
      LOG_FATAL(to_string("", true, true));
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    return true;
  }());
  if (not _manager->params()->glmb_distribution.post_process_update.enable)
  {
    return 0;
  }
  switch (_manager->params()->glmb_distribution.post_process_update.method)
  {
    case GLMB_POST_PROCESSING_METHOD::DISSERTATION_HERRMANN:
      return post_process_update_diss_herrmann();
      break;
    case GLMB_POST_PROCESSING_METHOD::STANDARD:
      return post_process_update_standard();
      break;
    default:
      LOG_FATAL("post process method not known! Use standard post processing");
      return post_process_update_standard();
  }
}

}  // namespace ttb
