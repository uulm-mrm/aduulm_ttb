#include "tracking_lib/MultiObjectStateDistributions/Hypothesis.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Misc/Numeric.h"
#include "tracking_lib/Misc/ProportionalAllocation.h"

#include <set>

namespace ttb
{

template class Hypothesis<double>;
template class Hypothesis<AD>;

template <class UDT>
IDGenerator<HypothesisId> Hypothesis<UDT>::_idGenerator{};

template <class UDT>
Hypothesis<UDT>::Hypothesis(std::vector<StateId> tracks, UDT weight_log, HypothesisId origin, UDT origin_weight_log)
  : _prior_id{ origin }, _weightLog{ weight_log }, _prior_weight_log{ origin_weight_log }, _tracks{ std::move(tracks) }
{
  std::ranges::sort(_tracks);
}

template <class UDT>
std::string Hypothesis<UDT>::toString(std::string const& prefix) const
{
  std::string out = prefix + "Hypothesis\n" + prefix + "|\tId " + std::to_string(_id.value_) + "|\tOriginId " +
                    std::to_string(_prior_id.value_) + "\n" + prefix + "|\t#Tracks: " + std::to_string(_tracks.size()) +
                    "\n" + prefix + "|\tWeight " + std::to_string(getWeight()) + "\n" + prefix + "|\tPrior Weight " +
                    std::to_string(std::exp(_prior_weight_log)) + "\n" + prefix + "|\tTracks: ";
  for (StateId id : _tracks)
  {
    out += std::to_string(id.value_) + " ";
  }
  out += '\n';
  for (auto const& [source, val] : _point_weight)
  {
    out += std::format("{}|\t{}: Point Weight {}\n", prefix, source, val);
  }
  for (auto const& [source, val] : _soft_clutter_weight)
  {
    out += std::format("{}|\t{}: Soft Clutter {}\n", prefix, source, val);
  }
  for (auto const& [source, val] : _soft_detection_weight)
  {
    out += std::format("{}|\t{}: Soft detection {}\n", prefix, source, val);
  }
  for (auto const& [source, val] : _num_clutter)
  {
    out += std::format("{}|\t{}: #Clutter {}\n", prefix, source, val);
  }
  for (auto const& [source, val] : _num_detected)
  {
    out += std::format("{}|\t{}: #Detections {}\n", prefix, source, val);
  }
  return out;
}

template <class UDT>
UDT Hypothesis<UDT>::getWeight() const
{
  return std::exp(_weightLog);
}

template <class UDT>
std::size_t Hypotheses<UDT>::max_tracks() const
{
  if (_hyps.empty())
  {
    return 0;
  }
  return std::ranges::max(_hyps, {}, [](Hypothesis<UDT> const& hyp) { return hyp._tracks.size(); })._tracks.size();
}


template class Hypotheses<double>;
template class Hypotheses<AD>;

template <class UDT>
std::vector<StateId> Hypotheses<UDT>::all_tracks() const
{
  std::vector<StateId> tracks;
  for (Hypothesis<UDT> const& hyp : _hyps)
  {
    for (StateId id : hyp._tracks)
    {
      tracks.push_back(id);
    }
  }
  std::ranges::sort(tracks);
  auto const [first, last] = std::ranges::unique(tracks);
  tracks.erase(first, last);
  return tracks;
}

template <class UDT>
std::unordered_map<Label, std::vector<StateId>> Hypotheses<UDT>::label_2_tracks(TTBManager<UDT> const& manager) const
{
  std::unordered_map<Label, std::vector<StateId>> l2t;
  for (StateId id : all_tracks())
  {
    l2t[manager.get_state(id)._label].push_back(id);
  }
  return l2t;
}

template <class UDT>
void Hypotheses<UDT>::add(Hypothesis<UDT> hyp)
{
  assert([&] {
    if (not std::ranges::is_sorted(hyp._tracks))
    {
      LOG_FATAL("Hypothesis Tracks not sorted");
      LOG_FATAL(hyp.toString());
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    return true;
  }());
  if (auto const it = _hyps.template get<StateIds>().find(hyp._tracks); it != _hyps.template get<StateIds>().end())
  {
    Hypothesis update = *it;
    update._weightLog = std::log(std::exp(update._weightLog) + std::exp(hyp._weightLog));
    _hyps.template get<StateIds>().replace(it, std::move(update));
  }
  else
  {
    _hyps.insert(std::move(hyp));
  }
}

template <class UDT>
UDT Hypotheses<UDT>::sum_hypotheses_weight() const
{
  UDT const log_weight = numeric::logsumexp(_hyps, [](Hypothesis<UDT> const& hyp) { return hyp._weightLog; });
  return std::exp(log_weight);
}

template <class UDT>
bool Hypotheses<UDT>::is_valid() const
{
  if (std::abs(sum_hypotheses_weight() - 1.0) > 1e-7)
  {
    LOG_FATAL(std::format("Hypotheses are not normalized. Weight {}. Hyps {}", sum_hypotheses_weight(), to_string()));
    return false;
  }
  return true;
}

template <class UDT>
void Hypotheses<UDT>::multiply_hypotheses_weight(UDT fac)
{
  UDT const log_fac = std::log(fac);
  for (auto it = _hyps.begin(); it != _hyps.end(); ++it)
  {
    _hyps.modify(it, [&](Hypothesis<UDT>& hyp) { hyp._weightLog += log_fac; });
  }
}

template <class UDT>
void Hypotheses<UDT>::limit_states_per_label(TTBManager<UDT> const& manager, std::size_t max_states)
{
  std::unordered_map<StateId, UDT> const& track_ex_prob = track_existence_prob();
  std::unordered_map<Label, std::vector<StateId>> best_tracks_per_label = label_2_tracks(manager);
  std::size_t const overall_max_states = max_states * best_tracks_per_label.size();
  for (std::vector<StateId>& tracks : best_tracks_per_label | std::views::values)
  {
    std::ranges::sort(tracks, std::greater{}, [&](StateId id) { return track_ex_prob.at(id); });
    tracks.erase(std::next(tracks.begin(), std::min(overall_max_states, tracks.size())), tracks.end());
  }
  Hypotheses new_hyps;
  for (Hypothesis<UDT> const& hyp : _hyps)
  {
    Hypothesis filtered = hyp;
    filtered._tracks.clear();
    for (StateId track : hyp._tracks)
    {
      std::vector<StateId> const& best_tracks = best_tracks_per_label.at(manager.get_state(track)._label);
      if (std::ranges::contains(best_tracks, track))
      {
        filtered._tracks.push_back(track);
      }
      else
      {
        filtered._tracks.push_back(best_tracks.front());
      }
    }
    std::ranges::sort(filtered._tracks);
    new_hyps.add(std::move(filtered));
  }
  *this = std::move(new_hyps);
}

template <class UDT>
Vector<UDT> Hypotheses<UDT>::cardinality_distribution() const
{
  Vector<UDT> cardDist = Vector<UDT>::Zero(static_cast<Index>(1 + max_tracks()));
  for (Hypothesis<UDT> const& hyp : _hyps)
  {
    cardDist(std::ssize(hyp._tracks)) += hyp.getWeight();
  }
  return cardDist;
}

template <class UDT>
std::unordered_map<Label, UDT> Hypotheses<UDT>::label_existence_prob(TTBManager<UDT> const& manager) const
{
  assert(is_valid());
  std::unordered_map<Label, UDT> out;
  for (auto const& [trackId, prob] : track_existence_prob())
  {
    assert(std::isfinite(prob));
    out[manager.get_state(trackId)._label] += prob;
  }
  for (auto& [label, prob] : out)
  {
    if (prob > 1 + 1e-7)
    {
      LOG_WARN("Invalid Probability " + std::to_string(prob) + " for Label " + std::to_string(label.value_) +
               ". Closely monitor. Clamping for now....");
    }
    prob = std::min(prob, 1.0);
  }
  return out;
}

template <class UDT>
std::unordered_map<StateId, UDT> Hypotheses<UDT>::track_existence_prob() const
{
  std::unordered_map<StateId, std::vector<UDT>> log_weights;
  {
    for (Hypothesis<UDT> const& hyp : _hyps)
    {
      for (StateId const track_id : hyp._tracks)
      {
        assert([&] {  // NOLINT
          if (not std::isfinite(hyp._weightLog) and hyp._weightLog != -std::numeric_limits<double>::infinity())
          {
            LOG_FATAL(hyp._weightLog);
            LOG_FATAL(to_string());
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
    UDT prob = std::exp(numeric::logsumexp(log_w));
    if (prob > 1 + 1e-7)
    {
      LOG_WARN("Invalid Probability " + std::to_string(prob) + " for Track " + std::to_string(track.value_) +
               ". Closely monitor. Clamping for now....");
    }
    prob = std::min(prob, 1.0);
    assert([&] {  // NOLINT
      if (not std::isfinite(prob))
      {
        for (UDT d : log_w)
        {
          LOG_FATAL(d);
        }
        LOG_FATAL(numeric::logsumexp(log_w));
        return false;
      }
      return true;
    }());
    out.emplace(track, prob);
  }
  return out;
}

template <class UDT>
std::pair<double, std::size_t> Hypotheses<UDT>::prune_max_hypotheses(std::size_t max_hypotheses)
{
  double psum = 0;
  std::size_t num_erased = 0;
  std::vector<std::pair<HypothesisId, UDT>> hyp_ids;
  hyp_ids.reserve(_hyps.size());
  std::ranges::for_each(_hyps, [&](Hypothesis<UDT> const& hyp) { hyp_ids.emplace_back(hyp._id, hyp._weightLog); });
  std::ranges::nth_element(hyp_ids,
                           std::next(hyp_ids.begin(), std::min(max_hypotheses, hyp_ids.size())),
                           std::ranges::greater{},
                           &std::pair<HypothesisId, UDT>::second);
  for (std::size_t i = max_hypotheses; i < hyp_ids.size(); ++i)
  {
    auto const it = _hyps.find(hyp_ids.at(i).first);
    assert(it != _hyps.end());
    psum += to_val(it->getWeight());
    ++num_erased;
    _hyps.erase(it);
  }
  return { psum, num_erased };
}
template <class UDT>
struct HypothesisIdAndWeightStruct
{
  HypothesisId _hyp_id;
  double _hyp_weight;
};

template <class UDT>
std::pair<double, std::size_t> Hypotheses<UDT>::prune_hypotheses_diss_herrmann(std::size_t max_comps_retain,
                                                                               std::size_t min_comps_per_card,
                                                                               double min_comps_per_card_thresh,
                                                                               double max_weight_ratio,
                                                                               bool alloc_min_comps_per_card_by_poisson)
{
  double psum = 0;
  std::size_t num_erased = 0;
  if (_hyps.empty())
    return { psum, num_erased };
  std::size_t max_card = max_tracks();
  std::size_t num_kept = 0;
  std::vector<HypothesisIdAndWeightStruct<UDT>> sorted_hypId_to_weight;
  sorted_hypId_to_weight.reserve(_hyps.size());
  std::unordered_map<HypothesisId, Cardinality> id_to_card;

  // save weight values
  for (const auto& hyp : _hyps)
  {
    Cardinality num_tracks = std::ssize(hyp._tracks);
    HypothesisIdAndWeightStruct<UDT> inf;
    inf._hyp_id = hyp._id;
    inf._hyp_weight = to_val(hyp.getWeight());
    id_to_card.emplace(hyp._id, num_tracks);
    sorted_hypId_to_weight.emplace_back(inf);
  }
  std::ranges::sort(sorted_hypId_to_weight,
                    [](const HypothesisIdAndWeightStruct<UDT>& a, const HypothesisIdAndWeightStruct<UDT>& b) {
                      return a._hyp_weight > b._hyp_weight;
                    });

  double min_weight_log = std::log(sorted_hypId_to_weight.front()._hyp_weight) - std::log(max_weight_ratio);
  Vector<UDT> const card_dist = cardinality_distribution();
  std::vector<std::size_t> card_num_kept(max_card + 1);
  Vector<UDT> cdn_min_hyps = Vector<UDT>::Zero(max_card + 1);
  if (alloc_min_comps_per_card_by_poisson && min_comps_per_card > 0)
  {
    Eigen::ArrayXf card_dist2(max_card + 1);
    for (std::size_t i = 0; i < max_card; ++i)
    {
      card_dist2(i) = to_val(card_dist(i));
    }
    std::size_t minNumCompsPerCardPoisson = 1;
    Eigen::ArrayXf linspace = Eigen::ArrayXf::LinSpaced(max_card + 1, 0, max_card);
    Eigen::ArrayXf prod = card_dist2 * linspace;
    double lambda = prod.sum();
    Vector<UDT> poiss_probs = Vector<UDT>::Zero(max_card + 1);
    poiss_probs(0) = std::exp(-lambda);
    for (std::size_t n = 1; n <= max_card; ++n)
    {
      poiss_probs(n) = lambda / (n)*poiss_probs(n - 1);
    }
    poiss_probs = poiss_probs * max_card * min_comps_per_card;
    cdn_min_hyps = poiss_probs.array().floor();
    for (std::size_t n = 0; n <= max_card; ++n)
    {
      if (poiss_probs(n) < minNumCompsPerCardPoisson)
      {
        cdn_min_hyps(n) = minNumCompsPerCardPoisson;
      }
    }
  }
  else
  {
    for (std::size_t n = 0; n <= max_card; ++n)
    {
      cdn_min_hyps(n) = min_comps_per_card;
    }
  }

  for (const auto hyp_info : sorted_hypId_to_weight)
  {
    HypothesisId hyp_id = hyp_info._hyp_id;
    UDT hypProb = hyp_info._hyp_weight;
    Cardinality card = id_to_card.at(hyp_id);
    UDT card_prob = card_dist(card);

    const bool maxNumCompsNotReached = num_kept < max_comps_retain;
    const bool hypWeightSufficient = std::log(hypProb) > min_weight_log;
    const bool hypWeightOK = !std::isinf(std::log(hypProb)) && hypProb > 0.;  // std::numeric_limits<double>::epsilon();
    const bool maxNumCompsPerCardNotReached = card_num_kept[card] < cdn_min_hyps(card);
    const bool cardWeightOK = std::log(card_prob) > std::log(min_comps_per_card_thresh);
    const bool inactivePruning = max_comps_retain <= 0;

    if (!((maxNumCompsNotReached && hypWeightSufficient) || (maxNumCompsPerCardNotReached && cardWeightOK) ||
          (inactivePruning && hypWeightOK)))
    {
      // weight of hypotheses to low
      auto const it = _hyps.find(hyp_id);
      assert(it != _hyps.end());
      psum += to_val(it->getWeight());
      ++num_erased;
      _hyps.erase(it);
    }
    else
    {
      card_num_kept[card]++;
      num_kept += 1;
    }
  }

  LOG_DEB("number kept hypotheses: " << num_kept << " out of: " << max_comps_retain << "num erased: " << num_erased
                                     << " psum: " << psum);
  return { psum, num_erased };
}

template <class UDT>
std::pair<double, std::size_t> Hypotheses<UDT>::prune_hypotheses_weight_threshold(double threshold)
{
  double psum = 0;
  std::size_t num_erased = 0;
  for (auto it = _hyps.begin(); it != _hyps.end();)
  {
    if (it->getWeight() < threshold)
    {
      psum += to_val(it->getWeight());
      it = _hyps.erase(it);
      ++num_erased;
    }
    else
    {
      ++it;
    }
  }
  return { psum, num_erased };
}

template <class UDT>
std::pair<double, std::size_t> Hypotheses<UDT>::prune_labels_existence_threshold(TTBManager<UDT> const& manager,
                                                                                 double threshold)
{
  auto const label_ex_prob = label_existence_prob(manager);
  Hypotheses pruned_hyps;
  double pruned_weight = 0;
  std::size_t num_labels_pruned = 0;
  for (Hypothesis<UDT> const& hyp : _hyps)
  {
    std::vector<StateId> pruned_tracks = hyp._tracks;
    std::erase_if(pruned_tracks, [&](StateId track) {
      double const ex_prob = to_val(label_ex_prob.at(manager.get_state(track)._label));
      if (ex_prob < threshold)
      {
        ++num_labels_pruned;
        pruned_weight += ex_prob;
        return true;
      }
      return false;
    });
    Hypothesis pruned_hyp = hyp;
    pruned_hyp._tracks = std::move(pruned_tracks);
    pruned_hyps.add(std::move(pruned_hyp));
  }
  *this = std::move(pruned_hyps);
  return { pruned_weight, num_labels_pruned };
}

template <class UDT>
std::vector<Hypotheses<UDT>> Hypotheses<UDT>::split(std::vector<std::vector<StateId>> const& split_ids) const
{
  ZoneScopedN("Hypotheses::split");
  assert([&] {
    std::set<StateId> seen_ids;
    for (std::vector<StateId> const& ids : split_ids)
    {
      for (StateId id : ids)
      {
        if (seen_ids.contains(id))
        {
          LOG_FATAL("splitIds are not mutually exclusive");
          return false;
        }
        seen_ids.insert(id);
      }
    }
    return true;
  }());
  std::vector<Hypotheses> split(split_ids.size() + 1);
  for (Hypothesis<UDT> const& ori : _hyps)
  {
    std::vector<std::vector<StateId>> split_tracks(split_ids.size() + 1);
    for (StateId id : ori._tracks)
    {
      bool found = false;
      for (auto const& [ind, split_id] : std::views::enumerate(split_ids))
        if (std::ranges::find(split_id, id) != split_id.end())
        {
          split_tracks.at(ind).push_back(id);
          found = true;
          break;
        }
      if (not found)
      {
        split_tracks.back().push_back(id);
      }
    }
    for (auto const& [ind, ts] : std::views::enumerate(split_tracks))
    {
      if (not ts.empty())
      {
        Hypothesis hyp(std::move(ts), ori._weightLog, ori._prior_id);
        hyp._id = ori._id;
        split.at(ind).add(std::move(hyp));
      }
    }
  }
  std::vector<Hypotheses> non_empty;
  for (Hypotheses& hyps : split)
  {
    if (not hyps._hyps.empty())
    {
      non_empty.push_back(std::move(hyps));
    }
  }
  if (non_empty.empty())
  {
    non_empty.push_back(Hypotheses{});
  }
  // the weight of every hypothesis that is not assigned to a split group, is assigned to the zero hyp
  UDT const ori_weight = sum_hypotheses_weight();
  for (Hypotheses& hyps : non_empty)
  {
    if (auto const it = _hyps.template get<StateIds>().find(std::vector<StateId>{}); it != _hyps.template get<StateIds>().end())
    {
      hyps.add(*it);
    }
    assert([&] {  // NOLINT
      if (ori_weight + 1e-10 < hyps.sum_hypotheses_weight())
      {
        LOG_FATAL(hyps.to_string());
        LOG_FATAL(to_string());
        LOG_FATAL(ori_weight);
        LOG_FATAL(hyps.sum_hypotheses_weight());
        return false;
      }
      if (hyps._hyps.empty())
      {
        LOG_FATAL(to_string());
        LOG_FATAL("all empty hyp");
        return false;
      }
      return true;
    }());
    if (UDT const split_weight = hyps.sum_hypotheses_weight(); ori_weight > split_weight)
    {
      UDT const missing_weight = std::log(ori_weight - split_weight);
      Hypothesis hyp({}, missing_weight);
      hyps.add(std::move(hyp));
    }
  }
  assert([&] {  // NOLINT
    for (Hypotheses const& hyp : non_empty)
    {
      if (std::abs(hyp.sum_hypotheses_weight() - sum_hypotheses_weight()) > 1e-10)
      {
        LOG_FATAL(to_string());
        LOG_FATAL(hyp.to_string());
        return false;
      }
      if (not std::isfinite(hyp.sum_hypotheses_weight()))
      {
        LOG_FATAL(hyp.to_string());
        return false;
      }
    }
    return true;
  }());
  return non_empty;
}

template <class UDT>
std::string Hypotheses<UDT>::to_string() const
{
  std::string out = std::format("Hypotheses\n");
  for (Hypothesis<UDT> const& hyp : _hyps)
  {
    out += hyp.toString();
  }
  return out;
}

template <class UDT>
void Hypotheses<UDT>::normalize_hypotheses_weight()
{
  assert([&] {  // NOLINT
    if (sum_hypotheses_weight() == 0)
    {
      LOG_FATAL("#Hypotheses: " << _hyps.size());
      LOG_FATAL(std::stacktrace::current());
      for (Hypothesis<UDT> const& hypothesis : _hyps)
      {
        LOG_FATAL(hypothesis.toString());
      }
      return false;
    }
    return true;
  }());
  UDT const log_weight = numeric::logsumexp(_hyps, [](Hypothesis<UDT> const& hyp) { return hyp._weightLog; });
  for (auto it = _hyps.begin(); it != _hyps.end(); ++it)
  {
    _hyps.modify(it, [&](Hypothesis<UDT>& hyp) { hyp._weightLog -= log_weight; });
  }
  assert([&] {  // NOLINT
    if (std::abs(sum_hypotheses_weight() - 1) > 1e-5)
    {
      LOG_FATAL("normalization of hypotheses weights failed. Weight after norm: " << sum_hypotheses_weight());
      return false;
    }
    return true;
  }());
}

template Hypotheses<double> merge(std::vector<Hypotheses<double>> const& independent_hypotheses, std::size_t max_merged_size);
template Hypotheses<AD> merge(std::vector<Hypotheses<AD>> const& independent_hypotheses, std::size_t max_merged_size);
template <class UDT>
Hypotheses<UDT> merge(std::vector<Hypotheses<UDT>> const& independent_hypotheses, std::size_t max_merged_size)
{
  ZoneScopedN("Hypotheses::merge");
  assert(not independent_hypotheses.empty());

  using Edge = std::pair<std::size_t, HypothesisId>;
  using Node = std::size_t;  // nodes have no meaning
  std::vector<Node> nodes;   // start = 0
  std::vector<graph::Edge<Edge, Node, double>> edges;
  for (auto const& [ind, hypotheses] : std::views::enumerate(independent_hypotheses))
  {
    for (Hypothesis<UDT> const& hyp : hypotheses._hyps)
    {
      edges.emplace_back(std::pair{ ind, hyp._id }, ind, ind + 1, to_val(hyp._weightLog));
    }
    nodes.push_back(ind);
  }
  nodes.push_back(independent_hypotheses.size());
  graph::DiGraph graph(std::move(nodes), std::move(edges));
  auto best_merged = graph.k_longest_paths(0, independent_hypotheses.size(), max_merged_size);
  Hypotheses<UDT> merged_hyps;
  for (auto const& sol : best_merged)
  {
    std::vector<StateId> merged_tracks;
    std::map<SourceId, UDT> _soft_detection_weight{};
    std::map<SourceId, UDT> _soft_clutter_weight{};
    std::map<SourceId, UDT> _point_weight{};
    UDT _prior_weight = 1;
    for (auto const& [ind, hyp_id] : sol.first)
    {
      Hypothesis<UDT> const& local_hyp = *independent_hypotheses.at(ind)._hyps.find(hyp_id);
      merged_tracks.insert(merged_tracks.end(), local_hyp._tracks.begin(), local_hyp._tracks.end());
      _prior_weight *= std::exp(local_hyp._prior_weight_log);
      for (auto const& [source, val] : local_hyp._soft_detection_weight)
      {
        auto [it, _] = _soft_detection_weight.emplace(source, 1);
        it->second *= val;
      }
      for (auto const& [source, val] : local_hyp._soft_clutter_weight)
      {
        auto [it, _] = _soft_clutter_weight.emplace(source, 1);
        it->second *= val;
      }
      for (auto const& [source, val] : local_hyp._point_weight)
      {
        auto [it, _] = _point_weight.emplace(source, 1);
        it->second *= val;
      }
    }
    Hypothesis<UDT> merged(std::move(merged_tracks), UDT{sol.second});
    merged._soft_detection_weight = std::move(_soft_detection_weight);
    merged._soft_clutter_weight = std::move(_soft_clutter_weight);
    merged._point_weight = std::move(_point_weight);
    merged._prior_weight_log = std::log(_prior_weight);
    merged_hyps.add(std::move(merged));
  }
  assert(!merged_hyps._hyps.empty() && " merged hypotheses are empty!");
  merged_hyps.normalize_hypotheses_weight();
  return merged_hyps;
}

}  // namespace ttb
