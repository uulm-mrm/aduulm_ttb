#include "tracking_lib/MultiObjectStateDistributions/GLMBDistribution.h"
#include "tracking_lib/MultiObjectStateDistributions/Utils.h"
#include "tracking_lib/Misc/MurtyAlgorithm.h"

#include "tracking_lib/Misc/ProportionalAllocation.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/States/Innovation.h"
#include "tracking_lib/Misc/GibbsSampler.h"
#include "tracking_lib/Misc/Numeric.h"
#include "tracking_lib/Misc/AngleNormalization.h"

#include <utility>
#include <ranges>
#include <unordered_set>
#include <set>
#include <tracy/tracy/Tracy.hpp>

namespace ttb
{
constexpr auto tracy_color = tracy::Color::RebeccaPurple;

IDGenerator<MODistributionId> GLMBDistribution::_idGenerator{};

std::string to_string(GLMBUpdateProfilerData const& data)
{
  std::string out = "GLMB Update\n";
  out += "\tDuration: " + std::to_string(to_milliseconds(data._duration)) + "ms\n";
  out += "\t#Tracks: " + std::to_string(data._numTracks) + "\n";
  out += "\t#existing Hypotheses: " + std::to_string(data._numExistingHyps) + "\n";
  out += "\t#update Hypotheses: " + std::to_string(data._numUpdateHyps) + "\n";
  return out;
}
std::string to_stringStatistics(std::vector<GLMBUpdateProfilerData> const& datas)
{
  std::string out = "GLMB Update Statistics\n";
  Duration meanDuration(0);
  double meanTracks = 0;
  double meanEx = 0;
  std::size_t maxEx = 0;
  double meanUpdate = 0;
  std::size_t maxNewHyps = 0;
  for (auto const& data : datas)
  {
    meanDuration += data._duration / datas.size();
    meanTracks += static_cast<double>(data._numTracks) / datas.size();
    meanEx += static_cast<double>(data._numExistingHyps) / datas.size();
    maxEx = std::max(maxEx, data._numExistingHyps);
    meanUpdate += static_cast<double>(data._numUpdateHyps) / datas.size();
    maxNewHyps = std::max(maxNewHyps, data._numUpdateHyps);
  }
  out += "\tMean Duration: " + std::to_string(to_milliseconds(meanDuration)) + "ms\n";
  out += "\tMean #Tracks: " + std::to_string(meanTracks) + "\n";
  out += "\tMean #prior Hypotheses: " + std::to_string(meanEx) + "\n";
  out += "\tMax #prior Hypotheses: " + std::to_string(maxEx) + "\n";
  out += "\tMean #new Hypotheses: " + std::to_string(meanUpdate) + "\n";
  out += "\tMax #new Hypotheses: " + std::to_string(maxNewHyps) + "\n";
  return out;
}

std::string to_string(GLMBCreateHypothesesProfilerData const& data)
{
  std::string out = "GLMB Create Hypotheses\n";
  out += "\tDuration: " + std::to_string(to_milliseconds(data._duration)) + "ms\n";
  out += "\t#Tracks: " + std::to_string(data._numTracks) + "\n";
  out += "\t#Hypotheses: " + std::to_string(data._numHypotheses) + "\n";
  return out;
}
std::string to_stringStatistics(std::vector<GLMBCreateHypothesesProfilerData> const& datas)
{
  std::string out = "GLMB Create Hypotheses Statistics\n";
  Duration meanDuration(0);
  double meanTracks = 0;
  std::size_t maxTracks = 0;
  double meanHypotheses = 0;
  std::size_t maxHyps = 0;
  for (auto const& data : datas)
  {
    meanDuration += data._duration / datas.size();
    meanTracks += static_cast<double>(data._numTracks) / datas.size();
    maxTracks = std::max(maxTracks, data._numTracks);
    meanHypotheses += static_cast<double>(data._numHypotheses) / datas.size();
    maxHyps = std::max(maxHyps, data._numHypotheses);
  }
  out += "\tMean Duration: " + std::to_string(to_milliseconds(meanDuration)) + "ms\n";
  out += "\tMean #Tracks: " + std::to_string(meanTracks) + "\n";
  out += "\tMax #Tracks: " + std::to_string(maxTracks) + "\n";
  out += "\tMean #Hypotheses: " + std::to_string(meanHypotheses) + "\n";
  out += "\tMax #Hypotheses: " + std::to_string(maxHyps) + "\n";
  return out;
}

GLMBDistribution::GLMBDistribution(TTBManager* manager) : _manager{ manager }
{
}

GLMBDistribution::GLMBDistribution(TTBManager* manager, std::vector<State> tracks)
  : _manager{ manager }, _tracks{ [&]() {
    std::unordered_map<StateId, State> out;
    for (State& track : tracks)
    {
      out.emplace(track._id, std::move(track));
    }
    return out;
  }() }
{
}

std::unordered_map<Label, std::vector<StateId>> const& GLMBDistribution::label2Tracks() const
{
  if (_label2Tracks.has_value())
  {
    return _label2Tracks.value();
  }
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution before label2Tracks.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
  std::unordered_map<Label, std::vector<StateId>> out;
  for (auto const& [track_id, track] : _tracks)
  {
    if (std::vector<StateId>& tmp = out[track._label]; std::ranges::find(tmp, track._id) == tmp.end())
    {
      tmp.push_back(track._id);
    }
  }
  _label2Tracks = std::move(out);
  return _label2Tracks.value();
}

std::unordered_map<Label, Probability> const& GLMBDistribution::label2ExProb() const
{
  if (_label2exProb.has_value())
  {
    return _label2exProb.value();
  }
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution before label2ExProb.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
  std::unordered_map<Label, Probability> out;
  for (auto const& [trackId, prob] : track2ExProb())
  {
    out[_tracks.at(trackId)._label] += prob;
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
  _label2exProb = std::move(out);
  return _label2exProb.value();
}

std::unordered_map<StateId, Probability> const& GLMBDistribution::track2ExProb() const
{
  if (_trackId2exProb.has_value())
  {
    return _trackId2exProb.value();
  }
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution before track2ExProb.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
  std::unordered_map<StateId, Probability> out;
  for (auto const& [hyp_id, hyp] : _hypotheses)
  {
    for (StateId const track_id : hyp._tracks)
    {
      out[track_id] += hyp.getWeight();
    }
  }
  for (auto& [track, prob] : out)
  {
    if (prob > 1 + 1e-7)
    {
      LOG_WARN("Invalid Probability " + std::to_string(prob) + " for Track " + std::to_string(track.value_) +
               ". Closely monitor. Clamping for now....");
    }
    prob = std::min(prob, 1.0);
  }
  _trackId2exProb = std::move(out);
  return _trackId2exProb.value();
}

std::unordered_map<HypothesisId, std::vector<HypothesisId>> const& GLMBDistribution::priorId2UpdatedHypId() const
{
  if (_priorId2updatedHypId.has_value())
  {
    return _priorId2updatedHypId.value();
  }

  std::unordered_map<HypothesisId, std::vector<HypothesisId>> out;
  for (auto const& [hyp_id, hyp] : _hypotheses)
  {
    out[hyp._origin_id].push_back(hyp_id);
  }
  _priorId2updatedHypId = std::move(out);
  return _priorId2updatedHypId.value();
}

Indices GLMBDistribution::numHypsPerCard() const
{
  if (_numHypsPerCard.has_value())
  {
    return _numHypsPerCard.value();
  }
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution before numHypsPerCard.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
  Indices out = Indices::Zero(static_cast<Index>(_tracks.size() + 1));
  for (auto const& [hyp_id, hyp] : _hypotheses)
  {
    ++out(static_cast<Index>(hyp._tracks.size()));
  }
  _numHypsPerCard = out;
  return _numHypsPerCard.value();
}

Vector const& GLMBDistribution::clutter_distribution(Index num_measurements) const
{
  ZoneScopedNC("GLMBDistribution::clutter_distribution", tracy_color);
  if (_clutter_dist.has_value())
  {
    return _clutter_dist.value();
  }
  assert(isValid());
  _clutter_dist = Vector::Zero(num_measurements + 1);

  for (Index tracks = 0; tracks <= detection_distribution().rows() - 1; ++tracks)
  {
    for (Index detected = 0; detected <= std::min(tracks, num_measurements); ++detected)
    {
      _clutter_dist.value()(num_measurements - detected) += detection_distribution()(tracks, detected);
    }
  }
  assert([&] {  // NOLINT
    if (std::abs(_clutter_dist.value().sum() - 1) > 1e-5)
    {
      LOG_FATAL("clutter dist does not sum up to 1: " << _clutter_dist.value());
      printCardandHypothesesNumbersPerCard(aduulm_logger::LoggerLevel::Fatal);
      return false;
    }
    return true;
  }());
  return _clutter_dist.value();
}

Matrix const& GLMBDistribution::detection_distribution() const
{
  ZoneScopedNC("GLMBDistribution::detection_distribution", tracy_color);
  if (_detection_probs.has_value())
  {
    return _detection_probs.value();
  }
  assert(isValid());
  _detection_probs = Matrix::Zero(static_cast<Index>(numberOfLabels() + 1), static_cast<Index>(numberOfLabels() + 1));
  for (auto const& [hyp_id, hyp] : _hypotheses)
  {
    Index detections = 0;
    Index tracks = 0;
    for (StateId trackId : hyp._tracks)
    {
      if (not _tracks.at(trackId)._detectable or _tracks.at(trackId)._meta_data._numUpdates <= 1)
      {
        continue;
      }
      tracks++;
      if (_tracks.at(trackId)._meta_data._lastAssociatedMeasurement != NOT_DETECTED)
      {
        detections++;
      }
    }
    _detection_probs.value()(tracks, detections) += hyp.getWeight();
  }
  assert([&] {  // NOLINT
    if (std::abs(_detection_probs.value().sum() - 1) > 1e-5)
    {
      LOG_FATAL("detection dist does not sum up to 1: " << _detection_probs.value().sum());
      printCardandHypothesesNumbersPerCard(aduulm_logger::LoggerLevel::Fatal);
      return false;
    }
    return true;
  }());
  return _detection_probs.value();
}

std::size_t GLMBDistribution::numberOfLabels() const
{
  std::vector<Label> labels;
  for (auto const& [track_id, track] : _tracks)
  {
    if (std::ranges::find(labels, track._label) == labels.end())
    {
      labels.push_back(track._label);
    }
  }
  return labels.size();
}

Vector const& GLMBDistribution::cardinalityDistribution() const
{
  if (_cacheCardDist.has_value())
  {
    return _cacheCardDist.value();
  }
  assert(isValid());
  Vector cardDist = Vector::Zero(static_cast<Index>(1 + numberOfLabels()));
  for (auto const& [hyp_id, hyp] : _hypotheses)
  {
    cardDist(static_cast<Index>(hyp._tracks.size())) += hyp.getWeight();
  }
  _cacheCardDist = std::move(cardDist);
  return _cacheCardDist.value();
}

double GLMBDistribution::estimatedNumberOfTracks() const
{
  Vector cardDist = cardinalityDistribution();
  double val = 0;
  for (Index n_elem = 0; n_elem != cardDist.rows(); n_elem++)
  {
    val += static_cast<double>(n_elem) * cardDist(n_elem);
  }
  return val;
}

std::unordered_map<HypothesisId, std::size_t> GLMBDistribution::getProportionalAllocation() const
{
  LOG_DEB("Proportional Allocation");
  Vector weights(_hypotheses.size());
  // save weight values
  for (const auto& [idx, hyp] : std::views::enumerate(_hypotheses))
  {
    weights(idx) = hyp.second.getWeight();
  }
  std::unordered_map<HypothesisId, std::size_t> numNewHyps;

  // calc for each hypothesis of this cardinality the value k for the murty algorithm
  Indices allocs =
      propAllocation(_manager->params().glmb_distribution.update.max_total_number_update_hypotheses,
                     weights,
                     _manager->params().glmb_distribution.update.num_update_hypotheses.equal_allocation_share_ratio);
  for (const auto& [idx, hyp] : std::views::enumerate(_hypotheses))
  {
    numNewHyps[hyp.second._id] = allocs(idx);
  }

  assert([&] {  // NOLINT
    std::size_t allocated_new_hyps = 0;
    for (auto const& [hyp, alloc] : numNewHyps)
    {
      allocated_new_hyps += alloc;
    }
    if (_manager->params().glmb_distribution.update.max_total_number_update_hypotheses != allocated_new_hyps)
    {
      LOG_FATAL("requested to create " << _manager->params().glmb_distribution.update.max_total_number_update_hypotheses
                                       << " new hypotheses but allocated " << allocated_new_hyps);
      return false;
    }
    return true;
  }());

  return numNewHyps;
}

std::size_t GLMBDistribution::maxNumUpdateHyps(HypothesisId predicted_hyp) const
{
  ZoneScopedNC("GLMBDistribution::numUpdatedHyps", tracy_color);
  if (_kBestCache.has_value())
  {
    return _kBestCache.value().at(predicted_hyp);
  }
  _kBestCache = getProportionalAllocation();
  return std::min(_kBestCache.value().at(predicted_hyp),
                  _manager->params().glmb_distribution.update.num_update_hypotheses.max_update_hypotheses);
}

GLMBDistribution::GenerateHypothesesInfos GLMBDistribution::createGraph() const
{
  ZoneScopedNC("GLMBDistribution::createGraph", tracy_color);
  std::vector<std::size_t> nodes(_tracks.size() + 1);
  std::iota(nodes.begin(), nodes.end(), 0);
  std::vector<graph::Edge<StateId, std::size_t, double>> edges;
  edges.reserve(2 * _tracks.size());
  // Add each track to the graph
  for (const auto& [idx, track] : std::views::enumerate(_tracks))
  {
    edges.emplace_back(track.second._id, idx, idx + 1, track.second._existenceProbability);
    edges.emplace_back(StateId{ 0 }, idx, idx + 1, 1 - track.second._existenceProbability);  ///< represents
                                                                                             ///< non-existence
  }
  return {
    ._graph = graph::DiGraph(std::move(nodes), std::move(edges)),
    ._startNode = 0,
    ._endNode = _tracks.size(),
  };
}

void add_hyp(Hypothesis hyp,
             std::unordered_map<HypothesisId, Hypothesis>& hypotheses,
             std::unordered_map<std::vector<StateId>, HypothesisId>& state_2_hypotheses)
{
  ZoneScopedNC("GLMBDistribution::add_hyp", tracy_color);
  std::vector<StateId> tracks = hyp._tracks;
  if (auto const it = state_2_hypotheses.find(tracks); it != state_2_hypotheses.end())
  {
    hypotheses.at(it->second)._weightLog += hyp._weightLog;
    return;
  }
  HypothesisId hyp_id = hyp._id;
  hypotheses.emplace(hyp_id, std::move(hyp));
  state_2_hypotheses.emplace(std::move(tracks), hyp_id);
}

void GLMBDistribution::generateHypothesesSampling()
{
  ZoneScopedNC("GLMBDistribution::generateHypothesesSampling", tracy_color);
  LOG_DEB("Use Sampling");
  double const num_all_hypotheses = std::pow(2, _tracks.size());
  LOG_DEB(
      "Want to create: " << _manager->params().glmb_distribution.lmb_2_glmb_conversion.sampling.max_number_hypotheses
                         << " out of: " << num_all_hypotheses);
  std::set<std::vector<StateId>> already_created_hyps_tracks;

  Vector include =
      0.5 * (Vector::Random(
                 static_cast<Index>(_tracks.size() *
                                    _manager->params().glmb_distribution.lmb_2_glmb_conversion.sampling.max_num_tries))
                 .array() +
             1);
  _numHypsPerCard = Indices::Zero(static_cast<Index>(_tracks.size() + 1));  // +1 for card 0
  double sum_hyp_weights = 0;
  bool emptyHyp_created = false;
  for (auto [ctr, i] = std::tuple{ Index{ 0 }, std::size_t{ 0 } };
       i < _manager->params().glmb_distribution.lmb_2_glmb_conversion.sampling.max_num_tries;
       ++i)
  {
    std::vector<StateId> tracks;

    double weight_log{ 0 };
    for (auto const& [track_id, track] : _tracks)
    {
      if (track._existenceProbability > include(ctr))
      {
        tracks.push_back(track_id);
        weight_log += std::log(track._existenceProbability);
      }
      else
      {
        weight_log += std::log(1 - track._existenceProbability);
      }
      ctr++;
    }
    if (auto [_, is_new] = already_created_hyps_tracks.insert(tracks); is_new)
    {
      if (tracks.empty())
      {
        emptyHyp_created = true;
      }
      _numHypsPerCard.value()(static_cast<Index>(tracks.size())) += 1;
      sum_hyp_weights += std::exp(weight_log);
      add_hyp(Hypothesis(std::move(tracks), weight_log), _hypotheses, _state_2_hypotheses);
      if (sum_hyp_weights >
              _manager->params().glmb_distribution.lmb_2_glmb_conversion.sampling.percentage_of_weight_hypotheses or
          _hypotheses.size() >=
              _manager->params().glmb_distribution.lmb_2_glmb_conversion.sampling.max_number_hypotheses)
      {
        break;
      }
    }
  }
  // Create empty Hypotheses, if not already created
  if (not emptyHyp_created)
  {
    _numHypsPerCard.value()(0) += 1;
    assert([&] {  // NOLINT
      if (_numHypsPerCard.value()(0) != 1)
      {
        LOG_FATAL("Multiple Hypotheses for 0 tracks .... " << _numHypsPerCard.value());
        return false;
      }
      return true;
    }());
    double empty_weight_log{ 0 };
    for (auto const& [track_id, track] : _tracks)
    {
      empty_weight_log += std::log(1 - track._existenceProbability);
    }
    add_hyp(Hypothesis({}, empty_weight_log), _hypotheses, _state_2_hypotheses);
  }
}

void GLMBDistribution::generateHypothesesKBest()
{
  ZoneScopedNC("GLMBDistribution::generateHypothesesKBest", tracy_color);
  LOG_DEB("Use KBest");
  double const num_all_hypotheses = std::pow(2, _tracks.size());
  LOG_DEB("Want to create: " << _manager->params().glmb_distribution.lmb_2_glmb_conversion.kBest.max_number_hypotheses
                             << " out of: " << num_all_hypotheses << " for " << _tracks.size() << " tracks");
  if (static_cast<double>(_manager->params().glmb_distribution.lmb_2_glmb_conversion.kBest.max_number_hypotheses) <
      num_all_hypotheses)
  {
    LOG_DEB("Pruning of hypotheses during LMB2GLMB");
  }
  // create Graph
  GenerateHypothesesInfos const infos = createGraph();
  std::vector<std::pair<std::vector<StateId>, double>> const solutions = infos._graph.k_shortest_paths(
      infos._startNode,
      infos._endNode,
      _manager->params().glmb_distribution.lmb_2_glmb_conversion.kBest.max_number_hypotheses);

  // Create Hypotheses
  _numHypsPerCard = Indices::Zero(static_cast<Index>(_tracks.size() + 1));  // +1 for card 0
  bool emptyHyp_created = false;
  for (const auto& [solution, weight] : solutions)
  {
    double weight_log = std::log(weight);
    std::vector<StateId> tracks;
    for (const auto& edge : solution)
    {
      if (edge != StateId{ 0 })
      {
        tracks.push_back(edge);
      }
    }
    if (tracks.empty())
    {
      emptyHyp_created = true;
    }
    _numHypsPerCard.value()(static_cast<Index>(tracks.size())) += 1;
    add_hyp(Hypothesis(std::move(tracks), weight_log), _hypotheses, _state_2_hypotheses);
  }
  // Create empty Hypotheses, if not already created
  if (not emptyHyp_created)
  {
    _numHypsPerCard.value()(0) += 1;
    assert(_numHypsPerCard.value()(0) == 1);
    double weight_log = 0;
    for (auto const& [track_id, track] : _tracks)
    {
      weight_log += std::log(track._existenceProbability);
    }
    add_hyp(Hypothesis({}, weight_log), _hypotheses, _state_2_hypotheses);
  }
}

void GLMBDistribution::generateHypothesesAll()
{
  ZoneScopedNC("GLMBDistribution::generateHypothesesAll", tracy_color);
  LOG_DEB("Want to generate ALL hypotheses");
  if (_tracks.size() > _manager->params().glmb_distribution.lmb_2_glmb_conversion.all.num_track_limit)
  {
    LOG_DEB("#Tracks > limit -> use fallback solution");
    switch (_manager->params().glmb_distribution.lmb_2_glmb_conversion.all.fallback_type)
    {
      case LMB_2_GLMB_CONVERISON_TYPE::SAMPLING:
        generateHypothesesSampling();
        return;
      case LMB_2_GLMB_CONVERISON_TYPE::K_BEST:
        generateHypothesesKBest();
        return;
      case LMB_2_GLMB_CONVERISON_TYPE::ALL:
        LOG_ERR("Use ALL as fallback for ALL, FIX CONFIG!. Using Sampling");
        generateHypothesesSampling();
        return;
    }
  }
  std::vector<std::pair<std::vector<StateId>, double>> hyps_tracks;
  double const num_all_hypotheses = std::pow(2, _tracks.size());
  LOG_DEB("CREATE " << num_all_hypotheses << " Hypotheses");
  _numHypsPerCard = Indices::Zero(static_cast<Index>(_tracks.size() + 1));
  for (std::size_t num = 0; static_cast<double>(num) < num_all_hypotheses; ++num)
  {
    std::vector<StateId> tracks;
    double weight_log{ 0 };
    for (auto const& [i, track] : std::views::enumerate(_tracks))
    {
      if (num & (1 << i))  // 2 bit representation of number, if 1 at pos i include this Track
      {
        tracks.push_back(track.first);
        weight_log += std::log(track.second._existenceProbability);
      }
      else
      {
        weight_log += std::log(1 - track.second._existenceProbability);
      }
    }
    LOG_DEB("Create Hypothesis with " << tracks.size() << " Tracks and labels "
                                      << " with weight: " << std::exp(weight_log));
    _numHypsPerCard.value()(static_cast<Index>(tracks.size())) += 1;
    add_hyp(Hypothesis(std::move(tracks), weight_log), _hypotheses, _state_2_hypotheses);
  }
}

void GLMBDistribution::generateHypotheses()
{
  ZoneScopedNC("GLMBDistribution::generateHypotheses", tracy_color);
  LOG_DEB("generateHypotheses");
  assert(numberOfLabels() == _tracks.size() && "Duplicate Labels in tracks, can not create Hypotheses for that");
  _hypotheses.clear();
  _state_2_hypotheses.clear();
  if (_tracks.empty())
  {
    LOG_DEB("No Tracks - generate empty Hypotheses");
    add_hyp(Hypothesis({}, 0), _hypotheses, _state_2_hypotheses);
  }
  else
  {
    LOG_DEB("Generate Hypotheses for " << _tracks.size() << " Tracks");
    switch (_manager->params().glmb_distribution.lmb_2_glmb_conversion.type)
    {
      case LMB_2_GLMB_CONVERISON_TYPE::SAMPLING:
        generateHypothesesSampling();
        break;
      case LMB_2_GLMB_CONVERISON_TYPE::K_BEST:
        generateHypothesesKBest();
        break;
      case LMB_2_GLMB_CONVERISON_TYPE::ALL:
        generateHypothesesAll();
        break;
    }
  }
  LOG_DEB("Created " + std::to_string(_hypotheses.size()) + " Hypotheses");
  normalizeHypothesesWeights();
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Generated Hypotheses are not valid");
      return false;
    }
    return true;
  }());
}

double GLMBDistribution::sumHypothesesWeights() const
{
  ZoneScopedNC("GLMBDistribution::sumHypothesesWeights", tracy_color);
  double const log_weight =
      numeric::logsumexp(_hypotheses, [](auto const& id_hyp) { return id_hyp.second._weightLog; });
  return std::exp(log_weight);
}

void GLMBDistribution::multiplyHypothesesWeights(double fac)
{
  ZoneScopedNC("GLMBDistribution::multiplyHypothesesWeights", tracy_color);
  double log_fac = std::log(fac);
  std::ranges::for_each(_hypotheses, [&](auto& hyp) { hyp.second._weightLog += log_fac; });
}

void GLMBDistribution::normalizeHypothesesWeights()
{
  ZoneScopedNC("GLMBDistribution::normalizeHypothesesWeights", tracy_color);
  assert([&] {  // NOLINT
    if (sumHypothesesWeights() == 0)
    {
      LOG_FATAL("#Hypotheses: " << _hypotheses.size());
      for (auto const& [id, hypothesis] : _hypotheses)
      {
        LOG_FATAL(hypothesis.toString());
      }
      return false;
    }
    return true;
  }());
  double const log_weight =
      numeric::logsumexp(_hypotheses, [](auto const& id_hyp) { return id_hyp.second._weightLog; });
  std::ranges::for_each(_hypotheses, [&](auto& hyp) { hyp.second._weightLog -= log_weight; });
  assert([&] {  // NOLINT
    if (std::abs(sumHypothesesWeights() - 1) > 1e-5)
    {
      LOG_FATAL("normalization of hypotheses weights failed. Weight after norm: " << sumHypothesesWeights());
      return false;
    }
    return true;
  }());
}

void GLMBDistribution::add_track(State state)
{
  ZoneScopedNC("GLMBDistribution::add_track", tracy_color);
  add_tracks({ std::move(state) });
}

void GLMBDistribution::add_tracks(std::vector<State> states)
{
  LOG_DEB("add_tracks with k_shortest path");
  ZoneScopedNC("GLMBDistribution::add_tracks", tracy_color);
  if (states.empty())
  {
    return;
  }
  // use the k-shortest path algorithm to come up with the most likely hypotheses
  using Edge = std::pair<StateId, HypothesisId>;
  using Node = std::size_t;  // nodes have no meaning
  Node start_node{ 0 };
  Node end_node{ 1 };
  std::vector<Node> nodes{ start_node, end_node };  // start = 0
  nodes.reserve(2 + _hypotheses.size() * states.size());
  std::vector<graph::Edge<Edge, Node, double>> edges;
  edges.reserve(_hypotheses.size() * (2 + states.size() * 2));
  std::size_t next_node = 2;
  if (_hypotheses.empty())
  {
    add_hyp(Hypothesis({}, 0), _hypotheses, _state_2_hypotheses);
  }
  for (auto const& [id, hyp] : _hypotheses)
  {
    edges.emplace_back(std::pair{ StateId{ 0 }, id }, start_node, next_node, hyp.getWeight());
    for (State const& state : states)
    {
      edges.emplace_back(
          std::pair{ state._id, HypothesisId{ 0 } }, next_node, next_node + 1, state._existenceProbability);
      edges.emplace_back(
          std::pair{ StateId{ 0 }, HypothesisId{ 0 } }, next_node, next_node + 1, 1 - state._existenceProbability);
      nodes.push_back(next_node);
      ++next_node;
    }
    edges.emplace_back(std::pair{ StateId{ 0 }, HypothesisId{ 0 } }, next_node, end_node, 1);
    nodes.push_back(next_node);
    ++next_node;
  }
  graph::DiGraph graph(std::move(nodes), std::move(edges));
  auto best_paths = graph.k_shortest_paths(
      start_node, end_node, _manager->params().glmb_distribution.update.max_total_number_update_hypotheses);
  std::unordered_map<HypothesisId, Hypothesis> new_hypotheses;
  std::unordered_map<std::vector<StateId>, HypothesisId> new_state_2_hypotheses;
  for (auto const& [path, weight] : best_paths)
  {
    assert(path.front().second != HypothesisId{ 0 });
    assert(path.size() - 2 == states.size());
    assert(path.front().first == StateId{ 0 });
    std::vector hyp_tracks = _hypotheses.at(path.front().second)._tracks;
    for (auto const& [state_id, _] : path)
    {
      if (state_id != StateId{ 0 })
      {
        hyp_tracks.push_back(state_id);
      }
    }
    add_hyp(Hypothesis(std::move(hyp_tracks), std::log(weight)), new_hypotheses, new_state_2_hypotheses);
  }
  for (auto& state : states)
  {
    StateId id = state._id;
    _tracks.emplace(id, std::move(state));
  }
  _hypotheses = std::move(new_hypotheses);
  _state_2_hypotheses = std::move(new_state_2_hypotheses);
  normalizeHypothesesWeights();
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution after add_tracks().");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
}

void GLMBDistribution::calcInnovation(MeasurementContainer const& Z)
{
  ZoneScopedNC("GLMBDistribution::calcInnovation", tracy_color);
  utils::innovate(_manager, _tracks, Z, [](auto& id_state) -> State& { return id_state.second; });
}

void GLMBDistribution::predict(Duration deltaT, EgoMotionDistribution const& egoDist)
{
  ZoneScopedNC("GLMBDistribution::predict", tracy_color);
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution before predict.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
  if (_hypotheses.empty())
  {
    add_hyp(Hypothesis({}, 0), _hypotheses, _state_2_hypotheses);
  }
  {
    ZoneScopedNC("GLMBDistribution::predict_states", tracy_color);
    utils::predict(
        _manager,
        _tracks,
        deltaT,
        egoDist,
        [](auto& id_track) -> State& { return id_track.second; },
        [&](State& state) {
          double const survival_probability = _manager->getPersistenceModel().getPersistenceProbability(state, deltaT);
          state._survival_probability = survival_probability;
          state._existenceProbability = std::numeric_limits<double>::quiet_NaN();
        });
  }
  {
    ZoneScopedNC("GLMBDistribution::predict_hypotheses", tracy_color);
    std::unordered_map<HypothesisId, Hypothesis> predicted_hyps;
    std::unordered_map<std::vector<StateId>, HypothesisId> predicted_state_2_hyps;
    if (_manager->params().glmb_distribution.update.joint_prediction_and_update)
    {
      // just take the old hypotheses with updated state ids
      for (auto const& [hyp_id, hyp] : _hypotheses)
      {
        Hypothesis new_hyp({}, hyp._weightLog, hyp._origin_id);
        for (StateId id : hyp._tracks)
        {
          new_hyp._tracks.push_back(_tracks.at(id)._id);
        }
        add_hyp(std::move(new_hyp), predicted_hyps, predicted_state_2_hyps);
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
      std::size_t const largest_num_tracks = std::ranges::max_element(_hypotheses, {}, [](auto const& id_hyp) {
                                               return id_hyp.second._tracks.size();
                                             })->second._tracks.size();
      nodes.reserve(2 + _hypotheses.size() * largest_num_tracks);
      std::vector<graph::Edge<Edge, Node, double>> edges;
      edges.reserve(_hypotheses.size() * (2 + largest_num_tracks * 2));
      std::size_t next_node = 2;
      for (auto const& [id, hyp] : _hypotheses)
      {
        if (largest_num_tracks > 0)
        {
          edges.emplace_back(std::pair{ StateId{ 0 }, id }, start_node, next_node, hyp.getWeight());
          for (std::size_t i = 0; i < largest_num_tracks; ++i)
          {
            if (i < hyp._tracks.size())
            {
              edges.emplace_back(std::pair{ hyp._tracks.at(i), HypothesisId{ 0 } },
                                 next_node,
                                 next_node + 1,
                                 _tracks.at(hyp._tracks.at(i))._survival_probability);
              edges.emplace_back(std::pair{ StateId{ 0 }, HypothesisId{ 0 } },
                                 next_node,
                                 next_node + 1,
                                 1 - _tracks.at(hyp._tracks.at(i))._survival_probability);
              nodes.push_back(next_node);
              ++next_node;
            }
            else  // add dummy edge so all sub-graphs have the same size, this is needed for the k_shortest_path
                  // algorithm
            {
              edges.emplace_back(std::pair{ StateId{ 0 }, HypothesisId{ 0 } }, next_node, next_node + 1, 1);
              nodes.push_back(next_node);
              ++next_node;
            }
          }
          edges.emplace_back(std::pair{ StateId{ 0 }, HypothesisId{ 0 } }, next_node, end_node, 1);
          nodes.push_back(next_node);
          ++next_node;
        }
        else
        {
          edges.emplace_back(std::pair{ StateId{ 0 }, id }, start_node, end_node, hyp.getWeight());
        }
      }
      graph::DiGraph graph(std::move(nodes), std::move(edges));
      auto best_paths = graph.k_shortest_paths(
          start_node, end_node, _manager->params().glmb_distribution.post_process_prediction.max_hypotheses);
      for (auto const& [path, weight] : best_paths)
      {
        assert(path.front().second != HypothesisId{ 0 });
        assert([&] {  // NOLINT
          if (path.size() - 2 != largest_num_tracks and not _hypotheses.at(path.front().second)._tracks.empty())
          {
            LOG_FATAL("path size: " << path.size());
            LOG_FATAL("#track size: " << _hypotheses.at(path.front().second)._tracks.size());
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
            hyp_tracks.push_back(_tracks.at(state_id)._id);
          }
        }
        add_hyp(Hypothesis(std::move(hyp_tracks), std::log(weight)), predicted_hyps, predicted_state_2_hyps);
      }
    }
    std::unordered_map<StateId, State> predicted_tracks;
    predicted_tracks.reserve(_tracks.size());
    for (auto& [_, state] : _tracks)
    {
      StateId const id = state._id;
      predicted_tracks.emplace(id, std::move(state));
    }
    _tracks = std::move(predicted_tracks);
    _hypotheses = std::move(predicted_hyps);
    _state_2_hypotheses = std::move(predicted_state_2_hyps);
    normalizeHypothesesWeights();
  }
  postProcessPrediction();
  reset_caches();
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution after predict.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
}

Matrix GLMBDistribution::buildCostMatrix(std::vector<StateId> const& tracks,
                                         MeasurementContainer const& measurements) const
{
  LOG_DEB("GLMBDistribution::build_cost_matrix");
  ZoneScopedNC("GLMBDistribution::build_cost_matrix", tracy_color);
  const std::size_t numMeasurements = measurements._data.size();

  Matrix neglogcostm{ [&] {
    if (_manager->params().glmb_distribution.update.joint_prediction_and_update)
    {
      return std::numeric_limits<double>::infinity() *
             Matrix::Ones(static_cast<Index>(tracks.size()), static_cast<Index>(numMeasurements + 2 * tracks.size()));
    }
    return std::numeric_limits<double>::infinity() *
           Matrix::Ones(static_cast<Index>(tracks.size()), static_cast<Index>(numMeasurements + tracks.size()));
  }() };
  for (auto [trackIndex, track_id] : std::views::enumerate(tracks))
  {
    State const& track = _tracks.at(track_id);
    Innovation const& innoMap = track._innovation.at(measurements._id);
    // Detection and Misdetection probability
    double const logPD = std::log(innoMap._detectionProbability);
    double const logPND = std::log(1 - innoMap._detectionProbability);
    double const predictedExProb = [&] {
      assert([&] {  // NOLINT
        if (std::isnan(track._existenceProbability) + std::isnan(track._survival_probability) != 1)
        {
          LOG_FATAL("state with both existence prob and survival prob in GLMB build cost. Set one to nan.");
          LOG_FATAL("state: " << track.toString());
          return false;
        }
        return true;
      }());
      // LMB filter -> track has existence probability
      if (not std::isnan(track._existenceProbability))
      {
        return track._existenceProbability;
      }
      // GLMB filter -> survival probability plays role of existence probability
      return track._survival_probability;
    }();
    assert([&] {  // NOLINT
      if (not std::isfinite(predictedExProb))
      {
        LOG_FATAL("track has non-finite existence prob: " << track.toString());
        return false;
      }
      return true;
    }());
    double const log_predictedExProb = std::log(predictedExProb);
    for (auto const& [measIndex, meas] : std::views::enumerate(measurements._data))
    {
      if (auto const& inno = innoMap._updates.find(meas._id); inno != innoMap._updates.end())
      {
        // track exists and is detected
        if (_manager->params().glmb_distribution.update.joint_prediction_and_update)
        {
          neglogcostm(trackIndex, measIndex) =
              -(log_predictedExProb + logPD + inno->second.log_likelihood - std::log(inno->second.clutter_intensity));
        }
        else
        {
          neglogcostm(trackIndex, measIndex) =
              -(logPD + inno->second.log_likelihood - std::log(inno->second.clutter_intensity));  // cost = -(logPD +
                                                                                                  // logLikelihood)
        }
      }
    }
    // track exists but mis detected
    if (_manager->params().glmb_distribution.update.joint_prediction_and_update)
    {
      neglogcostm(trackIndex, static_cast<Index>(numMeasurements) + trackIndex) = -(log_predictedExProb + logPND);
    }
    else
    {
      neglogcostm(trackIndex, static_cast<Index>(numMeasurements) + trackIndex) = -logPND;
    }
    // track does not exist, i.e., has died
    if (_manager->params().glmb_distribution.update.joint_prediction_and_update)
    {
      neglogcostm(trackIndex, static_cast<Index>(numMeasurements + tracks.size()) + trackIndex) =
          -std::log(1 - predictedExProb);
    }
  }
  return neglogcostm;
}

void GLMBDistribution::update(MeasurementContainer const& measurements)
{
  ZoneScopedNC("GLMBDistribution::update", tracy_color);
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution before update.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
  std::unordered_map<StateId, State> updated_tracks;
  std::unordered_map<HypothesisId, Hypothesis> updated_hypotheses;
  std::unordered_map<std::vector<StateId>, HypothesisId> updated_state_2_hypotheses;
  std::unordered_map<StateId, std::map<MeasurementId, StateId>> updated_tracks_map;  // which track and measurement
                                                                                     // created the updated track
  for (auto& [origin_id, origin] : _tracks)
  {
    LOG_DEB("_updated dists size: " << origin._innovation.at(measurements._id)._updates.size());
    for (auto& updated_state : origin._innovation.at(measurements._id)._updates)
    {
      StateId id = updated_state.second.updated_dist._id;
      updated_tracks_map[origin_id].emplace(updated_state.first, id);
      LOG_DEB("Track with id " << origin_id.value_ << " updated with Meas id: " << updated_state.first.value_
                               << " has new id: " << id);
      updated_tracks.emplace(id, std::move(updated_state.second.updated_dist));
    }
  }
  LOG_DEB("Updated tracks moved to _updated_tracks");

  if (_hypotheses.empty())
  {
    assert(_state_2_hypotheses.empty());
    LOG_DEB("Generate Hypotheses out of tracks");
    if (_manager->params().glmb_distribution.update.joint_prediction_and_update)
    {
      // create dummy hypothesis
      auto kv = std::views::keys(_tracks);
      add_hyp(Hypothesis(std::vector<StateId>{ kv.begin(), kv.end() }, 0), _hypotheses, _state_2_hypotheses);
    }
    else
    {
      LOG_DEB("Generate Hypotheses");
      generateHypotheses();
    }
  }
  LOG_DEB("GLMB density before update: " << toString());
  for (auto& [hyp_id, hyp] : _hypotheses)
  {
    LOG_DEB("Update hypothesis: " << hyp.toString());
    if (hyp._tracks.empty())
    {
      add_hyp(Hypothesis({}, hyp._weightLog, hyp_id), updated_hypotheses, updated_state_2_hypotheses);
      continue;
    }
    Matrix const costMatrix = buildCostMatrix(hyp._tracks, measurements);
    auto const [assignments, costs]{ [&] {
      std::size_t const k{ maxNumUpdateHyps(hyp_id) };
      switch (_manager->params().glmb_distribution.update.assignment_method)
      {
        case GLMB_ASSIGNMENT_METHOD::MURTY:
          return murty::getAssignments(costMatrix, k);
        case GLMB_ASSIGNMENT_METHOD::GIBBS_SAMPLING:
        {
          Eigen::VectorXi const initSol =
              Eigen::VectorXi::LinSpaced(static_cast<Index>(hyp._tracks.size()),  // all existent but not detected
                                         static_cast<int>(measurements._data.size()),
                                         static_cast<int>(measurements._data.size() + hyp._tracks.size() - 1));
          return gibbs::sample_assignments(
              costMatrix,
              k,
              initSol,
              static_cast<std::size_t>(static_cast<double>(k) *
                                       _manager->params().glmb_distribution.update.gibbs_sampling.max_trials_factor),
              _manager->params().glmb_distribution.update.gibbs_sampling.abort_after_ntimes_no_new_sol);
        }
      }
      assert(false);
      DEBUG_ASSERT_MARK_UNREACHABLE;
    }() };
    for (Index solI = 0; solI < assignments.cols(); solI++)
    {
      std::vector<StateId> updatedTrackList;
      for (auto const& [trackEnumId, track] : std::views::enumerate(hyp._tracks))
      {
        std::size_t const measIndex = assignments(trackEnumId, solI);
        LOG_DEB("Track: " << trackEnumId << "->" << measIndex);
        if (measIndex >= measurements._data.size() + hyp._tracks.size())  // track does not exist
        {
          continue;
        }
        MeasurementId const measId{ [&]() {
          if (measIndex < measurements._data.size())  // track detected
          {
            return measurements._data.at(measIndex)._id;
          }
          return NOT_DETECTED;
        }() };

        updatedTrackList.push_back(updated_tracks_map.at(track).at(measId));
      }
      add_hyp(Hypothesis(std::move(updatedTrackList), hyp._weightLog - costs(solI), hyp_id),
              updated_hypotheses,
              updated_state_2_hypotheses);
    }
  }
  _hypotheses = std::move(updated_hypotheses);
  _state_2_hypotheses = std::move(updated_state_2_hypotheses);
  _tracks = std::move(updated_tracks);
  normalizeHypothesesWeights();
  postProcessUpdate();
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("invalid after update");
      return false;
    }
    return true;
  }());
  reset_caches();
}

void GLMBDistribution::pm_fusion(std::vector<GLMBDistribution>&& updated_glmbs)
{
  ZoneScopedN("GLMBDistribution::pm_fusion");
  if (updated_glmbs.size() == 1)
  {
    _tracks = std::move(updated_glmbs.at(0)._tracks);
    _hypotheses = std::move(updated_glmbs.at(0)._hypotheses);
    return;
  }
  std::unordered_map<HypothesisId, Hypothesis> fused_hypotheses;
  std::unordered_map<StateId, State> fused_tracks;
  std::unordered_map<std::vector<StateId>, HypothesisId> fused_state_2_hypotheses;

  // do some checks, whether fusion is needed or not and save some informations
  std::map<Label, std::map<StateId, State>> updated_tracks_all_sensors;  // possible since stateId is globally unique,
                                                                         // because of idGenerator()
  std::map<StateId, Label> updated_track_id_2_label;
  bool empty_sensor_update = false;
  std::vector<StateId> already_used_track_ids;  // is used to check if track_id is really globally unique!
  for (auto& updated_glmb : updated_glmbs)
  {
    assert(updated_glmb.isValid() && " One ore more GLMB distributions are not valid...");
    if (updated_glmb._hypotheses.empty())
    {
      empty_sensor_update = true;
      if (!updated_glmb._tracks.empty())
      {
        LOG_WARN("No hypotheses left, but there are still tracks? Is this a bug?");
        throw std::runtime_error("No hypotheses left, but there are still tracks? Check if this is a bug!");
      }
    }
    std::vector<StateId> used_track_ids_sensor;
    for (auto const& [_, hyp] : updated_glmb._hypotheses)
    {
      for (auto const& track_id : hyp._tracks)
      {
        Label track_label = updated_glmb._tracks.at(track_id)._label;
        updated_track_id_2_label.emplace(track_id, track_label);
        if (updated_tracks_all_sensors.contains(track_label))
        {
          updated_tracks_all_sensors.at(track_label).emplace(track_id, std::move(updated_glmb._tracks.at(track_id)));
        }
        else
        {
          std::map<StateId, State> states;
          states.emplace(track_id, std::move(updated_glmb._tracks.at(track_id)));
          updated_tracks_all_sensors.emplace(track_label, std::move(states));
        }

        // Valid Checks (Todo(hermann): Do in assert or remove?)
        if (std::find(already_used_track_ids.begin(), already_used_track_ids.end(), track_id) !=
            already_used_track_ids.end())
        {
          throw std::runtime_error("track_id is already given in GLMBDistribution of other sensor! Fusion will be "
                                   "wrong!!!");
        }
        if (std::find(used_track_ids_sensor.begin(), used_track_ids_sensor.end(), track_id) ==
            used_track_ids_sensor.end())
        {
          used_track_ids_sensor.push_back(track_id);
        }
      }
    }
    already_used_track_ids.insert(
        already_used_track_ids.end(), used_track_ids_sensor.begin(), used_track_ids_sensor.end());
  }
  if (empty_sensor_update)
  {
    LOG_INF("Received at least one empty distribution => no fusion is performed and empty distribution is the result");
    _tracks.clear();
    _hypotheses.clear();
    reset_caches();
    return;
  }

  // create graphs for fusion with kBestSelection algorithm
  std::size_t num_sensors = updated_glmbs.size();
  std::vector<PMFusionInfos> fusionInfosVec = create_graphs_pm(updated_glmbs, num_sensors);
  if (fusionInfosVec.empty())
  {
    LOG_WARN("No Fusion result given since no prior hypothesis is given by all sensors!");
    _tracks.clear();
    _hypotheses.clear();
    reset_caches();
    return;
  }

  // K BestSelection algorithm
  LOG_DEB("start with kBestSelection algorithm, pmNumBestComponents_k: "
          << _manager->params().filter.lmb_fpm.pmNumBestComponents_k);
  std::vector<std::pair<PMFusionPart, std::vector<std::pair<std::vector<Edge_PM>, double>>>> solutionsVec;
  for (auto const& fusionInfos : fusionInfosVec)
  {
    if (fusionInfos._k == 0)
    {
      continue;
    }
    std::vector<std::pair<std::vector<Edge_PM>, double>> solutions =
        fusionInfos._graph.k_shortest_paths(fusionInfos._startNode, fusionInfos._endNode, fusionInfos._k);

    if (solutions.empty())
    {
      // There is no fused result for this label!
      LOG_ERR("Something is weird with the kBestSelectionAlgorithm! There should be at least one solution!");  // This
                                                                                                               // should
                                                                                                               // never
                                                                                                               // happen
      throw std::runtime_error("Something is weird with the kBestSelectionAlgorithm! There should be at least "
                               "one solution!");
    }
    solutionsVec.emplace_back(fusionInfos._partInfos, std::move(solutions));
  }
  // calculate fused hypotheses based on the kBestSelection results
  fuse_hypotheses_pm(solutionsVec,
                     updated_tracks_all_sensors,
                     updated_track_id_2_label,
                     num_sensors,
                     fused_hypotheses,           // remove from here?!
                     fused_tracks,               // this one also?
                     fused_state_2_hypotheses);  // this also?!

  _hypotheses = std::move(fused_hypotheses);
  _tracks = std::move(fused_tracks);
  _state_2_hypotheses = std::move(fused_state_2_hypotheses);
  make_valid();  // necessary because it can create empty states !!!!
  if (not _hypotheses.empty())
  {
    normalizeHypothesesWeights();
  }
  postProcessUpdate();
  reset_caches();
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution after pm_fusion.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
}

void GLMBDistribution::fuse_hypotheses_pm(
    const std::vector<std::pair<PMFusionPart, std::vector<std::pair<std::vector<Edge_PM>, double>>>>& solutionsVec,
    std::map<Label, std::map<StateId, State>>& updated_tracks_all_sensors,
    std::map<StateId, Label>& updated_track_id_2_label,
    const std::size_t num_sensors,
    std::unordered_map<HypothesisId, Hypothesis>& fused_hypotheses,
    std::unordered_map<StateId, State>& fused_tracks,
    std::unordered_map<std::vector<StateId>, HypothesisId>& fused_state_2_hypotheses)
{
  ZoneScopedN("GLMBDistribution::fuse_hypotheses_pm");
  std::unordered_map<std::vector<StateId>, StateId> updated_tracks_2_fused_track;  // parameter contains fused state ids
                                                                                   // and belonging eta
  for (auto const& [fusion_infos, solutions] : solutionsVec)
  {
    double prior_weight_log = _hypotheses.at(fusion_infos._priorHyp)._weightLog;
    for (auto const& sol : solutions)
    {
      double fused_hyp_weight_log = prior_weight_log;
      std::vector<StateId> fused_hyp_track_ids;
      std::stringstream fused_tracks_ids_str;
      for (auto const& prior_track_id : _hypotheses.at(fusion_infos._priorHyp)._tracks)
      {
        std::vector<State*> updated_tracks;  // todo(hermann): make unordered map for easy searching if this track is
                                             // already calculated
        const Label prior_label = _tracks.at(prior_track_id)._label;
        for (auto const& edge_id : sol.first)
        {
          HypothesisId updated_hyp_id = edge_id.second;
          std::vector<StateId> updated_track_ids_sensor =
              fusion_infos._edgeId2localHyp[edge_id.first].at(updated_hyp_id)->_tracks;
          bool foundLabel = false;
          for (auto const& updated_track_id : updated_track_ids_sensor)
          {
            if (updated_track_id_2_label.at(updated_track_id) == prior_label)
            {
              updated_tracks.push_back(&updated_tracks_all_sensors.at(prior_label).at(updated_track_id));
              foundLabel = true;
              break;
            }
          }
          if (!foundLabel)
          {
            throw std::runtime_error("Prior label is not given in updated hypothesis. There must be a bug!");
          }
        }
        // fusion of spatial densities
        StateId fused_track_id =
            fuse_tracks_pm(prior_track_id, updated_tracks, fused_tracks, updated_tracks_2_fused_track, num_sensors);
        fused_hyp_weight_log += std::log(std::any_cast<double>(
            fused_tracks.at(fused_track_id)._misc.at("eta")));  // see eq (11) and (15) in
                                                                // https://ieeexplore.ieee.org/document/10224121
        fused_tracks_ids_str << fused_track_id << ", ";
        fused_hyp_track_ids.push_back(fused_track_id);
      }
      add_hyp(Hypothesis(fused_hyp_track_ids, fused_hyp_weight_log), fused_hypotheses, fused_state_2_hypotheses);
    }
  }
}

StateId
GLMBDistribution::fuse_tracks_pm(StateId prior_track_id,
                                 std::vector<State*>& updated_tracks,
                                 std::unordered_map<StateId, State>& fused_tracks,
                                 std::unordered_map<std::vector<StateId>, StateId>& updated_tracks_2_fused_track,
                                 std::size_t num_sensors) const
{
  ZoneScopedN("GLMBDistribution::fuse_tracks_pm");
  auto const& stateModels = _manager->getStateModelMap();

  const Label priorLabel = _tracks.at(prior_track_id)._label;
  State fused_state = _manager->createState();

  std::vector<StateId> posterior_state_ids;
  for (const auto& updated_track : updated_tracks)
  {
    posterior_state_ids.push_back(updated_track->_id);
  }

  if (auto const fusedTrackIt = updated_tracks_2_fused_track.find(posterior_state_ids);
      fusedTrackIt != updated_tracks_2_fused_track.end())
  {
    // the fusion result for this combination is already calculated!
    return fusedTrackIt->second;
  }

  for (auto const& [model_id, mixture] : _tracks.at(prior_track_id)._state_dist)
  {
    BaseStateModel const& stateModel = *stateModels.at(model_id);

    const std::vector<BaseDistribution*>& priorMixtureComponents = mixture->dists();
    if (priorMixtureComponents.empty())
    {
      LOG_WARN("Skipping empty mixture in track fusion");
      continue;
    }
    double component_weight_sum = 0;
    double eta = 0;
    std::vector<std::unique_ptr<BaseDistribution>> fused_components;
    for (auto const& prior_component : priorMixtureComponents)
    {
      DistributionId prior_id = prior_component->id();
      std::vector<BaseDistribution*> posterior_comps;
      std::vector<DistributionId> posterior_comps_ids;
      bool found_prior_component_id = true;
      for (auto const& updated_track : updated_tracks)
      {
        const std::vector<BaseDistribution*>& posteriorMixtureComponents =
            updated_track->_state_dist.at(model_id)->dists();
        bool foundComponentSensor = false;
        for (auto const& posterior_component : posteriorMixtureComponents)
        {
          if ((posterior_component->priorId() == prior_id) ||
              ((posterior_component->priorId() == NO_DISTRIBUTION_ID_HISTORY) &&
               (posterior_component->id() == prior_id)))  // prior_id==loc_comp->priorId()
                                                          //  || misdetected component (corresponds
                                                          //  to priorId=0) with id==prior_id
          {
            if (foundComponentSensor)
            {
              throw std::runtime_error("Fitting posterior component should only be once in the mixture distribution!");
            }
            foundComponentSensor = true;
            posterior_comps_ids.push_back(posterior_component->id());
            posterior_comps.push_back(posterior_component);
          }
        }
        if (!foundComponentSensor)
        {
          // no results for this prior component are given -> no fusion result for this prior component!
          found_prior_component_id = false;
          break;
        }
      }

      if (!found_prior_component_id)
      {
        // no results for this prior component are given -> no fusion result for this prior component!
        continue;
      }
      // perform IMF Fusion
      const auto* priorDist = impl::smartPtrCast<GaussianDistribution>(prior_component,
                                                                       "prior component has unsupported "
                                                                       "Gaussian distribution type "
                                                                       "(fus_tracks_pm)");
      double prior_component_weight = priorDist->sumWeights();
      Matrix priorInfoMat = priorDist->getInfoMat();
      Vector priorInfoVec = priorDist->getInfoVec();
      if (stateModel.state_comps().indexOf(COMPONENT::ROT_Z).has_value())
      {
        angles::normalizeAngle(priorInfoVec(stateModel.state_comps().indexOf(COMPONENT::ROT_Z).value()));
      }

      Matrix infoMatSum;
      Vector infoVecSum;
      double weight_product = 1.0;
      double ny_product = 1.0;
      for (auto const& posterior_comp : posterior_comps)
      {
        const auto* localDist = impl::smartPtrCast<GaussianDistribution>(posterior_comp,
                                                                         "posterior component has unsupported "
                                                                         "Gaussian distribution type "
                                                                         "(fuse_tracks_pm)");
        const Matrix infoMat = localDist->getInfoMat();
        Vector infoVec = localDist->getInfoVec();
        if (stateModel.state_comps().indexOf(COMPONENT::ROT_Z).has_value())
        {
          angles::normalizeAngle(infoVec(stateModel.state_comps().indexOf(COMPONENT::ROT_Z).value()));
        }

        if (infoMatSum.size() == 0)
        {
          infoMatSum = infoMat;
        }
        else
        {
          infoMatSum = infoMatSum + infoMat;
        }

        if (infoVecSum.size() == 0)
        {
          infoVecSum = infoVec;
        }
        else
        {
          infoVecSum = infoVecSum + infoVec;
        }
        weight_product = weight_product * localDist->sumWeights();
        if (!localDist->_misc.contains("ny"))
        {
          LOG_ERR("Does not contain value for ny!!");
          throw std::runtime_error("Does not contain value for ny!!");
        }
        ny_product = ny_product * std::any_cast<double>(localDist->_misc.at("ny"));
      }
      Matrix diff = infoMatSum - static_cast<double>(num_sensors - 1) * priorInfoMat;
      Matrix fusedCov = diff.inverse();
      Vector fusedMean = fusedCov * (infoVecSum - static_cast<double>(num_sensors - 1) * priorInfoVec);

      if (fusedMean.hasNaN())
      {
        throw std::runtime_error("Nan!!!");
      }
      if (stateModel.state_comps().indexOf(COMPONENT::ROT_Z).has_value())
      {
        angles::normalizeAngle(fusedMean(stateModel.state_comps().indexOf(COMPONENT::ROT_Z).value()));
      }

      double norm_const = 1.0;  // Vercauteren approximation! Otherwise, this value has to be calculated!
      double fused_component_weight = norm_const * weight_product / std::pow(prior_component_weight, num_sensors - 1);
      component_weight_sum += fused_component_weight;
      eta += norm_const * ny_product *
             prior_component_weight;  // see eq (11) in https://ieeexplore.ieee.org/document/10224121
      auto fused_dist = std::make_unique<GaussianDistribution>(std::move(fusedMean), std::move(fusedCov));
      fused_dist->multiplyWeights(fused_component_weight);
      fused_components.push_back(std::move(fused_dist));
    }
    // normalize component weights
    LOG_DEB("Normalize weights with component_weight_sum: " << component_weight_sum);
    if (component_weight_sum > 0)
    {
      for (auto& comp : fused_components)
      {
        comp->multiplyWeights(1 / component_weight_sum);
        LOG_DEB("comp id: " << comp->id() << " comp weight: " << comp->sumWeights());
      }
    }
    fused_state._state_dist.at(stateModel.id())->merge(std::move(fused_components));
    fused_state._state_dist.at(stateModel.id())->_misc["eta"] = eta;
  }
  fused_state._label = priorLabel;
  fused_state._time = (*updated_tracks.begin())->_time;
  StateId fused_state_id = fused_state._id;
  // todo(hermann): Find solution for multi-model filter case!
  fused_state._misc["eta"] = std::any_cast<double>(fused_state._state_dist.begin()->second->_misc.at("eta"));
  fused_tracks.emplace(fused_state._id, std::move(fused_state));
  updated_tracks_2_fused_track.emplace(std::move(posterior_state_ids), fused_state._id);
  return fused_state_id;
}

std::size_t GLMBDistribution::scaled_by_prior_weight(const double weight) const
{
  return static_cast<std::size_t>(weight *
                                  static_cast<double>(_manager->params().filter.lmb_fpm.pmNumBestComponents_k));
}

std::size_t GLMBDistribution::scaled_by_prior_weight_poisson(std::map<HypothesisId, Index>& hypID2Alloc,
                                                             HypothesisId prior_id) const
{
  ZoneScopedN("LMB_FPM_Tracker::scaledByPriorWeightPoisson");
  Vector scaling_weights{ { 1 } };
  if (!hypID2Alloc.empty())
  {
    Indices slots = propAllocation(
        hypID2Alloc.at(prior_id), scaling_weights, _manager->params().filter.lmb_fpm.equal_allocation_share_ratio);
    return slots(0);
  }
  return 0;
}

std::size_t GLMBDistribution::get_k(const double weight,
                                    std::map<HypothesisId, Index>& hypID2Alloc,
                                    HypothesisId prior_id) const
{
  if (_manager->params().filter.lmb_fpm.calculate_poisson_k_best)
  {
    return scaled_by_prior_weight_poisson(hypID2Alloc, prior_id);
  }
  else
  {
    return scaled_by_prior_weight(weight);
  }
}

std::vector<GLMBDistribution::PMFusionInfos>
GLMBDistribution::create_graphs_pm(std::vector<GLMBDistribution>& updated_glmbs, std::size_t num_sensors) const
{
  ZoneScopedN("LMB_FPM_Tracker::createGraphs");
  std::vector<GLMBDistribution::PMFusionInfos> graphs;
  std::map<HypothesisId, Index> hypID2Alloc;
  std::vector<HypothesisId> hyp_ids;

  if (_manager->params().filter.lmb_fpm.calculate_poisson_k_best)
  {
    std::size_t num_prior_hyps = _hypotheses.size();
    Eigen::ArrayXf weights(num_prior_hyps);
    hyp_ids.resize(num_prior_hyps);
    std::size_t counter = 0;
    for (const auto& [prior_hyp_id, prior_hyp] : _hypotheses)
    {
      hyp_ids.at(counter) = prior_hyp._id;
      weights(counter) = prior_hyp.getWeight();
      counter++;
    }

    if (num_prior_hyps != 1)
    {
      Eigen::ArrayXf linspace = Eigen::ArrayXf::LinSpaced(num_prior_hyps, 0, num_prior_hyps - 1);
      Eigen::ArrayXf prod = weights * linspace;
      double lambda = prod.sum();
      Vector poiss_probs = Vector::Zero(num_prior_hyps);
      poiss_probs(0) = std::exp(-lambda);
      for (std::size_t n = 1; n <= num_prior_hyps - 1; ++n)
      {
        poiss_probs(n) = lambda / (n)*poiss_probs(n - 1);
      }
      Indices alloc = propAllocation(_manager->params().filter.lmb_fpm.pmNumBestComponents_k,
                                     poiss_probs,
                                     _manager->params().filter.lmb_fpm.equal_allocation_share_ratio);
      const auto alloc_slots = alloc.sum();
      assert(alloc_slots <= _manager->params().filter.lmb_fpm.pmNumBestComponents_k && "too much slots are allocated");
      if (static_cast<double>(alloc_slots) < _manager->params().filter.lmb_fpm.pmNumBestComponents_k * 0.85)
      {
        LOG_WARN("Number of allocated slots is very small (" << alloc_slots << " / "
                                                             << _manager->params().filter.lmb_fpm.pmNumBestComponents_k
                                                             << " are allocated).");
      }
      std::size_t counter2 = 0;
      for (auto const& val : alloc)
      {
        hypID2Alloc.emplace(hyp_ids.at(counter2), val);
        counter2++;
      }
    }
    else
    {
      hypID2Alloc.emplace(hyp_ids.at(0), _manager->params().filter.lmb_fpm.pmNumBestComponents_k);
    }
  }

  for (const auto& [prior_hyp_id, prior_hyp] : _hypotheses)
  {
    bool isempty = false;
    std::vector<std::map<HypothesisId, const Hypothesis*>> edgeId2localHyp(num_sensors);
    std::vector<std::size_t> nodes;
    std::vector<graph::Edge<Edge_PM, std::size_t, double>> edges;

    std::size_t startNode = 0;
    std::size_t endNode = 0;
    nodes.push_back(startNode);

    std::size_t node_ctr = 0;

    std::vector<std::size_t> nodes_tmp;
    std::vector<graph::Edge<Edge_PM, std::size_t, double>> edges_comp;
    std::size_t sensorId = 0;
    for (auto const& loc_dist : updated_glmbs)
    {
      std::vector<graph::Edge<Edge_PM, std::size_t, double>> edges_sensor;
      std::size_t node1 = node_ctr;
      node_ctr++;
      std::size_t node2 = node_ctr;
      std::unordered_map<HypothesisId, std::vector<HypothesisId>> priorId2UpdatedHypId =
          loc_dist.priorId2UpdatedHypId();
      if (!priorId2UpdatedHypId.contains(prior_hyp_id))
      {
        LOG_DEB("Sensor does not deliver results for prior hypothesis id " << prior_hyp_id);
        isempty = true;
        break;
      }
      for (const auto& loc_hyp_id : priorId2UpdatedHypId.at(prior_hyp_id))
      {
        if (loc_dist._hypotheses.at(loc_hyp_id)._origin_id != prior_hyp_id)
        {
          throw std::runtime_error("hypothesis id of sensor is not the same like prior hypothesis id. There must be a "
                                   "bug in priorId2UpdatedHypId()!");
        }
        edges_sensor.emplace_back(
            std::make_pair(sensorId, loc_hyp_id), node1, node2, loc_dist._hypotheses.at(loc_hyp_id).getWeight());
        edgeId2localHyp[sensorId].emplace(loc_hyp_id, &loc_dist._hypotheses.at(loc_hyp_id));
      }
      sensorId++;
      if (edges_sensor.empty())
      {
        // one sensor has no MixtureComponents with this prior component id!
        throw std::runtime_error("this case should be happened early - how can you end up here?");
      }
      nodes_tmp.push_back(node1);
      nodes_tmp.push_back(node2);
      endNode = node2;
      edges_comp.insert(
          edges_comp.end(), std::make_move_iterator(edges_sensor.begin()), std::make_move_iterator(edges_sensor.end()));
    }
    if (!isempty)
    {
      nodes.insert(nodes.end(), std::make_move_iterator(nodes_tmp.begin()), std::make_move_iterator(nodes_tmp.end()));
      edges.insert(edges.end(), std::make_move_iterator(edges_comp.begin()), std::make_move_iterator(edges_comp.end()));

      // remove duplicated nodes
      sort(nodes.begin(), nodes.end());
      nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());

      graph::DiGraph<std::size_t, Edge_PM, double> graph = graph::DiGraph(std::move(nodes), std::move(edges));
      // calculate k dependent on which mode
      std::size_t k = get_k(prior_hyp.getWeight(), hypID2Alloc, prior_hyp_id);
      if (prior_hyp._tracks.empty())
      {
        // we always need a solution for the empty hypotheses!! Otherwise, the tracks will never disappear later!!
        k = 1;
      }
      PMFusionPart partInfos;
      partInfos._priorHyp = prior_hyp_id;
      partInfos._edgeId2localHyp = std::move(edgeId2localHyp);
      graphs.emplace_back(std::move(graph), std::move(partInfos), startNode, endNode, k);
    }
  }
  return graphs;
}

std::vector<State> GLMBDistribution::getEstimate() const
{
  ZoneScopedNC("GLMBDistribution::getEstimate", tracy_color);
  LOG_DEB("Get Estimation");
  Vector cardDist = cardinalityDistribution();
  if (cardDist.rows() == 0)
  {
    return {};
  }
  switch (_manager->params().glmb_distribution.extraction.type)
  {
    case MO_DISTRIBUTION_EXTRACTION_TYPE::EXISTENCE_PROBABILITY:
    {
      std::vector<State> estimated_tracks;
      // for each label with existence prob > existence threshold find the track with the highest existence
      // probability
      for (auto const& [label, exProb] : label2ExProb())
      {
        if (exProb > _manager->params().glmb_distribution.extraction.threshold)
        {
          // find best track of label
          auto tracks = label2Tracks().at(label);
          auto bestTrack = std::ranges::max_element(tracks, [&](StateId a, StateId b) {
            State const& ta = _tracks.at(a);
            State const& tb = _tracks.at(b);
            return ta._existenceProbability < tb._existenceProbability;
          });
          estimated_tracks.push_back(_tracks.at(*bestTrack));
        }
      }
      return estimated_tracks;
    }
    case MO_DISTRIBUTION_EXTRACTION_TYPE::CARDINALITY:
    {
      std::vector<State> estimated_tracks;
      // auto maxCard = std::distance(m_cardDist.begin(), std::max_element(m_cardDist.begin(), m_cardDist.end()));
      // CARDINALITY card(maxCard);
      std::size_t maxCard = std::distance(cardDist.begin(), std::ranges::max_element(cardDist));

      // copy values for sorting in descending order
      std::vector<std::pair<Label, double>> sortList;
      for (auto const it : label2ExProb())
      {
        sortList.emplace_back(it);
      }
      if (auto endIt = std::next(sortList.begin(), std::min(maxCard, sortList.size())); endIt != sortList.end())
      {
        std::ranges::nth_element(sortList, endIt, [](auto const& a, auto const& b) { return a.second > b.second; });
        for (auto trackIt = sortList.begin(); trackIt != endIt; ++trackIt)
        {
          Label label = trackIt->first;
          assert(trackIt != sortList.end() && "too less tracks for extraction!");
          // find best track of label
          if (auto tracks = label2Tracks().at(label); not tracks.empty())
          {
            auto bestTrack = std::ranges::max_element(tracks, [&](StateId a, StateId b) {
              State const& ta = _tracks.at(a);
              State const& tb = _tracks.at(b);
              return track2ExProb().at(ta._id) < track2ExProb().at(tb._id);
            });
            State out = _tracks.at(*bestTrack);
            out._existenceProbability = track2ExProb().at(out._id);
            if (not out.isEmpty())
            {
              estimated_tracks.emplace_back(std::move(out));
            }
          }
        }
      }
      return estimated_tracks;
    }
    case MO_DISTRIBUTION_EXTRACTION_TYPE::BEST_HYPOTHESIS:
    {
      std::vector<State> estimated_tracks;
      if (auto bestHyp = std::ranges::max_element(
              _hypotheses, [&](auto const& a, auto const& b) { return a.second._weightLog < b.second._weightLog; });
          bestHyp != _hypotheses.end())
        for (StateId track_id : bestHyp->second._tracks)
        {
          if (not _tracks.at(track_id).isEmpty())
          {
            estimated_tracks.push_back(_tracks.at(track_id));
          }
        }
      return estimated_tracks;
    }
  }
  assert(false);
  DEBUG_ASSERT_MARK_UNREACHABLE;
}

std::map<MeasurementId, Probability> GLMBDistribution::probOfAssigment(MeasurementContainer const& measContainer) const
{
  ZoneScopedNC("GLMBDistribution::probOfAssigment", tracy_color);
  std::map<MeasurementId, Probability> rzMap;
  _trackId2exProb = std::unordered_map<StateId, double>{};
  _label2Tracks = std::unordered_map<Label, std::vector<StateId>>{};
  for (Measurement const& meas : measContainer._data)
  {
    rzMap.emplace(meas._id, 0);
  }
  for (auto const& [hyp_id, hyp] : _hypotheses)
  {
    for (StateId updated_track_id : hyp._tracks)
    {
      State const& track = _tracks.at(updated_track_id);
      _trackId2exProb.value()[track._id] += hyp.getWeight();
      if (std::vector<StateId>& tmp = _label2Tracks.value()[track._label];
          std::ranges::find(tmp, updated_track_id) == tmp.end())
      {
        tmp.push_back(updated_track_id);
      }
      if (track._meta_data._lastAssociatedMeasurement != NOT_DETECTED)
      {
        rzMap.at(track._meta_data._lastAssociatedMeasurement) += hyp.getWeight();
      }
    }
  }
  for (auto& [track, prob] : _trackId2exProb.value())
  {
    if (prob > 1 + 1e-7)
    {
      LOG_WARN("Invalid Probability " + std::to_string(prob) + " for Track " + std::to_string(track.value_) +
               ". Closely monitor. Clamping for now....");
    }
    prob = std::min(prob, 1.0);
  }
  return rzMap;
}

bool GLMBDistribution::isValid() const
{
  ZoneScopedNC("GLMBDistribution::isValid", tracy_color);
  if (bool const tracks_valid =
          std::ranges::all_of(_tracks,
                              [&](auto const& track) {
                                if (track.first != track.second._id)
                                {
                                  LOG_FATAL("Track with id " + std::to_string(track.second._id.value_) +
                                            " stored in map under id: " + std::to_string(track.first.value_));
                                  return false;
                                }
                                if (not track.second.isValid())
                                {
                                  LOG_FATAL("Track: " + track.second.toString() + " not valid");
                                  return false;
                                }
                                return true;
                              });
      not tracks_valid)
  {
    LOG_FATAL("Tracks are not valid");
    return false;
  }

  if (double const hyp_sum = sumHypothesesWeights(); not _hypotheses.empty() and std::abs(hyp_sum - 1) > 1e-5)
  {
    LOG_WARN("Hyp sum not == 1: " << hyp_sum);
    return false;
  }
  for (auto const& [hyp_id, hyp] : _hypotheses)
  {
    if (hyp_id != hyp._id)
    {
      LOG_WARN("Hypotheses with id " + std::to_string(hyp._id.value_) +
               " stored in map under id: " + std::to_string(hyp_id.value_));
    }
    for (StateId id : hyp._tracks)
    {
      if (not _tracks.contains(id))
      {
        LOG_WARN("_tracks does not contain track: " << id.value_ << " which is in hypotheses: " << hyp.toString());
        LOG_WARN(toString());
        return false;
      }
    }
    if constexpr (bool constexpr perform_paranoid_checks = false; perform_paranoid_checks)
    {
      for (auto const& [o_hyp_id, o_hyp] : _hypotheses)
      {
        if (o_hyp_id == hyp_id)
        {
          continue;
        }
        auto otracks = o_hyp._tracks;
        std::ranges::sort(otracks);
        auto tracks = hyp._tracks;
        std::ranges::sort(tracks);
        if (otracks == tracks)
        {
          LOG_FATAL("different hypotheses with the same tracks");
          return false;
        }
      }
    }
  }
  if constexpr (bool constexpr perform_paranoid_checks = false; perform_paranoid_checks)
  {
    for (auto const& [id, hyp] : _hypotheses)
    {
      if (not _state_2_hypotheses.contains(hyp._tracks))
      {
        LOG_FATAL("_state_2_hypotheses does not contain hypotheses " + hyp.toString());
        return false;
      }
    }
    for (auto const& [tracks, id] : _state_2_hypotheses)
    {
      if (not _hypotheses.contains(id))
      {
        LOG_FATAL("_hypotheses does not contain id " + std::to_string(id.value_) + " which is in _state_2_hypotheses");
        return false;
      }
      if (auto it =
              std::ranges::find_if(_hypotheses, [&](auto const& id_hyp) { return id_hyp.second._tracks == tracks; });
          it == _hypotheses.end())
      {
        LOG_FATAL("_hypotheses does not contain any Hypothesis with tracks in _state_2_hypotheses");
        return false;
      }
    }
  }
  return true;
}

std::string GLMBDistribution::toString(std::string const& prefix) const
{
  std::string out = prefix + "GLMB Distribution\n";
  out += prefix + "|\tId: " + std::to_string(_id.value_) + '\n';
  for (auto const& [hip_id, hyp] : _hypotheses)
  {
    out += hyp.toString(prefix + "|\t");
  }
  for (auto const& [state_id, state] : _tracks)
  {
    out += prefix + "|\tState Id: " + std::to_string(state_id.value_) + "\n";
    out += prefix + "|\tState: " + state.toString() + "\n";
  }
  return out;
}

void GLMBDistribution::reset_caches() const
{
  _label2exProb.reset();
  _trackId2exProb.reset();
  _cacheCardDist.reset();
  _cacheCardDistLog.reset();
  _numHypsPerCard.reset();
  _label2Tracks.reset();
  _label2Hypotheses.reset();
  _clutter_dist.reset();
  _detection_probs.reset();
  _kBestCache.reset();
  _priorId2updatedHypId.reset();
}

std::size_t GLMBDistribution::erase_unreferenced_tracks()
{
  ZoneScopedNC("GLMBDistribution::erase_unreferenced_tracks", tracy_color);
  std::unordered_set<StateId> referenced_tracks;
  std::ranges::for_each(_hypotheses, [&](auto const& hyp) {
    std::ranges::for_each(hyp.second._tracks, [&](StateId id) { referenced_tracks.insert(id); });
  });
  std::size_t const num_deleted =
      erase_if(_tracks, [&](auto const& state) { return not referenced_tracks.contains(state.first); });
  return num_deleted;
}

std::size_t GLMBDistribution::prune_threshold(double threshold)
{
  reset_caches();
  return std::erase_if(_hypotheses, [&](auto const& hyp) {
    if (hyp.second.getWeight() < threshold)
    {
      _state_2_hypotheses.erase(hyp.second._tracks);
      return true;
    }
    return false;
  });
}

std::size_t GLMBDistribution::prune_max_hypotheses(std::size_t max_hypotheses)
{
  reset_caches();
  std::vector<std::pair<HypothesisId, double>> hyp_ids;
  hyp_ids.reserve(_hypotheses.size());
  std::ranges::for_each(_hypotheses,
                        [&](auto const& hyp) { hyp_ids.emplace_back(hyp.second._id, hyp.second._weightLog); });
  std::ranges::nth_element(
      hyp_ids,
      std::next(hyp_ids.begin(),
                std::min(_manager->params().glmb_distribution.post_process_prediction.max_hypotheses, hyp_ids.size())),
      std::ranges::greater{},
      &std::pair<HypothesisId, double>::second);
  for (std::size_t i = _manager->params().glmb_distribution.post_process_prediction.max_hypotheses; i < hyp_ids.size();
       ++i)
  {
    auto const it = _hypotheses.find(hyp_ids.at(i).first);
    assert(it != _hypotheses.end());
    if (!it->second._tracks.empty())
    {
      _state_2_hypotheses.erase(it->second._tracks);
      _hypotheses.erase(it);
    }
  }
  return std::max(0UL, hyp_ids.size() - _manager->params().glmb_distribution.post_process_prediction.max_hypotheses);
}

std::size_t GLMBDistribution::postProcessPrediction()
{
  ZoneScopedNC("GLMBDistribution::postProcessPrediction", tracy_color);
  LOG_DEB("GLMBDistribution::postProcessPrediction");
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution before postProcessPrediction.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
  if (not _manager->params().glmb_distribution.post_process_prediction.enable)
  {
    return 0;
  }
  std::size_t num_deleted =
      prune_threshold(_manager->params().glmb_distribution.post_process_prediction.pruning_threshold);

  LOG_DEB("#pruned threshold: " << num_deleted);
  num_deleted += prune_max_hypotheses(_manager->params().glmb_distribution.post_process_prediction.max_hypotheses);
  erase_unreferenced_tracks();
  if (not _hypotheses.empty())
  {
    normalizeHypothesesWeights();
  }
  reset_caches();
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution after postProcessPrediction.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
  return num_deleted;
}

void GLMBDistribution::make_valid()
{
  std::unordered_map<StateId, State> valid_tracks;
  std::vector<StateId> invalid_ids;
  for (auto& [id, state] : _tracks)
  {
    if (not state.isEmpty())
    {
      valid_tracks.emplace(id, std::move(state));
    }
    else
    {
      invalid_ids.push_back(id);
    }
  }
  std::unordered_map<HypothesisId, Hypothesis> valid_hyps;
  std::unordered_map<std::vector<StateId>, HypothesisId> state_2_hypotheses;
  for (auto& [hyp_id, hyp] : _hypotheses)
  {
    std::erase_if(hyp._tracks,
                  [&](StateId const& id) { return std::ranges::find(invalid_ids, id) != invalid_ids.end(); });
    add_hyp(std::move(hyp), valid_hyps, state_2_hypotheses);
  }
  _tracks = std::move(valid_tracks);
  _hypotheses = std::move(valid_hyps);
  reset_caches();
}

std::size_t GLMBDistribution::postProcessUpdate()
{
  ZoneScopedNC("GLMBDistribution::postProcessUpdate", tracy_color);
  LOG_DEB("GLMBDistribution::postProcessUpdate");
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution before postProcessUpdate.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
  if (not _manager->params().glmb_distribution.post_process_update.enable)
  {
    return 0;
  }
  std::size_t num_deleted = prune_threshold(_manager->params().glmb_distribution.post_process_update.pruning_threshold);

  num_deleted += prune_max_hypotheses(_manager->params().glmb_distribution.post_process_update.max_hypotheses);
  erase_unreferenced_tracks();
  if (not _hypotheses.empty())
  {
    normalizeHypothesesWeights();
  }
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL("Invalid GLMB distribution after postProcessPrediction.");
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
  reset_caches();
  return num_deleted;
}

void GLMBDistribution::printCardandHypothesesNumbersPerCard(aduulm_logger::LoggerLevel printLogLevel) const
{
  std::stringstream msg;
  if (printLogLevel > aduulm_logger::g_log_level)
  {
    return;
  }
  msg << "############################## GLMB Distribution " << _id.value_ << " ##############################"
      << std::endl;

  std::stringstream cardStr;
  std::stringstream cardLogStr;
  std::stringstream hypsPerCardNumStr;
  for (const auto& val : cardinalityDistribution())
  {
    cardStr << val << " ";
  }
  msg << "Cardinality Distribution: [ " << cardStr.str() << "]" << std::endl;

  for (const auto& val2 : cardinalityDistribution().array().log())
  {
    cardLogStr << val2 << " ";
  }
  msg << "Cardinality Distribution Log: [ " << cardLogStr.str() << "]" << std::endl;

  for (auto& numHyps : numHypsPerCard())
  {
    hypsPerCardNumStr << numHyps << " ";
  }
  msg << "Cardinality Distribution Num: [ " << hypsPerCardNumStr.str() << "]" << std::endl;
  //      for (auto& trkIt : m_trackMap) {
  //          trkIt.second->print();
  //      }
  for (const auto& [label, exProb] : label2ExProb())
  {
    msg << "Label: " << label << ", exProb: " << exProb << std::endl;
  }
  switch (printLogLevel)
  {
    case 1:
      LOG_ERR(msg.str());
      break;
    case 2:
      LOG_WARN(msg.str());
      break;
    case 3:
      LOG_INF(msg.str());
      break;
    default:
      LOG_DEB(msg.str());
      break;
  }
}

}  // namespace ttb
