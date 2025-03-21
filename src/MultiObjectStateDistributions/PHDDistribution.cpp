#include "tracking_lib/MultiObjectStateDistributions/PHDDistribution.h"

#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/Distributions/BaseDistribution.h"
#include "tracking_lib/States/Innovation.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/Graph/Graph.h"
#include "tracking_lib/MultiObjectStateDistributions/Utils.h"

#include <vector>
#include <utility>

#include <tracy/tracy/Tracy.hpp>

namespace ttb
{

auto constexpr tracy_color = tracy::Color::PaleTurquoise;

IDGenerator<MODistributionId> PHDDistribution::_idGenerator{};

PHDDistribution::PHDDistribution(TTBManager* manager) : _manager{ manager }
{
}

std::string PHDDistribution::toString(std::string const& prefix) const
{
  std::string out = prefix + "PHD Distribution\n";
  out += prefix + "|\tId: " + std::to_string(_id.value_) + "\n";
  for (auto const& track : _tracks)
  {
    out += track.toString(prefix + "|\t");
  }
  return out;
}

bool PHDDistribution::isValid() const
{
  return std::ranges::all_of(_tracks, [&](State const& track) {
    if (not track.isValid())
    {
      LOG_FATAL("Track " + track.toString() + "not valid");
      return false;
    }
    return true;
  });
}

double& PHDDistribution::weight(State& state)
{
  auto& val = state._misc["phd_weight"];
  if (not val.has_value())
  {
    val = 0.0;
  }
  return std::any_cast<double&>(val);
}

double PHDDistribution::weight(State const& state)
{
  return std::any_cast<double>(state._misc.at("phd_weight"));
}

void PHDDistribution::postProcessPrediction()
{
  LOG_DEB("PHDDistribution::postProcessPrediction");
  if (not _manager->params().phd_distribution.prediction.post_process.enable)
  {
    return;
  }
  merge(_manager->params().phd_distribution.prediction.post_process.merge_distance);
  double const weight_before = sum_weights();
  std::erase_if(_tracks, [&](State const& state) {
    return weight(state) < _manager->params().phd_distribution.prediction.post_process.pruning_threshold;
  });
  double const weight_after = sum_weights();
  if (weight_after > 0)
  {
    for (State& state : _tracks)
    {
      weight(state) *= weight_before / weight_after;
    }
  }
}

void PHDDistribution::postProcessUpdate()
{
  LOG_DEB("PHDDistribution::postProcessUpdate");
  if (not _manager->params().phd_distribution.update.post_process.enable)
  {
    return;
  }
  merge(_manager->params().phd_distribution.update.post_process.merge_distance);
  double const weight_before = sum_weights();
  std::erase_if(_tracks, [&](State const& state) {
    return weight(state) < _manager->params().phd_distribution.update.post_process.pruning_threshold;
  });
  double const weight_after = sum_weights();
  if (weight_after > 0)
  {
    for (State& state : _tracks)
    {
      weight(state) *= weight_before / weight_after;
    }
  }
}

std::vector<State> PHDDistribution::getEstimate() const
{
  LOG_DEB("PHDDistribution::getEstimate");
  double expected_tracks = std::accumulate(
      _tracks.begin(), _tracks.end(), 0.0, [](double old, State const& state) { return old + weight(state); });
  // sort trackMap by existence Prob in descending order
  std::vector<std::size_t> inds(_tracks.size());
  std::iota(inds.begin(), inds.end(), 0);
  std::ranges::sort(inds, [&](std::size_t a, std::size_t b) { return weight(_tracks.at(a)) > weight(_tracks.at(b)); });
  std::vector<State> out;
  for (std::size_t i = 0; i < std::min(static_cast<std::size_t>(std::round(expected_tracks)), _tracks.size()); ++i)
  {
    out.push_back(_tracks.at(inds.at(i)));
  }
  return out;
}

void PHDDistribution::predict(Duration deltaT, EgoMotionDistribution const& egoDist)
{
  ZoneScopedNC("PHDDistribution::predict", tracy_color);
  LOG_DEB("PHDDistribution::predict");
  utils::predict(_manager, _tracks, deltaT, egoDist, {}, [&](State& state) {
    weight(state) *= _manager->getPersistenceModel().getPersistenceProbability(state, deltaT);
  });
  postProcessPrediction();
}

void PHDDistribution::calcInnovation(MeasurementContainer const& Z)
{
  ZoneScopedNC("PHDDistribution::calcInnovation", tracy_color);
  LOG_DEB("PHDDistribution::calcInnovation");
  utils::innovate(_manager, _tracks, Z);
}

std::map<MeasurementId, Probability> const& PHDDistribution::meas_assignment_prob() const
{
  return _meas_assignment_prob;
}

double PHDDistribution::sum_weights() const
{
  return std::accumulate(
      _tracks.begin(), _tracks.end(), 0.0, [](double old, State const& state) { return old + weight(state); });
}

void PHDDistribution::merge(double max_merge_distance)
{
  ZoneScopedNC("PHDDistribution::merge", tracy_color);
  LOG_DEB("PHDDistribution::merge");
  using Node = std::size_t;
  using Edge = graph::Edge<std::size_t, Node, int>;
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::size_t edge_ctr = 0;
  for (auto const& [id, state] : std::views::enumerate(_tracks))
  {
    nodes.push_back(id);
    auto [model, dist] = state.getEstimate();
    Vector const& mean = dist->mean();
    Matrix const& cov = dist->covariance();
    for (auto const& [other_id, other_state] : std::views::enumerate(_tracks))
    {
      if (other_id < id)
      {
        continue;
      }
      auto [other_model, other_dist] = other_state.getEstimate();
      if (model != other_model)
      {
        continue;
      }
      double const mhd = (mean - other_dist->mean()).transpose() *
                         (cov + other_dist->covariance()).llt().solve(mean - other_dist->mean());
      if (mhd > max_merge_distance)
      {
        continue;
      }
      edges.emplace_back(edge_ctr++, id, other_id, 1);
    }
  }
  std::vector<std::vector<Node>> const components = graph::Graph(std::move(nodes), std::move(edges)).components();
  std::vector<State> merged_tracks;
  for (std::vector<Node> const& comp : components)
  {
    assert(not comp.empty());
    State merged = _tracks.at(comp.front());
    BaseDistribution const& first_dist = _tracks.at(comp.front()).bestState().second;
    double sum_weight = weight(_tracks.at(comp.front()));
    Vector merged_mean = first_dist.mean() * sum_weight;
    Matrix merged_cov = first_dist.covariance() * sum_weight;

    for (std::size_t i = 1; i < comp.size(); ++i)
    {
      State const& state = _tracks.at(comp.at(i));
      if (merged._meta_data._numUpdates < state._meta_data._numUpdates)
      {
        merged._meta_data._numUpdates = state._meta_data._numUpdates;
        merged._label = state._label;
      }
      BaseDistribution const& dist = state.bestState().second;
      double const w = weight(state);
      sum_weight += w;
      merged_mean += dist.mean() * w;
      merged_cov += dist.covariance() * w;
    }
    merged_mean /= sum_weight;
    for (std::size_t i : comp)
    {
      State const& state = _tracks.at(i);
      BaseDistribution const& dist = state.bestState().second;
      double const w = weight(state);
      merged_cov += (merged_mean - dist.mean()) * (merged_mean - dist.mean()).transpose() * w;
    }
    merged_cov /= sum_weight;
    merged._state_dist.at(_tracks.at(comp.front()).bestState().first)
        ->set(std::move(merged_mean), std::move(merged_cov));
    weight(merged) = sum_weight;
    merged_tracks.push_back(std::move(merged));
  }
  _tracks = std::move(merged_tracks);
}

void PHDDistribution::update(MeasurementContainer const& measurementContainer)
{
  ZoneScopedNC("PHDDistribution::update", tracy_color);
  LOG_DEB("PHDDistribution::update");
  std::vector<State> updated_tracks;
  _meas_assignment_prob.clear();
  for (Measurement const& meas : measurementContainer._data)
  {
    _meas_assignment_prob[meas._id] = 0;
  }
  for (State& state : _tracks)
  {
    auto& [updates, detectionProbability] = state._innovation.at(measurementContainer._id);
    for (auto& [meas_id, update] : updates)
    {
      if (meas_id == NOT_DETECTED)
      {
        auto& [updated_dist, log_likelihood, clutter_intensity] = update;
        weight(updated_dist) *= 1 - detectionProbability;
        updated_tracks.push_back(std::move(updated_dist));
      }
      else
      {
        _meas_assignment_prob[meas_id] = 1;
        auto& [updated_dist, log_likelihood, clutter_intensity] = update;
        weight(updated_dist) *= detectionProbability * std::exp(log_likelihood);
        double denum = clutter_intensity;
        for (State& other_state : _tracks)
        {
          auto& [other_updates, other_detectionProbability] = other_state._innovation.at(measurementContainer._id);
          if (auto const it = other_updates.find(meas_id); it != other_updates.end())
          {
            denum += other_detectionProbability * std::exp(it->second.log_likelihood);
          }
        }
        weight(updated_dist) /= denum;
        updated_tracks.push_back(std::move(updated_dist));
      }
    }
  }
  _tracks = std::move(updated_tracks);
  postProcessUpdate();
}

void PHDDistribution::addTracks(std::vector<State> tracks)
{
  LOG_DEB("PHDDistribution::addTracks");
  for (State& state : tracks)
  {
    weight(state) = state._existenceProbability;
    _tracks.push_back(state);
  }
}

}  // namespace ttb