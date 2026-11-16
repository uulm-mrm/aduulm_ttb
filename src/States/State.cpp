#include "tracking_lib/States/State.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/Transformations/Transformation.h"
#include "tracking_lib/Misc/ProportionalAllocation.h"

#include <boost/geometry/util/range.hpp>
#include <tracy/tracy/Tracy.hpp>
#include <tracy/tracy/TracyC.h>

#include <stacktrace>

namespace ttb
{

constexpr auto tracy_color = tracy::Color::LightBlue;

template class State<double>;
template class State<AD>;

template <class UDT>
IDGenerator<Label> State<UDT>::_labelGenerator{};
template <class UDT>
IDGenerator<StateId> State<UDT>::_idGenerator{};

template <class UDT>
State<UDT>::State(TTBManager<UDT> const* manager,
                  std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> dist)
  : _manager{ manager }, _state_dist{ build_consistent_state_dist(std::move(dist)) }, _classification{ *_manager }
{
  LOG_DEB("create new state");
  assert(isEmpty() or isValid());
}

template <class UDT>
std::map<StateModelId, std::pair<UDT, Distribution<UDT>>> State<UDT>::build_consistent_state_dist(
    std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> dist) const
{
  LOG_DEB("build_consistent_state_dist");
  ZoneScopedN("State::build_consistent_state_dist");
  if (dist.empty())
  {
    return {};
  }

  UDT total_weight = 0;
  for (auto const& [model_id, gauss_comps] : dist)
  {
    for (auto const& [weight, gauss] : gauss_comps)
    {
      total_weight += weight;
    }
  }
  assert(total_weight > 0);
  std::map<StateModelId, std::pair<UDT, Distribution<UDT>>> sd_map;
  std::shared_ptr<Params const> const params = _manager->params();
  std::vector<StateModelId> const& use_models = params->state.multi_model.use_state_models;
  for (auto& [state_model_id, gauss_comps] : dist)
  {
    UDT model_weight = 0;
    for (auto const& [weight, gauss] : gauss_comps)
    {
      model_weight += weight;
    }
    if (std::ranges::find(use_models, state_model_id) != use_models.end())
    {
      if (not params->state_model_params(state_model_id).distribution.mixture)  // use gaussian
      {
        if (gauss_comps.size() == 1)
        {
          LOG_DEB("only one component. Weight should be 1: model weight "
                  << model_weight << " total weight: " << total_weight << " diff " << model_weight / total_weight);
          sd_map.emplace(state_model_id, std::pair{ model_weight / total_weight, gauss_comps.front().second });
        }
        else
        {
          LOG_DEB("model weight " << model_weight << " total weight: " << total_weight);
          MixtureDistribution mixture(std::move(gauss_comps));
          if (not mixture.is_empty())
          {
            sd_map.emplace(
                state_model_id,
                std::pair{ model_weight / total_weight,
                           GaussianDistribution(mixture.mean(_manager->state_comps(state_model_id)),
                                                mixture.covariance(_manager->state_comps(state_model_id))) });
          }
        }
      }
      else  // use mixture
      {
        LOG_DEB("model weight " << model_weight << " total weight: " << total_weight << " diff "
                                << model_weight / total_weight);
        MixtureDistribution mixture(std::move(gauss_comps));
        if (not mixture.is_empty())
        {
          sd_map.emplace(state_model_id, std::pair{ model_weight / total_weight, std::move(mixture) });
        }
      }
    }
    else
    {
      LOG_DEB("model weight " << model_weight << " total weight: " << total_weight << " diff "
                              << model_weight / total_weight);
      if (gauss_comps.size() == 1)
      {
        sd_map.emplace(state_model_id, std::pair{ model_weight / total_weight, gauss_comps.front().second });
      }
      else
      {
        MixtureDistribution mixture(std::move(gauss_comps));
        if (not mixture.is_empty())
        {
          sd_map.emplace(state_model_id, std::pair{ model_weight / total_weight, std::move(mixture) });
        }
      }
    }
  }
  UDT weight_sum = 0;
  for (auto const& [model, weight_dist] : sd_map)
  {
    weight_sum += weight_dist.first;
  }
  for (auto& elems : sd_map)
  {
    elems.second.first /= weight_sum;
  }
  return sd_map;
}

template <class UDT>
bool State<UDT>::isValid() const
{
  ZoneScopedN("State::isValid");
  if (isEmpty())
  {
    LOG_FATAL("State is empty!");
    LOG_FATAL(toString());
    LOG_FATAL(std::stacktrace::current());
    return false;
  }
  if (_type == STATE_TYPE::ESTIMATION)
  {
    if (_state_dist.size() != 1)
    {
      LOG_FATAL("multiple state models in the estimation");
      return false;
    }
    if (not std::holds_alternative<GaussianDistribution<UDT>>(_state_dist.begin()->second.second._distribution))
    {
      LOG_FATAL("Distribution of estimation is not Gaussian");
      return false;
    }
  }
  if (_type == STATE_TYPE::MEASUREMENT)
  {
    if (_state_dist.size() != 1)
    {
      LOG_FATAL("State is Measurement but has multiple distributions. This must not happen");
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
  }
  std::vector<MeasurementId> used_to_update;
  std::map<SourceId, UDT> sum;
  for (auto const& [source_id, weights] : _meta_data._meas_update_weight)
  {
    for (auto const& [meas_id, weight] : weights)
    {
      if (std::ranges::find(used_to_update, meas_id) != used_to_update.end())
      {
        LOG_FATAL(std::format(
            "Measurement {} is used multiple times for update of this State {}",
            [&] -> std::string {
              if (meas_id == NOT_DETECTED)
              {
                return "NOT_DETECTED";
              }
              return std::to_string(meas_id.value_);
            }(),
            toString()));
        LOG_FATAL(std::stacktrace::current());
        return false;
      }
      sum[source_id] += weight;
    }
  }
  for (SourceId const& source_id : _meta_data._meas_update_weight | std::views::keys)
  {
    if (std::abs(sum.at(source_id) - sumWeights()) > 1e-3)
    {
      LOG_FATAL(std::format(
          "inconsistent update_weight {} and distribution_weight {}\n{}", sum.at(source_id), sumWeights(), toString()));
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    if (std::abs(sum.at(source_id) - 1) > 1e-5 and _meta_data._numUpdates > 0)
    {
      LOG_FATAL(std::format("Sum of update weight is not near 1: {}, State {}", sum.at(source_id), toString()));
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
  }

  for (auto const& [model, dist] : _state_dist)
  {
    if (not dist.second.is_valid())
    {
      LOG_FATAL("Invalid Distribution for state model " << model);
      LOG_FATAL(toString());
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
  }
  if (std::abs(sumWeights() - 1) > 1e-5 or not std::isfinite(sumWeights()))
  {
    LOG_FATAL("Weight of StateDist not near 1: " + std::to_string(sumWeights()) + "\n" + toString());
    LOG_FATAL(std::stacktrace::current());
    return false;
  }
  return true;
}

template <class UDT>
bool State<UDT>::isEmpty() const
{
  return std::ranges::all_of(_state_dist, [](auto const& elem) { return elem.second.first == 0; });
}

template <class UDT>
void State<UDT>::predict(Duration deltaT, EgoMotionDistribution const& ego)
{
  ZoneScopedN("State::predict");
  if (deltaT < 0ms)
  {
    LOG_WARN_THROTTLE(1, "Prediction into past is not supported");
    _state_dist.clear();
    return;
  }
  if (deltaT < 1ms)
  {
    return;
  }
  if (deltaT > 100s)
  {
    LOG_FATAL(std::format("requested to predict the state {} into the future. This is way too large ... Delete this "
                          "state",
                          deltaT));
    _state_dist.clear();
    return;
  }
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL(toString("Invalid State: "));
      return false;
    }
    return true;
  }());
  if (_manager->params()->state.multi_model.enable_markov_transition)
  {
    markovTransition();
  }
  std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> predicted_comps;
  for (auto const& [stateModelId, dist] : _state_dist)
  {
    StateModel<UDT> const& sm = _manager->getStateModel(stateModelId);
    for (auto const& [weight, dist_mixture] : dist.second.to_gaussian_components())
    {
      predicted_comps[stateModelId].emplace_back(dist.first * weight, sm.predict(deltaT, dist_mixture, ego));
    }
  }
  _state_dist = build_consistent_state_dist(std::move(predicted_comps));
  _classification.discount();
  _meta_data._durationSinceLastAssociation += deltaT;
  _time += deltaT;
  ++_meta_data._numPredictions;
  _all_prior_ids.push_back(_id);
  _id = _idGenerator.getID();
  postProcess();
  assert([&] {  // NOLINT
    if (isEmpty())
    {
      return true;
    }
    if (not isValid())
    {
      LOG_FATAL("State after prediction not valid: " << toString());
      return false;
    }
    return true;
  }());
}

template <class UDT>
void State<UDT>::compensateEgoMotion(Duration dt, EgoMotionDistribution const& egoMotion)
{
  // hpx::tracy::region("StateDistribution::compensateEgoMotion", 0, 0);
  std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> compensated;
  for (auto& [stateModelId, dist] : _state_dist)
  {
    for (auto const& [weight, dist_mixture] : dist.second.to_gaussian_components())
    {
      compensated[stateModelId].emplace_back(
          weight, _manager->getStateModel(stateModelId).compensateEgoMotion(dt, dist_mixture, egoMotion));
    }
  }
  _state_dist = build_consistent_state_dist(std::move(compensated));
  _all_prior_ids.push_back(_id);
  _id = _idGenerator.getID();
}

template <class UDT>
void State<UDT>::markovTransition()
{
  ZoneScopedN("State::markovTransition");
  std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> new_state;
  Matrix<double> const& markovMatrix = _manager->getTransition().transitionMatrix(*this);
  for (auto const [i, origin_modelId] : std::views::enumerate(_manager->params()->state.multi_model.use_state_models))
  {
    StateModel<UDT> const& origin_stateModel = _manager->getStateModel(origin_modelId);
    for (auto const [j, target_modelId] : std::views::enumerate(_manager->params()->state.multi_model.use_state_models))
    {
      if (markovMatrix(i, j) > 0)
      {
        StateModel<UDT> const& target_stateModel = _manager->getStateModel(target_modelId);
        if (auto const it = _state_dist.find(origin_modelId); it != _state_dist.end())
        {
          for (auto const& [weight, distComp] : it->second.second.to_gaussian_components())
          {
            GaussianDistribution transformedDist =
                transformation::transform(distComp, origin_stateModel, target_stateModel);
            new_state[target_modelId].emplace_back(it->second.first * weight * markovMatrix(i, j),
                                                   std::move(transformedDist));
          }
        }
      }
    }
  }
  _state_dist = build_consistent_state_dist(std::move(new_state));
}

template <class UDT>
void State<UDT>::postProcess()
{
  ZoneScopedN("State::postProcess");
  // can make state invalid if all components get pruned
  // delete state models with small weight
  std::erase_if(_state_dist, [&](auto const& elem) {
    return elem.second.first < _manager->params()->state_model_params(elem.first).distribution.post_process.min_weight;
  });
  for (auto& [id, dist] : _state_dist)
  {
    DistributionParams const& dist_params = _manager->params()->state_model_params(id).distribution;
    if (dist_params.post_process.enable)
    {
      dist.second.merge_components(dist_params.post_process.merging_distance,
                                   _manager->getStateModel(id).state_comps());

      dist.second.prune_weight(dist_params.post_process.min_weight);
      dist.second.prune_var(dist_params.post_process.max_variance);

      dist.second.truncate(dist_params.post_process.max_components);
    }
  }
  // the post_processing of the distribution can delete it entirely, so it is "empty" -> delete the state models with
  // the empty distributions
  std::erase_if(_state_dist, [](auto const& elem) { return elem.second.second.is_empty(); });
  if (sumWeights() > 1e-10)
  {
    multiplyWeights(1 / sumWeights());
  }
  else
  {
    _state_dist.clear();
  }
  _all_prior_ids.push_back(_id);
  _id = _idGenerator.getID();
  assert(isEmpty() or isValid());
}

template <class UDT>
void State<UDT>::approximateToOneComponent()
{
  for (auto& [id, dist] : _state_dist)
  {
    dist.second.merge_components(std::numeric_limits<double>::infinity(), _manager->getStateModel(id).state_comps());
  }
  _all_prior_ids.push_back(_id);
  _id = _idGenerator.getID();
  assert(isEmpty() or isValid());
}

template <class UDT>
std::string State<UDT>::toString(std::string_view prefix) const
{
  constexpr std::size_t detection_hist_display_size = 10;
  std::string detection_hist_string;
  auto it = _meta_data._detection_hist.begin();
  if (_meta_data._detection_hist.size() > detection_hist_display_size)
  {
    it = _meta_data._detection_hist.end() - detection_hist_display_size;
  }
  for (; it != _meta_data._detection_hist.end(); ++it)
  {
    detection_hist_string += std::to_string(*it) + " ";
  }
  std::vector<StateId> prior_ids_show;
  for (StateId id : _all_prior_ids | std::views::reverse | std::views::take(10))
  {
    prior_ids_show.push_back(id);
  }
  std::string out = std::format("{}StateDistribution\n"
                                "{}|\tID:     {}\n"
                                "{}|\tPriorID {}\n"
                                "{}|\tPriorIDs {}\n"
                                "{}|\tType: {}\n"
                                "{}|\tLabel {}\n"
                                "{}|\tClass {}\n"
                                "{}|\t#Updates {}\n"
                                "{}|\t#Predictions {}\n"
                                "{}|\tTime: {}\n"
                                "{}|\tTime of last assoc.       {}\n"
                                "{}|\tDuration since last assoc {}\n",
                                prefix,
                                prefix,
                                _id,
                                prefix,
                                _prior_id,
                                prefix,
                                prior_ids_show,
                                prefix,
                                _type,
                                prefix,
                                _label,
                                prefix,
                                _classification.getEstimate(),
                                prefix,
                                _meta_data._numUpdates,
                                prefix,
                                _meta_data._numPredictions,
                                prefix,
                                _time,
                                prefix,
                                _meta_data._timeOfLastAssociation,
                                prefix,
                                _meta_data._durationSinceLastAssociation);
  for (auto const& [source_id, likelihood] : _meta_data._last_meas_log_likelihood)
  {
    out += std::format("{}|\tMeas Likelihood source {}: {}\n", prefix, source_id, likelihood);
  }
  for (auto const& [source_id, detectable] : _meta_data._detectable)
  {
    out += std::format("{}|\tDetectable for source {}: {}\n", prefix, source_id, detectable);
  }
  out += std::format("{}|\tExistence score {}\n"
                     "{}|\tStage {}\n"
                     "{}|\tDetection Hist {}\n",
                     prefix,
                     _score,
                     prefix,
                     _stage,
                     prefix,
                     detection_hist_string);
  for (auto const& [source_id, weights] : _meta_data._meas_update_weight)
  {
    out += std::format("{}|\tUpdated from Source {}\n", prefix, source_id);
    for (auto const& [meas_id, weight] : weights)
    {
      out += std::format("{}|\t|\t meas id {}: {}\n", prefix, meas_id, weight);
    }
  }
  for (auto const& [id, dist] : _state_dist)
  {
    out += std::format("{}|\tState Model Id {} {}\n"
                       "{}{}",
                       prefix,
                       id,
                       dist.first,
                       _manager->state_comps(id).toString(std::string(prefix) + "|\t|\t"),
                       dist.second.to_string(std::string(prefix) + "|\t|\t"));
  }
  out += std::format("{}|\tInnovations\n", prefix);
  return out;
}

template <class UDT>
std::pair<StateModelId, Distribution<UDT> const&> State<UDT>::bestState() const
{
  assert([&] {
    if (_state_dist.empty())
    {
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    return true;
  }());
  auto const& it = std::ranges::max_element(
      _state_dist, [](auto const& a, auto const& b) { return a.second.first < b.second.first; });
  return { it->first, it->second.second };
}

template <class UDT>
std::pair<StateModelId, GaussianDistribution<UDT>> State<UDT>::getEstimate() const
{
  if (_type == STATE_TYPE::ESTIMATION or _type == STATE_TYPE::MEASUREMENT)
  {
    assert(_state_dist.size() == 1);
    auto const it = _state_dist.begin();
    if (std::holds_alternative<GaussianDistribution<UDT>>(it->second.second._distribution))
    {
      return { it->first, std::get<GaussianDistribution<UDT>>(it->second.second._distribution) };
    }
    if (std::holds_alternative<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(it->second.second._distribution))
    {
      std::optional<GaussianDistribution<UDT>> best;
      UDT best_weight = 0;
      for (auto const& [weight, dist] : it->second.second.to_gaussian_components())
      {
        if (weight > best_weight)
        {
          best_weight = weight;
          best = dist;
        }
      }
      assert(best.has_value());
      return { it->first, std::move(best.value()) };
    }
    LOG_FATAL("unknown");
    assert(false);
    std::unreachable();
  }
  // hpx::tracy::region("State::getEstimate", 0, 0);

  auto const& [stateModelId, baseDist] = bestState();

  if (_manager->params()->state.estimation.transform_output_state)
  {
    StateModelId target = _manager->params()->state.estimation.output_state_model;
    if (std::holds_alternative<GaussianDistribution<UDT>>(baseDist._distribution))
    {
      return std::pair{ target,
                        transformation::transform(std::get<GaussianDistribution<UDT>>(baseDist._distribution),
                                                  _manager->getStateModel(stateModelId),
                                                  _manager->getStateModel(target)) };
    }
    return std::pair{
      target,
      transformation::transform(
          std::get<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(baseDist._distribution).best_component(),
          _manager->getStateModel(stateModelId),
          _manager->getStateModel(target))
    };
  }
  if (std::holds_alternative<GaussianDistribution<UDT>>(baseDist._distribution))
  {
    return std::pair{ stateModelId, std::get<GaussianDistribution<UDT>>(baseDist._distribution) };
  }
  return std::pair{
    stateModelId, std::get<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(baseDist._distribution).best_component()
  };
}

template <class UDT>
UDT State<UDT>::sumWeights() const
{
  UDT sum = 0;
  for (auto const& [modelID, dist] : _state_dist)
  {
    sum += dist.first;
  }
  return sum;
}

template <class UDT>
void State<UDT>::multiplyWeights(UDT factor)
{
  for (auto& dist : _state_dist | std::views::values)
  {
    dist.first *= factor;
  }
  _all_prior_ids.push_back(_id);
  _id = _idGenerator.getID();
}

template <class UDT>
std::tuple<StateDetection<UDT>,
           std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>>,
           std::vector<MeasClutter<UDT>>>
State<UDT>::innovate(DataSourceContainer<UDT> const& measContainer) const
{
  ZoneScopedN("State::innovate");
  assert(isValid());
  return _manager->getMeasModel(measContainer._source_id).calculate_innovation(measContainer, *this);
}

template <class UDT>
void State<UDT>::performStageUpdate(std::size_t num_sensors)
{
  if (_manager->params()->state.stage_management.use_score_based_stage_logic)
  {
    double const alpha = _manager->params()->state.stage_management.false_track_confirmation_probability;
    double const beta = _manager->params()->state.stage_management.true_track_deletion_probability;
    double const track_confirmation_score = log((1 - beta) / alpha);
    if (_score > track_confirmation_score)
    {
      _stage = STAGE::CONFIRMED;
    }
    else
    {
      _stage = STAGE::PRELIMINARY;
    }
  }
  if (_manager->params()->state.stage_management.use_history_based_stage_logic)
  {
    performHistoryStageLogic(num_sensors);
  }
}

template <class UDT>
void State<UDT>::performHistoryStageLogic(Index num_sensors)
{
  if (_meta_data._detection_hist.size() >=
      num_sensors * _manager->params()->state.stage_management.deletion_history_threshold.N)
  {
    bool tobedeleted =
        std::count(_meta_data._detection_hist.end() -
                       num_sensors * _manager->params()->state.stage_management.deletion_history_threshold.N,
                   _meta_data._detection_hist.end(),
                   false) >=
        static_cast<Index>(num_sensors * _manager->params()->state.stage_management.deletion_history_threshold.M);
    if (tobedeleted)
    {
      _stage = STAGE::DEAD;
    }
  }
  if (_stage == STAGE::TENTATIVE)
  {
    auto [first_stage_confirmation_M, first_stage_confirmation_N] =
        _manager->params()->state.stage_management.first_stage_confirmation_history_threshold;
    if (first_stage_confirmation_N == 0)
    {
      _stage = STAGE::PRELIMINARY;
    }
    else if (_meta_data._detection_hist.size() >= num_sensors * first_stage_confirmation_M &&
             first_stage_confirmation_N > 0)
    {
      if (std::count(_meta_data._detection_hist.end() -
                         std::min(_meta_data._detection_hist.size(), num_sensors * first_stage_confirmation_N),
                     _meta_data._detection_hist.end(),
                     true) >= static_cast<Index>(num_sensors * first_stage_confirmation_M))
      {
        _stage = STAGE::PRELIMINARY;
        _meta_data._detection_hist = {};
      }
    }
  }
  else if (_stage == STAGE::PRELIMINARY)
  {
    auto [second_stage_confirmation_M, second_stage_confirmation_N] =
        _manager->params()->state.stage_management.second_stage_confirmation_history_threshold;
    if (second_stage_confirmation_N == 0)
    {
      _stage = STAGE::CONFIRMED;
    }
    else if (_meta_data._detection_hist.size() >= num_sensors * second_stage_confirmation_M &&
             second_stage_confirmation_N > 0)
    {
      if (std::count(_meta_data._detection_hist.end() -
                         std::min(_meta_data._detection_hist.size(), num_sensors * second_stage_confirmation_N),
                     _meta_data._detection_hist.end(),
                     true) >= static_cast<Index>(num_sensors * second_stage_confirmation_M))
      {
        _stage = STAGE::CONFIRMED;
        _meta_data._detection_hist = {};
      }
    }
  }
}

template State<double> State<double>::transform(SE3Trafo const& trafo) const;
template State<AD> State<AD>::transform(SE3Trafo const& trafo) const;
template <class UDT>
State<UDT> State<UDT>::transform(SE3Trafo const& trafo) const
{
  State transformed = *this;
  std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> transformed_state;
  for (auto const& [model_id, dist] : _state_dist)
  {
    for (auto const& [weight, dist_comp] : dist.second.to_gaussian_components())
    {
      auto t = transformation::transform(
          dist_comp.mean(), dist_comp.covariance(), _manager->getStateModel(model_id).state_comps(), trafo);
      transformed_state[model_id].emplace_back(
          weight * dist.first, GaussianDistribution<UDT>(std::move(t.mean), std::move(t.cov), dist_comp.ref_point()));
    }
  }
  transformed._state_dist = transformed.build_consistent_state_dist(std::move(transformed_state));
  return transformed;
}

template <class UDT>
double kl_divergence(State<UDT> const& first, State<UDT> const& second)
{
  double w = 0;
  double kl_dist = 0;
  for (auto const& [model_id, dist] : first._state_dist)
  {
    if (auto const it = second._state_dist.find(model_id); it != second._state_dist.end())
    {
      if (dist.first > 0 and it->second.first > 0)
      {
        kl_dist += dist.first * kl_divergence(dist.second, it->second.second);
      }
      w += dist.first;
    }
  }
  kl_dist /= w;
  return kl_dist;
}

/// needed information's for the State posterior fusion
using EdgeId = std::tuple<std::size_t, DistributionId, StateModelId>;
struct FPMFusionInfos
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
FPMFusionInfos create_graph_fuse_posteriors_GKBest(State<UDT> const& prior, std::vector<State<UDT>> const& posteriors)
{
  ZoneScopedN("State::create_fpm_fusion_graph");
  std::vector<graph::Edge<EdgeId, std::size_t, double>> edges;  // edge.first ==0. prior
  // distribution id, else
  // posteriors(edge.first-1)
  // distribution id
  assert([&] {  // NOLINT
    if (posteriors.empty())
    {
      LOG_FATAL("received empty posteriors");
      LOG_FATAL(prior.toString());
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    return true;
  }());
  constexpr std::size_t start_node = 0;
  constexpr std::size_t end_node = 1;
  std::vector nodes{ start_node, end_node };  // nodes have no meaning
  std::size_t next_node = 2;
  for (auto const& [prior_model_id, prior_dist] : prior._state_dist)
  {
    for (auto const& [weight, prior_comp] : prior_dist.second.to_gaussian_components())
    {
      nodes.push_back(next_node);
      next_node++;
      edges.emplace_back(std::tuple{ std::size_t{ 0 }, prior_comp.id(), prior_model_id },
                         start_node,
                         next_node - 1,
                         -std::log(to_val(weight)) * (posteriors.size() - 1.0));
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
        if (auto const it = post._state_dist.find(prior_model_id); it != post._state_dist.end())
        {
          for (auto const& [post_weight, post_comp] : it->second.second.to_gaussian_components())
          {
            if (post_comp.prior_id() == prior_comp.id())
            {
              assert(post_weight > 0);
              edges.emplace_back(std::tuple{ static_cast<std::size_t>(post_ind + 1), post_comp.id(), prior_model_id },
                                 next_node - 1,
                                 edge_end_node,
                                 std::log(to_val(post_weight)));
              added_edge = true;
            }
          }
        }
        if (not added_edge)
        {
          edges.emplace_back(std::tuple{ static_cast<std::size_t>(post_ind + 1), NO_DISTRIBUTION, prior_model_id },
                             next_node - 1,
                             edge_end_node,
                             -std::numeric_limits<double>::infinity());
        }
        nodes.push_back(next_node);
        next_node++;
      }
    }
  }
  auto graph = graph::DiGraph(std::move(nodes), std::move(edges));
  return { ._graph = std::move(graph), ._startNode = start_node, ._endNode = end_node, ._k = 0 };
}

template <class UDT>
std::vector<FPMFusionInfos> create_graph_fuse_posteriors_KBest(State<UDT> const& prior,
                                                               std::vector<State<UDT>> const& posteriors,
                                                               std::size_t num_components,
                                                               PM_MULTI_SENSOR_UPDATE_METHOD method,
                                                               double equal_allocation_share_ratio)
{
  ZoneScopedN("State::create_fpm_fusion_graph");
  assert([&] {  // NOLINT
    if (posteriors.empty())
    {
      LOG_FATAL("received empty posteriors");
      LOG_FATAL(prior.toString());
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    return true;
  }());

  std::map<DistributionId, Index> componentID2Alloc;
  std::vector<DistributionId> component_ids;

  if (PM_MULTI_SENSOR_UPDATE_METHOD::K_BEST_POISSON == method)
  {
    std::size_t num_prior_comps = prior.bestState().second.to_gaussian_components().size();
    Eigen::ArrayXf weights(num_prior_comps);
    component_ids.resize(num_prior_comps);
    std::size_t counter = 0;
    for (const auto& [weight, prior_comp] : prior.bestState().second.to_gaussian_components())
    {
      component_ids.at(counter) = prior_comp.id();
      weights(counter) = to_val(weight);
      counter++;
    }

    if (num_prior_comps != 1)
    {
      Eigen::ArrayXf linspace = Eigen::ArrayXf::LinSpaced(num_prior_comps, 0, num_prior_comps - 1);
      Eigen::ArrayXf prod = weights * linspace;
      double lambda = prod.sum();
      Vector<UDT> poiss_probs = Vector<UDT>::Zero(num_prior_comps);
      UDT sum_poiss = 0.0;
      poiss_probs(0) = std::exp(-lambda);
      sum_poiss += poiss_probs(0);
      for (std::size_t n = 1; n <= num_prior_comps - 1; ++n)
      {
        poiss_probs(n) = lambda / (n)*poiss_probs(n - 1);
        sum_poiss += poiss_probs(n);
      }
      Vector<UDT> poiss_probs_norm = Vector<UDT>::Zero(num_prior_comps);
      for (std::size_t n = 0; n <= num_prior_comps - 1; ++n)
      {
        poiss_probs_norm(n) = poiss_probs(n) / sum_poiss;
      }
      Indices alloc = propAllocation(num_components, poiss_probs_norm, equal_allocation_share_ratio);
      const auto alloc_slots = alloc.sum();
      assert(alloc_slots <= num_components && "too much slots are allocated");
      if (static_cast<double>(alloc_slots) < num_components * 0.85)
      {
        LOG_WARN("Number of allocated slots is very small (" << alloc_slots << " / " << num_components
                                                             << " are allocated).");
      }
      std::size_t counter2 = 0;
      for (auto const& val : alloc)
      {
        componentID2Alloc.emplace(component_ids.at(counter2), val);
        counter2++;
      }
    }
    else
    {
      componentID2Alloc.emplace(component_ids.at(0), num_components);
    }
  }

  std::vector<FPMFusionInfos> graphs;
  for (auto const& [prior_model_id, prior_dist] : prior._state_dist)
  {
    for (auto const& [weight, prior_comp] : prior_dist.second.to_gaussian_components())
    {
      // create for every prior comp separate graph
      LOG_DEB("prior id: " << prior_comp.id());
      std::vector<graph::Edge<EdgeId, std::size_t, double>> edges;  // edge.first ==0. prior
      // distribution id, else
      // posteriors(edge.first-1)
      // distribution id
      constexpr std::size_t start_node = 0;
      constexpr std::size_t end_node = 1;
      std::vector nodes{ start_node, end_node };  // nodes have no meaning
      std::size_t next_node = 2;

      nodes.push_back(next_node);
      next_node++;
      edges.emplace_back(std::tuple{ std::size_t{ 0 }, prior_comp.id(), prior_model_id },
                         start_node,
                         next_node - 1,
                         -std::log(to_val(weight)) * (posteriors.size() - 1.0));
      const std::size_t k = [&] {
        if (method == PM_MULTI_SENSOR_UPDATE_METHOD::K_BEST_POISSON)
        {
          Vector<UDT> scaling_weights{ { 1 } };
          if (!componentID2Alloc.empty())
          {
            Indices slots =
                propAllocation(componentID2Alloc.at(prior_comp.id()), scaling_weights, equal_allocation_share_ratio);
            return static_cast<std::size_t>(slots(0));
          }
          return static_cast<std::size_t>(0);
        }
        LOG_DEB("weight: " << weight << " num_comps: " << num_components);
        return static_cast<std::size_t>(weight * num_components);
      }();
      LOG_DEB("results to k=" << k << " for id " << prior_comp.id());
      if (k == 0)
      {
        // calculation of graph is not necessary...
        FPMFusionInfos infos;
        infos._k = 0;
        graphs.emplace_back(std::move(infos));
        LOG_DEB("continue");
        continue;
      }
      LOG_DEB("calculate graph...");
      for (auto const& [post_ind, post] : std::views::enumerate(posteriors))
      {
        std::size_t const edge_end_node = [&] {
          if (post_ind + 1 < static_cast<Index>(posteriors.size()))
          {
            return next_node;
          }
          LOG_DEB("edge_end_node: " << end_node);
          return end_node;
        }();
        bool added_edge = false;
        if (auto const it = post._state_dist.find(prior_model_id); it != post._state_dist.end())
        {
          for (auto const& [post_weight, post_comp] : it->second.second.to_gaussian_components())
          {
            if (post_comp.prior_id() == prior_comp.id())
            {
              LOG_DEB("found fitting posterior: " << post_comp.id());
              LOG_DEB("emplace sn " << next_node - 1 << " eN " << edge_end_node << " weight " << post_weight);
              assert(post_weight > 0);
              edges.emplace_back(std::tuple{ static_cast<std::size_t>(post_ind + 1), post_comp.id(), prior_model_id },
                                 next_node - 1,
                                 edge_end_node,
                                 std::log(to_val(post_weight)));
              added_edge = true;
            }
          }
        }
        if (not added_edge)
        {
          edges.emplace_back(std::tuple{ static_cast<std::size_t>(post_ind + 1), NO_DISTRIBUTION, prior_model_id },
                             next_node - 1,
                             edge_end_node,
                             -std::numeric_limits<double>::infinity());
        }
        nodes.push_back(next_node);
        next_node++;
      }
      auto single_graph = graph::DiGraph(std::move(nodes), std::move(edges));
      FPMFusionInfos infos;
      infos._startNode = start_node;
      infos._endNode = end_node;
      infos._graph = std::move(single_graph);
      infos._k = k;
      graphs.emplace_back(std::move(infos));
    }
  }
  return graphs;
}

template State<double> merge(TTBManager<double> const& manager, std::vector<std::pair<double, StateId>> const& states);
template State<AD> merge(TTBManager<AD> const& manager, std::vector<std::pair<AD, StateId>> const& states);
template <class UDT>
State<UDT> merge(TTBManager<UDT> const& manager, std::vector<std::pair<UDT, StateId>> const& states)
{
  ZoneScopedN("State::merge");
  assert([&] {  // NOLINT
    if (not std::ranges::all_of(states, [&](std::pair<UDT, StateId> const& elem) {
          State<UDT> const& state = manager.get_state(elem.second);
          return state.isValid() and elem.first >= 0;
        }))
    {
      LOG_FATAL("Invalid State to merge");
      return false;
    }
    return true;
  }());
  UDT const merge_weight_sum = std::accumulate(
      states.begin(), states.end(), UDT{ 0.0 }, [](UDT old, auto const& elem) { return old + elem.first; });
  assert(to_val(merge_weight_sum) > 0);
  std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> merged_map;
  {
    ZoneScopedN("State::merge::build_merged_state");
    for (auto const& [merge_weight, state_id] : states)
    {
      State<UDT> const& state = manager.get_state(state_id);
      for (auto const& [model_id, weight_dist] : state._state_dist)
      {
        for (auto const& [mixture_weight, gauss] : weight_dist.second.to_gaussian_components())
        {
          merged_map[model_id].emplace_back(merge_weight * mixture_weight * weight_dist.first, gauss);
        }
      }
    }
  }
  State merged = [&] {
    ZoneScopedN("State::merge::create_state_dist");
    return State(&manager, std::move(merged_map));
  }();
  {
    ZoneScopedN("State::merge::metadata");
    for (auto const& [merge_weight, state_id] : states)
    {
      State<UDT> const& state = manager.get_state(state_id);
      for (auto const& [source_id, meass] : state._meta_data._meas_update_weight)
      {
        for (auto const& [meas_id, weight] : meass)
        {
          auto const it = std::ranges::find(
              merged._meta_data._meas_update_weight[source_id], meas_id, &std::pair<MeasurementId, UDT>::first);
          if (it != merged._meta_data._meas_update_weight.at(source_id).end())
          {
            it->second += merge_weight * weight / merge_weight_sum;
          }
          else
          {
            merged._meta_data._meas_update_weight.at(source_id).emplace_back(meas_id,
                                                                             merge_weight * weight / merge_weight_sum);
          }
        }
      }
    }
    merged._prior_id = manager.get_state(states.front().second)._prior_id;
    merged._label = manager.get_state(states.front().second)._label;
    for (auto const& [weight, state_id] : states)
    {
      State<UDT> const& state = manager.get_state(state_id);
      if (merged._prior_id == NO_PRIOR or merged._prior_id != state._prior_id)
      {
        merged._prior_id = NO_PRIOR;
      }
      if (merged._label != state._label)
      {
        merged._label = State<UDT>::_labelGenerator.getID();
      }
      merged._classification.merge(state._classification);
      merged._time = std::max(merged._time, state._time);
      merged._meta_data._timeOfLastAssociation =
          std::max(merged._meta_data._timeOfLastAssociation, state._meta_data._timeOfLastAssociation);
      merged._meta_data._durationSinceLastAssociation =
          std::min(merged._meta_data._durationSinceLastAssociation, state._meta_data._durationSinceLastAssociation);
      for (StateId prior_id : state._all_prior_ids | std::views::reverse | std::views::take(5))
      {
        if (std::ranges::find(merged._all_prior_ids, prior_id) == merged._all_prior_ids.end())
        {
          merged._all_prior_ids.push_back(prior_id);
        }
      }
      for (auto const& [source_id, nis_values] : state._meta_data._nis)
      {
        for (auto const& [model, values] : nis_values)
        {
          auto insert = values;
          for (auto& val : insert)
          {
            std::get<0>(val) *= weight / merge_weight_sum;
          }
          merged._meta_data._nis[source_id][model].insert(
              merged._meta_data._nis[source_id][model].end(), insert.begin(), insert.end());
        }
      }
      auto misc_copy = state._misc;
      merged._misc.merge(misc_copy);
      merged._meta_data._numPredictions = std::max(merged._meta_data._numPredictions, state._meta_data._numPredictions);
      merged._meta_data._numUpdates = std::max(merged._meta_data._numUpdates, state._meta_data._numUpdates);
      for (auto const& [source_id, log_likelihood] : state._meta_data._last_meas_log_likelihood)
      {
        merged._meta_data._last_meas_log_likelihood[source_id] += weight / merge_weight_sum * std::exp(log_likelihood);
      }
      for (auto const& [source_id, detectable] : state._meta_data._detectable)
      {
        merged._meta_data._detectable[source_id] = merged._meta_data._detectable[source_id] or detectable;
      }
    }
    for (auto& elem : merged._meta_data._last_meas_log_likelihood)
    {
      elem.second = std::log(elem.second);
    }
  }
  {
    ZoneScopedN("State::merge::post_process");
    merged.postProcess();
  }
  assert(merged.isEmpty() or merged.isValid());
  return merged;
}

template std::optional<State<double>> fuse_posteriors(State<double> const& prior,
                                                      std::vector<State<double>> const& posteriors,
                                                      std::size_t max_components,
                                                      PM_MULTI_SENSOR_UPDATE_METHOD method,
                                                      double equal_allocation_share_ratio,
                                                      bool is_pm_glmb);
template std::optional<State<AD>> fuse_posteriors(State<AD> const& prior,
                                                  std::vector<State<AD>> const& posteriors,
                                                  std::size_t max_components,
                                                  PM_MULTI_SENSOR_UPDATE_METHOD method,
                                                  double equal_allocation_share_ratio,
                                                  bool is_pm_glmb);
template <class UDT>
std::optional<State<UDT>> fuse_posteriors(State<UDT> const& prior,
                                          std::vector<State<UDT>> const& posteriors,
                                          std::size_t max_components,
                                          PM_MULTI_SENSOR_UPDATE_METHOD method,
                                          double equal_allocation_share_ratio,
                                          bool is_pm_glmb)
{
  ZoneScopedN("State::fuse_posteriors");
  assert([&] {  // NOLINT
    if (not prior.isValid())
    {
      LOG_FATAL("pior track s is invalid");
      return false;
    }
    for (State<UDT> const& post : posteriors)
    {
      if (not post.isValid())
      {
        LOG_FATAL("post is invalid:");
        LOG_FATAL(post.toString());
        return false;
      }
    }
    std::vector<SourceId> seen_ids;
    for (State<UDT> const& state : posteriors)
    {
      if (state._meta_data._last_meas_log_likelihood.size() != 1 or state._meta_data._meas_update_weight.size() != 1)
      {
        LOG_FATAL("more than 1 last associated measurement in posterior");
        return false;
      }
      if (std::ranges::find_if(seen_ids, [&](SourceId const& id) {
            return id == state._meta_data._meas_update_weight.begin()->first;
          }) != seen_ids.end())
      {
        LOG_FATAL(std::format("Same data source container {} used by multiple posteriors",
                              state._meta_data._meas_update_weight.begin()->first));
        LOG_FATAL(prior.toString());
        for (State<UDT> const& post : posteriors)
        {
          LOG_FATAL(post.toString());
        }
        return false;
      }
      seen_ids.push_back(state._meta_data._meas_update_weight.begin()->first);
    }
    if (not std::ranges::all_of(posteriors, [&](State<UDT> const& state) { return state._prior_id == prior._id; }))
    {
      LOG_FATAL("Posteriors have different prior_id compared to the id of the prior");
      return false;
    }
    if (not std::ranges::all_of(posteriors, [&](State<UDT> const& state) { return state._label == prior._label; }))
    {
      LOG_FATAL("Posteriors have different labels compared to given prior");
      return false;
    }
    if (is_pm_glmb)
    {
      for (auto const& [prior_model_id, prior_dist] : prior._state_dist)
      {
        for (auto const& [post_ind, post] : std::views::enumerate(posteriors))
        {
          if (post._state_dist.find(prior_model_id)->second.second.to_gaussian_components().size() !=
              prior_dist.second.to_gaussian_components().size())
          {
            LOG_FATAL("Number of prior gaussian components in PM-GLMB fusion must be equal to number of posterior "
                      "gaussian components of each sensor!! ");
            LOG_ERR("number prior: " << prior_dist.second.to_gaussian_components().size());
            LOG_ERR("number post: " << post._state_dist.find(prior_model_id)
                                           ->second.second.to_gaussian_components()
                                           .size());
            LOG_FATAL(prior.toString());
            LOG_FATAL(post._state_dist.find(prior_model_id)->second.second.to_string("post:"));
            return false;
          }
        }
      }
    }
    return true;
  }());

  // Fusion
  LOG_DEB("Fuse states");
  UDT weightSum = 0;
  std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> fusedComponents;
  UDT eta = 0.0;
  if (method == PM_MULTI_SENSOR_UPDATE_METHOD::GENERALIZED_K_BEST)
  {
    // Use GeneralizedKBestSelection algorithm
    FPMFusionInfos fusionInfos = create_graph_fuse_posteriors_GKBest(prior, posteriors);
    std::vector<std::pair<std::vector<EdgeId>, double>> solutions =
        fusionInfos._graph.k_longest_paths(fusionInfos._startNode, fusionInfos._endNode, max_components);

    // calculate fused result of each selection
    if (solutions.empty())
    {
      // This case happens when not all posteriors have a value for one prior component! (Fused mixture is empty
      // due to pruning in local updates) This happens in very complex scenarios like bearing only! Todo: Make
      // sure there is at least one common component! (Must be probably fixed in innovation part...) this fused
      // component does not exist which is worse....
      LOG_WARN("Fused mixture is empty due to pruning in posteriors");
      return std::nullopt;
    }
    for (auto const& [edges, weight_log] : solutions)
    {
      UDT prior_weight = 0.0;
      if (weight_log == -std::numeric_limits<double>::infinity())
      {
        // solution with weight 0 is possible if one prior component is not detected by one sensor. Then no fused result
        // exists in this case
        continue;
      }
      assert(std::get<0>(edges.front()) == 0);  // first comes prior;
      Matrix<UDT> merged_precision;
      Vector<UDT> merged_mean_inv;
      UDT ny_product = 1.0;
      StateModelId const state_model_id = std::get<2>(edges.front());
      type_safe::deferred_construction<DistributionId> prior_id;
      for (auto const& [ind, distribution_id, model_id] : edges)
      {
        assert(model_id == state_model_id);
        if (ind == 0)
        {
          auto prior_dists = prior._state_dist.at(state_model_id).second.to_gaussian_components();
          auto const prior_it = std::ranges::find_if(
              prior_dists, [distribution_id](auto const& dist) { return dist.second.id() == distribution_id; });
          assert(prior_it != prior_dists.end());
          prior_id = prior_it->second.id();
          prior_weight = prior_it->first;
          merged_precision = (1.0 - posteriors.size()) * prior_it->second.covariance().inverse();
          merged_mean_inv =
              (1.0 - posteriors.size()) * prior_it->second.covariance().inverse() * prior_it->second.mean();
        }
        else
        {
          assert(model_id == state_model_id);
          auto post_dists = posteriors.at(ind - 1)._state_dist.at(state_model_id).second.to_gaussian_components();
          auto const post_it = std::ranges::find_if(
              post_dists, [distribution_id](auto const& dist) { return dist.second.id() == distribution_id; });
          assert(post_it != post_dists.end());
          assert([&] {  // NOLINT
            if (post_it->second.prior_id() != prior_id.value())
            {
              LOG_FATAL(std::format("component has different prior id {} than the prior component {}",
                                    post_it->second.prior_id().value_,
                                    prior_id.value().value_));
              LOG_FATAL(prior.toString());
              LOG_FATAL(posteriors.at(ind - 1).toString());
              return false;
            }
            return true;
          }());
          merged_precision += post_it->second.covariance().inverse();
          merged_mean_inv += post_it->second.covariance().inverse() * post_it->second.mean();
          if (!post_it->second.misc().contains("ny"))
          {
            LOG_WARN("Does not contain value for ny!! is needed for pm_lmb fusion. Now ny=1 is used!");
          }
          else
          {
            ny_product *= std::any_cast<UDT>(post_it->second.misc().at("ny"));
          }
        }
      }
      Matrix<UDT> fusedCov = merged_precision.inverse();
      Vector<UDT> fusedMean = fusedCov * merged_mean_inv;
      UDT norm_const = 1.0;  // Vercauteren approximation! Otherwise, this value has to be calculated!
      assert([&] {           // NOLINT
        if (prior_weight == 0.0)
        {
          LOG_FATAL("No prior gaussian component given!");
          return false;
        }
        return true;
      }());
      eta += norm_const * ny_product * prior_weight;  // see eq (11) in https://ieeexplore.ieee.org/document/10224121
      weightSum += std::exp(weight_log);
      GaussianDistribution fused_dist(std::move(fusedMean), std::move(fusedCov));
      assert([&] {  // NOLINT
        if (not fused_dist.is_valid())
        {
          LOG_FATAL("fused Gaussian Distribution is not valid");
          LOG_FATAL(fused_dist.to_string());
          LOG_FATAL("solution:");
          for (auto const& [ind_, distribution_id_, model_id_] : edges)
          {
            LOG_FATAL(std::format(
                "Edge: Ind {}, distribution id {}, model id {}", ind_, distribution_id_.value_, model_id_.value_));
          }
          LOG_FATAL("prior_state");
          LOG_FATAL(prior.toString());
          LOG_FATAL("posterior");
          for (State<UDT> const& post : posteriors)
          {
            LOG_FATAL(post.toString());
          }
          return false;
        }
        return true;
      }());
      fusedComponents[state_model_id].emplace_back(std::exp(weight_log), std::move(fused_dist));
    }
  }
  else
  {
    // use kBestSelection algorithm
    std::vector<FPMFusionInfos> fusionInfosVec =
        create_graph_fuse_posteriors_KBest(prior, posteriors, max_components, method, equal_allocation_share_ratio);
    if (fusionInfosVec.empty())
    {
      LOG_WARN("No Fusion result for prior " << prior.toString()
                                             << " since no prior mixture component is given by all sensors!");
      return std::nullopt;
    }
    LOG_DEB("calculated " << fusionInfosVec.size() << " graphs!");
    for (auto const& fusionInfos : fusionInfosVec)
    {
      LOG_DEB("calculate " << fusionInfos._k << " solutions");
      if (fusionInfos._k == 0)
      {
        continue;
      }
      std::vector<std::pair<std::vector<EdgeId>, double>> solutions =
          fusionInfos._graph.k_longest_paths(fusionInfos._startNode, fusionInfos._endNode, fusionInfos._k);
      // calculate fused result of each selection
      if (solutions.empty())
      {
        // There is no fused result for this label, but there should be one!
        LOG_DEB("no solution found for one prior component: " << prior.toString());
        continue;
      }
      LOG_DEB("fusion graph with sN " << fusionInfos._startNode << " eN " << fusionInfos._endNode
                                      << " graph: " << fusionInfos._graph.toString());
      for (auto const& [edges, weight_log] : solutions)
      {
        UDT prior_weight = 0.0;
        if (weight_log == -std::numeric_limits<double>::infinity())
        {
          // solution with weight 0 is possible if one prior component is not detected by one sensor. Then no fused
          // result exists in this case
          continue;
        }
        assert(std::get<0>(edges.front()) == 0);  // first comes prior;
        Matrix<UDT> merged_precision;
        Vector<UDT> merged_mean_inv;
        UDT ny_product = 1.0;
        StateModelId const state_model_id = std::get<2>(edges.front());
        type_safe::deferred_construction<DistributionId> prior_id;
        for (auto const& [ind, distribution_id, model_id] : edges)
        {
          assert(model_id == state_model_id);
          if (ind == 0)
          {
            auto prior_dists = prior._state_dist.at(state_model_id).second.to_gaussian_components();
            auto const prior_it = std::ranges::find_if(
                prior_dists, [distribution_id](auto const& dist) { return dist.second.id() == distribution_id; });
            assert(prior_it != prior_dists.end());
            prior_id = prior_it->second.id();
            prior_weight = prior_it->first;
            LOG_DEB("calc fused result for prior id " << prior_id.value());
            merged_precision = (1.0 - posteriors.size()) * prior_it->second.covariance().inverse();
            merged_mean_inv =
                (1.0 - posteriors.size()) * prior_it->second.covariance().inverse() * prior_it->second.mean();
          }
          else
          {
            assert(model_id == state_model_id);
            auto post_dists = posteriors.at(ind - 1)._state_dist.at(state_model_id).second.to_gaussian_components();
            auto const post_it = std::ranges::find_if(
                post_dists, [distribution_id](auto const& dist) { return dist.second.id() == distribution_id; });
            assert(post_it != post_dists.end());
            assert([&] {  // NOLINT
              if (post_it->second.prior_id() != prior_id.value())
              {
                LOG_FATAL(std::format("component has different prior id {} than the prior component {}",
                                      post_it->second.prior_id().value_,
                                      prior_id.value().value_));
                LOG_FATAL(prior.toString());
                LOG_FATAL(posteriors.at(ind - 1).toString());
                return false;
              }
              return true;
            }());
            merged_precision += post_it->second.covariance().inverse();
            merged_mean_inv += post_it->second.covariance().inverse() * post_it->second.mean();
            if (!post_it->second.misc().contains("ny"))
            {
              LOG_WARN("Does not contain value for ny!! is needed for pm_lmb fusion. Now ny=1 is used!");
            }
            else
            {
              ny_product *= std::any_cast<UDT>(post_it->second.misc().at("ny"));
            }
          }
        }
        Matrix<UDT> fusedCov = merged_precision.inverse();
        Vector<UDT> fusedMean = fusedCov * merged_mean_inv;
        weightSum += exp(weight_log);
        UDT norm_const = 1.0;  // Vercauteren approximation! Otherwise, this value has to be calculated!
        assert([&] {           // NOLINT
          if (prior_weight == 0.0)
          {
            LOG_FATAL("No prior gaussian component given!");
            return false;
          }
          return true;
        }());
        eta += norm_const * ny_product * prior_weight;  // see eq (11) in https://ieeexplore.ieee.org/document/10224121
        GaussianDistribution fused_dist(std::move(fusedMean), std::move(fusedCov));
        assert([&] {  // NOLINT
          if (not fused_dist.is_valid())
          {
            LOG_FATAL("fused Gaussian Distribution is not valid");
            LOG_FATAL(fused_dist.to_string());
            LOG_FATAL("solution:");
            for (auto const& [ind_, distribution_id_, model_id_] : edges)
            {
              LOG_FATAL(std::format(
                  "Edge: Ind {}, distribution id {}, model id {}", ind_, distribution_id_.value_, model_id_.value_));
            }
            LOG_FATAL("prior_state");
            LOG_FATAL(prior.toString());
            LOG_FATAL("posterior");
            for (State<UDT> const& post : posteriors)
            {
              LOG_FATAL(post.toString());
            }
            return false;
          }
          return true;
        }());
        LOG_DEB("emplace fused dist for prior id " << prior_id.value() << " with weight_log " << weight_log);
        fusedComponents[state_model_id].emplace_back(std::exp(weight_log), std::move(fused_dist));
      }
    }
  }

  if (weightSum == 0)
  {
    LOG_WARN("Got solution with weight sum 0, because posteriors do not share at least one component with common prior "
             "id! Currently no good solution is implemented...");
    return std::nullopt;
  }
  for (const auto& [model_id, comps] : fusedComponents)
  {
    for (const auto& comp : comps)
    {
      LOG_DEB("fused comp: " << comp.second.to_string());
    }
  }
  State fused(prior._manager, std::move(fusedComponents));
  fused._misc["dist_weight_sum"] = weightSum;
  fused._misc["eta"] = eta;
  fused._label = prior._label;
  fused._type = STATE_TYPE::POSTERIOR;
  fused._prior_id = prior._id;
  fused._time = posteriors.front()._time;
  fused._meta_data._numPredictions = prior._meta_data._numPredictions;
  fused._meta_data._numUpdates = prior._meta_data._numUpdates + posteriors.size();
  fused._meta_data._last_meas_log_likelihood.clear();
  fused._meta_data._meas_update_weight.clear();
  fused._meta_data._detectable.clear();
  fused._all_prior_ids = prior._all_prior_ids;
  for (State<UDT> const& state : posteriors)
  {
    for (auto const& it : state._meta_data._last_meas_log_likelihood)
    {
      assert(not fused._meta_data._last_meas_log_likelihood.contains(it.first));
      fused._meta_data._last_meas_log_likelihood.insert(it);
    }
    for (auto const& it : state._meta_data._meas_update_weight)
    {
      assert(not fused._meta_data._meas_update_weight.contains(it.first));
      fused._meta_data._meas_update_weight.insert(it);
    }
    for (auto const& it : state._meta_data._nis)
    {
      assert(not fused._meta_data._nis.contains(it.first));
      fused._meta_data._nis.insert(it);
    }
    for (auto const& it : state._meta_data._detectable)
    {
      assert([&] {  // NOLINT
        if (fused._meta_data._detectable.contains(it.first))
        {
          LOG_FATAL("Fused state already contained detectable info from meas source: " << it.first);
          LOG_FATAL(fused.toString());
          LOG_FATAL(state.toString());
          LOG_FATAL(std::stacktrace::current());
          return false;
        }
        return true;
      }());
      fused._meta_data._detectable.insert(it);
    }
    fused._meta_data._timeOfLastAssociation =
        std::max(fused._meta_data._timeOfLastAssociation, state._meta_data._timeOfLastAssociation);
    fused._meta_data._durationSinceLastAssociation =
        std::min(fused._meta_data._durationSinceLastAssociation, state._meta_data._durationSinceLastAssociation);
  }
  fused.postProcess();
  assert([&] {  // NOLINT
    if (not fused.isValid())
    {
      LOG_FATAL("fused track is not valid");
      LOG_FATAL(fused.toString());
      LOG_FATAL("prior");
      LOG_FATAL(prior.toString());
      LOG_FATAL("posteriors");
      for (State<UDT> const& post : posteriors)
      {
        LOG_FATAL(post.toString());
      }
      return false;
    }
    return true;
  }());
  assert(fused._meta_data._last_meas_log_likelihood.size() == posteriors.size());
  assert(fused._meta_data._meas_update_weight.size() == posteriors.size());
  assert(fused._meta_data._detectable.size() == posteriors.size());
  return { std::move(fused) };
}

template std::optional<State<double>> fuse_posteriors_pu(State<double> const& prior,
                                                         std::vector<State<double>>& posteriors,
                                                         bool use_best_component,
                                                         std::size_t num_components);
template std::optional<State<AD>> fuse_posteriors_pu(State<AD> const& prior,
                                                     std::vector<State<AD>>& posteriors,
                                                     bool use_best_component,
                                                     std::size_t num_components);
template <class UDT>
std::optional<State<UDT>> fuse_posteriors_pu(State<UDT> const& prior,
                                             std::vector<State<UDT>>& posteriors,
                                             bool use_best_component,
                                             std::size_t num_components)
{
  ZoneScopedN("State::fuse_posteriors");
  assert([&] {  // NOLINT
    if (not prior.isValid())
    {
      LOG_FATAL("pior track s is invalid");
      return false;
    }
    for (State<UDT> const& post : posteriors)
    {
      if (not post.isValid())
      {
        LOG_FATAL("post is invalid:");
        LOG_FATAL(post.toString());
        return false;
      }
    }
    std::vector<SourceId> seen_ids;
    for (State<UDT> const& state : posteriors)
    {
      if (state._meta_data._last_meas_log_likelihood.size() != 1 or state._meta_data._meas_update_weight.size() != 1)
      {
        LOG_FATAL("more than 1 last associated measurement in posterior");
        return false;
      }
      if (std::ranges::find_if(seen_ids, [&](SourceId const& id) {
            return id == state._meta_data._meas_update_weight.begin()->first;
          }) != seen_ids.end())
      {
        LOG_FATAL(std::format("Same data source container {} used by multiple posteriors",
                              state._meta_data._meas_update_weight.begin()->first));
        LOG_FATAL(prior.toString());
        for (State<UDT> const& post : posteriors)
        {
          LOG_FATAL(post.toString());
        }
        return false;
      }
      seen_ids.push_back(state._meta_data._meas_update_weight.begin()->first);
    }
    if (not std::ranges::all_of(posteriors, [&](State<UDT> const& state) { return state._label == prior._label; }))
    {
      LOG_FATAL("Posteriors have different labels compared to given prior");
      return false;
    }
    return true;
  }());

  // Fusion
  LOG_DEB("Fuse states");
  struct fused_infos
  {
    GaussianDistribution<UDT> _fused_dist;
    double _weight_log;
  };
  std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> fusedComponents;
  double eta = 0.0;
  for (auto const& [prior_model_id, prior_dist] : prior._state_dist)
  {
    assert(prior_dist.second.to_gaussian_components().size() == 1 && "prior dist does have more than one component, "
                                                                     "fused result can not be calculated with "
                                                                     "fuse_posteriors_pu!");
    std::vector<fused_infos> fusion_results;
    eta = 0.0;
    for (auto const& [prior_weight, prior_comp] : prior_dist.second.to_gaussian_components())
    {
      // create graph
      LOG_DEB("prior id: " << prior_comp.id());
      std::vector<graph::Edge<EdgeId, std::size_t, double>> edges;  // edge.first ==0. prior
      // distribution id, else
      // posteriors(edge.first-1)
      // distribution id
      constexpr std::size_t start_node = 0;
      constexpr std::size_t end_node = 1;
      std::vector nodes{ start_node, end_node };  // nodes have no meaning
      std::size_t next_node = 2;

      nodes.push_back(next_node);
      next_node++;
      edges.emplace_back(std::tuple{ std::size_t{ 0 }, prior_comp.id(), prior_model_id },
                         start_node,
                         next_node - 1,
                         -std::log(to_val(prior_weight)) * (posteriors.size() - 1.0));
      for (auto const& [post_ind, post] : std::views::enumerate(posteriors))
      {
        std::size_t const edge_end_node = [&] {
          if (post_ind + 1 < static_cast<Index>(posteriors.size()))
          {
            return next_node;
          }
          LOG_DEB("edge_end_node: " << end_node);
          return end_node;
        }();
        bool added_edge = false;
        if (auto const it = post._state_dist.find(prior_model_id); it != post._state_dist.end())
        {
          for (auto const& [post_weight, post_comp] : it->second.second.to_gaussian_components())
          {
            LOG_DEB("emplace sn " << next_node - 1 << " eN " << edge_end_node << " weight " << post_weight);
            assert(post_weight > 0);
            edges.emplace_back(std::tuple{ static_cast<std::size_t>(post_ind + 1), post_comp.id(), prior_model_id },
                               next_node - 1,
                               edge_end_node,
                               std::log(to_val(post_weight)));
            added_edge = true;
          }
        }
        if (not added_edge)
        {
          edges.emplace_back(std::tuple{ static_cast<std::size_t>(post_ind + 1), NO_DISTRIBUTION, prior_model_id },
                             next_node - 1,
                             edge_end_node,
                             -std::numeric_limits<double>::infinity());
        }
        nodes.push_back(next_node);
        next_node++;
      }
      auto graph = graph::DiGraph(std::move(nodes), std::move(edges));

      std::vector<std::pair<std::vector<EdgeId>, double>> solutions =
          graph.k_longest_paths(start_node, end_node, num_components);
      // calculate fused result of each selection
      if (solutions.empty())
      {
        // There is no fused result for this label, but there should be one!
        LOG_FATAL("no solution found for prior component: " << prior.toString());
        LOG_FATAL("This case should not happen! Check!");
        return std::nullopt;
      }

      // calculate fused stuff!
      for (auto const& [sol_edges, weight_log] : solutions)
      {
        Matrix<UDT> merged_precision;
        Vector<UDT> merged_mean_inv;
        type_safe::deferred_construction<double> g_merged_log;
        assert(weight_log != -std::numeric_limits<double>::infinity() && "result with weight 0 should not happen!");
        if (weight_log == -std::numeric_limits<double>::infinity())
        {
          LOG_FATAL("result with weight 0 should not happen! Check this!");
          continue;
        }
        assert(std::get<0>(sol_edges.front()) == 0);  // first comes prior;
        for (auto const& [ind, distribution_id, model_id] : sol_edges)
        {
          if (ind == 0)
          {
            // prior values are already used...
            merged_precision = (1.0 - posteriors.size()) * prior_comp.covariance().inverse();
            merged_mean_inv = (1.0 - posteriors.size()) * prior_comp.covariance().inverse() * prior_comp.mean();
            g_merged_log =
                to_val((1.0 - posteriors.size()) *
                       (-0.5 * prior_comp.mean().transpose() * prior_comp.covariance().inverse() * prior_comp.mean() -
                        0.5 * std::log((2 * std::numbers::pi * prior_comp.covariance()).determinant())));
          }
          else
          {
            assert(model_id == prior_model_id);
            auto post_dists = posteriors.at(ind - 1)._state_dist.at(prior_model_id).second.to_gaussian_components();
            auto const post_it = std::ranges::find_if(
                post_dists, [distribution_id](auto const& dist) { return dist.second.id() == distribution_id; });
            assert(post_it != post_dists.end());
            merged_precision += post_it->second.covariance().inverse();
            Vector<UDT> hC = post_it->second.covariance().inverse() * post_it->second.mean();
            merged_mean_inv += hC;
            g_merged_log.value() +=
                to_val(-0.5 * post_it->second.mean().transpose() * hC -
                       0.5 * std::log((2 * std::numbers::pi * post_it->second.covariance()).determinant()) +
                       std::log(post_it->first));
          }
        }
        Matrix<UDT> fusedCov = merged_precision.inverse();
        Vector<UDT> fusedMean = fusedCov * merged_mean_inv;
        g_merged_log.value() += to_val(0.5 * fusedMean.transpose() * merged_precision * fusedMean +
                                       0.5 * std::log((2 * std::numbers::pi * fusedCov.determinant())));
        GaussianDistribution fused_dist(std::move(fusedMean), std::move(fusedCov));

        eta += std::exp(g_merged_log.value());
        assert([&] {  // NOLINT
          if (not fused_dist.is_valid())
          {
            LOG_FATAL("fused Gaussian Distribution is not valid");
            LOG_FATAL(fused_dist.to_string());
            LOG_FATAL("prior_state");
            LOG_FATAL(prior.toString());
            LOG_FATAL("posterior");
            for (State<UDT> const& post : posteriors)
            {
              LOG_FATAL(post.toString());
            }
            return false;
          }
          return true;
        }());
        fusion_results.emplace_back(std::move(fused_dist), std::exp(g_merged_log.value()));
      }
    }
    double max_g =
        std::max_element(fusion_results.begin(), fusion_results.end(), [](const fused_infos& a, const fused_infos& b) {
          return a._weight_log < b._weight_log;
        })->_weight_log;
    for (auto const& fused : fusion_results)
    {
      double fused_weight = std::exp(fused._weight_log - max_g);
      fusedComponents[prior_model_id].emplace_back(fused_weight, std::move(fused._fused_dist));
    }
  }

  State fused(prior._manager, std::move(fusedComponents));
  fused._misc["eta"] = eta;
  fused._label = prior._label;
  fused._type = STATE_TYPE::POSTERIOR;
  fused._prior_id = prior._id;
  fused._time = posteriors.front()._time;
  fused._meta_data._numPredictions = prior._meta_data._numPredictions;
  fused._meta_data._numUpdates = prior._meta_data._numUpdates + posteriors.size();
  fused._meta_data._last_meas_log_likelihood.clear();
  fused._meta_data._meas_update_weight.clear();
  fused._meta_data._detectable.clear();
  fused._all_prior_ids = prior._all_prior_ids;
  for (State<UDT> const& state : posteriors)
  {
    for (auto const& it : state._meta_data._last_meas_log_likelihood)
    {
      assert(not fused._meta_data._last_meas_log_likelihood.contains(it.first));
      fused._meta_data._last_meas_log_likelihood.insert(it);
    }
    for (auto const& it : state._meta_data._meas_update_weight)
    {
      assert(not fused._meta_data._meas_update_weight.contains(it.first));
      fused._meta_data._meas_update_weight.insert(it);
    }
    for (auto const& it : state._meta_data._nis)
    {
      assert(not fused._meta_data._nis.contains(it.first));
      fused._meta_data._nis.insert(it);
    }
    for (auto const& it : state._meta_data._detectable)
    {
      assert([&] {  // NOLINT
        if (fused._meta_data._detectable.contains(it.first))
        {
          LOG_FATAL("Fused state already contained detectable info from meas source: " << it.first);
          LOG_FATAL(fused.toString());
          LOG_FATAL(state.toString());
          LOG_FATAL(std::stacktrace::current());
          return false;
        }
        return true;
      }());
      fused._meta_data._detectable.insert(it);
    }
    fused._meta_data._timeOfLastAssociation =
        std::max(fused._meta_data._timeOfLastAssociation, state._meta_data._timeOfLastAssociation);
    fused._meta_data._durationSinceLastAssociation =
        std::min(fused._meta_data._durationSinceLastAssociation, state._meta_data._durationSinceLastAssociation);
  }
  if (use_best_component)
  {
    for (auto& [_, fused_dist] : fused._state_dist | std::views::values)
    {
      fused_dist.truncate(1);
    }
    fused._all_prior_ids.push_back(fused._id);
    fused._id = fused._idGenerator.getID();
  }
  else
  {
    fused.approximateToOneComponent();
  }

  assert([&] {  // NOLINT
    if (not fused.isValid())
    {
      LOG_FATAL("fused track is not valid");
      LOG_FATAL(fused.toString());
      LOG_FATAL("prior");
      LOG_FATAL(prior.toString());
      LOG_FATAL("posteriors");
      for (State<UDT> const& post : posteriors)
      {
        LOG_FATAL(post.toString());
      }
      return false;
    }
    return true;
  }());
  assert(fused._meta_data._last_meas_log_likelihood.size() == posteriors.size());
  assert(fused._meta_data._meas_update_weight.size() == posteriors.size());
  assert(fused._meta_data._detectable.size() == posteriors.size());
  return { std::move(fused) };
}

template std::optional<State<double>>
create_meas_state(TTBManager<double>* manager,
                  std::vector<std::pair<double, GaussianDistribution<double>>> dist,
                  Time time,
                  Components const& meas_comps);
template std::optional<State<AD>> create_meas_state(TTBManager<AD>* manager,
                                                    std::vector<std::pair<AD, GaussianDistribution<AD>>> dist,
                                                    Time time,
                                                    Components const& meas_comps);
template <class UDT>
std::optional<State<UDT>> create_meas_state(TTBManager<UDT>* manager,
                                            std::vector<std::pair<UDT, GaussianDistribution<UDT>>> dist,
                                            Time time,
                                            Components const& meas_comps)
{
  StateModelId const state_model_id = manager->state_model_id(meas_comps);
  State state(manager, [&] {
    std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> birth_dist;
    birth_dist[state_model_id] = std::move(dist);
    return birth_dist;
  }());
  state._type = STATE_TYPE::MEASUREMENT;
  state._time = time;
  if (not state.isValid())
  {
    LOG_ERR("Created state out of measurement is not valid");
    return {};
  }
  return { std::move(state) };
}

template std::optional<State<double>>
create_meas_state(TTBManager<double>* manager, Distribution<double> dist, Time time, Components const& meas_comps);
template std::optional<State<AD>>
create_meas_state(TTBManager<AD>* manager, Distribution<AD> dist, Time time, Components const& meas_comps);
template <class UDT>
std::optional<State<UDT>>
create_meas_state(TTBManager<UDT>* manager, Distribution<UDT> dist, Time time, Components const& meas_comps)
{
  StateModelId const state_model_id = manager->state_model_id(meas_comps);
  State state(manager, [&] {
    std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> birth_dist;
    for (auto& [weight, gauss] : dist.to_gaussian_components())
    {
      birth_dist[state_model_id].emplace_back(weight, std::move(gauss));
    }
    return birth_dist;
  }());
  state._type = STATE_TYPE::MEASUREMENT;
  state._time = time;
  if (not state.isValid())
  {
    LOG_ERR("Created state out of measurement is not valid");
    return {};
  }
  return { std::move(state) };
}

template std::optional<State<double>> create_meas_state(TTBManager<double>* manager,
                                                        GaussianDistribution<double> const& gauss_dist,
                                                        Time time,
                                                        Components const& meas_comps);
template std::optional<State<AD>> create_meas_state(TTBManager<AD>* manager,
                                                    GaussianDistribution<AD> const& gauss_dist,
                                                    Time time,
                                                    Components const& meas_comps);
template <class UDT>
std::optional<State<UDT>> create_meas_state(TTBManager<UDT>* manager,
                                            GaussianDistribution<UDT> const& gauss_dist,
                                            Time time,
                                            Components const& meas_comps)
{
  return create_meas_state(manager, Distribution<UDT>(gauss_dist), time, meas_comps);
}

template <class UDT>
UDT State<UDT>::_get_weight() const
{
  return std::exp(_log_weight);
}

}  // namespace ttb
