#include "tracking_lib/States/State.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Distributions/BaseDistribution.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/PersistenceModels/BasePersistenceModel.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/States//Innovation.h"
#include "tracking_lib/Transformations/Transformation.h"
#include "tracking_lib/Distributions/MixtureDistribution.h"

#include <boost/geometry/util/range.hpp>
#include <tracy/tracy/Tracy.hpp>

namespace ttb
{

constexpr auto tracy_color = tracy::Color::LightBlue;

IDGenerator<Label> State::_labelGenerator{};
IDGenerator<StateId> State::_idGenerator{};

State::State(TTBManager* manager, std::map<StateModelId, std::unique_ptr<BaseDistribution>> dist)
  : _manager{ manager }, _state_dist{ std::move(dist) }, _classification{ _manager }
{
}

State::State(State const& other)
  : _manager{ other._manager }
  , _id{ other._id }
  , _label{ other._label }
  , _state_dist{ [&]() {
    std::map<StateModelId, std::unique_ptr<BaseDistribution>> out;
    for (auto const& [id, bsd] : other._state_dist)
    {
      out.emplace(id, bsd->clone());
    }
    return out;
  }() }
  , _innovation{ other._innovation }
  , _classification{ other._classification }
  , _existenceProbability{ other._existenceProbability }
  , _survival_probability{ other._survival_probability }
  , _stage{ other._stage }
  , _score{ other._score }
  , _time{ other._time }
  , _meta_data{ other._meta_data }
  , _weight_mis_detection{ other._weight_mis_detection }
  , _nis{ other._nis }
  , _detectable{ other._detectable }
  , _misc{ other._misc }
{
}

State& State::operator=(State const& other)
{
  State tmp(other);
  *this = std::move(tmp);
  return *this;
}

State::~State() = default;

State::State(State&& other) noexcept = default;

State& State::operator=(State&& other) noexcept = default;

bool State::isValid() const
{
  if (isEmpty())
  {
    LOG_ERR("State is empty!");
    return false;
  }
  for (auto const& [model, dist] : _state_dist)
  {
    if (not dist->isValid())
    {
      LOG_ERR("Invalid Distribution for state model " + _manager->getStateModelMap().at(model)->toString());
      return false;
    }
  }
  if (std::abs(sumWeights() - 1) > 1e-5)
  {
    LOG_ERR("Weight of StateDist not near 1: " + std::to_string(sumWeights()) + "\n" + toString());
    return false;
  }
  if (_existenceProbability > 1 or _existenceProbability < 0)
  {
    LOG_ERR("Existence Prob not valid: " + std::to_string(_existenceProbability));
    return false;
  }
  return true;
}

bool State::isEmpty() const
{
  return std::ranges::all_of(_state_dist, [](auto const& elem) { return elem.second->dists().empty(); });
}

void State::predict(Duration deltaT, EgoMotionDistribution const& ego)
{
  ZoneScopedNC("State::predict", tracy_color);
  if (deltaT < 0ms)
  {
    LOG_WARN_THROTTLE(1, "Prediction into past is not supported");
    return;
  }
  LOG_DEB("Predict State Distribution with " << _state_dist.size() << " State Models");
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL(toString("Invalid State: "));
      return false;
    }
    return true;
  }());
  if (_manager->params().state.multi_model.enable_markov_transition)
  {
    LOG_DEB("Perform the markov state transition");
    markovTransition();
  }
  for (auto& [stateModelId, dist] : _state_dist)
  {
    BaseStateModel const& sm = _manager->getStateModel(stateModelId);
    for (BaseDistribution* dist_mixture : dist->dists())
    {
      assert(dist_mixture->type() == DISTRIBUTION_TYPE::GAUSSIAN);
      if (dist_mixture->refPoint() != REFERENCE_POINT::CENTER)
      {
        auto trafo = transformation::transform(dist_mixture->mean(),
                                               dist_mixture->covariance(),
                                               sm.state_comps(),
                                               dist_mixture->refPoint(),
                                               REFERENCE_POINT::CENTER);
        dist_mixture->set(std::move(trafo.value().mean), std::move(trafo.value().cov));
        dist_mixture->set(REFERENCE_POINT::CENTER);
      }
      sm.predict(deltaT, *dist_mixture, ego);
    }
  }
  _classification.discount();
  _meta_data._durationSinceLastAssociation += deltaT;
  _time += deltaT;
  _meta_data._numPredictions++;
  _id = _idGenerator.getID();
  postProcess();
  assert([&] {  // NOLINT
    if (isValid() or isEmpty())
    {
      return true;
    }
    LOG_FATAL("State after prediction not valid: " << toString());
    return false;
  }());
}

void State::compensateEgoMotion(Duration dt, EgoMotionDistribution const& egoMotion)
{
  ZoneScopedNC("StateDistribution::compensateEgoMotion", tracy_color);
  LOG_DEB("Predict State Distribution with " << _state_dist.size() << " State Models");
  for (auto& [stateModelId, dist] : _state_dist)
  {
    BaseStateModel const& sm = *_manager->getStateModelMap().at(stateModelId);
    for (auto& dist_mixture : dist->dists())
    {
      assert(dist_mixture->type() == DISTRIBUTION_TYPE::GAUSSIAN);
      sm.compensateEgoMotion(dt, *dist_mixture, egoMotion);
    }
  }
  _id = _idGenerator.getID();
}

void State::merge(State other, bool enablePostProcess)
{
  for (auto& [other_state_id, other_dist] : other._state_dist)
  {
    _state_dist.at(other_state_id)->merge(std::move(other_dist));
  }
  _classification.merge(other._classification);
  _time = std::max(_time, other._time);
  _meta_data._timeOfLastAssociation =
      std::max(_meta_data._timeOfLastAssociation, other._meta_data._timeOfLastAssociation);
  _meta_data._durationSinceLastAssociation =
      std::min(_meta_data._durationSinceLastAssociation, other._meta_data._durationSinceLastAssociation);
  _id = _idGenerator.getID();
  for (auto& [other_model, values] : other._nis)
  {
    _nis[other_model].insert(_nis[other_model].end(), values.begin(), values.end());
  }
  _misc.merge(other._misc);
  _detectable = _detectable or other._detectable;
  _meta_data._numPredictions = std::max(_meta_data._numPredictions, other._meta_data._numPredictions);
  _meta_data._numUpdates = std::max(_meta_data._numUpdates, other._meta_data._numUpdates);
  if (enablePostProcess)
  {
    postProcess();
  }
}

double State::sumNisWeights() const
{
  double sum = 0;
  for (auto const& [model, values] : _nis)
  {
    for (auto const& [weight, _] : values)
    {
      sum += weight;
    }
  }
  return sum;
}

void State::multiplyNisWeights(double factor)
{
  for (auto& [model, values] : _nis)
  {
    for (auto& [weight, _] : values)
    {
      weight *= factor;
    }
  }
}

void State::merge(std::vector<State> others)
{
  ZoneScopedNC("StateDistribution::merge", tracy_color);
  ZoneText("StateDistribution", 17);
  ZoneValue(_id.value_);
  for (State& other : others)
  {
    merge(std::move(other), false);
  }
  postProcess();
}

void State::markovTransition()
{
  ZoneScopedNC("State::markovTransition", tracy_color);
  State newState = _manager->createState();
  Matrix const& markovMatrix = _manager->getTransition().transitionMatrix(*this);
  std::size_t i = 0;
  for (StateModelId origin_modelId : _manager->params().state.multi_model.use_state_models)
  {
    BaseStateModel const& origin_stateModel = _manager->getStateModel(origin_modelId);
    std::size_t j = 0;
    for (StateModelId target_modelId : _manager->params().state.multi_model.use_state_models)
    {
      if (markovMatrix(i, j) > 0)
      {
        BaseStateModel const& target_stateModel = _manager->getStateModel(target_modelId);
        for (BaseDistribution* distComp : _state_dist.at(origin_modelId)->dists())
        {
          std::unique_ptr<BaseDistribution> transformedDist =
              transformation::transform(*distComp, origin_stateModel, target_stateModel);
          transformedDist->multiplyWeights(markovMatrix(i, j));
          if (transformedDist->sumWeights() > 0)
          {
            newState._state_dist.at(target_modelId)->merge(std::move(transformedDist));
          }
        }
      }
      ++j;
    }
    ++i;
  }
  _state_dist = std::move(newState._state_dist);
}

void State::postProcess()
{
  // can make state invalid if all components get pruned
  ZoneScopedNC("State::postProcess", tracy_color);
  for (auto& [id, dist] : _state_dist)
  {
    DistributionParams const& params = _manager->state_model_params(id).distribution;
    if (params.post_process.enable)
    {
      // keep weight ratio of the different state_models the same
      double const weightBefore = dist->sumWeights();
      dist->mergeComponents(params.post_process.merging_distance, _manager->getStateModel(id).state_comps());

      dist->pruneWeight(params.post_process.min_weight);
      dist->pruneVar(params.post_process.max_variance);

      dist->truncate(params.post_process.max_components);
      if (double const weightAfter = dist->sumWeights(); weightAfter > 0)
      {
        dist->multiplyWeights(weightBefore / weightAfter);
      }
    }
  }
  if (double const weight = sumWeights(); weight > 0)
  {
    multiplyWeights(1 / weight);
  }
  _id = _idGenerator.getID();
}

void State::resetPriorId()
{
  for (auto& [_, dist] : _state_dist)
  {
    dist->resetPriorId();
  }
}

std::string State::toString(std::string prefix) const
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
  std::string out =
      prefix + "State Distribution\n" + prefix + "|\tId: " + std::to_string(_id.value_) + '\n' + prefix +
      "|\tLabel: " + std::to_string(_label.value_) + '\n' + prefix +
      "|\tEx. Probability: " + std::to_string(_existenceProbability) + '\n' + prefix +
      "|\tSurvival Probability: " + std::to_string(_survival_probability) + '\n' + prefix +
      "|\t#Updates: " + std::to_string(_meta_data._numUpdates) + '\n' + prefix +
      "|\t#Predictions: " + std::to_string(_meta_data._numPredictions) + '\n' + prefix +
      "|\tTime: " + std::to_string(to_milliseconds(_time.time_since_epoch())) + "ms" + '\n' + prefix +
      "|\tTime of last assoc. measurement: " +
      std::to_string(to_milliseconds(_meta_data._timeOfLastAssociation.time_since_epoch())) + "ms" + '\n' + prefix +
      "|\tDuration since last assoc.: " + std::to_string(to_milliseconds(_meta_data._durationSinceLastAssociation)) +
      "ms" + '\n' + prefix +
      "|\tLast assoc. measurement: " + std::to_string(_meta_data._lastAssociatedMeasurement.value_) + '\n' + prefix +
      "|\tExistence score: " + std::to_string(_score) + '\n' + prefix + "|\tStage: " + to_string(_stage) + '\n' +
      prefix + "|\tDetection history (associations) of size " + std::to_string(detection_hist_display_size) + " : " +
      detection_hist_string + " \n";

  for (auto const& [id, dist] : _state_dist)
  {
    out += prefix + "|\tState Model Id: " + std::to_string(id.value_) + "\n" +
           _manager->getStateModel(id).state_comps().toString(prefix + "|\t") + dist->toString(prefix + "|\t|\t");
  }
  out += prefix + "|\tInnovations\n";
  for (auto const& [meas_model_id, inno] : _innovation)
  {
    if (not inno._updates.empty())
    {
      out += prefix + "|\t\tMeas Model: " + meas_model_id.value_ + "\n" + inno.toString(prefix + "|\t\t");
    }
  }
  return out;
}

std::pair<StateModelId, BaseDistribution const&> State::bestState() const
{
  LOG_DEB("State::bestState");
  auto const& it = std::max_element(_state_dist.begin(), _state_dist.end(), [](auto const& a, auto const& b) {
    return a.second->sumWeights() < b.second->sumWeights();
  });
  return { it->first, *it->second };
}

std::pair<StateModelId, std::unique_ptr<BaseDistribution>> State::getEstimate() const
{
  LOG_DEB("State::getEstimate");
  ZoneScopedNC("State::getEstimate", tracy_color);
  if (_estimationCache.has_value())
  {
    return { _estimationCache.value().first, _estimationCache.value().second->clone() };
  }
  for (auto& [model, distribution] : _state_dist)
  {
    for (BaseDistribution* dist : distribution->dists())
    {
      if (dist->refPoint() != REFERENCE_POINT::CENTER)
      {
        auto trafo = transformation::transform(dist->mean(),
                                               dist->covariance(),
                                               _manager->getStateModel(model).state_comps(),
                                               dist->refPoint(),
                                               REFERENCE_POINT::CENTER);
        dist->set(std::move(trafo.value().mean), std::move(trafo.value().cov));
        dist->set(REFERENCE_POINT::CENTER);
      }
    }
  }
  _estimationCache = [&] -> std::optional<std::pair<StateModelId, std::unique_ptr<BaseDistribution>>> {
    switch (_manager->params().state.estimation.type)
    {
      case STATE_DISTRIBUTION_EXTRACTION_TYPE::BEST_STATE_MODEL:
      {
        auto const& [stateModelId, baseDist] = bestState();
        switch (_manager->state_model_params(stateModelId).distribution.extraction_type)
        {
          case DISTRIBUTION_EXTRACTION::BEST_COMPONENT:
          {
            if (_manager->params().state.estimation.transform_output_state)
            {
              StateModelId target = _manager->params().state.estimation.output_state_model;
              return std::pair{ target,
                                transformation::transform(baseDist.bestComponent(),
                                                          _manager->getStateModel(stateModelId),
                                                          _manager->getStateModel(target)) };
            }
            return std::pair{ stateModelId, baseDist.bestComponent().clone() };
          }
          case DISTRIBUTION_EXTRACTION::MIXTURE:
          {
            auto est = std::make_unique<GaussianDistribution>(baseDist.mean(), baseDist.covariance());
            if (_manager->params().state.estimation.transform_output_state)
            {
              StateModelId target = _manager->params().state.estimation.output_state_model;
              return std::pair{ target,
                                transformation::transform(
                                    *est, _manager->getStateModel(stateModelId), _manager->getStateModel(target)) };
            }
            return std::pair{ stateModelId, std::move(est) };
          }
        }
        assert(false);
        DEBUG_ASSERT_MARK_UNREACHABLE;
      }
      case STATE_DISTRIBUTION_EXTRACTION_TYPE::AVERAGE:
      {
        if (not _manager->params().state.estimation.transform_output_state)
        {
          LOG_FATAL("Average need a target state model");
          throw std::runtime_error("Average need a target state model");
        }
        StateModelId target = _manager->params().state.estimation.output_state_model;
        auto mixture = std::make_unique<MixtureDistribution>();
        for (auto const& [modelId, dist] : _state_dist)
        {
          for (BaseDistribution* distComp : dist->dists())
          {
            mixture->merge(transformation::transform(
                *distComp, _manager->getStateModel(modelId), _manager->getStateModel(target)));
          }
        }
        switch (_manager->state_model_params(target).distribution.extraction_type)
        {
          case DISTRIBUTION_EXTRACTION::BEST_COMPONENT:
            return std::pair{ target, mixture->bestComponent().clone() };
          case DISTRIBUTION_EXTRACTION::MIXTURE:
            return std::pair{ target, std::make_unique<GaussianDistribution>(mixture->mean(), mixture->covariance()) };
        }
      }
        assert(false);
        DEBUG_ASSERT_MARK_UNREACHABLE;
    }
    assert(false);
    DEBUG_ASSERT_MARK_UNREACHABLE;
  }();
  return { _estimationCache.value().first, _estimationCache.value().second->clone() };
}

double State::sumWeights() const
{
  double sum = 0;
  for (auto const& [modelID, dist] : _state_dist)
  {
    sum += dist->sumWeights();
  }
  return sum;
}

void State::multiplyWeights(double factor)
{
  for (auto const& [_, dist] : _state_dist)
  {
    dist->multiplyWeights(factor);
  }
  _id = _idGenerator.getID();
}

void State::innovate(MeasurementContainer const& measContainer)
{
  ZoneScopedNC("State::innovate", tracy_color);
  LOG_DEB("Calculate Innovation for State Distribution");
  BaseMeasurementModel const& measModel = *_manager->getMeasModelMap().at(measContainer._id);
  _innovation.at(measContainer._id) = measModel.calculateInnovation(measContainer, *this);
  std::string info_str = "Id: " + std::to_string(_id.value_);
  ZoneText(info_str.c_str(), info_str.size());
  info_str = "Label: " + std::to_string(_label.value_);
  ZoneText(info_str.c_str(), info_str.size());
  info_str = "Existence Prob: " + std::to_string(_existenceProbability);
  ZoneText(info_str.c_str(), info_str.size());
  info_str = "Time: " + std::to_string(to_seconds(_time.time_since_epoch()));
  ZoneText(info_str.c_str(), info_str.size());
  info_str = "#Updates: " + std::to_string(_meta_data._numUpdates);
  ZoneText(info_str.c_str(), info_str.size());
  info_str = "#Predictions: " + std::to_string(_meta_data._numPredictions);
  ZoneText(info_str.c_str(), info_str.size());
  info_str = "Duration since last assoc.: " + std::to_string(to_milliseconds(_meta_data._durationSinceLastAssociation));
  ZoneText(info_str.c_str(), info_str.size());
  info_str = "Weight Miss " + std::to_string(_weight_mis_detection);
  ZoneText(info_str.c_str(), info_str.size());
}

void State::performStageUpdate(std::size_t num_sensors)
{
  if (_manager->params().state.stage_management.use_score_based_stage_logic)
  {
    LOG_DEB("Looking at Track: " << _label);
    double alpha = _manager->params().state.stage_management.false_track_confirmation_probability;
    double beta = _manager->params().state.stage_management.true_track_deletion_probability;
    double track_confirmation_score = log((1 - beta) / alpha);
    double track_deletion_score = log(beta / (1 - alpha));
    LOG_DEB("track_score: " << _score);
    LOG_DEB("track_confirmation_score: " << track_confirmation_score);
    LOG_DEB("track_deletion_score: " << track_deletion_score);
    if (_score > track_confirmation_score)
    {
      _stage = STAGE::CONFIRMED;
    }
    _stage = STAGE::PRELIMINARY;
  }
  if (_manager->params().state.stage_management.use_history_based_stage_logic)
  {
    performHistoryStageLogic(num_sensors);
  }
}

void State::performHistoryStageLogic(std::size_t num_sensors)
{
  LOG_DEB("Looking at Track " << _label);
  LOG_DEB("Track " << _label << " has a history of " << _meta_data._detection_hist.size() << " with "
                   << std::count(_meta_data._detection_hist.begin(), _meta_data._detection_hist.end(), false)
                   << " misses and "
                   << std::count(_meta_data._detection_hist.begin(), _meta_data._detection_hist.end(), true)
                   << " hits");
  if (_meta_data._detection_hist.size() >=
      num_sensors * _manager->params().state.stage_management.deletion_history_threshold.N)
  {
    bool tobedeleted =
        std::count(_meta_data._detection_hist.end() -
                       num_sensors * _manager->params().state.stage_management.deletion_history_threshold.N,
                   _meta_data._detection_hist.end(),
                   false) >= num_sensors * _manager->params().state.stage_management.deletion_history_threshold.M;
    if (tobedeleted)
    {
      _stage = STAGE::DEAD;
      LOG_DEB("Track " << _label << " will be deleted!");
      LOG_DEB("Detection history of this track (end -  "
              << _meta_data._detection_hist.at(_meta_data._detection_hist.size() - 3));
      LOG_DEB("Detection history of this track "
              << _meta_data._detection_hist.at(_meta_data._detection_hist.size() - 2));
      LOG_DEB("Detection history of this track "
              << _meta_data._detection_hist.at(_meta_data._detection_hist.size() - 1));
    }
  }
  if (_stage == STAGE::TENTATIVE)
  {
    LOG_DEB("Track " << _label << " is tentative");
    LOG_DEB("Track " << _label << " has a measurement history of " << _meta_data._detection_hist.size() << " with "
                     << std::count(_meta_data._detection_hist.begin(), _meta_data._detection_hist.end(), false)
                     << " misses and "
                     << std::count(_meta_data._detection_hist.begin(), _meta_data._detection_hist.end(), true)
                     << " hits");
    auto [first_stage_confirmation_M, first_stage_confirmation_N] =
        _manager->params().state.stage_management.first_stage_confirmation_history_threshold;
    if (first_stage_confirmation_N == 0)
    {
      _stage = STAGE::PRELIMINARY;
      LOG_DEB("Track " << _label
                       << " will reach the second confirmation stage, "
                          "because first_stage_history_threshold_N = 0");
    }
    else if (_meta_data._detection_hist.size() >= num_sensors * first_stage_confirmation_M &&
             first_stage_confirmation_N > 0)
    {
      if (std::count(_meta_data._detection_hist.end() -
                         std::min(_meta_data._detection_hist.size(), num_sensors * first_stage_confirmation_N),
                     _meta_data._detection_hist.end(),
                     true) >= num_sensors * first_stage_confirmation_M)
      {
        LOG_DEB("Track " << _label
                         << " will reach the second initiation stage (preliminary)"
                            " and detection history will be reset!");
        _stage = STAGE::PRELIMINARY;
        _meta_data._detection_hist = {};
      }
    }
  }
  else if (_stage == STAGE::PRELIMINARY)
  {
    LOG_DEB("Track " << _label << " is preliminary");
    LOG_DEB("Track " << _label << " has a measurement history of " << _meta_data._detection_hist.size() << " with "
                     << std::count(_meta_data._detection_hist.begin(), _meta_data._detection_hist.end(), false)
                     << " misses and "
                     << std::count(_meta_data._detection_hist.begin(), _meta_data._detection_hist.end(), true)
                     << " hits");
    auto [second_stage_confirmation_M, second_stage_confirmation_N] =
        _manager->params().state.stage_management.second_stage_confirmation_history_threshold;
    if (second_stage_confirmation_N == 0)
    {
      _stage = STAGE::CONFIRMED;
      LOG_DEB("Track " << _label << " is now confirmed, because second_stage_confirmation_N = 0");
    }
    else if (_meta_data._detection_hist.size() >= num_sensors * second_stage_confirmation_M &&
             second_stage_confirmation_N > 0)
    {
      if (std::count(_meta_data._detection_hist.end() -
                         std::min(_meta_data._detection_hist.size(), num_sensors * second_stage_confirmation_N),
                     _meta_data._detection_hist.end(),
                     true) >= num_sensors * second_stage_confirmation_M)
      {
        LOG_DEB("Track " << _label << " will now be confirmed and detection history will be reset!");
        _stage = STAGE::CONFIRMED;
        _meta_data._detection_hist = {};
      }
    }
    LOG_DEB("Track " << _label << " has track stage " << to_string(_stage));
    LOG_DEB("Track " << _label << " has a measurement history of " << _meta_data._detection_hist.size() << " with "
                     << std::count(_meta_data._detection_hist.begin(), _meta_data._detection_hist.end(), false)
                     << " misses and "
                     << std::count(_meta_data._detection_hist.begin(), _meta_data._detection_hist.end(), true)
                     << " hits");
  }
}

}  // namespace ttb
