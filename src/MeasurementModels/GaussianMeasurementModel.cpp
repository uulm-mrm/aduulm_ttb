#include "tracking_lib/MeasurementModels/GaussianMeasurementModel.h"

#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/Measurements/Measurement.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Distributions/BaseDistribution.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/States/Innovation.h"
#include "tracking_lib/Transformations/Transformation.h"
#include "tracking_lib/OcclusionModels/NoOcclusionModel.h"

#include <memory>
#include <utility>
#include <boost/math/distributions/chi_squared.hpp>
#include <tracy/tracy/Tracy.hpp>

namespace ttb
{

constexpr const auto tracy_color = tracy::Color::MediumVioletRed;

GaussianMeasurementModel::GaussianMeasurementModel(TTBManager* manager, MeasModelId id)
  : _manager{ manager }
  , _id{ std::move(id) }
  , _model_comps{ Components(_manager->meas_model_params(_id).components) }
  , _occlusion_model{ [&]() -> std::unique_ptr<BaseOcclusionModel> {
    switch (_manager->meas_model_params(_id).occlusion.model)
    {
      case OCCLUSION_MODEL_TYPE::NO_OCCLUSION:
        return std::make_unique<NoOcclusionModel>(_manager);
      case OCCLUSION_MODEL_TYPE::LINE_OF_SIGHT_OCCLUSION:
        return {};
    }
    DEBUG_ASSERT_MARK_UNREACHABLE;
  }() }
{
  LOG_INF("Measurement Model " + _id.value_ + " with components: " + _model_comps.toString());
}

double GaussianMeasurementModel::gatingMHD2Distance(std::size_t dimMeasSpace) const
{
  try
  {
    return boost::math::quantile(boost::math::chi_squared_distribution<>(dimMeasSpace),
                                 _manager->meas_model_params(_id).gating_prob);
  }
  catch (std::overflow_error const& e)
  {
    return std::numeric_limits<double>::infinity();
  }
}
std::tuple<Probability, bool>
GaussianMeasurementModel::detection_probability(MeasurementContainer const& meas_container, State const& state) const
{
  ZoneScopedNC("GaussianMeasurementModel::detection_prob", tracy_color);
  double pD{ 0 };
  bool detectable = false;
  for (auto const& [model_id, model_dist] : state._state_dist)
  {
    BaseStateModel const& stateModel = _manager->getStateModel(model_id);
    Components measComps = transformation::transformableComps(
                               stateModel.state_comps(),
                               transformation::TransformOptions{
                                   _manager->state_model_params(model_id).assume_orient_in_velocity_direction })
                               .all.intersection(_model_comps);
    for (const auto& dist_comp : model_dist->dists())
    {
      assert(dist_comp->type() == DISTRIBUTION_TYPE::GAUSSIAN);
      assert(dist_comp->refPoint() == REFERENCE_POINT::CENTER);
      Vector const& x = dist_comp->mean();
      auto z_pred{ [&] -> std::optional<Vector> {
        std::optional<Vector> predictedMeas =
            transformation::transform(x, stateModel.state_comps(), measComps, meas_container._sensorInfo._to_sensor_cs);
        if (predictedMeas.has_value())
        {
          return predictedMeas.value();
        }
        return std::nullopt;
      }() };
      if (z_pred.has_value())
      {
        pD += dist_comp->sumWeights() *
              getDetectionProbability(z_pred.value(), measComps, meas_container._sensorInfo._sensor_fov);
        if (meas_container._sensorInfo._sensor_fov.has_value())
        {
          if (meas_container._sensorInfo._sensor_fov.value().contains(z_pred.value(), measComps))
          {
            detectable = true;
          }
        }
        else
        {
          detectable = true;
        }
      }
    }
  }
  pD *= (1 - _occlusion_model->occlusionProb(state, {}));  // Todo: alex (this works for now, but when LOF
  // occlusion gets implemented, adapt the interface
  // of the method to get the to the current time
  // predicted estimated tracks of the last cycle)
  return { pD, detectable };
}

GaussianMeasurementModel::PredMeasCache
GaussianMeasurementModel::create_predicted_measurement(MeasurementContainer const& meas_container,
                                                       State const& state) const
{
  ZoneScopedNC("GaussianMeasurementModel::create_predicted_measurement", tracy_color);
  std::vector<PredMeasCache::key_type> unique_preds;
  for (Measurement const& meas : meas_container._data)
  {
    for (BaseDistribution const* meas_dist_comp : meas._dist->dists())
    {
      for (const auto& [stateModelId, stateDist] : state._state_dist)
      {
        BaseStateModel const& stateModel = _manager->getStateModel(stateModelId);
        Components const predictAbleStateComps =
            transformation::transformableComps(
                stateModel.state_comps(),
                transformation::TransformOptions{
                    _manager->state_model_params(stateModelId).assume_orient_in_velocity_direction })
                .all;
        Components const measComps = meas._meas_comps.intersection(predictAbleStateComps);
        if (measComps._comps.empty())
        {
          LOG_WARN("No Component of the Measurement can be used to update.\n" +
                   stateModel.state_comps().toString("State Model Comps: ") + "\n" +
                   meas._meas_comps.toString("Meas Comps"));
          continue;
        }
        // iterate over all components
        for (BaseDistribution const* state_dist_comp : stateDist->dists())
        {
          LOG_DEB("Iterate over all Reference Points");
          std::vector<REFERENCE_POINT> const observableRefPoints = observableReferencePoints(
              state_dist_comp->mean(), stateModel.state_comps(), meas, *meas_dist_comp, meas_container._sensorInfo);
          for (REFERENCE_POINT measRefPoint : observableRefPoints)
          {
            LOG_DEB("MeasRefPoint:" + to_string(measRefPoint));
            std::tuple key{ measComps._comps, measRefPoint, state_dist_comp->id() };
            if (auto it = std::ranges::find(unique_preds, key); it == unique_preds.end())
            {
              unique_preds.emplace_back(std::move(key));
            }
          }
        }
      }
    }
  }
  PredMeasCache predMeasCache;
  auto const work = [&](PredMeasCache::key_type const& pred) {
    for (const auto& [stateModelId, stateDist] : state._state_dist)
    {
      BaseStateModel const& stateModel = _manager->getStateModel(stateModelId);
      auto const comps = stateDist->dists();
      if (auto it = std::ranges::find_if(comps,
                                         [&](BaseDistribution const* dist) { return dist->id() == std::get<2>(pred); });
          it != comps.end())
      {
        auto pred_state = transformation::transform((*it)->mean(),
                                                    (*it)->covariance(),
                                                    stateModel.state_comps(),
                                                    Components(std::get<0>(pred)),
                                                    (*it)->refPoint(),
                                                    std::get<1>(pred),
                                                    meas_container._sensorInfo._to_sensor_cs);
        if (not pred_state.has_value())
        {
          return;
        }
        predMeasCache.emplace(pred,
                              PredictedMeasurement{ .z_pred = std::move(pred_state.value().mean),
                                                    .R_pred = std::move(pred_state.value().cov),
                                                    .H = std::move(pred_state.value().T),
                                                    .T = std::move(pred_state.value().cross_cov) });
        return;
      }
    }
    assert(false);
  };
  std::ranges::for_each(unique_preds, work);

  return predMeasCache;
}

Innovation GaussianMeasurementModel::calculateInnovation(MeasurementContainer const& measContainer,
                                                         State const& dist) const
{
  ZoneScopedNC("GaussianMeasurementModel::innovation", tracy_color);
  ZoneText("StateDistribution", 17);
  ZoneValue(dist._id.value_);
  ZoneText("#Measurements", 13);
  ZoneValue(measContainer._data.size());
  LOG_DEB("Calculate Innovation for Gaussian Measurement Model for " << measContainer._data.size() << " Measurements");
  LOG_DEB("State Distribution:\n" << dist.toString());
  assert([&] {  // NOLINT
    if (not dist.isValid())
    {
      LOG_FATAL(dist.toString());
      return false;
    }
    return true;
  }());
  assert(_id == measContainer._id and "MeasurementModel got the wrong MeasurementContainer");
  auto const& stateModels = _manager->getStateModelMap();
  /// Set detection probability
  auto const [detection_prob, detectable] = detection_probability(measContainer, dist);
  assert([&]() {  // NOLINT
    if (detection_prob > 1.)
    {
      LOG_FATAL("Detection probability > 1.0: " << detection_prob);
      return false;
    }
    if (detection_prob < 0.)
    {
      LOG_FATAL("Detection probability < 0.0: " << detection_prob);
      return false;
    }
    return true;
  }());

  Innovation innovation{ ._detectionProbability = detection_prob };

  PredMeasCache predMeasCache = create_predicted_measurement(measContainer, dist);
  auto const single_meas_update = [&](Measurement const& meas) {
    ZoneScopedNC("GaussianMeasurementModel::SingleMeasUpdate", tracy_color);
    if (_manager->meas_model_params(id()).filter_outside_fov)
    {
      if (measContainer._sensorInfo._sensor_fov.has_value())
      {
        if (not measContainer._sensorInfo._sensor_fov.value().contains(meas._dist->dists().front()->mean(),
                                                                       meas._meas_comps))
        {
          LOG_WARN_THROTTLE(5, "Measurement outside FOV");
          LOG_WARN_THROTTLE(5, "Measurement:\n" + meas.toString("\t"));
          LOG_WARN_THROTTLE(5, "FOV:\n" + measContainer._sensorInfo._sensor_fov.value().toString("\t"));
          return;
        }
      }
    }
    Innovation::Update update{ .updated_dist = _manager->createState() };
    update.updated_dist._label = dist._label;
    update.updated_dist._time = meas._time;
    update.updated_dist._existenceProbability = dist._existenceProbability;
    update.updated_dist._survival_probability = dist._survival_probability;
    update.updated_dist._meta_data._numPredictions = dist._meta_data._numPredictions;
    update.updated_dist._meta_data._numUpdates = dist._meta_data._numUpdates + 1;
    update.updated_dist._meta_data._timeOfLastAssociation = meas._time;
    update.updated_dist._meta_data._durationSinceLastAssociation = Duration::zero();
    update.updated_dist._meta_data._lastAssociatedMeasurement = meas._id;
    update.updated_dist._misc = dist._misc;
    update.updated_dist._score = dist._score;
    update.updated_dist._nis = {};
    update.updated_dist._detectable = detectable;
    update.updated_dist._stage = dist._stage;
    update.updated_dist._meta_data._detection_hist = dist._meta_data._detection_hist;
    update.updated_dist._meta_data._detection_hist.emplace_back(true);

    LOG_DEB("Calculate Innovation for Measurement:\n" + meas.toString());
    double const kappa = getClutterIntensity(meas, measContainer._sensorInfo);

    update.updated_dist._classification = dist._classification;

    update.clutter_intensity = kappa;

    // iterate over map with state models
    LOG_DEB("Iterate over all State Models");
    for (const auto& [stateModelId, stateDist] : dist._state_dist)
    {
      BaseStateModel const& stateModel = *stateModels.at(stateModelId);
      Components const predictAbleStateComps =
          transformation::transformableComps(
              stateModel.state_comps(),
              transformation::TransformOptions{
                  _manager->state_model_params(stateModelId).assume_orient_in_velocity_direction })
              .all;
      Components const measComps = meas._meas_comps.intersection(predictAbleStateComps);
      if (measComps._comps.empty())
      {
        LOG_WARN("No Component of the Measurement can be used to update.\n" +
                 stateModel.state_comps().toString("State Model Comps: ") + "\n" +
                 meas._meas_comps.toString("Meas Comps"));
        continue;
      }
      // iterate over all components
      for (BaseDistribution const* state_dist_comp : stateDist->dists())
      {
        for (BaseDistribution const* meas_dist_comp : meas._dist->dists())
        {
          LOG_DEB("Iterate over all Reference Points");
          std::vector<REFERENCE_POINT> const observableRefPoints = observableReferencePoints(
              state_dist_comp->mean(), stateModel.state_comps(), meas, *meas_dist_comp, measContainer._sensorInfo);
          for (REFERENCE_POINT measRefPoint : observableRefPoints)
          {
            if (auto it = predMeasCache.find({ measComps._comps, measRefPoint, state_dist_comp->id() });
                it != predMeasCache.end())
            {
              PredictedMeasurement const& pred_meas = it->second;

              GaussianUpdateInfo const info = calcUpdateInfo(meas, *meas_dist_comp, pred_meas, measComps);
              if (info.isGated)
              {
                continue;
              }
              double const likelihood =
                  getMeasurementLikelihood(info.MHD2, _manager->meas_model_params(_id).gating_prob, info.S);
              Update single_update = calculateUpdate(
                  state_dist_comp->mean(), state_dist_comp->covariance(), stateModel.state_comps(), pred_meas, info);
              double const newWeight = state_dist_comp->sumWeights() * meas_dist_comp->sumWeights() /
                                       static_cast<double>(observableRefPoints.size()) * likelihood;
              auto updateComp = std::make_unique<GaussianDistribution>(
                  std::move(single_update.mean), std::move(single_update.var), newWeight, state_dist_comp->refPoint());
              updateComp->setPriorId(state_dist_comp->id());
              updateComp->_misc["ny"] = likelihood * detection_prob / kappa;
              update.updated_dist._state_dist.at(stateModelId)->merge(std::move(updateComp));
              update.updated_dist._nis[stateModelId].emplace_back(newWeight, [&] {
                std::map<COMPONENT, Nis> elem_wise_nis;
                Vector nis = info.residuum.array() * info.residuum.array() * info.iS.diagonal().array();
                Index ind = 0;
                for (COMPONENT comp : measComps._comps)
                {
                  elem_wise_nis.emplace(comp, nis(ind));
                  ind++;
                }
                return elem_wise_nis;
              }());
            }
          }  // Iterate over Measurement Mixture Comps
        }  // Iterate over Reference Points
      }  // Iterate over State Mixture Comps
    }  // Iterate over State Models
    if (not update.updated_dist.isEmpty())
    {
      update.updated_dist._classification.update(meas._classification);
      double const weight = update.updated_dist.sumWeights();
      update.log_likelihood = std::log(weight);
      update.updated_dist._score =
          update.updated_dist._score + std::log(detection_prob) + std::log(weight) - std::log(kappa);
      update.updated_dist.multiplyWeights(1 / weight);  // normalize dist
      assert([&] {                                      // NOLINT
        if (std::abs(weight - update.updated_dist.sumNisWeights()) > 1e-5)
        {
          LOG_FATAL("different nis and state dist weights. Must not happen");
          return false;
        }
        return true;
      }());
      update.updated_dist.multiplyNisWeights(1 / weight);
      update.updated_dist.postProcess();
      assert([&] {  // NOLINT
        if (update.updated_dist.isValid() or update.updated_dist.isEmpty())
        {
          return true;
        }
        LOG_FATAL("Update of state: " + dist.toString() + " with Measurement " + meas.toString() + " is INVALID");
        LOG_FATAL("Update: " + update.updated_dist.toString());
        return false;
      }());
      if (not update.updated_dist.isEmpty())
      {
        innovation._updates.emplace(meas._id, std::move(update));
      }
    }
  };

  std::ranges::for_each(measContainer._data, single_meas_update);

  /// add original Distribution to model a mis detection
  State misdetection = dist;
  misdetection._meta_data._lastAssociatedMeasurement = NOT_DETECTED;
  misdetection._id = State::_idGenerator.getID();
  misdetection._time = measContainer._time;
  misdetection._score = misdetection._score + std::log(1 - detection_prob);
  misdetection._meta_data._detection_hist.emplace_back(false);
  misdetection._nis = {};
  misdetection._detectable = detectable;
  for (const auto& [model_id, mixture_dist] : misdetection._state_dist)
  {
    for (auto& dist_val : mixture_dist->dists())
    {
      dist_val->_misc["ny"] = 1 - detection_prob;
    }
  }
  for (auto& [model, inno] : misdetection._innovation)  // we do not want to copy the innovations of the state
  {
    inno._updates.clear();
  }
  innovation._updates.emplace(NOT_DETECTED,
                              Innovation::Update{
                                  .updated_dist = std::move(misdetection),
                              });
  LOG_DEB("Innovation done");
  return innovation;
}
MEASUREMENT_MODEL_TYPE GaussianMeasurementModel::type() const noexcept
{
  return MEASUREMENT_MODEL_TYPE::GAUSSIAN;
}

TTBManager* GaussianMeasurementModel::manager() const
{
  return _manager;
}

std::string GaussianMeasurementModel::toString() const
{
  return _id.value_;
}

Components const& GaussianMeasurementModel::meas_model_comps() const
{
  return _model_comps;
}

MeasModelId GaussianMeasurementModel::id() const
{
  return _id;
}

BaseMeasurementModel::DefaultVal GaussianMeasurementModel::defaultVal(COMPONENT comp, CLASS clazz) const
{
  return default_val(_manager->meas_model_params(_id).default_values, comp, clazz);
}

double GaussianMeasurementModel::getDetectionProbability(Vector const& predictedMeasurement,
                                                         Components const& comps,
                                                         std::optional<FieldOfView> const& fov) const
{
  if (fov.has_value())
  {
    if (fov.value().contains(predictedMeasurement, comps))
    {
      LOG_DEB("predicted meas inside fov");
      return _manager->meas_model_params(_id).detection.prob;
    }
    LOG_DEB("predicted meas outside fov");
    return _manager->meas_model_params(_id).detection.prob_min;
  }
  LOG_DEB("no fov");
  return _manager->meas_model_params(_id).detection.prob;
}

double GaussianMeasurementModel::getClutterIntensity(Measurement const& meas,
                                                     SensorInformation const& sensorInformation) const
{
  LOG_DEB("getClutterIntensity");
  if (_manager->meas_model_params(_id).clutter.rate.has_value())
  {
    if (not sensorInformation._sensor_fov.has_value())
    {
      LOG_FATAL("Can not use clutter rate since no field of view associated to measurement: "
                << meas.toString() << " from Model " << _id.value_);
      throw std::runtime_error("Can not use clutter rate since no field of view associated to measurement: " +
                               meas.toString());
    }
    double const fov_size = sensorInformation._sensor_fov.value().area(meas._meas_comps);
    return _manager->meas_model_params(_id).clutter.rate.value() / fov_size;
  }
  return _manager->meas_model_params(_id).clutter.intensity.value();
}

std::optional<State> GaussianMeasurementModel::createState(Measurement const& meas, bool force) const
{
  LOG_DEB("createState based on measurement: " + meas.toString());
  State new_state = _manager->createState();
  auto const& stateModels = _manager->getStateModelMap();
  for (auto& [stateId, dist] : new_state._state_dist)
  {
    for (BaseDistribution* meas_comp : meas._dist->dists())
    {
      assert(meas_comp->type() == DISTRIBUTION_TYPE::GAUSSIAN);
      auto transformedMeas = transformation::transform(
          meas_comp->mean(), meas_comp->covariance(), meas._meas_comps, stateModels.at(stateId)->state_comps());
      if (transformedMeas.has_value())
      {
        new_state._state_dist.at(stateId)->merge(
            std::make_unique<GaussianDistribution>(std::move(transformedMeas.value().mean),
                                                   std::move(transformedMeas.value().cov),
                                                   meas_comp->sumWeights(),
                                                   meas_comp->refPoint()));
      }
      else if (force)
      {
        Components const initAbleComps = stateModels.at(stateId)->state_comps().intersection(
            transformation::transformableComps(
                meas._meas_comps,
                transformation::TransformOptions{
                    _manager->state_model_params(stateId).assume_orient_in_velocity_direction })
                .all);
        transformedMeas =
            transformation::transform(meas_comp->mean(), meas_comp->covariance(), meas._meas_comps, initAbleComps);
        assert([&] {  // NOLINT
          if (not transformedMeas.has_value())
          {
            LOG_FATAL("transformation has no value although transformed to transformable Components - inconsisten "
                      "Implementations");
            return false;
          }
          return true;
        }());
        Vector mean(stateModels.at(stateId)->state_comps()._comps.size());
        Matrix cov = Matrix::Zero(stateModels.at(stateId)->state_comps()._comps.size(),
                                  stateModels.at(stateId)->state_comps()._comps.size());
        auto const sub_inds = stateModels.at(stateId)->state_comps().indexOf(initAbleComps._comps).value();
        mean(sub_inds) = transformedMeas.value().mean;
        cov(sub_inds, sub_inds) = transformedMeas.value().cov;
        for (COMPONENT comp : stateModels.at(stateId)->state_comps().diff(initAbleComps)._comps)
        {
          auto ind = stateModels.at(stateId)->state_comps().indexOf(comp).value();
          DefaultVal val = defaultVal(comp, meas._classification.getEstimate());
          mean(ind) = val.mean.value();
          cov(ind, ind) = val.var.value();
        }
        new_state._state_dist.at(stateId)->merge(std::make_unique<GaussianDistribution>(
            std::move(mean), std::move(cov), meas_comp->sumWeights(), meas_comp->refPoint()));
      }
      else
      {
        return std::nullopt;
      }
    }
  }

  new_state._classification.update(meas._classification);

  std::size_t ctr = 0;
  for (StateModelId const modelId : _manager->params().state.multi_model.use_state_models)
  {
    new_state._state_dist.at(modelId)->multiplyWeights(_manager->params().state.multi_model.birth_weights.at(ctr));
    ctr++;
  }
  new_state._time = meas._time;
  new_state._meta_data._timeOfLastAssociation = meas._time;
  new_state._meta_data._lastAssociatedMeasurement = meas._id;
  new_state._misc["origin"] = id().value_;
  LOG_DEB("init State: " << new_state.toString());
  assert([&] {  // NOLINT
    if (not new_state.isValid())
    {
      LOG_FATAL("new_state is invalid. " << new_state.toString());
      return false;
    }
    return true;
  }());
  return new_state;
}

GaussianMeasurementModel::Update GaussianMeasurementModel::calculateUpdate(Vector const& x,
                                                                           Matrix const& P,
                                                                           Components const& stateComps,
                                                                           PredictedMeasurement const& pred_meas,
                                                                           GaussianUpdateInfo const& info) const
{
  if (pred_meas.H.has_value())
  {
    ZoneScopedNC("GaussianMeasurementModel::linearUpdate", tracy_color);
    LOG_DEB("Calculate the Linear Kalman Update");
    Matrix const I = Matrix::Identity(pred_meas.H.value().cols(), pred_meas.H.value().cols());
    Matrix const K = P * pred_meas.H.value().transpose() * info.iS;
    Vector x_upd = x + K * info.residuum;
    Matrix tmp_eigen = I - K * pred_meas.H.value();
    Eigen::MatrixXd P_upd = tmp_eigen * P * tmp_eigen.transpose() + K * info.R * K.transpose();
    if (stateComps.indexOf(COMPONENT::ROT_Z).has_value())
    {
      angles::normalizeAngle(x_upd(stateComps.indexOf(COMPONENT::ROT_Z).value()));
    }
    return { .mean = std::move(x_upd), .var = std::move(P_upd) };
  }
  {
    ZoneScopedNC("GaussianMeasurementModel::nonlinearUpdate", tracy_color);
    LOG_DEB("Calculate Updated Gaussian based on Unscented Trafo");
    Matrix const K = pred_meas.T.value() * info.iS;
    Vector x_upd = x + K * info.residuum;
    Matrix P_upd = P - K * info.S * K.transpose();
    if (stateComps.indexOf(COMPONENT::ROT_Z).has_value())
    {
      angles::normalizeAngle(x_upd(stateComps.indexOf(COMPONENT::ROT_Z).value()));
    }
    return { .mean = std::move(x_upd), .var = std::move(P_upd) };
  }
}

GaussianUpdateInfo GaussianMeasurementModel::calcUpdateInfo(Measurement const& meas,
                                                            BaseDistribution const& meas_dist,
                                                            PredictedMeasurement const& pred_meas,
                                                            Components const& measComps) const
{
  ZoneScopedNC("GaussianMeasurementModel::calcUpdateInfo", tracy_color);
  assert(meas_dist.type() == DISTRIBUTION_TYPE::GAUSSIAN);
  auto const meas_inds = meas._meas_comps.indexOf(measComps._comps).value();

  GaussianUpdateInfo info;
  info.R = meas_dist.covariance()(meas_inds, meas_inds);
  info.residuum = getResiduum(pred_meas.z_pred, meas_dist.mean()(meas_inds), measComps.indexOf(COMPONENT::ROT_Z));
  info.S = info.R + pred_meas.R_pred;
  info.iS = info.S.inverse();
  info.MHD2 = info.residuum.transpose() * info.iS * info.residuum;
  info.isGated = info.MHD2 > gatingMHD2Distance(measComps._comps.size());

  std::string const info_str = "MHD: " + std::to_string(info.MHD2);
  ZoneText(info_str.c_str(), info_str.size());
  std::string const gating_str = "Gating: " + std::to_string(gatingMHD2Distance(measComps._comps.size()));
  ZoneText(gating_str.c_str(), gating_str.size());
  return info;
}

std::optional<State> GaussianMeasurementModel::initStage2(std::vector<Measurement> const& oldMeasurements,
                                                          Measurement const& meas,
                                                          SensorInformation const& sensorInfo) const
{
  ZoneScopedNC("GenericBoxModel::initStage2", tracy_color);
  LOG_DEB("Generic Model initStage2");
  // Todo(alex) this considers only the first Component of the old and new measurement !
  if (oldMeasurements.empty())
  {
    LOG_DEB("Not Birth Candidates");
    std::string info_str = "No birth candidates given";
    ZoneText(info_str.c_str(), info_str.size());
    return {};
  }

  // Search nearest birth candidate to measurement on ground plane (2D)
  double minDist = std::numeric_limits<double>::infinity();
  std::optional<Measurement> bestOldMeas;
  for (Measurement const& oldMeas : oldMeasurements)
  {
    // Ignore birth candidates whose reference point differs from the measured rp
    if (_manager->meas_model_params(_id).check_matching_reference_point_init2)
    {
      if (oldMeas._dist->dists().front()->refPoint() != meas._dist->dists().front()->refPoint())
      {
        LOG_DEB("Reference of birth candidate (" << to_string(oldMeas._dist->refPoint()) + ") and measurement ("
                                                 << to_string(meas._dist->refPoint())
                                                 << ") do not coincide, skipping measurement");
        continue;
      }
    }
    else
    {
      LOG_WARN_THROTTLE(10, "Consider to enable check_matching_reference_point_init2");
    }
    if (_manager->meas_model_params(_id).check_matching_classification_init2)
    {
      // Ignore birth candidates whose type probability for the measurement type is below 25 percent
      if (CLASS old_class = oldMeas._classification.getEstimate(), new_class = meas._classification.getEstimate();
          old_class != CLASS::NOT_CLASSIFIED and new_class != CLASS::NOT_CLASSIFIED and old_class != new_class)
      {
        LOG_DEB("The measured class " << to_string(new_class) << "does not fit to the old meas class "
                                      << to_string(old_class) << " - SKIP");
        continue;
      }
    }
    auto const meas_xy_inds = meas._meas_comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value();
    auto const old_meas_xy_inds = oldMeas._meas_comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value();
    double const dist =
        (meas._dist->dists().front()->mean()(meas_xy_inds) - oldMeas._dist->dists().front()->mean()(old_meas_xy_inds))
            .norm();
    if (dist < minDist)
    {
      minDist = dist;
      bestOldMeas = oldMeas;
    }
  }
  // nothing found
  if (not bestOldMeas.has_value())
  {
    std::string info_str = "No fitting old measurement";
    ZoneText(info_str.c_str(), info_str.size());
    return {};
  }
  double const dt = to_seconds(meas._time - bestOldMeas.value()._time);
  CLASS const type{ meas._classification.getEstimate() };
  State new_state = _manager->createState();
  for (auto const& [stateModelId, new_dist] : new_state._state_dist)
  {
    BaseStateModel const& stateModel = _manager->getStateModel(stateModelId);
    // Create track
    std::size_t const dim = stateModel.state_comps()._comps.size();
    auto [mean, cov]{ [&] {
      std::optional<transformation::Transformed> tranformedMeas = transformation::transform(meas._dist->mean(),
                                                                                            meas._dist->covariance(),
                                                                                            meas._meas_comps,
                                                                                            stateModel.state_comps(),
                                                                                            sensorInfo._to_sensor_cs);
      if (tranformedMeas.has_value())
      {
        return std::pair(std::move(tranformedMeas.value().mean), std::move(tranformedMeas.value().cov));
      }
      Vector initX = Vector::Zero(dim);
      Matrix initP = Matrix::Zero(dim, dim);
      for (COMPONENT comp : stateModel.state_comps()._comps)
      {
        // switch could be removed because all cases do the same
        auto ind = stateModel.state_comps().indexOf(comp).value();
        switch (comp)
        {
          case COMPONENT::ROT_Z:
          {
            auto meas_rot_z_ind = meas._meas_comps.indexOf(COMPONENT::ROT_Z);
            if (meas_rot_z_ind.has_value())
            {
              initX(ind) = meas._dist->dists().front()->mean()(meas_rot_z_ind.value());
              initP(ind, ind) =
                  meas._dist->dists().front()->covariance()(meas_rot_z_ind.value(), meas_rot_z_ind.value());
              angles::normalizeAngle(initX(ind));
            }
            else
            {
              LOG_DEB("Use vel_x, vel_y to set up ROT_Z");
              auto meas_xy_ind = meas._meas_comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value();
              auto best_old_meas_xy_ind =
                  bestOldMeas.value()._meas_comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value();
              Vector const vel_xy = meas._dist->dists().front()->mean()(meas_xy_ind) -
                                    bestOldMeas.value()._dist->dists().front()->mean()(best_old_meas_xy_ind);
              initX(ind) = std::atan2(vel_xy(1), vel_xy(0));
              auto init = defaultVal(comp, CLASS::UNKNOWN);
              initP(ind, ind) = init.var.value();
            }
            break;
          }
          case COMPONENT::VEL_X:
          {
            auto meas_vel_x_ind = meas._meas_comps.indexOf(COMPONENT::VEL_X);
            if (meas_vel_x_ind.has_value())
            {
              initX(ind) = meas._dist->dists().front()->mean()(meas_vel_x_ind.value());
              initP(ind, ind) =
                  meas._dist->dists().front()->covariance()(meas_vel_x_ind.value(), meas_vel_x_ind.value());
            }
            else
            {
              auto meas_x_ind = meas._meas_comps.indexOf(COMPONENT::POS_X).value();
              auto best_old_meas_x_ind = bestOldMeas.value()._meas_comps.indexOf(COMPONENT::POS_X).value();
              double const deltax = meas._dist->dists().front()->mean()(meas_x_ind) -
                                    bestOldMeas.value()._dist->dists().front()->mean()(best_old_meas_x_ind);
              initX(ind) = deltax / dt;
              initP(ind, ind) =
                  (meas._dist->dists().front()->covariance()(meas_x_ind, meas_x_ind) +
                   bestOldMeas.value()._dist->dists().front()->covariance()(best_old_meas_x_ind, best_old_meas_x_ind)) /
                  (dt * dt);
            }
            break;
          }
          case COMPONENT::VEL_Y:
          {
            auto meas_vel_x_ind = meas._meas_comps.indexOf(COMPONENT::VEL_Y);
            if (meas_vel_x_ind.has_value())
            {
              initX(ind) = meas._dist->dists().front()->mean()(meas_vel_x_ind.value());
              initP(ind, ind) =
                  meas._dist->dists().front()->covariance()(meas_vel_x_ind.value(), meas_vel_x_ind.value());
            }
            else
            {
              auto meas_x_ind = meas._meas_comps.indexOf(COMPONENT::POS_Y).value();
              auto best_old_meas_x_ind = bestOldMeas.value()._meas_comps.indexOf(COMPONENT::POS_Y).value();
              double const deltax = meas._dist->dists().front()->mean()(meas_x_ind) -
                                    bestOldMeas.value()._dist->dists().front()->mean()(best_old_meas_x_ind);
              initX(ind) = deltax / dt;
              initP(ind, ind) =
                  (meas._dist->dists().front()->covariance()(meas_x_ind, meas_x_ind) +
                   bestOldMeas.value()._dist->dists().front()->covariance()(best_old_meas_x_ind, best_old_meas_x_ind)) /
                  (dt * dt);
            }
            break;
          }
          case COMPONENT::VEL_ABS:
          {
            auto meas_vel_abs_ind = meas._meas_comps.indexOf(COMPONENT::VEL_ABS);
            if (meas_vel_abs_ind.has_value())  // Orientation is actually measured
            {
              initX(ind) = meas._dist->dists().front()->mean()(meas_vel_abs_ind.value());
              initP(ind, ind) =
                  meas._dist->dists().front()->covariance()(meas_vel_abs_ind.value(), meas_vel_abs_ind.value());
            }
            else
            {
              auto meas_xy_ind = meas._meas_comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value();
              auto best_old_meas_xy_ind =
                  bestOldMeas.value()._meas_comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value();
              Vector const vxy = meas._dist->dists().front()->mean()(meas_xy_ind) -
                                 bestOldMeas.value()._dist->dists().front()->mean()(best_old_meas_xy_ind) / dt;
              Matrix const P = (meas._dist->dists().front()->covariance()(meas_xy_ind, meas_xy_ind) +
                                bestOldMeas.value()._dist->dists().front()->covariance()(best_old_meas_xy_ind,
                                                                                         best_old_meas_xy_ind)) /
                               (dt * dt);
              auto vel_abs = transformation::transform(vxy,
                                                       P,
                                                       Components({ COMPONENT::VEL_X, COMPONENT::VEL_Y }),
                                                       Components({ COMPONENT::VEL_ABS }),
                                                       sensorInfo._to_sensor_cs);
              initX(ind) = vel_abs.value().mean(0);
              initP(ind, ind) = vel_abs.value().cov(0, 0);
            }
            break;
          }
          default:
          {
            auto comp_ind = meas._meas_comps.indexOf(comp);
            if (comp_ind.has_value())
            {
              initX(ind) = meas._dist->dists().front()->mean()(comp_ind.value());
              initP(ind, ind) = meas._dist->dists().front()->covariance()(comp_ind.value(), comp_ind.value());
            }
            else
            {
              LOG_INF_THROTTLE(10,
                               "Using Measurement Model default values to initialize component: " + to_string(comp));
              BaseMeasurementModel::DefaultVal init = defaultVal(comp, type);
              initX(ind) = init.mean.value();
              initP(ind, ind) = init.var.value();
            }
          }
        }
      }
      return std::pair(initX, initP);
    }() };

    std::optional<double> max_speed = getConstraints(type, COMPONENT::VEL_ABS);
    auto const vel_abs = transformation::transform(
        mean, stateModel.state_comps(), Components({ COMPONENT::VEL_ABS }), sensorInfo._to_sensor_cs);
    if (max_speed.has_value() and vel_abs.has_value())
    {
      if ((vel_abs.value().array() > max_speed.value()).any())
      {
        LOG_DEB("Vel abs of initialized state to big: " + std::to_string(vel_abs.value()(0)) +
                " Threshold: " + std::to_string(max_speed.value()));
        std::string const info_str = "Vel abs of initialized state to big: " + std::to_string(vel_abs.value()(0)) +
                                     " Threshold: " + std::to_string(max_speed.value());
        ZoneText(info_str.c_str(), info_str.size());
        return {};
      }
    }

    // Calculate reference points
    std::vector<REFERENCE_POINT> refPoints =
        observableReferencePoints(mean, stateModel.state_comps(), meas, *meas._dist->dists().front(), sensorInfo);

    // Create state distribution of newly born object (one mixture per reference point)
    std::vector<Vector> vecX;
    std::vector<Matrix> vecP;
    std::vector<double> vecWeights;
    std::vector<REFERENCE_POINT> vecRP;
    for (REFERENCE_POINT compRP : refPoints)
    {
      auto trans = transformation::transform(mean, stateModel.state_comps(), compRP, REFERENCE_POINT::CENTER);
      auto width = stateModel.state_comps().indexOf(COMPONENT::WIDTH);
      auto length = stateModel.state_comps().indexOf(COMPONENT::LENGTH);
      auto height = stateModel.state_comps().indexOf(COMPONENT::HEIGHT);
      if ((width.has_value() and trans(width.value()) < 0.) or (length.has_value() and trans(length.value()) < 0.) or
          (height.has_value() and trans(height.value()) < 0.))
      {
        LOG_WARN("negative extents in initStage2 objects");
        std::string info_str = "negative extent in birth state";
        ZoneText(info_str.c_str(), info_str.size());
      }
      vecX.push_back(trans);
      vecP.push_back(cov);
      vecRP.push_back(REFERENCE_POINT::CENTER);
      vecWeights.push_back(1.0 / static_cast<double>(refPoints.size()));
    }
    for (std::size_t i = 0; i < vecX.size(); i++)
    {
      new_state._state_dist.at(stateModelId)
          ->merge(std::make_unique<GaussianDistribution>(
              std::move(vecX.at(i)), std::move(vecP.at(i)), vecWeights.at(i), vecRP.at(i)));
    }
  }
  std::size_t ctr = 0;
  for (StateModelId const modelId : _manager->params().state.multi_model.use_state_models)
  {
    new_state._state_dist.at(modelId)->multiplyWeights(_manager->params().state.multi_model.birth_weights.at(ctr));
    ctr++;
  }

  new_state._classification.update(meas._classification);

  new_state._time = meas._time;
  new_state._meta_data._lastAssociatedMeasurement = meas._id;
  new_state._meta_data._timeOfLastAssociation = meas._time;
  for (auto const& [modelId, dist] : new_state._state_dist)
  {
    std::string info_str =
        "Weight of State Model: " + std::to_string(modelId.value_) + ": " + std::to_string(dist->sumWeights());
    ZoneText(info_str.c_str(), info_str.size());
  }
  assert([&] {
    if (not new_state.isValid())
    {
      LOG_FATAL(new_state.toString());
      return false;
    }
    return true;
  }());
  return new_state;
}

std::optional<double> GaussianMeasurementModel::getConstraints(CLASS type, COMPONENT comp) const
{
  if (not _manager->meas_model_params(_id).check_constraints)
  {
    return {};
  }
  for (auto const& cons : _manager->meas_model_params(_id).constraints)
  {
    if (type == cons.type)
    {
      if (cons.max_vals.contains(to_string(comp)))
      {
        return cons.max_vals.at(to_string(comp));
      }
      return {};
    }
  }
  return {};
}

std::vector<REFERENCE_POINT>
GaussianMeasurementModel::observableReferencePoints(Vector const& x,
                                                    Components const& comps,
                                                    Measurement const& meas,
                                                    BaseDistribution const& meas_dist,
                                                    SensorInformation const& sensorInfo) const
{
  LOG_DEB("observableReferencePoints");
  std::vector<REFERENCE_POINT> out;
  if (not _manager->meas_model_params(_id).force_estimate_rp)
  {
    if (meas._ref_point_measured)
    {
      out.push_back(meas_dist.refPoint());
      if (_manager->meas_model_params(_id).consider_inverse_rp)
      {
        auto invRP = inverseRP(meas_dist.refPoint());
        if (std::find(out.begin(), out.end(), invRP) == out.end())
        {
          out.push_back(invRP);
        }
      }
      return out;
    }
  }
  else if (_manager->meas_model_params(_id).mult_hyp_rp_estimation)
  {
    out.push_back(REFERENCE_POINT::BACK_LEFT);
    out.push_back(REFERENCE_POINT::BACK_RIGHT);
    out.push_back(REFERENCE_POINT::FRONT_LEFT);
    out.push_back(REFERENCE_POINT::FRONT_RIGHT);
    return out;
  }

  // Estimate the detected reference point by the viewing angle.
  auto idxP = comps.indexOf(COMPONENT::ROT_Z);
  if (not idxP.has_value())
  {
    LOG_DEB("Can not estimate Reference Points based on Angles because state model has no orientation");
    return { REFERENCE_POINT::CENTER };
  }

  Vector const pole{ { sensorInfo._to_sensor_cs.translation()(0), sensorInfo._to_sensor_cs.translation()(1) } };
  double aspect_angle = atan2(pole(1) - x(comps.indexOf(COMPONENT::POS_Y).value()),
                              pole(0) - x((comps.indexOf(COMPONENT::POS_X).value()))) -
                        x(idxP.value());
  angles::normalizeAngle(aspect_angle);

  // Edges are problematic, since only one side is constrained
  if (_manager->meas_model_params(_id).estimate_edge_rps)
  {
    // find corresponding reference point
    if (aspect_angle < -std::numbers::pi * 3 / 4)
    {
      out.push_back(REFERENCE_POINT::BACK_LEFT);
      out.push_back(REFERENCE_POINT::BACK);
      out.push_back(REFERENCE_POINT::BACK_RIGHT);
      return out;
    }
    if (aspect_angle < -std::numbers::pi / 2)
    {
      out.push_back(REFERENCE_POINT::BACK);
      out.push_back(REFERENCE_POINT::BACK_RIGHT);
      out.push_back(REFERENCE_POINT::RIGHT);
      return out;
    }
    if (aspect_angle < -std::numbers::pi / 4)
    {
      out.push_back(REFERENCE_POINT::BACK_RIGHT);
      out.push_back(REFERENCE_POINT::RIGHT);
      out.push_back(REFERENCE_POINT::FRONT_RIGHT);
      return out;
    }
    if (aspect_angle < 0)
    {
      out.push_back(REFERENCE_POINT::RIGHT);
      out.push_back(REFERENCE_POINT::FRONT_RIGHT);
      out.push_back(REFERENCE_POINT::FRONT);
      return out;
    }
    if (aspect_angle < std::numbers::pi / 4)
    {
      out.push_back(REFERENCE_POINT::FRONT_RIGHT);
      out.push_back(REFERENCE_POINT::FRONT);
      out.push_back(REFERENCE_POINT::FRONT_LEFT);
      return out;
    }
    if (aspect_angle < std::numbers::pi / 2)
    {
      out.push_back(REFERENCE_POINT::FRONT);
      out.push_back(REFERENCE_POINT::FRONT_LEFT);
      out.push_back(REFERENCE_POINT::LEFT);
      return out;
    }
    if (aspect_angle < std::numbers::pi * 3 / 4)
    {
      out.push_back(REFERENCE_POINT::FRONT_LEFT);
      out.push_back(REFERENCE_POINT::LEFT);
      out.push_back(REFERENCE_POINT::BACK_LEFT);
      return out;
    }

    out.push_back(REFERENCE_POINT::LEFT);
    out.push_back(REFERENCE_POINT::BACK_LEFT);
    out.push_back(REFERENCE_POINT::BACK);
    return out;
  }
  if (_manager->meas_model_params(_id).force_estimate_rp)
  {
    out.push_back(REFERENCE_POINT::BACK_LEFT);
    out.push_back(REFERENCE_POINT::BACK_RIGHT);
    out.push_back(REFERENCE_POINT::FRONT_LEFT);
    out.push_back(REFERENCE_POINT::FRONT_RIGHT);
    return out;
  }
  // only vertices are allowed
  // find corresponding reference point
  if (aspect_angle < -std::numbers::pi / 2)
  {
    out.push_back(REFERENCE_POINT::BACK_LEFT);
    out.push_back(REFERENCE_POINT::BACK_RIGHT);
    out.push_back(REFERENCE_POINT::FRONT_RIGHT);
    return out;
  }
  if (aspect_angle < 0)
  {
    out.push_back(REFERENCE_POINT::BACK_RIGHT);
    out.push_back(REFERENCE_POINT::FRONT_RIGHT);
    out.push_back(REFERENCE_POINT::FRONT_LEFT);
    return out;
  }
  if (aspect_angle < std::numbers::pi / 2)
  {
    out.push_back(REFERENCE_POINT::FRONT_RIGHT);
    out.push_back(REFERENCE_POINT::FRONT_LEFT);
    out.push_back(REFERENCE_POINT::BACK_LEFT);
    return out;
  }
  out.push_back(REFERENCE_POINT::FRONT_LEFT);
  out.push_back(REFERENCE_POINT::BACK_LEFT);
  out.push_back(REFERENCE_POINT::BACK_RIGHT);
  return out;
}

Vector GaussianMeasurementModel::getResiduum(Vector const& z_pred, Vector const& z, std::optional<Index> rot_Ind)
{
  Eigen::VectorXd res = z - z_pred;
  if (rot_Ind.has_value())
  {
    angles::normalizeAngle(res(rot_Ind.value()));
  }
  return res;
}

double GaussianMeasurementModel::getMeasurementLikelihood(double mhd2, double gatingConf, Matrix const& S)
{
  return exp(-0.5 * mhd2) / std::sqrt((2 * std::numbers::pi * S).determinant()) / gatingConf;
}

}  // namespace ttb