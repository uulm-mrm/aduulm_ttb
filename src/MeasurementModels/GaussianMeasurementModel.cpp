#include "tracking_lib/MeasurementModels/GaussianMeasurementModel.h"

#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/Transformations/Transformation.h"

#include <aduulm_logger/aduulm_logger.hpp>
#include <memory>
#include <utility>
#include <boost/math/distributions/chi_squared.hpp>
#include <tracy/tracy/TracyC.h>

namespace ttb
{

template class GaussianMeasurementModel<double>;
template class GaussianMeasurementModel<AD>;

template <class UDT>
GaussianMeasurementModel<UDT>::GaussianMeasurementModel(TTBManager<UDT> const* manager, SourceId id)
  : _manager{ manager }
  , _id{ std::move(id) }
  , _model_comps{ Components(_manager->params()->meas_model_params(_id).components) }
  , _detection_model_params{ give_init_AD_detection_model_params() }
{
  LOG_INF("Measurement Model " + _id.value_ + " with components: " + _model_comps.toString());
}

template <class UDT>
UDT GaussianMeasurementModel<UDT>::gatingMHD2Distance(std::size_t dimMeasSpace) const
{
  try
  {
    return boost::math::quantile(boost::math::chi_squared_distribution<>(static_cast<double>(dimMeasSpace)),
                                 _manager->params()->meas_model_params(_id).gating_prob);
  }
  catch (std::overflow_error const&)
  {
    return std::numeric_limits<double>::infinity();
  }
  catch (...)
  {
    throw;
  }
}

template <class UDT>
std::tuple<Probability<UDT>, bool>
GaussianMeasurementModel<UDT>::detection_probability(DataSourceContainer<UDT> const& meas_container,
                                                     State<UDT> const& state) const
{
  ZoneScopedN("GaussianMeasurementModel::detection_prob");
  UDT pD{ 0 };
  bool detectable = false;
  for (auto const& [model_id, model_dist] : state._state_dist)
  {
    Components const measComps =
        transformation::transformableComps(
            _manager->state_comps(model_id),
            transformation::TransformOptions{
                manager()->params()->state_model_params(model_id).assume_orient_in_velocity_direction })
            .all.intersection(_model_comps);

    for (const auto& [weight, gauss] : model_dist.second.to_gaussian_components())
    {
      assert(gauss.ref_point() == REFERENCE_POINT::CENTER);
      Vector<UDT> const& x = gauss.mean();
      auto z_pred{ [&] -> std::optional<Vector<UDT>> {
        std::optional<Vector<UDT>> predictedMeas = transformation::transform(
            x, _manager->state_comps(model_id), measComps, meas_container._sensorInfo._to_sensor_cs);
        if (predictedMeas.has_value())
        {
          return predictedMeas.value();
        }
        return std::nullopt;
      }() };
      if (z_pred.has_value())
      {
        pD += model_dist.first * weight *
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
  return { pD, detectable };
}

template <class UDT>
GaussianMeasurementModel<UDT>::PredMeasCache
GaussianMeasurementModel<UDT>::create_predicted_measurement(DataSourceContainer<UDT> const& meas_container,
                                                            State<UDT> const& state) const
{
  ZoneScopedN("GaussianMeasurementModel::create_predicted_measurement");
  std::vector<std::tuple<std::vector<COMPONENT>, REFERENCE_POINT, DistributionId>> unique_preds;
  for (State<UDT> const& meas : meas_container._measurements)
  {
    assert(meas._type == STATE_TYPE::MEASUREMENT);

    for (auto const& [meas_comp_weight, meas_dist_comp] :
         meas._state_dist.begin()->second.second.to_gaussian_components())
    {
      for (const auto& [stateModelId, stateDist] : state._state_dist)
      {
        Components const predictAbleStateComps =
            transformation::transformableComps(
                _manager->state_comps(stateModelId),
                transformation::TransformOptions{
                    manager()->params()->state_model_params(stateModelId).assume_orient_in_velocity_direction })
                .all;
        Components const measComps =
            _manager->state_comps(meas._state_dist.begin()->first).intersection(predictAbleStateComps);
        if (measComps._comps.empty())
        {
          LOG_WARN("No Component of the Measurement can be used to update.\n" +
                   _manager->state_comps(stateModelId).toString("State Model Comps: ") + "\n" +
                   _manager->state_comps(meas._state_dist.begin()->first).toString("Meas Comps"));
          continue;
        }
        // iterate over all components
        for (auto const& [state_comp_weight, state_dist_comp] : stateDist.second.to_gaussian_components())
        {
          std::vector<REFERENCE_POINT> const observableRefPoints =
              observableReferencePoints(state_dist_comp.mean(),
                                        _manager->state_comps(stateModelId),
                                        meas,
                                        meas_dist_comp.ref_point(),
                                        meas_container._sensorInfo);
          for (REFERENCE_POINT measRefPoint : observableRefPoints)
          {
            std::tuple key{ measComps._comps, measRefPoint, state_dist_comp.id() };
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
  for (std::tuple<std::vector<COMPONENT>, REFERENCE_POINT, DistributionId> const& pred : unique_preds)
  {
    for (const auto& [stateModelId, stateDist] : state._state_dist)
    {
      auto const& comps = stateDist.second.to_gaussian_components();
      if (auto it =
              std::ranges::find_if(comps, [&](auto const& dist) { return dist.second.id() == std::get<2>(pred); });
          it != comps.end())
      {
        std::optional<transformation::Transformed<UDT>> pred_state =
            transformation::transform<UDT>(it->second.mean(),
                                           it->second.covariance(),
                                           _manager->state_comps(stateModelId),
                                           Components(std::get<0>(pred)),
                                           it->second.ref_point(),
                                           std::get<1>(pred),
                                           meas_container._sensorInfo._to_sensor_cs);
        if (not pred_state.has_value())
        {
          break;
        }
        predMeasCache.emplace(pred,
                              PredictedMeasurement{ .z_pred = std::move(pred_state.value().mean),
                                                    .R_pred = std::move(pred_state.value().cov),
                                                    .H = std::move(pred_state.value().T),
                                                    .T = std::move(pred_state.value().cross_cov) });
        break;
      }
    }
  }
  return predMeasCache;
}

template <class UDT>
Vector<UDT> GaussianMeasurementModel<UDT>::give_init_AD_detection_model_params() const
{
  Eigen::Vector<UDT, -1> params = Eigen::Vector<UDT, -1>::Random(n_params);
  Index off = 0;
  Eigen::Map<Eigen::Vector<UDT, -1>> cx(params.data() + off, n_elems);
  off += n_elems;
  Eigen::Map<Eigen::Vector<UDT, -1>> cy(params.data() + off, n_elems);
  off += n_elems;
  Eigen::Map<Eigen::Vector<UDT, -1>> log_b(params.data() + off, n_log_b);
  off += n_log_b;
  Eigen::Map<Eigen::Vector<UDT, -1>> w(params.data() + off, n_w);
  off += n_w;
  assert(off + 1 == params.rows());
  // assert(log_b.rows()==1);
  UDT& bias = params(off);
  constexpr double xmin = -1.0;
  constexpr double xmax = 1.0;
  constexpr double ymin = -1.0;
  constexpr double ymax = 1.0;
  constexpr Index NX = 8;
  constexpr Index NY = 8;
  assert(NX * NY == n_elems);
  constexpr double dx = (xmax - xmin) / (NX - 1);
  constexpr double dy = (ymax - ymin) / (NY - 1);
  // Roughly one grid spacing.
  constexpr double sigma = std::max(dx, dy);
  std::mt19937 rng(123);
  std::normal_distribution weight_dist(0.0, 0.01);

  int k = 0;
  for (int iy = 0; iy < NY; ++iy)
  {
    for (int ix = 0; ix < NX; ++ix)
    {
      cx(k) = xmin + ix * dx;
      cy(k) = ymin + iy * dy;

      w(k) = weight_dist(rng);
      log_b(k) = std::log(1.0 / (sigma * sigma));

      ++k;
    }
  }
  bias = 0;
  return params;
}

template <class UDT>
std::tuple<StateDetection<UDT>,
           std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>>,
           std::vector<MeasClutter<UDT>>>
GaussianMeasurementModel<UDT>::calculateInnovation(DataSourceContainer<UDT> const& measurements,
                                                   State<UDT> const& dist) const
{
  ZoneScopedN("GaussianMeasurementModel::innovation");
  assert([&] {  // NOLINT
    if (not dist.isValid())
    {
      LOG_FATAL(dist.toString());
      return false;
    }
    return true;
  }());
  /// Set detection probability
  auto const [detection_prob, detectable] = detection_probability(measurements, dist);
  assert([&] {  // NOLINT
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

  PredMeasCache predMeasCache = create_predicted_measurement(measurements, dist);
  std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>> state_meas_innovations;
  std::vector<MeasClutter<UDT>> meas_innovations;
  for (State<UDT> const& meas : measurements._measurements)
  {
    UDT const kappa = getClutterIntensity(meas, measurements._sensorInfo);
    meas_innovations.push_back({ .meas_id = meas._id, .clutter_intensity = kappa });
    if (manager()->params()->meas_model_params(id()).filter_outside_fov)
    {
      if (measurements._sensorInfo._sensor_fov.has_value())
      {
        if (not measurements._sensorInfo._sensor_fov.value().contains(
                meas._state_dist.begin()->second.second.to_gaussian_components().front().second.mean(),
                _manager->state_comps(meas._state_dist.begin()->first)))
        {
          continue;
        }
      }
    }
    std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> updated_state_dist;
    std::map<StateModelId, std::vector<std::tuple<UDT, std::map<COMPONENT, UDT>>>> nis;
    UDT meas_likelihood = 0;
    for (const auto& [stateModelId, stateDist] : dist._state_dist)
    {
      Components const predictAbleStateComps =
          transformation::transformableComps(
              _manager->state_comps(stateModelId),
              transformation::TransformOptions{
                  manager()->params()->state_model_params(stateModelId).assume_orient_in_velocity_direction })
              .all;
      Components const measComps =
          _manager->state_comps(meas._state_dist.begin()->first).intersection(predictAbleStateComps);
      if (measComps._comps.empty())
      {
        LOG_WARN("No Component of the Measurement can be used to update.\n" +
                 _manager->state_comps(stateModelId).toString("State Model Comps: ") + "\n" +
                 _manager->state_comps(meas._state_dist.begin()->first).toString("Meas Comps"));
        continue;
      }
      // iterate over all components
      for (auto const& [state_dist_weight, state_dist_comp] : stateDist.second.to_gaussian_components())
      {
        for (auto const& [meas_dist_weight, meas_dist_comp] :
             meas._state_dist.begin()->second.second.to_gaussian_components())
        {
          std::vector<REFERENCE_POINT> const observableRefPoints =
              observableReferencePoints(state_dist_comp.mean(),
                                        _manager->state_comps(stateModelId),
                                        meas,
                                        meas_dist_comp.ref_point(),
                                        measurements._sensorInfo);
          for (REFERENCE_POINT measRefPoint : observableRefPoints)
          {
            if (auto it = predMeasCache.find({ measComps._comps, measRefPoint, state_dist_comp.id() });
                it != predMeasCache.end())
            {
              PredictedMeasurement<UDT> const& pred_meas = it->second;

              auto const calc_update_info = [&](GaussianDistribution<UDT> const& meas_dist) {
                // hpx::tracy::region("GaussianMeasurementModel::calcUpdateInfo", 0, 0);
                assert(meas._type == STATE_TYPE::MEASUREMENT);
                Indices const meas_inds =
                    _manager->state_comps(meas._state_dist.begin()->first).indexOf(measComps).value();
                GaussianUpdateInfo<UDT> info;
                info.R = meas_dist.covariance()(meas_inds, meas_inds);
                info.residuum =
                    getResiduum(pred_meas.z_pred, meas_dist.mean()(meas_inds), measComps.indexOf(COMPONENT::ROT_Z));
                info.S = info.R + pred_meas.R_pred;
                info.iS = info.S.inverse();
                info.MHD2 = info.residuum.transpose() * info.iS * info.residuum;
                info.isGated = info.MHD2 > gatingMHD2Distance(measComps._comps.size());
                return info;
              };
              GaussianUpdateInfo const info = calc_update_info(meas_dist_comp);
              if (info.isGated)
              {
                continue;
              }
              UDT const likelihood =
                  getMeasurementLikelihood(info.MHD2, manager()->params()->meas_model_params(_id).gating_prob, info.S);
              Update single_update = calculateUpdate(state_dist_comp.mean(),
                                                     state_dist_comp.covariance(),
                                                     _manager->state_comps(stateModelId),
                                                     pred_meas,
                                                     info);
              UDT const newWeight = stateDist.first * state_dist_weight * meas_dist_weight /
                                    static_cast<UDT>(observableRefPoints.size()) * likelihood;
              meas_likelihood += newWeight;
              auto updateComp = std::pair{ newWeight,
                                           GaussianDistribution(std::move(single_update.mean),
                                                                std::move(single_update.var),
                                                                state_dist_comp.ref_point()) };
              updateComp.second.prior_id() = state_dist_comp.id();
              updateComp.second.misc()["ny"] = UDT{ likelihood * detection_prob / kappa };
              updated_state_dist[stateModelId].push_back(std::move(updateComp));
              nis[stateModelId].emplace_back(newWeight, [&] {
                std::map<COMPONENT, UDT> elem_wise_nis;
                Vector<UDT> nis_ = info.residuum.array() * info.residuum.array() * info.iS.diagonal().array();
                Index ind = 0;
                for (COMPONENT comp : measComps._comps)
                {
                  elem_wise_nis.emplace(comp, nis_(ind));
                  ind++;
                }
                return elem_wise_nis;
              }());
            }
          }  // Iterate over Measurement Mixture Comps
        }  // Iterate over Reference Points
      }  // Iterate over State Mixture Comps
    }  // Iterate over State Models

    if (not updated_state_dist.empty())
    {
      State updated_dist(dist._manager, std::move(updated_state_dist));
      if (not updated_dist.isEmpty())
      {
        updated_dist._prior_id = dist._id;
        updated_dist._label = dist._label;
        updated_dist._type = STATE_TYPE::POSTERIOR;
        updated_dist._time = meas._time;
        updated_dist._meta_data._numPredictions = dist._meta_data._numPredictions;
        updated_dist._meta_data._numUpdates = dist._meta_data._numUpdates + 1;
        updated_dist._meta_data._timeOfLastAssociation = meas._time;
        updated_dist._meta_data._durationSinceLastAssociation = Duration::zero();
        updated_dist._misc = dist._misc;
        updated_dist._score = dist._score;
        assert(not updated_dist._meta_data._nis.contains(measurements._source_id));
        updated_dist._meta_data._nis.emplace(measurements._source_id, std::move(nis));
        updated_dist._meta_data._detectable = { { measurements._source_id, detectable } };
        updated_dist._stage = dist._stage;
        updated_dist._meta_data._detection_hist = dist._meta_data._detection_hist;
        updated_dist._meta_data._detection_hist.emplace_back(true);
        updated_dist._classification = dist._classification;
        updated_dist._log_weight = dist._log_weight;

        if (meas._meas_classification.has_value())
        {
          updated_dist._classification.update(meas._meas_classification.value());
        }
        updated_dist._meta_data._last_meas_log_likelihood = { { measurements._source_id, std::log(meas_likelihood) } };
        updated_dist._score += std::log(detection_prob) + std::log(meas_likelihood) - std::log(kappa);
        updated_dist._meta_data._meas_update_weight = { { measurements._source_id, { { meas._id, 1.0 } } } };
        assert(updated_dist.isValid());
        updated_dist.postProcess();
        if (not updated_dist.isEmpty())
        {
          assert([&] {  // NOLINT
            if (updated_dist.isValid())
            {
              return true;
            }
            LOG_FATAL("Update of state: " + dist.toString() + " with Measurement " + meas.toString() + " is INVALID");
            LOG_FATAL("Update: " + updated_dist.toString());
            return false;
          }());
          state_meas_innovations.emplace_back(dist._id, meas._id, std::move(updated_dist), std::log(meas_likelihood));
        }
      }
    }
  }
  /// add original Distribution to model a mis detection
  State misdetection = dist;
  misdetection._type = STATE_TYPE::POSTERIOR;
  misdetection._meta_data._last_meas_log_likelihood = { { measurements._source_id,
                                                          UDT{ -std::numeric_limits<double>::infinity() } } };
  misdetection._id = State<UDT>::_idGenerator.getID();
  misdetection._prior_id = dist._id;
  misdetection._time = measurements._time;
  misdetection._score = misdetection._score + std::log(1 - detection_prob);
  misdetection._meta_data._detection_hist.emplace_back(false);
  misdetection._meta_data._nis = {};
  misdetection._meta_data._detectable = { { measurements._source_id, detectable } };
  misdetection._meta_data._meas_update_weight = { { measurements._source_id, { { NOT_DETECTED, 1.0 } } } };
  misdetection._log_weight = dist._log_weight;

  for (auto& [model_id, mixture_dist] : misdetection._state_dist)
  {
    mixture_dist.second.give_new_id();
    for (auto const& [missed_dist, ori_dist] :
         std::views::zip(mixture_dist.second.to_gaussian_components(),
                         dist._state_dist.at(model_id).second.to_gaussian_components()))
    {
      missed_dist.second.prior_id() = ori_dist.second.id();
      missed_dist.second.misc()["ny"] = 1 - detection_prob;
    }
  }
  state_meas_innovations.emplace_back(
      dist._id, NOT_DETECTED, std::move(misdetection), UDT{ std::numeric_limits<double>::quiet_NaN() });
  LOG_DEB(std::format("Detection probability of state: {} #measurement {} / #updates {}",
                      detection_prob,
                      measurements._measurements.size(),
                      state_meas_innovations.size() - 1));
  return { StateDetection{ .state_id = dist._id, .detection_probability = detection_prob },
           std::move(state_meas_innovations),
           std::move(meas_innovations) };
}

template <class UDT>
MEASUREMENT_MODEL_TYPE GaussianMeasurementModel<UDT>::type() const noexcept
{
  return MEASUREMENT_MODEL_TYPE::GAUSSIAN;
}

template <class UDT>
TTBManager<UDT> const* GaussianMeasurementModel<UDT>::manager() const
{
  return _manager;
}

template <class UDT>
std::string GaussianMeasurementModel<UDT>::toString() const
{
  return _id.value_;
}

template <class UDT>
Components const& GaussianMeasurementModel<UDT>::meas_model_comps() const
{
  return _model_comps;
}

template <class UDT>
SourceId GaussianMeasurementModel<UDT>::id() const
{
  return _id;
}

template <class UDT>
UDT GaussianMeasurementModel<UDT>::getDetectionProbability(Vector<UDT> const& predictedMeasurement,
                                                           Components const& comps,
                                                           std::optional<Region> const& fov) const
{
  if (not _manager->params()->meas_model_params(_id).detection.learn_model)
  {
    if (fov.has_value())
    {
      if (fov.value().contains(predictedMeasurement, comps))
      {
        return manager()->params()->meas_model_params(_id).detection.prob;
      }
      return manager()->params()->meas_model_params(_id).detection.prob_min;
    }
    return manager()->params()->meas_model_params(_id).detection.prob;
  }
  auto const ind_xy = comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
  assert(ind_xy.has_value());
  Eigen::Vector<UDT, -1> state_in = Eigen::Vector<UDT, -1>::Zero(2);
  state_in(0) = predictedMeasurement(ind_xy.value()(0)) / 500;
  state_in(1) = predictedMeasurement(ind_xy.value()(1)) / 500;
  // state_in(0) = 1;
  // state_in(1) = 1;
  auto const& params = _detection_model_params;
  Index off = 0;
  Eigen::Map<Eigen::Vector<UDT, -1> const> const cx(params.data() + off, n_elems);
  off += n_elems;
  Eigen::Map<Eigen::Vector<UDT, -1> const> const cy(params.data() + off, n_elems);
  off += n_elems;
  Eigen::Map<Eigen::Vector<UDT, -1> const> const log_b(params.data() + off, n_log_b);
  off += n_log_b;
  Eigen::Map<Eigen::Vector<UDT, -1> const> const w(params.data() + off, n_w);
  off += n_w;
  assert(off + 1 == params.rows());
  UDT const& bias = params(off - 1);
  UDT theta = bias;
  for (int i = 0; i < n_elems; ++i)
  {
    UDT const dx = state_in(0) - cx(i);
    UDT const dy = state_in(1) - cy(i);
    UDT const r2 = dx * dx + dy * dy;
    theta += w(i) * exp(-exp(log_b(i)) * r2);
  }
  return 1.0 / (1.0 + exp(-theta));
}

template <class UDT>
UDT GaussianMeasurementModel<UDT>::getClutterIntensity(State<UDT> const& meas,
                                                       SensorInformation const& sensorInformation) const
{
  assert(meas._type == STATE_TYPE::MEASUREMENT);
  if (manager()->params()->meas_model_params(_id).clutter.rate.has_value())
  {
    if (not sensorInformation._sensor_fov.has_value())
    {
      LOG_FATAL("Can not use clutter rate since no field of view associated to measurement: "
                << meas.toString() << " from Model " << _id.value_);
      throw std::runtime_error("Can not use clutter rate since no field of view associated to measurement: " +
                               meas.toString());
    }
    double const fov_size =
        sensorInformation._sensor_fov.value().area(_manager->state_comps(meas._state_dist.begin()->first));
    return manager()->params()->meas_model_params(_id).clutter.rate.value() / fov_size;
  }
  return manager()->params()->meas_model_params(_id).clutter.intensity.value();
}

template <class UDT>
[[nodiscard]] std::pair<Vector<double>, Components> from_hull(State<UDT> const& meas)
{
  assert(meas._misc.contains("HULL"));
  auto const& poly = std::any_cast<Polygon::SinglePolygon>(meas._misc.at("HULL"));
  double best_area = std::numeric_limits<double>::infinity();
  double best_rot_z = 0;
  double pos_x = 0;
  double pos_y = 0;
  double length = 0;
  double width = 0;
  for (std::size_t i = 0; i < poly.outer().size(); ++i)
  {
    Vector2 const p0{ { poly.outer().at(i)._point.x(), poly.outer().at(i)._point.y() } };
    Point const& p1 = poly.outer().at((i + 1) % poly.outer().size());
    Vector2 const edge = Vector2{ { p1._point.x(), p1._point.y() } } - p0;
    Vector2 const tangent = edge.normalized();
    double const rot_z = std::atan2(tangent(1), tangent(0));
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double const c = std::cos(rot_z);
    double const s = std::sin(rot_z);
    Matrix<double> const rot{ { c, s }, { -s, c } };

    for (std::size_t j = 0; j < poly.outer().size(); ++j)
    {
      Vector<double> const vertex =
          rot * Vector<double>{ { poly.outer().at(j)._point.x(), poly.outer().at(j)._point.y() } };
      min_x = std::min(min_x, vertex(0));
      min_y = std::min(min_y, vertex(1));
      max_x = std::max(max_x, vertex(0));
      max_y = std::max(max_y, vertex(1));
    }
    assert(max_x >= min_x);
    assert(max_y >= min_y);
    if (double const area = (max_x - min_x) * (max_y - min_y); area < best_area)
    {
      best_area = area;
      best_rot_z = rot_z;
      length = max_x - min_x;
      width = max_y - min_y;
      Vector<double> pos{ { 0.5 * (max_x + min_x), 0.5 * (max_y + min_y) } };
      pos = rot.transpose() * pos;
      pos_x = pos(0);
      pos_y = pos(1);
    }
  }
  Components const comps({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::LENGTH, COMPONENT::WIDTH, COMPONENT::ROT_Z });
  Vector<double> state(comps._comps.size());
  state(comps.indexOf(COMPONENT::POS_X).value()) = pos_x;
  state(comps.indexOf(COMPONENT::POS_Y).value()) = pos_y;
  state(comps.indexOf(COMPONENT::ROT_Z).value()) = best_rot_z;
  state(comps.indexOf(COMPONENT::LENGTH).value()) = length;
  state(comps.indexOf(COMPONENT::WIDTH).value()) = width;
  return { state, comps };
}

template <class UDT>
std::optional<State<UDT>> GaussianMeasurementModel<UDT>::createState(State<UDT> const& meas, bool force) const
{
  assert(meas._type == STATE_TYPE::MEASUREMENT);
  std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> new_state_dist;
  Components const& meas_comps = _manager->state_comps(meas._state_dist.begin()->first);
  for (auto const [state_model_index, state_id] :
       std::views::enumerate(manager()->params()->state.multi_model.use_state_models))
  {
    double const state_model_weight = manager()->params()->state.multi_model.birth_weights.at(state_model_index);
    for (auto const& [meas_weight, meas_comp] : meas._state_dist.begin()->second.second.to_gaussian_components())
    {
      auto transformedMeas = transformation::transform(
          meas_comp.mean(), meas_comp.covariance(), meas_comps, _manager->state_comps(state_id));
      if (transformedMeas.has_value())
      {
        if (meas._misc.contains("HULL"))
        {
          // set POS_X, POS_Y, LENGTH, WIDTH, ROT_Z from HULL
          auto const [hull_state, co] = from_hull(meas);
          for (auto const [hull_state_i, comp] : std::views::enumerate(co._comps))
          {
            auto const ind = _manager->state_comps(state_id).indexOf(comp);
            if (ind.has_value())
            {
              transformedMeas.value().mean(ind.value()) = hull_state(hull_state_i);
            }
          }
        }
        new_state_dist[state_id].emplace_back(meas_weight * state_model_weight,
                                              GaussianDistribution(std::move(transformedMeas.value().mean),
                                                                   std::move(transformedMeas.value().cov),
                                                                   meas_comp.ref_point()));
      }
      else if (force)
      {
        Components const initAbleComps = _manager->state_comps(state_id).intersection(
            transformation::transformableComps(
                meas_comps,
                transformation::TransformOptions{
                    manager()->params()->state_model_params(state_id).assume_orient_in_velocity_direction })
                .all);
        transformedMeas =
            transformation::transform(meas_comp.mean(), meas_comp.covariance(), meas_comps, initAbleComps);
        assert([&] {  // NOLINT
          if (not transformedMeas.has_value())
          {
            LOG_FATAL("transformation has no value although transformed to transformable Components - inconsisten "
                      "Implementations");
            return false;
          }
          return true;
        }());
        Vector<UDT> mean(_manager->state_comps(state_id)._comps.size());
        Matrix<UDT> cov = Matrix<UDT>::Zero(_manager->state_comps(state_id)._comps.size(),
                                            _manager->state_comps(state_id)._comps.size());
        auto const sub_inds = _manager->state_comps(state_id).indexOf(initAbleComps).value();
        mean(sub_inds) = transformedMeas.value().mean;
        cov(sub_inds, sub_inds) = transformedMeas.value().cov;
        for (COMPONENT comp : _manager->state_comps(state_id).diff(initAbleComps)._comps)
        {
          auto ind = _manager->state_comps(state_id).indexOf(comp).value();
          DefaultVal val = default_val(
              manager()->params()->meas_model_params(id()).default_values, comp, meas._classification.getEstimate());
          mean(ind) = val.mean.value();
          cov(ind, ind) = val.var.value();
        }
        if (meas._misc.contains("HULL"))
        {
          // set POS_X, POS_Y, LENGTH, WIDTH, ROT_Z from HULL
          auto const [hull_state, co] = from_hull(meas);
          for (auto const [hull_state_i, comp] : std::views::enumerate(co._comps))
          {
            auto const ind = _manager->state_comps(state_id).indexOf(comp);
            if (ind.has_value())
            {
              mean(ind.value()) = hull_state(hull_state_i);
            }
          }
        }
        new_state_dist[state_id].emplace_back(
            meas_weight * state_model_weight,
            GaussianDistribution(std::move(mean), std::move(cov), meas_comp.ref_point()));
      }
      else
      {
        return std::nullopt;
      }
    }
  }
  State new_state(meas._manager, std::move(new_state_dist));
  if (meas._meas_classification.has_value())
  {
    new_state._classification.update(meas._meas_classification.value());
  }
  new_state._time = meas._time;
  new_state._meta_data._timeOfLastAssociation = meas._time;
  new_state._meta_data._meas_update_weight = { { DUMMY_SOURCE_ID, { { meas._id, 1.0 } } } };
  new_state._misc["origin"] = id().value_;
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

template <class UDT>
GaussianMeasurementModel<UDT>::Update
GaussianMeasurementModel<UDT>::calculateUpdate(Vector<UDT> const& x,
                                               Matrix<UDT> const& P,
                                               Components const& stateComps,
                                               PredictedMeasurement<UDT> const& pred_meas,
                                               GaussianUpdateInfo<UDT> const& info) const
{
  if (pred_meas.H.has_value())
  {
    // hpx::tracy::region("GaussianMeasurementModel::linearUpdate", 0, 0);
    Matrix<UDT> const I = Matrix<UDT>::Identity(pred_meas.H.value().cols(), pred_meas.H.value().cols());
    Matrix<UDT> const K = P * pred_meas.H.value().transpose() * info.iS;
    Vector<UDT> x_upd = x + K * info.residuum;
    Matrix<UDT> tmp_eigen = I - K * pred_meas.H.value();
    Matrix<UDT> P_upd = tmp_eigen * P * tmp_eigen.transpose() + K * info.R * K.transpose();
    if (stateComps.indexOf(COMPONENT::ROT_Z).has_value())
    {
      angles::normalizeAngle(x_upd(stateComps.indexOf(COMPONENT::ROT_Z).value()));
    }
    return { .mean = std::move(x_upd), .var = std::move(P_upd) };
  }
  {
    // hpx::tracy::region("GaussianMeasurementModel::nonlinearUpdate", 0, 0);
    assert(pred_meas.T.has_value());
    Matrix<UDT> const K = pred_meas.T.value() * info.iS;
    Vector<UDT> x_upd = x + K * info.residuum;
    Matrix<UDT> P_upd = P - K * info.S * K.transpose();
    if (stateComps.indexOf(COMPONENT::ROT_Z).has_value())
    {
      angles::normalizeAngle(x_upd(stateComps.indexOf(COMPONENT::ROT_Z).value()));
    }
    return { .mean = std::move(x_upd), .var = std::move(P_upd) };
  }
}

template <class UDT>
std::optional<UDT> GaussianMeasurementModel<UDT>::getConstraints(CLASS type, COMPONENT comp) const
{
  if (not manager()->params()->meas_model_params(_id).check_constraints)
  {
    return {};
  }
  for (auto const& cons : manager()->params()->meas_model_params(_id).constraints)
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

template <class UDT>
std::vector<REFERENCE_POINT>
GaussianMeasurementModel<UDT>::observableReferencePoints(Vector<UDT> const& x,
                                                         Components const& comps,
                                                         State<UDT> const& meas,
                                                         REFERENCE_POINT meas_ref_point,
                                                         SensorInformation const& sensorInfo) const
{
  assert(meas._type == STATE_TYPE::MEASUREMENT);
  std::vector<REFERENCE_POINT> out;
  if (not manager()->params()->meas_model_params(_id).force_estimate_rp)
  {
    out.push_back(meas_ref_point);
    if (manager()->params()->meas_model_params(_id).consider_inverse_rp)
    {
      REFERENCE_POINT const invRP = inverseRP(meas_ref_point);
      if (std::ranges::find(out, invRP) == out.end())
      {
        out.push_back(invRP);
      }
    }
    return out;
  }
  else if (manager()->params()->meas_model_params(_id).mult_hyp_rp_estimation)
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
    return { REFERENCE_POINT::CENTER };
  }

  Vector<UDT> const pole{ { sensorInfo._to_sensor_cs.translation()(0), sensorInfo._to_sensor_cs.translation()(1) } };
  UDT aspect_angle = atan2(pole(1) - x(comps.indexOf(COMPONENT::POS_Y).value()),
                           pole(0) - x((comps.indexOf(COMPONENT::POS_X).value()))) -
                     x(idxP.value());
  angles::normalizeAngle(aspect_angle);

  // Edges are problematic, since only one side is constrained
  if (manager()->params()->meas_model_params(_id).estimate_edge_rps)
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
  if (manager()->params()->meas_model_params(_id).force_estimate_rp)
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

template <class UDT>
Vector<UDT> GaussianMeasurementModel<UDT>::getResiduum(Vector<UDT> const& z_pred,
                                                       Vector<UDT> const& z,
                                                       std::optional<Index> rot_Ind)
{
  Vector<UDT> res = z - z_pred;
  if (rot_Ind.has_value())
  {
    angles::normalizeAngle(res(rot_Ind.value()));
  }
  return res;
}
template <class UDT>
UDT GaussianMeasurementModel<UDT>::getMeasurementLikelihood(UDT mhd2, UDT gatingConf, Matrix<UDT> const& S)
{
  return exp(-0.5 * mhd2) / sqrt((2 * std::numbers::pi * S).determinant()) / gatingConf;
}

}  // namespace ttb
