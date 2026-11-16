#include "tracking_lib/BirthModels/ROIBirthModel.h"
#include "tracking_lib/BirthModels/BaseBirthModel.h"

#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/Transformations/Transformation.h"

#include <boost/geometry.hpp>
#include <boost/math/distributions/chi_squared.hpp>

namespace ttb
{

template class ROIBirthModel<double>;
template class ROIBirthModel<AD>;

template <class UDT>
ROIBirthModel<UDT>::ROIBirthModel(TTBManager<UDT>* manager) : _manager{ manager }
{
  LOG_DEB("Init ROI birth model");
}
template <class UDT>
BIRTH_MODEL_TYPE ROIBirthModel<UDT>::type() noexcept
{
  return BIRTH_MODEL_TYPE::ROI;
}

template <class UDT>
TTBManager<UDT>* ROIBirthModel<UDT>::manager() const
{
  return _manager;
}

template <class UDT>
BirthTracks<UDT> ROIBirthModel<UDT>::getBirthTracks(DataSourceContainer<UDT> const& measContainer,
                                                    std::map<MeasurementId, UDT> const& /*unused*/,
                                                    std::vector<State<UDT>> const& existingTracks)
{
  if (not manager()->params()->meas_model_params(measContainer._source_id).can_init)
  {
    LOG_DEB("Meas Model: " + measContainer._source_id.value_ + " can not init");
    if (manager()->params()->show_gui)
    {
      _manager->vizu()._meas_model_data.wlock().data[measContainer._source_id]._birth_tracks.emplace_back(
          measContainer._time, std::vector<BernoulliState<UDT>>{});
    }
    return {};
  }
  loadBirthLocations(measContainer);
  BirthTracks<UDT> birthTracks;
  for (std::size_t i = 0; i < _birthLocations.size(); ++i)
  {
    std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> birth_dist;
    std::size_t ctr = 0;
    for (StateModelId modelId : manager()->params()->state.multi_model.use_state_models)
    {
      StateModel<UDT> const& sm = _manager->getStateModel(modelId);
      Vector<UDT> mean(sm.state_comps()._comps.size());
      Matrix<UDT> cov = Matrix<UDT>::Zero(sm.state_comps()._comps.size(), sm.state_comps()._comps.size());
      for (COMPONENT comp : sm.state_comps()._comps)
      {
        auto ind = sm.state_comps().indexOf(comp).value();
        mean(ind) = _birthLocations.at(i).at(comp);
        cov(ind, ind) = _birthVariances.at(i).at(comp);
      }

      std::optional<std::size_t> ix = sm.state_comps().indexOf(COMPONENT::POS_X);
      std::optional<std::size_t> iy = sm.state_comps().indexOf(COMPONENT::POS_Y);

      if (ix && iy)
      {
        // Map from fixed (POS_X, POS_Y) order to dynamic state vector indices
        std::array<std::size_t, 2> target_indices = { *ix, *iy };

        // 2x2 covariance matrix calculated from ROI polygon (fixed order: POS_X, POS_Y)
        double confidence = manager()->params()->birth_model->roi_model.confidence_prob;
        Eigen::Matrix2d cov_local = computeCovariance(_point, _roi, confidence);

        // Insert into full covariance matrix at correct dynamic positions
        for (int a = 0; a < 2; ++a)
        {
          for (int b = 0; b < 2; ++b)
          {
            cov(target_indices[a], target_indices[b]) = cov_local(a, b);
          }
        }
      }
      else
      {
        LOG_ERR("Position components (POS_X or POS_Y) are missing in the state vector.");
      }
      birth_dist[modelId].emplace_back(manager()->params()->state.multi_model.birth_weights.at(ctr),
                                       GaussianDistribution(mean, cov));
      ++ctr;
    }
    State newState(_manager, std::move(birth_dist));
    UDT const newState_existenceProbability = getBirthProbability<UDT>(*_manager, newState, existingTracks);
    assert([&] {  // NOLINT
      if (std::isnan(newState_existenceProbability))
      {
        LOG_FATAL("newState \n " << newState.toString());
        return false;
      }
      return true;
    }());
    newState._score = log(newState_existenceProbability / (1 - newState_existenceProbability));
    newState._type = STATE_TYPE::BIRTH;
    if (newState_existenceProbability > 0)
    {
      LOG_DEB("New State: " << newState.toString());
      birthTracks.birth_tracks.emplace_back(std::move(newState), newState_existenceProbability);
      birthTracks.birth_measurements.emplace_back(std::nullopt);
    }
  }
  if (manager()->params()->show_gui)
  {
    _manager->vizu()._meas_model_data.wlock().data[measContainer._source_id]._birth_tracks.emplace_back(
        measContainer._time, birthTracks.birth_tracks);
  }
  return birthTracks;
}

template <class UDT>
void ROIBirthModel<UDT>::loadBirthLocations(DataSourceContainer<UDT> const& measurementContainer)
{
  Components allComps = _manager->state_comps(manager()->params()->state.multi_model.use_state_models.front());
  Components roiComps{ std::vector<COMPONENT>() };
  std::map<COMPONENT, double> roiCenter;
  std::optional<Region> roi;
  if (manager()->params()->birth_model->roi_model.use_sensors_fov_instead)
  {
    roi = measurementContainer._sensorInfo._sensor_fov;
  }
  else
  {
    roi = measurementContainer._predictionInfo._roi;
  }
  if (roi.has_value())
  {
    LOG_DEB("roi:\n" << roi->toString());
    for (auto const& [comps, polygon] : roi->_polygons)
    {
      roiComps = roiComps.merge(comps);
      roiCenter = getCenterOfROI(roiComps, roi.value());
      _roi = polygon.multiPolygon.front();
      LOG_DEB("polygon:\n" << polygon.toString());
    }
    LOG_DEB("_roi:\n" << boost::geometry::wkt(_roi));
  }
  else
  {
    LOG_WARN("No ROI could be defined");
  }

  LOG_DEB("roiComps: " << roiComps.toString());

  for (StateModelId modelId : manager()->params()->state.multi_model.use_state_models)
  {
    allComps = allComps.merge(_manager->state_comps(modelId));
  }
  int bIdx = 0;
  for (StaticBirthLocation const& location : manager()->params()->birth_model->static_model.locations)
  {
    std::map<COMPONENT, double> birthState;
    std::map<COMPONENT, double> birthVar;
    for (COMPONENT comp : allComps._comps)
    {
      if (not location.mean.contains(to_string(comp)))
      {
        LOG_FATAL("Location " + std::to_string(bIdx) + " MEAN does not contain needed comp: " + to_string(comp));
      }
      if (std::ranges::contains(roiComps._comps, comp))
      {
        birthState.emplace(comp, roiCenter.at(comp));
      }
      else
      {
        birthState.emplace(comp, location.mean.at(to_string(comp)));
      }

      if (not location.var.contains(to_string(comp)))
      {
        LOG_FATAL("Location " + std::to_string(bIdx) + " VAR does not contain needed comp: " + to_string(comp));
      }
      birthVar.emplace(comp, location.var.at(to_string(comp)));
    }
    _birthLocations = { std::move(birthState) };
    _birthVariances = { std::move(birthVar) };
    bIdx++;
  }

  if (_birthLocations.empty())
  {
    LOG_FATAL("No birth locations specified");
  }
  if (_birthVariances.empty())
  {
    LOG_ERR("No birth covariance specified");
    throw std::runtime_error("No birth covariance specified");
  }
  LOG_DEB("birth means:");
  for (auto const& loc : _birthLocations)
  {
    LOG_DEB("Mean x: " << loc.at(COMPONENT::POS_X) << " \tMean y: " << loc.at(COMPONENT::POS_Y));
  }
  for (auto const& var : _birthVariances)
  {
    LOG_DEB("VAR x: " << var.at(COMPONENT::POS_X) << " \tVAR y: " << var.at(COMPONENT::POS_Y));
  }
  LOG_DEB("Allocated birth locations: " << _birthLocations.size());
  LOG_DEB("Allocated birth variances: " << _birthVariances.size());
}

template <class UDT>
std::map<COMPONENT, double> ROIBirthModel<UDT>::getCenterOfROI(Components const& comps, Region const& roi)
{
  std::map<COMPONENT, double> roiCenter = roi.center(comps);
  if (roiCenter.contains(COMPONENT::POS_X))
  {
    _point._point.x() = roiCenter.at(COMPONENT::POS_X);
  }
  if (roiCenter.contains(COMPONENT::POS_Y))
  {
    _point._point.y() = roiCenter.at(COMPONENT::POS_Y);
  }
  LOG_DEB(_point._point.x() << ", " << _point._point.y());
  return roiCenter;
}

template <class UDT>
Eigen::Matrix2d ROIBirthModel<UDT>::computeCovariance(Point const& center,
                                                      Polygon::SinglePolygon const& polygon,
                                                      double confidence)
{
  const auto& outer = polygon.outer();
  std::size_t n = outer.size();
  if (n >= 2 && boost::geometry::equals(outer.front(), outer.back()))
  {
    --n;
  }

  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (std::size_t i = 0; i < n; ++i)
  {
    const auto& pt = outer[i];
    Eigen::Vector2d diff(pt._point.x() - center._point.x(), pt._point.y() - center._point.y());
    cov += diff * diff.transpose();
  }
  cov /= static_cast<double>(n);

  const boost::math::chi_squared dist(2);
  const double chi2_val = quantile(dist, confidence);  // i.e. 5.991 for 95%

  double scale = 1.0;
  double step = 0.95;

  Eigen::VectorXd dummy_weights;
  Vector<UDT> center_vec(2);
  center_vec << center._point.x(), center._point.y();

  for (int tries = 0; tries < 50; ++tries)
  {
    Eigen::Matrix2d scaledCov = scale * chi2_val * cov;

    Vector<double> x(2);
    x << center._point.x(), center._point.y();
    Matrix<double> P = scaledCov;

    auto [sigmaPoints, weights] = transformation::createSigmaPoints(x, P).value();

    bool all_inside = true;
    for (int i = 0; i < sigmaPoints.cols(); ++i)
    {
      double px = to_val(sigmaPoints(0, i));
      double py = to_val(sigmaPoints(1, i));
      if (!boost::geometry::covered_by(Point(px, py), polygon))
      {
        all_inside = false;
        break;
      }
    }

    if (all_inside)
    {
      break;  // scale is valid
    }

    scale *= step;  // reduce and repeat
  }

  Eigen::Matrix2d finalCov = scale * chi2_val * cov;
  LOG_INF("Final scaled Covariance: " << finalCov);
  return finalCov;
}

template <class UDT>
Eigen::Matrix2d ROIBirthModel<UDT>::computeInnerCovariance(Point const& center,
                                                           Polygon::SinglePolygon const& polygon,
                                                           double confidence_level)
{
  Eigen::Vector2d dir_x(1.0, 0.0);
  Eigen::Vector2d dir_y(0.0, 1.0);
  Eigen::Vector2d c(center._point.x(), center._point.y());

  auto maxReach = [&](const Eigen::Vector2d& dir) -> double {
    double max_dist = 0.0;
    for (double d = 0.0; d < 1000.0; d += 0.1)
    {
      Eigen::Vector2d p = c + d * dir;
      Point pt(p.x(), p.y());
      if (!boost::geometry::within(pt, polygon))
        break;
      max_dist = d;
    }
    return max_dist;
  };

  double max_x = maxReach(dir_x);
  double max_y = maxReach(dir_y);

  boost::math::chi_squared chi2(2);
  double k_squared = quantile(chi2, confidence_level);

  double var_x = std::pow(max_x, 2) / k_squared;
  double var_y = std::pow(max_y, 2) / k_squared;

  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  cov(0, 0) = var_x * 0.8;
  cov(1, 1) = var_y * 0.8;

  LOG_DEB("cov: \n" << cov);

  return cov;
}
template <class UDT>
void ROIBirthModel<UDT>::reset()
{
  LOG_DEB("Reset ROI Birth Model");
}

}  // namespace ttb