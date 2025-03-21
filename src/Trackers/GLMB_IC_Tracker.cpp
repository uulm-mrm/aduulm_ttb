#include "tracking_lib/Trackers/GLMB_IC_Tracker.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/SelfAssessment/SelfAssessment.h"

#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <tracy/tracy/Tracy.hpp>

auto constexpr tracy_color = tracy::Color::Orange3;

namespace ttb
{

GLMB_IC_Tracker::GLMB_IC_Tracker(TTBManager* manager)
  : _manager{ manager }, _distribution{ _manager }, _parameter_estimation{ _manager }
{
}

void GLMB_IC_Tracker::cycle(Time time, std::vector<MeasurementContainer>&& measContainers)
{
  ZoneScopedNC("GLMB_IC_Tracker::cycle", tracy_color);
  LOG_DEB("GLMB_IC_Tracker::cycle");
  assert(_distribution.isValid());

  if (measContainers.empty())
  {
    _distribution.predict(time - _time, EgoMotionDistribution::zero());
    _time = time;
  }
  auto const start_glmb_ic_cycle_time = _time;
  for (auto const& measurementContainer : measContainers)
  {
    ZoneScopedNC("GLMB_IC_Tracker::ic_cycle", tracy_color);
    ZoneText(measurementContainer._id.value_.c_str(), measurementContainer._id.value_.size());
    auto const meas_update_start = std::chrono::high_resolution_clock::now();
    Duration deltaT = measurementContainer._time - _time;
    if (deltaT < 0ms)
    {
      LOG_FATAL("Receiving Measurement Container from the ancient past .... - MUST NOT HAPPEN");
      LOG_FATAL("Filter Time: " + to_string(_time) +
                " Measurement Container Time: " + to_string(measurementContainer._time));
      LOG_FATAL("MeasurementContainer: " << measurementContainer._id.value_);
      throw std::runtime_error("Receiving Measurement Container from the ancient past .... - MUST NOT HAPPEN");
    }
    LOG_DEB("Processing Msg. of sensor " + measurementContainer._id.value_
            << " with " << measurementContainer._data.size() << " Detections");
    _distribution.predict(deltaT, measurementContainer._egoMotion);
    _time = measurementContainer._time;

    if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::STATIC)
    {
      LOG_DEB("Add Static Birth Tracks");
      _distribution.add_tracks(_manager->getBirthModel().getBirthTracks(measurementContainer, {}, {}));
      _distribution.postProcessPrediction();
    }
    _distribution.calcInnovation(measurementContainer);
    _distribution.update(measurementContainer);
    _parameter_estimation.update(
        measurementContainer,
        _distribution.clutter_distribution(static_cast<Index>(measurementContainer._data.size())),
        _distribution.detection_distribution());
    if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
    {
      LOG_DEB("Add Dynamic Birth Tracks");
      _distribution.add_tracks(_manager->getBirthModel().getBirthTracks(
          measurementContainer, _distribution.probOfAssigment(measurementContainer), {}));
      _distribution.postProcessUpdate();
    }
    _time = measurementContainer._time;
    if (_manager->params().show_gui)
    {
      std::lock_guard lock(_manager->vizu().add_data_mutex);
      _manager->vizu()._meas_model_data[measurementContainer._id]._computation_time.emplace_back(
          start_glmb_ic_cycle_time, std::chrono::high_resolution_clock::now() - meas_update_start);
    }
  }
}

std::vector<State> GLMB_IC_Tracker::getEstimate() const
{
  return _distribution.getEstimate();
}

void GLMB_IC_Tracker::reset()
{
  _time = Time{ 0s };
  _distribution = GLMBDistribution(_manager);
}

TTBManager* GLMB_IC_Tracker::manager() const
{
  return _manager;
}

Time GLMB_IC_Tracker::time() const
{
  return _time;
}

FILTER_TYPE GLMB_IC_Tracker::type() const
{
  return FILTER_TYPE::GLMB_IC;
}

}  // namespace ttb
