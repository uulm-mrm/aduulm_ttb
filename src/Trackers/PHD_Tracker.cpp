#include "tracking_lib/Trackers/PHD_Tracker.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/BirthModels/StaticBirthModel.h"
#include "tracking_lib/States/Innovation.h"

#include <memory>
#include <utility>

#include <tracy/tracy/Tracy.hpp>

namespace ttb
{
constexpr auto tracy_color = tracy::Color::Goldenrod;

PHD_Tracker::PHD_Tracker(TTBManager* manager) : _manager{ manager }, _distribution{ _manager }
{
}

FILTER_TYPE PHD_Tracker::type() const
{
  return FILTER_TYPE::PHD;
}

TTBManager* PHD_Tracker::manager() const
{
  return _manager;
}

Time PHD_Tracker::time() const
{
  return _time;
}

void PHD_Tracker::cycle(Time time, std::vector<MeasurementContainer>&& measContainerList)
{
  ZoneScopedNC("PHD_Tracker::cycle", tracy_color);
  assert(_distribution.isValid() and "Invalid PHD Dist");
  LOG_DEB("PHD CYCLE with " << measContainerList.size() << " Measurement Container");
  if (measContainerList.empty())
  {
    _distribution.predict(time - _time, EgoMotionDistribution::zero());
    _time = time;
  }
  Time const start_cycle_time = _time;
  for (auto const& measurementContainer : measContainerList)
  {
    ZoneScopedNC("PHD_Tracker::ic_cycle", tracy_color);
    ZoneText(measurementContainer._id.value_.c_str(), measurementContainer._id.value_.size());
    auto const meas_update_start = std::chrono::high_resolution_clock::now();
    assert(_distribution.isValid() && "cycle start");
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
      _distribution.addTracks(
          _manager->getBirthModel().getBirthTracks(measurementContainer, {}, _distribution._tracks));
    }
    _distribution.calcInnovation(measurementContainer);
    _distribution.update(measurementContainer);

    if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
    {
      LOG_DEB("Add Dynamic Birth Tracks");
      _distribution.addTracks(_manager->getBirthModel().getBirthTracks(
          measurementContainer, _distribution.meas_assignment_prob(), _distribution._tracks));
    }
    assert(_distribution.isValid() && "before postprocessing");
    _time = measurementContainer._time;
    if (_manager->params().show_gui)
    {
      std::lock_guard lock(_manager->vizu().add_data_mutex);
      _manager->vizu()._meas_model_data[measurementContainer._id]._computation_time.emplace_back(
          start_cycle_time, std::chrono::high_resolution_clock::now() - meas_update_start);
    }
  }
}

std::vector<State> PHD_Tracker::getEstimate() const
{
  std::vector<State> tracks = _distribution.getEstimate();
  for (State& track : tracks)
  {
    track._misc["origin"] = std::string("PHD");
  }
  return tracks;
}

void PHD_Tracker::reset()
{
  LOG_DEB("Reset PHD Tracker");
  _distribution = PHDDistribution(_manager);
  _time = Time{ 0s };
}

}  // namespace ttb
