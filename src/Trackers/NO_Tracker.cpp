#include "tracking_lib/Trackers/NO_Tracker.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"

namespace ttb
{
NO_Tracker::NO_Tracker(TTBManager* manager) : _manager{ manager }
{
}

FILTER_TYPE NO_Tracker::type() const
{
  return FILTER_TYPE::NO;
}

TTBManager* NO_Tracker::manager() const
{
  return _manager;
}

Time NO_Tracker::time() const
{
  return _time;
}

void NO_Tracker::cycle(Time time, std::vector<MeasurementContainer>&& measContainerList)
{
  LOG_DEB("cycle start");
  LOG_DEB("Received " << measContainerList.size() << " Measurement Containers");
  for (auto const& measurementContainer : measContainerList)
  {
    _tracks[measurementContainer._id];
    _tracks.at(measurementContainer._id).clear();
    LOG_DEB("Process Measurement Container with " << measurementContainer._data.size() << " Measurements");
    auto const& meas_model = *_manager->getMeasModelMap().at(measurementContainer._id);
    for (auto const& measurement : measurementContainer._data)
    {
      auto state_dist = meas_model.createState(measurement, true).value();
      _tracks.at(measurementContainer._id).push_back(std::move(state_dist));
      _time = std::max(_time, measurement._time);
    }
  }
}

std::vector<State> NO_Tracker::getEstimate() const
{
  LOG_DEB("Estimation with " + std::to_string(_tracks.size()) + " Tracks");
  std::vector<State> tracks;
  for (auto const& [model_id, states] : _tracks)
  {
    for (State const& state : states)
    {
      tracks.push_back(state);
    }
  }
  return tracks;
}

void NO_Tracker::reset()
{
}

}  // namespace ttb