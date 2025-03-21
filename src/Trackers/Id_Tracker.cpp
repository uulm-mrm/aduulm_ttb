#include "tracking_lib/Trackers/Id_Tracker.h"
//
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/States/Innovation.h"

namespace ttb
{

Id_Tracker::Id_Tracker(TTBManager* manager) : _manager{ manager }
{
}

FILTER_TYPE Id_Tracker::type() const
{
  return FILTER_TYPE::ID_TRACKER;
}

TTBManager* Id_Tracker::manager() const
{
  return _manager;
}

Time Id_Tracker::time() const
{
  return _time;
}

void Id_Tracker::cycle(Time time, std::vector<MeasurementContainer>&& measContainerList)
{
  LOG_DEB("Ushift cycle");
  LOG_DEB("Old #Tracks: " << _tracks.size());
  if (measContainerList.empty())
  {
    for (auto it = _tracks.begin(); it != _tracks.end();)
    {
      it->second.predict(time - _time, EgoMotionDistribution::zero());
      if (it->second.isEmpty())
      {
        it = _tracks.erase(it);
      }
      else
      {
        ++it;
      }
    }
    _time = time;
    return;
  }
  for (MeasurementContainer const& measurementContainer : measContainerList)
  {
    for (auto it = _tracks.begin(); it != _tracks.end();)
    {
      it->second.predict(measurementContainer._time - _time, measurementContainer._egoMotion);
      if (it->second.isEmpty())
      {
        it = _tracks.erase(it);
      }
      else
      {
        ++it;
      }
    }
    _time = measurementContainer._time;
    auto const& measModel = *_manager->getMeasModelMap().at(measurementContainer._id);
    std::vector<MeasurementId> assocMeas;
    for (auto& [object_id, track] : _tracks)
    {
      Innovation inno = measModel.calculateInnovation(measurementContainer, track);
      for (Measurement const& measurement : measurementContainer._data)
      {
        if (not measurement._objectId.has_value())
        {
          LOG_FATAL("Received Measurement without Object Id. Ignoring: " + measurement.toString());
          assocMeas.push_back(measurement._id);
          continue;
        }
        if (object_id == measurement._objectId.value())
        {
          LOG_DEB("Meas belongs to object");
          assocMeas.push_back(measurement._id);
          if (inno._updates.contains(measurement._id))
          {
            track = std::move(inno._updates.at(measurement._id).updated_dist);
          }
          break;
        }
      }
    }
    LOG_DEB("Set up new Tracks");
    for (Measurement const& measurement : measurementContainer._data)
    {
      if (std::ranges::find(assocMeas, measurement._id) == assocMeas.end())
      {
        LOG_DEB("Measurement: " << measurement.toString() << " used to set up new Track");
        State state = measModel.createState(measurement, true).value();
        state._label = Label{ measurement._objectId.value().value_ };
        _tracks.emplace(measurement._objectId.value(), std::move(state));
      }
    }
  }
}

std::vector<State> Id_Tracker::getEstimate() const
{
  LOG_DEB("get Estimate");
  std::vector<State> estimate;
  for (auto const& [object_id, track] : _tracks)
  {
    estimate.push_back(track);
  }
  LOG_DEB("Estimated " << estimate.size() << " Tracks");
  return estimate;
}

void Id_Tracker::reset()
{
  _tracks.clear();
}

}  // namespace ttb