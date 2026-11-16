#include "tracking_lib/Trackers/NO_Tracker.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"

namespace ttb
{

template class NO_Tracker<double>;
template class NO_Tracker<AD>;

template <class UDT>
NO_Tracker<UDT>::NO_Tracker(TTBManager<UDT>* manager) : _manager{ manager }
{
}

template <class UDT>
FILTER_TYPE NO_Tracker<UDT>::type() const
{
  return FILTER_TYPE::NO;
}

template <class UDT>
TTBManager<UDT>* NO_Tracker<UDT>::manager() const
{
  return _manager;
}

template <class UDT>
Time NO_Tracker<UDT>::time() const
{
  return _time;
}

template <class UDT>
void NO_Tracker<UDT>::cycle(std::vector<DataSourceContainer<UDT>> measContainerList)
{
  LOG_DEB("cycle start");
  LOG_DEB("Received " << measContainerList.size() << " Measurement Containers");
  for (auto const& measurementContainer : measContainerList)
  {
    _tracks[measurementContainer._source_id];
    _tracks.at(measurementContainer._source_id).clear();
    LOG_DEB("Process Measurement Container with " << measurementContainer._measurements.size() << " Measurements");
    auto const& meas_model = _manager->getMeasModel(measurementContainer._source_id);
    for (auto const& measurement : measurementContainer._measurements)
    {
      auto state_dist = meas_model.create_state(measurement, true).value();
      _tracks.at(measurementContainer._source_id).push_back(std::move(state_dist));
      _time = std::max(_time, measurement._time);
    }
  }
}

template <class UDT>
std::vector<BernoulliState<UDT>> NO_Tracker<UDT>::get_estimate() const
{
  LOG_DEB("Estimation with " + std::to_string(_tracks.size()) + " Tracks");
  std::vector<BernoulliState<UDT>> tracks;
  for (auto const& [model_id, states] : _tracks)
  {
    for (State<UDT> const& state : states)
    {
      tracks.emplace_back(state, 1);
    }
  }
  return tracks;
}

template <class UDT>
std::vector<StateId> NO_Tracker<UDT>::all_tracks() const
{
  return {};
}

template <class UDT>
void NO_Tracker<UDT>::reset()
{
}

}  // namespace ttb
