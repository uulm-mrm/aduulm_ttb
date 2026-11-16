#include "tracking_lib/Trackers/Id_Tracker.h"
//
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/MultiObjectStateDistributions/Utils.h"

namespace ttb
{

template class Id_Tracker<double>;
template class Id_Tracker<AD>;

template <class UDT>
Id_Tracker<UDT>::Id_Tracker(TTBManager<UDT>* manager) : _manager{ manager }
{
}

template <class UDT>
FILTER_TYPE Id_Tracker<UDT>::type() const
{
  return FILTER_TYPE::ID_TRACKER;
}

template <class UDT>
TTBManager<UDT>* Id_Tracker<UDT>::manager() const
{
  return _manager;
}

template <class UDT>
Time Id_Tracker<UDT>::time() const
{
  return _time;
}

template <class UDT>
void Id_Tracker<UDT>::cycle(std::vector<DataSourceContainer<UDT>> measContainerList)
{
  LOG_DEB("Old #Tracks: " << _tracks.size());
  for (DataSourceContainer<UDT> const& measurementContainer : measContainerList)
  {
    for (auto it = _tracks.begin(); it != _tracks.end();)
    {
      it->second.predict(measurementContainer._time - _time, measurementContainer._predictionInfo._egoMotion);
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
    std::vector<MeasurementId> assocMeas;
    for (auto const& [_, state] : _tracks)
    {
      _manager->database().states.insert(state);
    }
    utils::innovate_db(
        _manager,
        [&] {
          std::vector<StateId> ids;
          for (StateId id :
               _tracks | std::views::values | std::views::transform([](State<UDT> const& state) { return state._id; }))
          {
            ids.push_back(id);
          }
          return ids;
        }(),
        measurementContainer);
    for (auto& [object_id, track] : _tracks)
    {
      for (State<UDT> const& measurement : measurementContainer._measurements)
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
          if (_manager->database().state_updates.contains(
                  std::tuple{ track._id, measurementContainer._id, measurement._id }))
          {
            track = _manager->get_state(
                _manager->database()
                    .state_updates.find(std::tuple{ track._id, measurementContainer._id, measurement._id })
                    ->updated_state_id);
          }
          break;
        }
      }
    }
    LOG_DEB("Set up new Tracks");
    for (State<UDT> const& measurement : measurementContainer._measurements)
    {
      if (std::ranges::find(assocMeas, measurement._id) == assocMeas.end())
      {
        LOG_DEB("Measurement: " << measurement.toString() << " used to set up new Track");
        State state = _manager->getMeasModel(measurementContainer._source_id).create_state(measurement, true).value();
        state._label = Label{ measurement._objectId.value().value_ };
        _tracks.emplace(measurement._objectId.value(), std::move(state));
      }
    }
  }
}

template <class UDT>
std::vector<BernoulliState<UDT>> Id_Tracker<UDT>::get_estimate() const
{
  LOG_DEB("get Estimate");
  std::vector<BernoulliState<UDT>> estimate;
  for (State<UDT> const& track : _tracks | std::views::values)
  {
    estimate.emplace_back(track, 1);
  }
  LOG_DEB("Estimated " << estimate.size() << " Tracks");
  return estimate;
}

template <class UDT>
std::vector<StateId> Id_Tracker<UDT>::all_tracks() const
{
  return {};
}

template <class UDT>
void Id_Tracker<UDT>::reset()
{
  *this = Id_Tracker(_manager);
}

}  // namespace ttb
