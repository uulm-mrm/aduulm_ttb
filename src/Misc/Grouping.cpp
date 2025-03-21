#include "tracking_lib/Misc/Grouping.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/States/Innovation.h"
#include "tracking_lib/Graph/Graph.h"

#include <tracy/tracy/Tracy.hpp>

namespace ttb::grouping
{

auto constexpr tracy_color = tracy::Color::Ivory4;

std::vector<Group> group(std::vector<State> const& states, MeasurementContainer const& measurement_container)
{
  ZoneScopedNC("Grouping::group", tracy_color);
  // build graph
  std::map<MeasurementId, std::vector<StateId>> measToTrack;  // this meas is used to update all this Tracks
  for (State const& track : states)
  {
    for (auto const& [measId, _] : track._innovation.at(measurement_container._id)._updates)
    {
      if (measId == NOT_DETECTED)
      {
        continue;
      }
      LOG_DEB("measID " << measId.value_ << " associated to track label " << track._label.value_ << " and track id "
                        << track._id.value_);
      measToTrack[measId].push_back(track._id);
    }
  }
  std::vector<StateId> nodes;
  // the tracks are connected, if they have a common measurement
  std::vector<graph::Edge<MeasurementId, StateId, double>> edges;
  for (State const& track1 : states)
  {
    nodes.push_back(track1._id);
    for (auto const& [measId, _] : track1._innovation.at(measurement_container._id)._updates)
    {
      if (measId == NOT_DETECTED)
      {
        continue;
      }
      for (StateId track2_id : measToTrack.at(measId))
      {
        if (track1._id != track2_id)
        {
          edges.emplace_back(measId, track1._id, track2_id, 1);
        }
      }
    }
  }
  std::vector<std::vector<StateId>> const comps = graph::DiGraph(nodes, edges).components();

  std::vector<Group> groups;
  groups.reserve(comps.size());
  std::vector<MeasurementId> addedMeasurements;
  addedMeasurements.reserve(measurement_container._data.size());
  for (std::vector<StateId> const& comp : comps)
  {
    Group group{ .tracks = {},
                 .measurement_container = { ._id = measurement_container._id,
                                            ._egoMotion = measurement_container._egoMotion,
                                            ._time = measurement_container._time,
                                            ._sensorInfo = measurement_container._sensorInfo } };
    group.tracks.reserve(states.size());
    group.measurement_container._data.reserve(measurement_container._data.size());
    for (StateId trackId : comp)
    {
      auto const it = std::ranges::find_if(states, [trackId](State const& track) { return track._id == trackId; });
      assert(it != states.end());
      group.tracks.push_back(*it);
      for (auto const& [measId, _] : it->_innovation.at(measurement_container._id)._updates)
      {
        if (measId == NOT_DETECTED)
        {
          continue;
        }
        if (std::ranges::find_if(group.measurement_container._data, [measId](Measurement const& meas) {
              return meas._id == measId;
            }) == group.measurement_container._data.end())
        {
          // adds measurement to this container since it was not associated until now
          auto measIt = std::ranges::find_if(measurement_container._data,
                                             [measId](Measurement const& meas) { return meas._id == measId; });
          assert(measIt != measurement_container._data.end());
          group.measurement_container._data.emplace_back(*measIt);
          addedMeasurements.push_back(measId);
        }
      }
    }
    groups.push_back(std::move(group));
  }
  MeasurementContainer nonAssoc{ ._id = measurement_container._id,
                                 ._egoMotion = measurement_container._egoMotion,
                                 ._time = measurement_container._time,
                                 ._sensorInfo = measurement_container._sensorInfo };
  assert(measurement_container._data.size() >= addedMeasurements.size());
  nonAssoc._data.reserve(measurement_container._data.size() - addedMeasurements.size());
  for (Measurement const& meas : measurement_container._data)  // collect all non-associated measurement so far and
                                                               // create a separate MeasurementContainer for them
  {
    if (std::ranges::find(addedMeasurements, meas._id) == addedMeasurements.end())
    {
      nonAssoc._data.emplace_back(meas);
    }
  }
  assert([&] {  // NOLINT
    std::size_t num_split_meas = 0;
    for (auto const& [_, meas_container] : groups)
    {
      num_split_meas += meas_container._data.size();
    }
    num_split_meas += nonAssoc._data.size();
    return num_split_meas == measurement_container._data.size();
  }());
  assert([&] {  // NOLINT
    std::size_t num_split_tracks = 0;
    for (auto const& [tracks, _] : groups)
    {
      num_split_tracks += tracks.size();
    }
    return num_split_tracks == states.size();
  }());
  LOG_DEB("Group " + std::to_string(measurement_container._data.size()) + " Measurements into " +
          std::to_string(groups.size()) + " Track Groups, and " + std::to_string(nonAssoc._data.size()) +
          " non-assoc. Measurements");
  groups.push_back({ .tracks = {}, .measurement_container = std::move(nonAssoc) });
  return groups;
}

}  // namespace ttb::grouping