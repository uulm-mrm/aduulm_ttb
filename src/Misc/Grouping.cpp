#include "tracking_lib/Misc/Grouping.h"

#include "tracking_lib/States/State.h"
#include "tracking_lib/Graph/Graph.h"
#include <tracy/tracy/Tracy.hpp>

namespace ttb::grouping
{

template std::vector<Group2<double>> group_db(TTBManager<double> const& manager,
                                              std::vector<StateId> const& states,
                                              DataSourceContainer<double> const& measurement_container);
template std::vector<Group2<AD>> group_db(TTBManager<AD> const& manager,
                                          std::vector<StateId> const& states,
                                          DataSourceContainer<AD> const& measurement_container);
template <class UDT>
std::vector<Group2<UDT>> group_db(TTBManager<UDT> const& manager,
                                  std::vector<StateId> const& states,
                                  DataSourceContainer<UDT> const& measurement_container)
{
  std::vector<MSGroup2<UDT>> groups = group_db(manager, states, std::span{ &measurement_container, 1 });
  std::vector<Group2<UDT>> better_groups;
  for (MSGroup2<UDT>& g : groups)
  {
    better_groups.emplace_back(std::move(g.tracks), std::move(g.measurement_containers.front()));
  }
  return better_groups;
}


template std::vector<MSGroup2<double>> group_db(TTBManager<double> const& manager,
                                    std::vector<StateId> const& states,
                                    std::span<DataSourceContainer<double> const> measurement_containers);
template std::vector<MSGroup2<AD>> group_db(TTBManager<AD> const& manager,
                                    std::vector<StateId> const& states,
                                    std::span<DataSourceContainer<AD> const> measurement_containers);
template <class UDT>
std::vector<MSGroup2<UDT>> group_db(TTBManager<UDT> const& manager,
                                    std::vector<StateId> const& states,
                                    std::span<DataSourceContainer<UDT> const> measurement_containers)
{
  ZoneScopedN("Grouping::group");
  // build graph
  std::map<MeasurementId, std::vector<StateId>> measToTrack;  // this meas is used to update all this Tracks
  for (StateId track_id : states)
  {
    for (DataSourceContainer<UDT> const& data_source_container : measurement_containers)
    {
      for (State<UDT> const& meas : data_source_container._measurements)
      {
        if (manager.database().state_updates.contains(std::tuple{ track_id, data_source_container._id, meas._id }))
        {
          measToTrack[meas._id].push_back(track_id);
        }
      }
    }
  }
  std::vector<StateId> nodes;
  // the tracks are connected, if they have a common measurement
  std::vector<graph::Edge<std::pair<DataSourceContainerId, MeasurementId>, StateId, double>> edges;
  for (StateId track1_id : states)
  {
    nodes.push_back(track1_id);
    for (DataSourceContainer<UDT> const& data_source_container : measurement_containers)
    {
      for (State<UDT> const& meas : data_source_container._measurements)
      {
        if (manager.database().state_updates.contains(std::tuple{ track1_id, data_source_container._id, meas._id }))
        {
          for (StateId track2_id : measToTrack.at(meas._id))
          {
            LOG_DEB("meas " << meas._id << " is used to update state: " << track2_id);
            edges.emplace_back(std::pair{ data_source_container._id, meas._id }, track1_id, track2_id, 1);
          }
        }
      }
    }
  }
  graph::DiGraph const graph(std::move(nodes), std::move(edges));
  std::vector<std::vector<StateId>> const comps = graph.components();

  std::vector<MSGroup2<UDT>> groups;
  std::unordered_set<MeasurementId> addedMeasurements;
  for (std::vector<StateId> const& comp : comps)
  {
    MSGroup2<UDT> group{ .tracks = {}, .measurement_containers = [&] {
                          std::vector<DataSourceContainer<UDT>> cont;
                          for (DataSourceContainer<UDT> const& data_source_container : measurement_containers)
                          {
                            cont.push_back({ ._source_id = data_source_container._source_id,
                                             ._time = data_source_container._time,
                                             ._sensorInfo = data_source_container._sensorInfo,
                                             ._predictionInfo = data_source_container._predictionInfo,
                                             ._id = data_source_container._id });
                          }
                          return cont;
                        }() };

    for (StateId trackId : comp)
    {
      auto const it = std::ranges::find(states, trackId);
      assert(it != states.end());
      group.tracks.push_back(*it);
      for (auto const& [data_source_id, meas_id] : graph.edges_from(trackId))
      {
        if (not addedMeasurements.contains(meas_id))
        {
          addedMeasurements.insert(meas_id);
          std::size_t const ind =
              std::distance(measurement_containers.begin(),
                            std::ranges::find(measurement_containers, data_source_id, &DataSourceContainer<UDT>::_id));
          assert(std::ranges::find(measurement_containers[ind]._measurements, meas_id, &State<UDT>::_id) !=
                 measurement_containers[ind]._measurements.end());
          group.measurement_containers.at(ind)._measurements.push_back(
              *std::ranges::find(measurement_containers[ind]._measurements, meas_id, &State<UDT>::_id));
        }
      }
    }
    groups.push_back(std::move(group));
  }
  MSGroup2<UDT> non_assoc{ .tracks = {}, .measurement_containers = [&] {
                            std::vector<DataSourceContainer<UDT>> cont;
                            for (DataSourceContainer<UDT> const& data_source_container : measurement_containers)
                            {
                              cont.push_back({ ._source_id = data_source_container._source_id,
                                               ._time = data_source_container._time,
                                               ._sensorInfo = data_source_container._sensorInfo,
                                               ._predictionInfo = data_source_container._predictionInfo,
                                               ._id = data_source_container._id });
                            }
                            return cont;
                          }() };
  for (auto const& [ind, data_source_container] : std::views::enumerate(measurement_containers))
  {
    for (State<UDT> const& state : data_source_container._measurements)
      if (not addedMeasurements.contains(state._id))
      {
        non_assoc.measurement_containers.at(ind)._measurements.emplace_back(state);
      }
  }
  if (std::ranges::any_of(non_assoc.measurement_containers, [](DataSourceContainer<UDT> const& data_source_container) {
        return not data_source_container._measurements.empty();
      }))
  {
    groups.push_back(std::move(non_assoc));
  }
  if (groups.empty())
  {
    assert(states.empty() and
           std::ranges::all_of(measurement_containers, [&](DataSourceContainer<UDT> const& data_source_container) {
             return data_source_container._measurements.empty();
           }));
    std::vector<DataSourceContainer<UDT>> containers;
    containers.insert(containers.end(), measurement_containers.begin(), measurement_containers.end());
    groups.emplace_back(MSGroup2{ {}, std::move(containers) });
  }
  LOG_DEB(std::format("Group {} tracks into {} groups", states.size(), groups.size()));
  assert([&] {  // NOLINT
    std::size_t num_split_meas = 0;
    for (auto const& [_, meas_containers] : groups)
    {
      for (DataSourceContainer<UDT> const& meas_container : meas_containers)
      {
        num_split_meas += meas_container._measurements.size();
      }
    }
    std::size_t ori_num_meas = 0;
    for (DataSourceContainer<UDT> const& data_source_container : measurement_containers)
    {
      ori_num_meas += data_source_container._measurements.size();
    }
    return num_split_meas == ori_num_meas;
  }());
  assert([&] {  // NOLINT
    std::size_t num_split_tracks = 0;
    for (auto const& [tracks, _] : groups)
    {
      num_split_tracks += tracks.size();
    }
    return num_split_tracks == states.size();
  }());
  assert([&] {  // NOLINT
    for (auto const& [ind, data_source_container] : std::views::enumerate(measurement_containers))
    {
      for (State<UDT> const& state : data_source_container._measurements)
      {
        int found = 0;
        for (MSGroup2<UDT> const& group : groups)
        {
          auto const it =
              std::ranges::find(group.measurement_containers.at(ind)._measurements, state._id, &State<UDT>::_id);
          if (it != group.measurement_containers.at(ind)._measurements.end())
          {
            ++found;
          }
        }
        if (found != 1)
        {
          LOG_FATAL(std::format("Measurement {} is {} times in the splitted measurement container", state._id, found));
          return false;
        }
      }
    }
    return true;
  }());
  assert(not groups.empty());
  return groups;
}


template std::vector<Group<double>> group(std::vector<State<double>> const& states,
                              DataSourceContainer<double> const& measurement_container);
template std::vector<Group<AD>> group(std::vector<State<AD>> const& states,
                              DataSourceContainer<AD> const& measurement_container);
template <class UDT>
std::vector<Group<UDT>> group(std::vector<State<UDT>> const& states,
                              DataSourceContainer<UDT> const& measurement_container)
{
  std::vector<MSGroup<UDT>> ms_groups = group<UDT>(states, std::span(&measurement_container, 1));
  std::vector<Group<UDT>> groups;
  for (MSGroup<UDT>& ms_group : ms_groups)
  {
    assert(ms_group.measurement_containers.size() == 1);
    groups.emplace_back(std::move(ms_group.tracks), std::move(ms_group.measurement_containers.front()));
  }
  return groups;
}

template <class UDT>
std::vector<MSGroup<UDT>> group(std::span<State<UDT> const> states,
                                std::span<DataSourceContainer<UDT> const> measurement_containers)
{
  ZoneScopedN("Grouping::group");
  // build graph
  std::map<MeasurementId, std::vector<StateId>> measToTrack;  // this meas is used to update all this Tracks
  for (State<UDT> const& track : states)
  {
    assert(track._type == STATE_TYPE::UPDATED_PRIOR);
    for (DataSourceContainer<UDT> const& data_source_container : measurement_containers)
    {
      for (State<UDT> const& meas : data_source_container._measurements)
      {
        if (track._manager->database().state_updates.contains(
                std::tuple{ track._id, data_source_container._id, meas._id }))
        {
          measToTrack[meas._id].push_back(track._id);
        }
      }
    }
  }
  std::vector<StateId> nodes;
  // the tracks are connected, if they have a common measurement
  std::vector<graph::Edge<std::pair<DataSourceContainerId, MeasurementId>, StateId, double>> edges;
  for (State<UDT> const& track1 : states)
  {
    nodes.push_back(track1._id);
    for (DataSourceContainer<UDT> const& data_source_container : measurement_containers)
    {
      for (State<UDT> const& meas : data_source_container._measurements)
      {
        if (track1._manager->database().state_updates.contains(
                std::tuple{ track1._id, data_source_container._id, meas._id }))
        {
          for (StateId track2_id : measToTrack.at(meas._id))
          {
            LOG_DEB("meas " << meas._id << " is used to update state: " << track2_id);
            edges.emplace_back(std::pair{ data_source_container._id, meas._id }, track1._id, track2_id, 1);
          }
        }
      }
    }
  }
  graph::DiGraph const graph(std::move(nodes), std::move(edges));
  std::vector<std::vector<StateId>> const comps = graph.components();

  std::vector<MSGroup<UDT>> groups;
  std::unordered_set<MeasurementId> addedMeasurements;
  for (std::vector<StateId> const& comp : comps)
  {
    MSGroup<UDT> group{ .tracks = {}, .measurement_containers = [&] {
                         std::vector<DataSourceContainer<UDT>> cont;
                         for (DataSourceContainer<UDT> const& data_source_container : measurement_containers)
                         {
                           cont.push_back({ ._source_id = data_source_container._source_id,
                                            ._time = data_source_container._time,
                                            ._sensorInfo = data_source_container._sensorInfo,
                                            ._predictionInfo = data_source_container._predictionInfo,
                                            ._id = data_source_container._id });
                         }
                         return cont;
                       }() };

    for (StateId trackId : comp)
    {
      auto const it = std::ranges::find(states, trackId, &State<UDT>::_id);
      assert(it != states.end());
      group.tracks.push_back(*it);
      for (auto const& [data_source_id, meas_id] : graph.edges_from(trackId))
      {
        if (not addedMeasurements.contains(meas_id))
        {
          addedMeasurements.insert(meas_id);
          std::size_t const ind =
              std::distance(measurement_containers.begin(),
                            std::ranges::find(measurement_containers, data_source_id, &DataSourceContainer<UDT>::_id));
          assert(std::ranges::find(measurement_containers[ind]._measurements, meas_id, &State<UDT>::_id) !=
                 measurement_containers[ind]._measurements.end());
          group.measurement_containers.at(ind)._measurements.push_back(
              *std::ranges::find(measurement_containers[ind]._measurements, meas_id, &State<UDT>::_id));
        }
      }
    }
    groups.push_back(std::move(group));
  }
  MSGroup non_assoc{ .tracks = {}, .measurement_containers = [&] {
                      std::vector<DataSourceContainer<UDT>> cont;
                      for (DataSourceContainer<UDT> const& data_source_container : measurement_containers)
                      {
                        cont.push_back({ ._source_id = data_source_container._source_id,
                                         ._time = data_source_container._time,
                                         ._sensorInfo = data_source_container._sensorInfo,
                                         ._predictionInfo = data_source_container._predictionInfo,
                                         ._id = data_source_container._id });
                      }
                      return cont;
                    }() };
  for (auto const& [ind, data_source_container] : std::views::enumerate(measurement_containers))
  {
    for (State<UDT> const& state : data_source_container._measurements)
      if (not addedMeasurements.contains(state._id))
      {
        non_assoc.measurement_containers.at(ind)._measurements.emplace_back(state);
      }
  }
  if (std::ranges::any_of(non_assoc.measurement_containers, [](DataSourceContainer<UDT> const& data_source_container) {
        return not data_source_container._measurements.empty();
      }))
  {
    groups.push_back(std::move(non_assoc));
  }
  if (groups.empty())
  {
    assert(states.empty() and
           std::ranges::all_of(measurement_containers, [&](DataSourceContainer<UDT> const& data_source_container) {
             return data_source_container._measurements.empty();
           }));
    std::vector<DataSourceContainer<UDT>> containers;
    containers.insert(containers.end(), measurement_containers.begin(), measurement_containers.end());
    groups.emplace_back(MSGroup{ {}, std::move(containers) });
  }
  LOG_DEB(std::format("Group {} tracks into {} groups", states.size(), groups.size()));
  assert([&] {  // NOLINT
    std::size_t num_split_meas = 0;
    for (auto const& [_, meas_containers] : groups)
    {
      for (DataSourceContainer<UDT> const& meas_container : meas_containers)
      {
        num_split_meas += meas_container._measurements.size();
      }
    }
    std::size_t ori_num_meas = 0;
    for (DataSourceContainer<UDT> const& data_source_container : measurement_containers)
    {
      ori_num_meas += data_source_container._measurements.size();
    }
    return num_split_meas == ori_num_meas;
  }());
  assert([&] {  // NOLINT
    std::size_t num_split_tracks = 0;
    for (auto const& [tracks, _] : groups)
    {
      num_split_tracks += tracks.size();
    }
    return num_split_tracks == states.size();
  }());
  assert([&] {  // NOLINT
    for (auto const& [ind, data_source_container] : std::views::enumerate(measurement_containers))
    {
      for (State<UDT> const& state : data_source_container._measurements)
      {
        int found = 0;
        for (MSGroup<UDT> const& group : groups)
        {
          auto const it =
              std::ranges::find(group.measurement_containers.at(ind)._measurements, state._id, &State<UDT>::_id);
          if (it != group.measurement_containers.at(ind)._measurements.end())
          {
            ++found;
          }
        }
        if (found != 1)
        {
          LOG_FATAL(std::format("Measurement {} is {} times in the splitted measurement container", state._id, found));
          return false;
        }
      }
    }
    return true;
  }());
  assert(not groups.empty());
  return groups;
}

}  // namespace ttb::grouping
