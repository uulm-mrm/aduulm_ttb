#include "tracking_lib/DataContainer/Database.h"

namespace ttb
{

template class Database<double>;
template class Database<AD>;

template <class UDT>
void Database<UDT>::reset()
{
  *this = Database();
}

template <class UDT>
void Database<UDT>::cut_data_containers_to_max_size()
{
  ZoneScopedN("Database::cut_to_max_size");
  std::vector<StateId> deleted_meas;
  {
    std::size_t const before = data_containers.size();
    if (not data_containers.empty())
    {
      Time const last_time = data_containers.template get<Time>().rbegin()->_time;
      for (auto it = data_containers.template get<Time>().begin();
           it != data_containers.template get<Time>().end() and last_time - it->_time > keep_data;)
      {
        for (State<UDT> const& state : it->_measurements)
        {
          deleted_meas.push_back(state._id);
        }
        for (BernoulliState<UDT> const& state : it->_tracks)
        {
          deleted_meas.push_back(state.state._id);
        }
        it = data_containers.template get<Time>().erase(it);
      }
    }
    if (before != data_containers.size())
    {
      LOG_DEB(std::format(
          "Container {} size before / after  {} {}", "DataSourceContainer", before, data_containers.size()));
    }
  }
  std::vector<StateId> deleted_states;
  {
    std::size_t const before = states.size();
    if (not states.empty())
    {
      Time const last_time = states.template get<Time>().rbegin()->_time;
      for (auto it = states.template get<Time>().begin(); it != states.template get<Time>().end() and last_time - it->_time > keep_data;)
      {
        deleted_states.push_back(it->_id);
        it = states.template get<Time>().erase(it);
      }

      if (before != states.size())
      {
        LOG_DEB(std::format("State Container {} size before / after  {} {}", "States", before, states.size()));
      }
    }
  }
  {
    std::size_t const before = state_detection_probabilities.size();
    for (StateId state_id : deleted_states)
    {
      auto [first, second] = state_detection_probabilities.template get<StateId>().equal_range(state_id);
      for (auto it = first; it != second;)
      {
        it = state_detection_probabilities.template get<StateId>().erase(it);
      }
    }
    if (before != state_detection_probabilities.size())
    {
      LOG_DEB(std::format("Container {} size before / after  {} {}",
                          "state_detection_probabilities",
                          before,
                          state_detection_probabilities.size()));
    }
  }
  {
    std::size_t const before = state_meas_log_likelihood.size();
    for (StateId state_id : deleted_states)
    {
      auto [first, second] = state_meas_log_likelihood.template get<StateId>().equal_range(state_id);
      for (auto it = first; it != second;)
      {
        it = state_meas_log_likelihood.template get<StateId>().erase(it);
      }
    }
    if (before != state_meas_log_likelihood.size())
    {
      LOG_DEB(std::format("Container {} size before / after  {} {}",
                          "state_meas_log_likelihood",
                          before,
                          state_meas_log_likelihood.size()));
    }
  }
  {
    std::size_t const before = state_updates.size();
    for (StateId state_id : deleted_states)
    {
      auto [first, second] = state_updates.template get<StateId>().equal_range(state_id);
      for (auto it = first; it != second;)
      {
        it = state_updates.template get<StateId>().erase(it);
      }
    }
    if (before != state_updates.size())
    {
      LOG_DEB(std::format("Container {} size before / after  {} {}", "state_updates", before, state_updates.size()));
    }
  }
  {
    std::size_t const before = relevant_meas.size();
    for (StateId state_id : deleted_states)
    {
      auto [first, second] = relevant_meas.get<StateId>().equal_range(state_id);
      for (auto it = first; it != second;)
      {
        it = relevant_meas.get<StateId>().erase(it);
      }
    }
    if (before != relevant_meas.size())
    {
      LOG_DEB(std::format("Container {} size before / after  {} {}", "relevant_meas", before, relevant_meas.size()));
    }
  }
  {
    std::size_t const before = meas_clutter_intensity.size();
    for (StateId state_id : deleted_meas)
    {
      auto [first, second] = meas_clutter_intensity.equal_range(state_id);
      for (auto it = first; it != second;)
      {
        it = meas_clutter_intensity.template get<StateId>().erase(it);
      }
    }
    if (before != meas_clutter_intensity.size())
    {
      LOG_DEB(std::format(
          "Container {} size before / after  {} {}", "meas_clutter_intensity", before, meas_clutter_intensity.size()));
    }
  }
}

}  // namespace ttb
