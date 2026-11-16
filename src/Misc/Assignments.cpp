
#include "tracking_lib/Misc/Assignments.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include <ranges>

namespace ttb
{

template double cost_assignment(Cost const& cost, Assignment const& assignment);
template AD cost_assignment(Cost const& cost, Assignment const& assignment);
template <class UDT>
UDT cost_assignment(Cost const& cost, Assignment const& assignment)
{
  UDT cost_prod = 1;
  for (StateAssignment const& state_assignment : assignment)
  {
    if (auto const cost_it = std::ranges::find(cost, state_assignment.id, &StateLikelihood::id);
        not state_assignment.alive)
    {
      assert(cost_it != cost.end());
      cost_prod *= 1 - cost_it->surv_likelihood;
    }
    else
    {
      assert(cost_it != cost.end());
      for (auto const& [source_id, meas_id] : state_assignment.meas_assignments)
      {
        auto const source_it =
            std::ranges::find(cost_it->alive,
                              source_id,
                              &std::pair<DataSourceContainerId, std::vector<std::pair<MeasurementId, double>>>::first);
        assert(source_it != cost_it->alive.end());
        auto const meas_it = std::ranges::find(source_it->second, meas_id, &std::pair<MeasurementId, double>::first);
        assert(meas_it != source_it->second.end());
        cost_prod *= meas_it->second;
      }
    }
  }
  return cost_prod;
}

template SoftCosts<double> soft_cost_assignment(TTBManager<double> const& manager,
                                                std::span<DataSourceContainer<double> const> data_containers,
                                                Cost const& soft_cost,
                                                Assignment const& assignment);
template SoftCosts<AD> soft_cost_assignment(TTBManager<AD> const& manager,
                                            std::span<DataSourceContainer<AD> const> data_containers,
                                            Cost const& soft_cost,
                                            Assignment const& assignment);
template <class UDT>
SoftCosts<UDT> soft_cost_assignment(TTBManager<UDT> const& manager,
                                    std::span<DataSourceContainer<UDT> const> data_containers,
                                    Cost const& soft_cost,
                                    Assignment const& assignment)
{
  SoftCosts<UDT> soft_costs;
  for (DataSourceContainer<UDT> const& data_source_container : data_containers)
  {
    std::size_t num_detected = 0;
    UDT soft_clutter_likelihood = 1;
    UDT soft_detection_likelihood = 1;
    if (not data_source_container._sensorInfo._sensor_fov.has_value())
    {
      LOG_FATAL("could not compute self assessment without fov");
      LOG_FATAL(std::stacktrace::current());
      std::terminate();
    }
    for (StateAssignment const& state_assignment : assignment)
    {
      if (auto const cost_it = std::ranges::find(soft_cost, state_assignment.id, &StateLikelihood::id);
          not state_assignment.alive)
      {
        assert(cost_it != soft_cost.end());
        soft_clutter_likelihood *= 1 - cost_it->surv_likelihood;
        soft_detection_likelihood *= 1 - cost_it->surv_likelihood;
      }
      else
      {
        assert(cost_it != soft_cost.end());

        for (auto const& [source_id, meas_id] : state_assignment.meas_assignments)
        {
          if (source_id != data_source_container._id)
          {
            continue;
          }
          auto const source_it = std::ranges::find(
              cost_it->alive,
              source_id,
              &std::pair<DataSourceContainerId, std::vector<std::pair<MeasurementId, double>>>::first);
          assert(source_it != cost_it->alive.end());
          auto const meas_it = std::ranges::find(source_it->second, meas_id, &std::pair<MeasurementId, double>::first);
          assert(meas_it != source_it->second.end());
          UDT const pd = manager.database()
                             .state_detection_probabilities.find(std::tuple{ state_assignment.id, source_it->first })
                             ->detection_probability;
          if (meas_it->first == NOT_DETECTED)
          {
            soft_clutter_likelihood *= (1 - pd);
          }
          else
          {
            soft_clutter_likelihood *= pd * meas_it->second;
            soft_detection_likelihood *= meas_it->second;
            num_detected += 1;
          }
        }
      }
    }
    std::size_t const num_clutter = data_source_container._measurements.size() - num_detected;
    soft_costs.point_likelihood[data_source_container._source_id] = 666;
    soft_costs.soft_clutter_likelihood[data_source_container._source_id] = soft_clutter_likelihood;
    soft_costs.soft_detection_likelihood[data_source_container._source_id] = soft_detection_likelihood;
    soft_costs.num_clutter[data_source_container._source_id] = num_clutter;
    soft_costs.num_detected[data_source_container._source_id] = num_detected;
  }
  return soft_costs;
}

std::string to_string(Assignment const& assignment)
{
  std::string out;
  for (StateAssignment const& state_assignment : assignment)
  {
    out += "State " + std::to_string(state_assignment.id.value_) + ": ";
    if (not state_assignment.alive)
    {
      out += "DEAD\n";
    }
    else
    {
      for (auto const& [id, meas_id] : state_assignment.meas_assignments)
      {
        out += "Source " + std::to_string(id.value_) + " -> " + std::to_string(meas_id.value_);
      }
      out += "\n";
    }
  }
  return out;
}

std::string to_string(Cost const& cost)
{
  std::string out;
  for (StateLikelihood const& state_cost : cost)
  {
    out += "State " + std::to_string(state_cost.id.value_) + ": \n";
    out += "\asurv: " + std::to_string(state_cost.surv_likelihood) + "\n";
    for (auto const& [id, meas_assignments] : state_cost.alive)
    {
      out += "\tid: " + std::to_string(id.value_) + "\n";
      for (auto const& [meas_id, meas_cost] : meas_assignments)
      {
        out += "\t\tmeas: " + std::to_string(meas_id.value_) + " -> " + std::to_string(meas_cost) + "\n";
      }
    }
  }
  return out;
}

bool is_valid(Assignment const& assignment)
{
  std::vector<MeasurementId> used_meas;
  for (auto const& [state_id, alive, meas_assign] : assignment)
  {
    if (not alive and not meas_assign.empty())
    {
      LOG_FATAL("track not alive but meas assigned");
      return false;
    }
    for (auto const& [source_id, meas_id] : meas_assign)
    {
      if (meas_id != NOT_DETECTED)
      {
        if (std::ranges::find(used_meas, meas_id) != used_meas.end())
        {
          LOG_FATAL("Meas " << meas_id << " is used multiple times!!!");
          return false;
        }
        used_meas.push_back(meas_id);
      }
    }
  }
  return true;
}

}  // namespace ttb
