#pragma once

#include "tracking_lib/DataContainer/DataSourceContainer.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

struct StateAssignment
{
  StateId id{};
  bool alive{};
  std::vector<std::pair<DataSourceContainerId, MeasurementId>> meas_assignments{};
  auto operator<=>(StateAssignment const&) const = default;
};

/// assign the list of states this measurement from this data_source_container
using Assignment = std::vector<StateAssignment>;

struct StateLikelihood
{
  StateId id{};
  double surv_likelihood{ 1 };  // probability that this state survives
  std::vector<std::pair<DataSourceContainerId, std::vector<std::pair<MeasurementId, double>>>> alive{};
};
using Cost = std::vector<StateLikelihood>;

[[nodiscard]] bool is_valid(Assignment const& assignment);

template <class UDT>
[[nodiscard]] UDT cost_assignment(Cost const& cost, Assignment const& assignment);

template <class UDT>
struct SoftCosts
{
  std::map<SourceId, UDT> point_likelihood;
  std::map<SourceId, UDT> soft_detection_likelihood;
  std::map<SourceId, UDT> soft_clutter_likelihood;
  std::map<SourceId, std::size_t> num_clutter;
  std::map<SourceId, std::size_t> num_detected;
};
template <class UDT>
[[nodiscard]] SoftCosts<UDT> soft_cost_assignment(TTBManager<UDT> const& manager,
                                                  std::span<DataSourceContainer<UDT> const> data_containers,
                                                  Cost const& soft_cost,
                                                  Assignment const& assignment);

[[nodiscard]] std::string to_string(Assignment const& assignment);

[[nodiscard]] std::string to_string(Cost const& cost);

}  // namespace ttb