#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/DataContainer/DataSourceContainer.h"

namespace ttb::grouping
{

template <class UDT>
struct Group
{
  std::vector<State<UDT>> tracks;
  DataSourceContainer<UDT> measurement_container;
};

/// Group the given states into independent parts.
/// This is  based on the measurements used to update the states inside the innovation
/// map.
/// If two tracks have been updated with a common measurement, they belong to the same group.
/// It also creates the corresponding measurementContainer with all measurements that have been used to update that
/// group, this Container may be empty.
/// All (to no track) non-associated measurements are collected in the last Group
template <class UDT>
std::vector<Group<UDT>> group(std::vector<State<UDT>> const& states,
                              DataSourceContainer<UDT> const& measurement_container);

template <class UDT>
struct MSGroup
{
  std::vector<State<UDT>> tracks;
  std::vector<DataSourceContainer<UDT>> measurement_containers;
};

template <class UDT>
struct MSGroup2
{
  std::vector<StateId> tracks;
  std::vector<DataSourceContainer<UDT>> measurement_containers;
};

/// Group the given states into independent parts using the measurements from multiple data containers
/// All (to no track) non-associated measurements are collected in the last Group
template <class UDT>
std::vector<MSGroup<UDT>> group(std::span<State<UDT> const> states,
                                std::span<DataSourceContainer<UDT> const> measurement_containers);

template <class UDT>
struct Group2
{
  std::vector<StateId> tracks;
  DataSourceContainer<UDT> measurement_container;
};

template <class UDT>
std::vector<Group2<UDT>> group_db(TTBManager<UDT> const& manager,
                                  std::vector<StateId> const& states,
                                  DataSourceContainer<UDT> const& measurement_container);

template <class UDT>
std::vector<MSGroup2<UDT>> group_db(TTBManager<UDT> const& manager,
                                    std::vector<StateId> const& states,
                                    std::span<DataSourceContainer<UDT> const> measurement_containers);

}  // namespace ttb::grouping