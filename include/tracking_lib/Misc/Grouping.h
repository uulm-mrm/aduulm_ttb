#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"

namespace ttb::grouping
{
struct Group
{
  std::vector<State> tracks;
  MeasurementContainer measurement_container;
};

/// Group the given states into independent parts.
/// This is  based on the measurements used to update the states inside the innovation
/// map.
/// If two tracks have been updated with a common measurement, they belong to the same group.
/// It also creates the corresponding measurementContainer with all measurements that have been used to update that
/// group, this Container may be empty.
/// All (to no track) non-associated measurements are collected in the last MeasurementContainer with an empty vector of
/// corresponding states.
std::vector<Group> group(std::vector<State> const& states, MeasurementContainer const& measurement_container);

}  // namespace ttb::grouping