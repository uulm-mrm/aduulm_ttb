#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Misc/Region.h"

namespace ttb
{

/// Dynamic Sensor Information with respect to the tracking frame
class SensorInformation
{
public:
  ///  The transformation from tracking coordinates to sensor coordinates
  SE3Trafo _to_sensor_cs{ SE3Trafo::Identity() };
  /// Field of view of the sensor in tracking coords
  std::optional<Region> _sensor_fov{};
  /// Field of view in sensor coords
  std::optional<Region> _sensor_fov_in_sensor_cs{};
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
};

}  // namespace ttb