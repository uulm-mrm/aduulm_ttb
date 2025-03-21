#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Measurements/Measurement.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/Measurements/SensorInformation.h"

namespace ttb
{

/// A MeasurementContainer is a collection of individual measurements of the same source of the same time
/// together with other information needed to use this information
class MeasurementContainer
{
public:
  [[nodiscard]] std::string toString() const;
  /// unique id connecting this data to a MeasurementModel
  MeasModelId _id;
  /// the individual measurements
  std::vector<Measurement> _data{};
  /// eqo motion of the tracking frame
  EgoMotionDistribution _egoMotion{ EgoMotionDistribution::zero() };
  /// time of the measurements
  Time _time;
  /// other information about the data source, e.g., the source frame
  SensorInformation _sensorInfo{};
};

}  // namespace ttb