#include "tracking_lib/Measurements/MeasurementContainer.h"

namespace ttb
{

std::string MeasurementContainer::toString() const
{
  std::string out{ "MeasurementContainer\n\tMeasModel ID: " + _id.value_ + "\n\tTime: " + to_string(_time) +
                   "\n\tEgo Motion: " + _egoMotion.toString() };
  for (Measurement const& meas : _data)
  {
    out += "\tMeasurement: " + meas.toString();
  }
  out += _sensorInfo.toString();
  return out;
}

}  // namespace ttb