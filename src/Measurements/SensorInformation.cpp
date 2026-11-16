#include "tracking_lib/Measurements/SensorInformation.h"
#include <regex>

namespace ttb
{

std::string SensorInformation::toString(std::string const& prefix) const
{
  std::string out = prefix + "Sensor Information\n";
  std::ostringstream pos_ss;
  pos_ss << prefix + "|\tPosition: " << _to_sensor_cs.translation();
  auto pos_str = pos_ss.str();
  pos_str = std::regex_replace(pos_str, std::regex("\n"), "\n" + prefix + "|\t          ");
  out += pos_str + "\n";
  std::ostringstream rot_ss;
  rot_ss << prefix + "|\tRotation: " << _to_sensor_cs.rotation();
  auto rot_str = rot_ss.str();
  rot_str = std::regex_replace(rot_str, std::regex("\n"), "\n" + prefix + "|\t          ");
  out += rot_str + "\n";
  if (_sensor_fov.has_value())
  {
    out += _sensor_fov.value().toString(prefix + "|\t");
  }
  return out;
}
}  // namespace ttb