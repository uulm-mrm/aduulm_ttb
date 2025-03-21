#include "tracking_lib/States/Innovation.h"

namespace ttb
{

std::string Innovation::toString(std::string prefix) const
{
  std::string out = prefix + "Innovation\n";
  for (auto const& [meas_id, update] : _updates)
  {
    out += prefix + "|\tMeasurement ID: " + std::to_string(meas_id.value_) + '\n' + prefix +
           "|\tLoglikelihood: " + std::to_string(update.log_likelihood) + '\n' + prefix +
           "|\tDetection Prob: " + std::to_string(_detectionProbability) + '\n' + prefix +
           "|\tClutter Intensity: " + std::to_string(update.clutter_intensity) + '\n' +
           update.updated_dist.toString(prefix + "|\t");
  }
  return out;
}

}  // namespace ttb
