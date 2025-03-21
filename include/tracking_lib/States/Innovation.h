#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/States/State.h"

namespace ttb
{

/// This class stores the state update of one track for a one (1) Measurement Model for ALL Measurements
class Innovation final
{
public:
  [[nodiscard]] std::string toString(std::string prefix = "") const;

  struct Update
  {
    State updated_dist;
    double log_likelihood{ std::numeric_limits<double>::quiet_NaN() };
    double clutter_intensity{ std::numeric_limits<double>::quiet_NaN() };
  };
  std::map<MeasurementId, Update> _updates{};
  /// the detection probability of the predicted measurement
  Probability _detectionProbability{ std::numeric_limits<double>::quiet_NaN() };
};

}  // namespace ttb