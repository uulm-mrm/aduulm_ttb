#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/States/State.h"

namespace ttb
{

/// A track container, similar to a MeasurementContainer, is a collection of tracks of a single Source of certain time
class StateContainer
{
public:
  [[nodiscard]] std::string toString(std::string prefix = "") const;
  SourceId _id;
  Time _time;
  EgoMotionDistribution _egoMotion{ EgoMotionDistribution::zero() };
  std::vector<State> _data{};
};

}  // namespace ttb