#pragma once

#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/Misc/Region.h"

namespace ttb
{

class PredictionInformation
{
public:
  /// eqo motion of the tracking frame
  EgoMotionDistribution _egoMotion{ EgoMotionDistribution::zero() };
  /// region of interest for this prediction step
  std::optional<Region> _roi{};
};

}  // namespace ttb