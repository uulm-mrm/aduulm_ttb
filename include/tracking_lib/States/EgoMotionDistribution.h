#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"

namespace ttb
{

/// This represents the Ego Motion of the Tracking Platform / Frame
class EgoMotionDistribution final
{
public:
  /// Ctor with a general distribution
  EgoMotionDistribution(GaussianDistribution<double> dist, Components stateComps);
  /// Ctor with a Gaussian Distribution

  /// zero, i.e, non-moving, ego motion
  [[nodiscard]] static EgoMotionDistribution zero();
  /// get the mean value for some Component
  [[nodiscard]] std::optional<double> meanOf(COMPONENT sc) const;
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
  /// data
  GaussianDistribution<double> _dist;
  Components _comps;
};

}  // namespace ttb
