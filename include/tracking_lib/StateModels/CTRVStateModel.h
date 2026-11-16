#pragma once

#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// constant turn rate and velocity state model
template <class UDT>
class CTRVStateModel
{
public:
  [[nodiscard]] static STATE_MODEL_TYPE type() noexcept;
  [[nodiscard]] Vector<UDT> applyKinematicModel(Duration dt, Vector<UDT> const& x, Vector<UDT> const& noise_values) const;
  GaussianDistribution<UDT> predict(Duration dt, GaussianDistribution<UDT> const& dist, EgoMotionDistribution const& ego) const;
  TTBManager<UDT> const& _manager;
  StateModelId _id;
};

}  // namespace ttb