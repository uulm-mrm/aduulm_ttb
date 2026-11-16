#pragma once
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// Constant turn rate and acceleration state model
template<class UDT>
class CTRAStateModel
{
public:
  [[nodiscard]] static STATE_MODEL_TYPE type() noexcept;
  [[nodiscard]] Vector<UDT> applyKinematicModel(Duration dt, Vector<UDT> const& x, Vector<UDT> const& noise) const;
  GaussianDistribution<UDT> predict(Duration dt, GaussianDistribution<UDT> const& dist, EgoMotionDistribution const& ego) const;
  TTBManager<UDT> const& _manager;
  StateModelId _id;
};

}  // namespace ttb