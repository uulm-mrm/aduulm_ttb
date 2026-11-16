#pragma once

#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// Constant Velocity State Model
template<class UDT>
class CVStateModel
{
public:
  [[nodiscard]] static STATE_MODEL_TYPE type() noexcept;
  [[nodiscard]] ProcessMatrices<UDT> processMatrix(Duration dt) const;
  [[nodiscard]] GaussianDistribution<UDT> predict(Duration dt,
                                           GaussianDistribution<UDT> const& dist,
                                           EgoMotionDistribution const& ego) const;
  TTBManager<UDT> const& _manager;
  StateModelId _id;
};

}  // namespace ttb