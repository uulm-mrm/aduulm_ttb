#pragma once

#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// Constant Position state model
template<class UDT>
class CPStateModel
{
public:
  [[nodiscard]] static STATE_MODEL_TYPE type();
  [[nodiscard]] GaussianDistribution<UDT> predict(Duration dt,
                                           GaussianDistribution<UDT> const& dist,
                                           EgoMotionDistribution const& ego) const;
  [[nodiscard]] ProcessMatrices<UDT> processMatrix(Duration dt) const;
  TTBManager<UDT> const& _manager;
  StateModelId _id;
};

}  // namespace ttb