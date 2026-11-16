#pragma once

#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// Independent split constant acceleration and turn rate state model.
/// Application for crab-walk and objects moving with a constant side slip angle.
/// This model was firstly developed by @hermann, because no other crab-walk model was known
/// See T. Woopen et al., "Toward Intelligent Automated Driving Functionalities for Multipurpose Vehicles in
/// UNICARagil," in Proceedings of the IEEE, vol. 114, no. 4, pp. 678-707, April 2026, doi: 10.1109/JPROC.2026.3714578.
/// https://ieeexplore.ieee.org/document/11643228
template <class UDT>
class ISCATRStateModel
{
public:
  [[nodiscard]] static STATE_MODEL_TYPE type() noexcept;
  [[nodiscard]] ProcessMatrices<UDT> processMatrix(Duration dt) const;
  GaussianDistribution<UDT> predict(Duration dt, GaussianDistribution<UDT> const& dist, EgoMotionDistribution const& ego) const;
  TTBManager<UDT> const& _manager;
  StateModelId _id;
};

}  // namespace ttb