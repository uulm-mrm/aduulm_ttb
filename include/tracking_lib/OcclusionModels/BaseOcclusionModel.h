#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// Occlusion model interface
class BaseOcclusionModel
{
public:
  virtual ~BaseOcclusionModel() = default;
  /// the type of the occlusion model
  [[nodiscard]] virtual OCCLUSION_MODEL_TYPE type() const = 0;
  /// The degree how much the given state is occluded by the other states
  [[nodiscard]] virtual Probability occlusionProb(State const& state, std::vector<State> const& otherStates) const = 0;
  /// string representation
  [[nodiscard]] virtual std::string toString(std::string const& prefix = "") const = 0;
};

}  // namespace ttb