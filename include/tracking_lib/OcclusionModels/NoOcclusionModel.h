#pragma once
#include "tracking_lib/OcclusionModels/BaseOcclusionModel.h"

#include "tracking_lib/TTBTypes/Params.h"

namespace ttb
{

/// the objects do not occlude each other
class NoOcclusionModel : public BaseOcclusionModel
{
public:
  explicit NoOcclusionModel(TTBManager* manager);

  ~NoOcclusionModel() override = default;

  [[nodiscard]] OCCLUSION_MODEL_TYPE type() const override;

  /// The degree how much the given state is occluded by the other states
  [[nodiscard]] Probability occlusionProb(State const& /*unused*/, std::vector<State> const& /*unused*/) const override;
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const override;

  TTBManager* _manager;
};

}  // namespace ttb