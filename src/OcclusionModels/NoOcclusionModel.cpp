#include "tracking_lib/OcclusionModels/NoOcclusionModel.h"

namespace ttb
{

NoOcclusionModel::NoOcclusionModel(TTBManager* manager) : _manager{ manager }
{
}

Probability NoOcclusionModel::occlusionProb(State const& /*unused*/, std::vector<State> const& /*unused*/) const
{
  return 0;
}

OCCLUSION_MODEL_TYPE NoOcclusionModel::type() const
{
  return OCCLUSION_MODEL_TYPE::NO_OCCLUSION;
}

std::string NoOcclusionModel::toString(std::string const& prefix) const
{
  return prefix + "NoOcclusionModel\n";
}

}  // namespace ttb