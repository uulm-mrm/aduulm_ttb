#include "tracking_lib/PersistenceModels/ConstantPersistenceModel.h"
#include "tracking_lib/TTBManager/TTBManager.h"

namespace ttb
{

ConstantPersistenceModel::ConstantPersistenceModel(ttb::TTBManager* manager) : _manager{ manager }
{
}

double ConstantPersistenceModel::getPersistenceProbability(State const& state, Duration dt) const
{
  return std::pow(_manager->params().persistence_model.constant.persistence_prob, to_seconds(dt));
}

PERSISTENCE_MODEL_TYPE ConstantPersistenceModel::type() const
{
  return PERSISTENCE_MODEL_TYPE::CONSTANT;
}

TTBManager* ConstantPersistenceModel::manager() const
{
  return _manager;
}

}  // namespace ttb