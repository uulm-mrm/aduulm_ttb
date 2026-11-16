#include "tracking_lib/PersistenceModels/ConstantPersistenceModel.h"
#include "tracking_lib/TTBManager/TTBManager.h"

namespace ttb
{

template class ConstantPersistenceModel<double>;
template class ConstantPersistenceModel<AD>;

template <class UDT>
ConstantPersistenceModel<UDT>::ConstantPersistenceModel(TTBManager<UDT>* manager) : _manager{ manager }
{
}

template <class UDT>
double ConstantPersistenceModel<UDT>::getPersistenceProbability(State<UDT> const& /**/,
                                                                Duration dt,
                                                                std::optional<Region> const& /**/) const
{
  return std::pow(_manager->params()->persistence_model.constant.persistence_prob, to_seconds(dt));
}

template <class UDT>
PERSISTENCE_MODEL_TYPE ConstantPersistenceModel<UDT>::type()
{
  return PERSISTENCE_MODEL_TYPE::CONSTANT;
}

template <class UDT>
TTBManager<UDT>* ConstantPersistenceModel<UDT>::manager() const
{
  return _manager;
}

}  // namespace ttb
