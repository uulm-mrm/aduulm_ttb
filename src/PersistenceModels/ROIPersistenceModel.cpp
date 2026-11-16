#include "tracking_lib/PersistenceModels/ROIPersistenceModel.h"
#include "tracking_lib/TTBManager/TTBManager.h"

namespace ttb
{

template class ROIPersistenceModel<double>;
template class ROIPersistenceModel<AD>;

template <class UDT>
ROIPersistenceModel<UDT>::ROIPersistenceModel(TTBManager<UDT>* manager) : _manager{ manager }
{
}

template <class UDT>
double ROIPersistenceModel<UDT>::getPersistenceProbability(State<UDT> const& state,
                                                           Duration dt,
                                                           std::optional<Region> const& roi) const
{
  double const inside = std::pow(_manager->params()->persistence_model.roi.inside_persistence_prob, to_seconds(dt));
  if (not roi.has_value())
  {
    return inside;
  }
  auto const [model_id, dist] = state.getEstimate();
  if (auto const ind = _manager->getStateModel(model_id).state_comps().indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
      ind.has_value())
  {
    if (roi.value().contains(dist.mean(), _manager->getStateModel(model_id).state_comps()))
    {
      return inside;
    }
  }
  return std::pow(_manager->params()->persistence_model.roi.outside_persistence_prob, to_seconds(dt));
}

template <class UDT>
PERSISTENCE_MODEL_TYPE ROIPersistenceModel<UDT>::type() const
{
  return PERSISTENCE_MODEL_TYPE::ROI;
}

template <class UDT>
TTBManager<UDT>* ROIPersistenceModel<UDT>::manager() const
{
  return _manager;
}

}  // namespace ttb
