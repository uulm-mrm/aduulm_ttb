#pragma once
#include "tracking_lib/PersistenceModels/ConstantPersistenceModel.h"
#include "tracking_lib/PersistenceModels/ROIPersistenceModel.h"

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Misc/Region.h"

namespace ttb
{

/// Model the Persistence Probability of a Track
template <class UDT>
class PersistenceModel
{
public:
  using AllModels = std::variant<ConstantPersistenceModel<UDT>, ROIPersistenceModel<UDT>>;
  /// Gets the persistence probability of a track
  [[nodiscard]] UDT getPersistenceProbability(State<UDT> const& state,
                                              Duration dt,
                                              std::optional<Region> const& roi) const
  {
    return std::visit([&](auto const& model) { return model.getPersistenceProbability(state, dt, roi); }, _model);
  }
  /// the type of the persistence model
  [[nodiscard]] PERSISTENCE_MODEL_TYPE type() const
  {
    return std::visit([&](auto const& model) { return model.type(); }, _model);
  }
  /// access to the manager
  [[nodiscard]] TTBManager<UDT>* manager() const
  {
    return std::visit([&](auto const& model) { return model.manager(); }, _model);
  }
  AllModels _model;
};

}  // namespace ttb