#pragma once

#include "tracking_lib/PersistenceModels/BasePersistenceModel.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// Model the Persistence Probability of a Track
class ConstantPersistenceModel : public BasePersistenceModel
{
public:
  explicit ConstantPersistenceModel(TTBManager* manager);

  ~ConstantPersistenceModel() override = default;

  /// Gets the persistence probability of a track
  [[nodiscard]] double getPersistenceProbability(State const& state, Duration dt) const override;

  [[nodiscard]] PERSISTENCE_MODEL_TYPE type() const override;

  [[nodiscard]] TTBManager* manager() const override;

  TTBManager* _manager;
};

}  // namespace ttb