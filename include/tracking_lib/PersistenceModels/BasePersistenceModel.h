#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// Model the Persistence Probability of a Track
class BasePersistenceModel
{
public:
  virtual ~BasePersistenceModel() = default;
  /// Gets the persistence probability of a track
  [[nodiscard]] virtual double getPersistenceProbability(State const& state, Duration dt) const = 0;
  /// the type of the persistence model
  [[nodiscard]] virtual PERSISTENCE_MODEL_TYPE type() const = 0;
  /// access to the manager
  [[nodiscard]] virtual TTBManager* manager() const = 0;
};

}  // namespace ttb