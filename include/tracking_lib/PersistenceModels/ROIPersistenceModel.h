#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Misc/Region.h"

namespace ttb
{

/// Model the Persistence Probability of a Track
template <class UDT>
class ROIPersistenceModel
{
public:
  explicit ROIPersistenceModel(TTBManager<UDT>* manager);

  /// Gets the persistence probability of a track
  [[nodiscard]] double getPersistenceProbability(State<UDT> const& state,
                                                 Duration dt,
                                                 std::optional<Region> const& roi) const;

  [[nodiscard]] PERSISTENCE_MODEL_TYPE type() const;

  [[nodiscard]] TTBManager<UDT>* manager() const;

  TTBManager<UDT>* _manager;
};

}  // namespace ttb