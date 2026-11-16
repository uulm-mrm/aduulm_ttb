#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Misc/Region.h"

namespace ttb
{

/// Model the Persistence Probability of a Track
template <class UDT>
class ConstantPersistenceModel
{
public:
  explicit ConstantPersistenceModel(TTBManager<UDT>* manager);

  /// Gets the persistence probability of a track
  [[nodiscard]] double getPersistenceProbability(State<UDT> const& /**/,
                                                 Duration dt,
                                                 std::optional<Region> const& /**/) const;

  [[nodiscard]] static PERSISTENCE_MODEL_TYPE type();

  [[nodiscard]] TTBManager<UDT>* manager() const;

  TTBManager<UDT>* _manager;
};

}  // namespace ttb