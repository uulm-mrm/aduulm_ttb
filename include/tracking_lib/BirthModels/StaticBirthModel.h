#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/States/State.h"

namespace ttb
{

/// Static Birth model that sets up new tracks at fixed points
template <class UDT>
class StaticBirthModel
{
public:
  explicit StaticBirthModel(TTBManager<UDT>* manager);
  [[nodiscard]] static BIRTH_MODEL_TYPE type() noexcept;
  [[nodiscard]] TTBManager<UDT>* manager() const;
  /// Returns birth Dists at predefined birth state distributions
  [[nodiscard]] BirthTracks<UDT> getBirthTracks(DataSourceContainer<UDT> const& measContainer,
                                                std::map<MeasurementId, UDT> const& /*unused*/,
                                                std::vector<State<UDT>> const& existingTracks);
  void reset();
  TTBManager<UDT>* _manager;
};

}  // namespace ttb
