#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/States/State.h"

namespace ttb
{

/// A dynamic Birth Model
/// It uses the assigment probabilities of the update to create new birth
/// The assignment expresses how much each measurement contributed to the update, see
/// S. Reuter, B. -T. Vo, B. -N. Vo and K. Dietmayer, "The Labeled Multi-Bernoulli Filter," in IEEE Transactions on
/// Signal Processing, vol. 62, no. 12, pp. 3246-3260, June15, 2014, doi: 10.1109/TSP.2014.2323064.
template <class UDT>
class DynamicBirthModel
{
public:
  explicit DynamicBirthModel(TTBManager<UDT>* manager);
  [[nodiscard]] BIRTH_MODEL_TYPE type() const;
  [[nodiscard]] TTBManager<UDT>* manager() const;
  [[nodiscard]] BirthTracks<UDT> getBirthTracks(DataSourceContainer<UDT> const& measContainer,
                                                std::map<MeasurementId, UDT> const& assignProb,
                                                std::vector<State<UDT>> const& existingTracks);
  void reset();
  TTBManager<UDT>* _manager;
  /// The birth candidates from previous time steps, needed for a two-step initialization
  std::map<SourceId, std::vector<State<UDT>>> _oldMeasurements;
};

}  // namespace ttb
