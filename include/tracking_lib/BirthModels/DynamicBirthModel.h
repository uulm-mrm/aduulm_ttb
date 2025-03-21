#pragma once

#include "tracking_lib/BirthModels/BaseBirthModel.h"
#include "tracking_lib/Measurements/Measurement.h"

namespace ttb
{
class TTBManager;
class State;
class BaseMeasurementModel;

/// A dynamic Birth Model
/// It uses the assigment probabilities of the update to create new birth
/// The assignment expresses how much each measurement contributed to the update, see
/// S. Reuter, B. -T. Vo, B. -N. Vo and K. Dietmayer, "The Labeled Multi-Bernoulli Filter," in IEEE Transactions on
/// Signal Processing, vol. 62, no. 12, pp. 3246-3260, June15, 2014, doi: 10.1109/TSP.2014.2323064.
class DynamicBirthModel final : public BaseBirthModel
{
public:
  explicit DynamicBirthModel(TTBManager* manager);
  ~DynamicBirthModel() override = default;
  [[nodiscard]] BIRTH_MODEL_TYPE type() const override;
  [[nodiscard]] TTBManager* manager() const override;
  [[nodiscard]] std::vector<State> getBirthTracks(MeasurementContainer const& measContainer,
                                                  std::map<MeasurementId, Probability> const& assignProb,
                                                  std::vector<State> const& existingTracks) override;
  void reset() noexcept override;
  TTBManager* _manager;
  /// The birth candidates from previous time steps, needed for a two-step initialization
  std::map<MeasModelId, std::vector<Measurement>> _oldMeasurements;
};

}  // namespace ttb
