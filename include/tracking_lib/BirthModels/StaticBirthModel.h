#pragma once

#include "tracking_lib/BirthModels/BaseBirthModel.h"

namespace ttb
{

/// Static Birth model that sets up new tracks at fixed points
class StaticBirthModel final : public BaseBirthModel
{
public:
  explicit StaticBirthModel(TTBManager* manager);
  [[nodiscard]] BIRTH_MODEL_TYPE type() const noexcept override;
  [[nodiscard]] TTBManager* manager() const override;
  /// Returns birth Dists at predefined birth state distributions
  [[nodiscard]] std::vector<State> getBirthTracks(MeasurementContainer const& measContainer,
                                                  std::map<MeasurementId, double> const& /*unused*/,
                                                  std::vector<State> const& existingTracks) override;
  void reset() noexcept override;
  /// read and store the static birth locations
  void loadBirthLocations();
  TTBManager* _manager;
  /// vector holding a list of birth locations
  std::vector<std::map<COMPONENT, double>> _birthLocations;
  /// vector holding a list of variance matrices
  std::vector<std::map<COMPONENT, double>> _birthVariances;
};

}  // namespace ttb
