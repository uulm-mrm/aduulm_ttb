#pragma once

#include "tracking_lib/StateModels/BaseStateModel.h"

namespace ttb
{

/// This represents a Nonlinear State Model
/// It uses an unscented transformation for the prediction
class NonLinearStateModel : public BaseStateModel
{
public:
  NonLinearStateModel(TTBManager* manager, StateModelId id, Components state_comps, Components noise_comps);
  ~NonLinearStateModel() override = default;
  /// Predicts the state using the given noise values
  [[nodiscard]] virtual Vector applyKinematicModel(Duration dt, Vector x, Vector noise_values) const = 0;
  /// predict the state with an unscented transformation
  void predict(Duration dt, BaseDistribution& dist, EgoMotionDistribution const& egoDist) const final;
  void compensateEgoMotion(Duration dt, BaseDistribution& dist, EgoMotionDistribution const& egoMotion) const final;
  [[nodiscard]] double std_noise(COMPONENT comp) const final;
  [[nodiscard]] Components const& state_comps() const final;
  [[nodiscard]] Components const& noise_comps() const final;
  [[nodiscard]] StateModelId id() const final;
  [[nodiscard]] std::string toString() const final;

  [[nodiscard]] TTBManager* manager() const final;

private:
  TTBManager* _manager;
  StateModelId _id;
  Components const _state_comps;
  Components const _noise_comps;
};

}  // namespace ttb