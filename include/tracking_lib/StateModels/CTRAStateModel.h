#pragma once

#include "tracking_lib/StateModels/NonLinearStateModel.h"

namespace ttb
{

Components const CTRA_state_comps = Components({ COMPONENT::POS_X,
                                                 COMPONENT::POS_Y,
                                                 COMPONENT::POS_Z,
                                                 COMPONENT::VEL_ABS,
                                                 COMPONENT::ACC_ABS,
                                                 COMPONENT::ROT_Z,
                                                 COMPONENT::VEL_ROT_Z });
Components const CTRA_noise_comps = Components({ COMPONENT::JERK_ABS, COMPONENT::ACC_ROT_Z });

/// modified TTB_EPS for case distinction in CTRA kinematic model, see
/// https://mrm-git.e-technik.uni-ulm.de/aduulm/source/tracking/issues/56
/// for detailed derivation and explanation
static constexpr double CTRA_EPS = 10 * 10 * 10 * TTB_EPS;

/// Constant turn rate and acceleration state model
class CTRAStateModel final : public NonLinearStateModel
{
public:
  CTRAStateModel(TTBManager* manager, StateModelId id);
  ~CTRAStateModel() override = default;
  [[nodiscard]] STATE_MODEL_TYPE type() const noexcept override;
  [[nodiscard]] Vector applyKinematicModel(Duration dt, Vector x, Vector noise) const override;
};

}  // namespace ttb