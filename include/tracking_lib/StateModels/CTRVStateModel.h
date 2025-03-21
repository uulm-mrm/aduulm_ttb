#pragma once

#include "tracking_lib/StateModels/NonLinearStateModel.h"

namespace ttb
{

Components const CTRV_state_comps = Components({ COMPONENT::POS_X,
                                                 COMPONENT::POS_Y,
                                                 COMPONENT::POS_Z,
                                                 COMPONENT::VEL_ABS,
                                                 COMPONENT::ROT_Z,
                                                 COMPONENT::VEL_ROT_Z });
Components const CTRV_noise_comps = Components({ COMPONENT::ACC_ABS, COMPONENT::ACC_ROT_Z });

// modified TTB_EPS for case distinction in CTRV kinematic model, see
// https://mrm-git.e-technik.uni-ulm.de/aduulm/source/tracking/issues/56
// for detailed derivation and explanation
static constexpr double CTRV_EPS = 3 * 10 * 10 * 10 * TTB_EPS;

/// constant turn rate and velocity state model
class CTRVStateModel final : public NonLinearStateModel
{
public:
  CTRVStateModel(TTBManager* manager, StateModelId id);
  ~CTRVStateModel() override = default;
  [[nodiscard]] STATE_MODEL_TYPE type() const noexcept override;
  [[nodiscard]] Vector applyKinematicModel(Duration dt, Vector x, Vector noise_values) const override;
};

}  // namespace ttb