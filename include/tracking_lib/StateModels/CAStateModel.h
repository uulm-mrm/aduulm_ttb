#pragma once

#include "tracking_lib/StateModels/LinearStateModel.h"

namespace ttb
{

Components const CA_state_comps = Components({ COMPONENT::POS_X,
                                               COMPONENT::POS_Y,
                                               COMPONENT::POS_Z,
                                               COMPONENT::VEL_X,
                                               COMPONENT::VEL_Y,
                                               COMPONENT::ACC_X,
                                               COMPONENT::ACC_Y });
Components const CA_noise_comps = Components({ COMPONENT::JERK_X, COMPONENT::JERK_Y, COMPONENT::VEL_Z });

/// Constant Acceleration State Model
class CAStateModel : public LinearStateModel
{
public:
  CAStateModel(TTBManager* manager, StateModelId id);
  ~CAStateModel() override = default;
  [[nodiscard]] STATE_MODEL_TYPE type() const noexcept override;
  [[nodiscard]] ProcessMatrices processMatrix(Duration dt) const override;
};

}  // namespace ttb