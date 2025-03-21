#pragma once

#include "tracking_lib/StateModels/LinearStateModel.h"

namespace ttb
{

Components const CTP_state_comps =
    Components({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z, COMPONENT::ROT_Z });
Components const CTP_noise_comps =
    Components({ COMPONENT::VEL_X, COMPONENT::VEL_Y, COMPONENT::VEL_Z, COMPONENT::VEL_ROT_Z });

/// Constant Turn and Position Model
class CTP final : public LinearStateModel
{
public:
  CTP(TTBManager* manager, StateModelId id);
  ~CTP() override = default;
  [[nodiscard]] STATE_MODEL_TYPE type() const noexcept override;
  [[nodiscard]] ProcessMatrices processMatrix(Duration dt) const override;
};

}  // namespace ttb