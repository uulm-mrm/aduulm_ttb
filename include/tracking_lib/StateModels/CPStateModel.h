#pragma once

#include "tracking_lib/StateModels/LinearStateModel.h"

namespace ttb
{

Components const CP_state_comps = Components({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
Components const CP_noise_comps = Components({ COMPONENT::VEL_X, COMPONENT::VEL_Y, COMPONENT::VEL_Z });

/// Constant Position state model
class CPStateModel final : public LinearStateModel
{
public:
  CPStateModel(TTBManager* manager, StateModelId id);
  ~CPStateModel() override = default;
  [[nodiscard]] STATE_MODEL_TYPE type() const noexcept override;
  [[nodiscard]] ProcessMatrices processMatrix(Duration dt) const override;
};

}  // namespace ttb