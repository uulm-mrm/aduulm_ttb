#pragma once

#include "tracking_lib/StateModels/LinearStateModel.h"

namespace ttb
{

Components const CV_state_comps =
    Components({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z, COMPONENT::VEL_X, COMPONENT::VEL_Y });
Components const CV_noise_comps = Components({ COMPONENT::ACC_X, COMPONENT::ACC_Y, COMPONENT::VEL_Z });

/// Constant Velocity State Model
class CVStateModel : public LinearStateModel
{
public:
  CVStateModel(TTBManager* manager, StateModelId id);
  ~CVStateModel() override = default;
  [[nodiscard]] STATE_MODEL_TYPE type() const noexcept override;
  [[nodiscard]] ProcessMatrices processMatrix(Duration dt) const override;
};

}  // namespace ttb