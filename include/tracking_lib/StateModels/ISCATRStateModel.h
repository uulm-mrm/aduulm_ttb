//
// Created by hermann on 6/7/22.
//

#pragma once

#include "tracking_lib/StateModels/LinearStateModel.h"

namespace ttb
{

Components const ISCATR_state_comps = Components({ COMPONENT::POS_X,
                                                   COMPONENT::POS_Y,
                                                   COMPONENT::POS_Z,
                                                   COMPONENT::VEL_X,
                                                   COMPONENT::VEL_Y,
                                                   COMPONENT::ACC_X,
                                                   COMPONENT::ACC_Y,
                                                   COMPONENT::ROT_Z,
                                                   COMPONENT::VEL_ROT_Z });
Components const ISCATR_noise_comps =
    Components({ COMPONENT::JERK_X, COMPONENT::JERK_Y, COMPONENT::VEL_Z, COMPONENT::ACC_ROT_Z });

/// Independent split constant acceleration and turn rate state model.
/// Application for crab-walk and objects moving with a constant side slip angle.
/// This model was firstly developed by @hermann, because no other crab-walk model was known
class ISCATRStateModel final : public LinearStateModel
{
public:
  ISCATRStateModel(TTBManager* manager, StateModelId id);
  ~ISCATRStateModel() override = default;
  [[nodiscard]] STATE_MODEL_TYPE type() const noexcept override;
  [[nodiscard]] ProcessMatrices processMatrix(Duration dt) const override;
};

}  // namespace ttb