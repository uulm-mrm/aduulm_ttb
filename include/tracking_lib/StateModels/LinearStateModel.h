#pragma once

#include "tracking_lib/StateModels/BaseStateModel.h"

namespace ttb
{

/// This represents all linear state Models
class LinearStateModel : public BaseStateModel
{
public:
  LinearStateModel(TTBManager* manager, StateModelId id, Components state_comps, Components noise_comps);
  ~LinearStateModel() override = default;
  /// The process matrix F, i.e., x_+ = F*x
  struct ProcessMatrices
  {
    Matrix Gamma;
    Matrix Q;
    Matrix F;
  };
  [[nodiscard]] virtual ProcessMatrices processMatrix(Duration dt) const = 0;
  /// predict the distribution by transforming the mean and variance
  void predict(Duration dt, BaseDistribution& dist, EgoMotionDistribution const& egoDist) const final;
  void compensateEgoMotion(Duration dt, BaseDistribution& dist, EgoMotionDistribution const& egoMotion) const final;
  double std_noise(COMPONENT comp) const final;
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