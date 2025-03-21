#pragma once

#include "tracking_lib/TTBTypes/Params.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/Transformations/Transformation.h"

namespace ttb
{

class TTBManager;
class EgoMotionDistribution;
class BaseDistribution;

struct EgoCompensated
{
  Vector state;  ///< the egoMotion compensated State
  Matrix cov;    ///< the egoMotion compensated covariance
};

/// Compensate the EgoMotion for the given state
/// Assumes:    EgoMotion in TrackingPlatform/Frame Coords, i.e., X-Axis to the front, Y-Axis to the left, Z-Asis to the
/// Top
[[nodiscard]] EgoCompensated compensateEgoMotion(Vector const& state,
                                                 Matrix const& cov,
                                                 Components const& comps,
                                                 Duration dt,
                                                 EgoMotionDistribution const& egoMotion);

/// State Model Interface
class BaseStateModel
{
public:
  virtual ~BaseStateModel() = default;
  /// State Model Type
  [[nodiscard]] virtual STATE_MODEL_TYPE type() const noexcept = 0;
  /// string representation
  [[nodiscard]] virtual std::string toString() const = 0;
  /// predict the distribution + compensate EgoMotion
  virtual void predict(Duration dt, BaseDistribution& dist, EgoMotionDistribution const& ego) const = 0;
  /// compensate EgoMotion only
  virtual void compensateEgoMotion(Duration dt,
                                   BaseDistribution& dist,
                                   EgoMotionDistribution const& egoMotion) const = 0;
  /// the state components
  [[nodiscard]] virtual Components const& state_comps() const = 0;
  /// the noise components
  [[nodiscard]] virtual Components const& noise_comps() const = 0;
  /// use this noise value for this component
  [[nodiscard]] virtual double std_noise(COMPONENT comp) const = 0;
  /// the unique id of this state model
  [[nodiscard]] virtual StateModelId id() const = 0;
  /// access to the manager
  [[nodiscard]] virtual TTBManager* manager() const = 0;
};

}  // namespace ttb