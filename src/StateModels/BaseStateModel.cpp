#include "tracking_lib/StateModels/BaseStateModel.h"
// ######################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/Transformations/Transformation.h"

namespace ttb
{

EgoCompensated compensateEgoMotion(Vector const& state,
                                   Matrix const& cov,
                                   Components const& comps,
                                   Duration dt,
                                   EgoMotionDistribution const& egoMotion)
{
  // modified TTB_EPS for case distinction in CTRV kinematic model, see
  // https://mrm-git.e-technik.uni-ulm.de/aduulm/source/tracking/issues/56
  // for detailed derivation and explanation
  static constexpr double CTRV_EPS = 3 * 10 * 10 * 10 * TTB_EPS;

  double const deltaT = to_seconds(dt);
  double const deltaX = egoMotion.meanOf(COMPONENT::VEL_X).value() * deltaT;
  double const deltaPhi = egoMotion.meanOf(COMPONENT::VEL_ROT_Z).value() * deltaT;
  double const dv = egoMotion.meanOf(COMPONENT::VEL_ABS).value();
  double const dphi = egoMotion.meanOf(COMPONENT::VEL_ROT_Z).value();
  // compensate translation of ego
  Vector3 translation = Vector3::Zero(3);
  ;
  Matrix33 rotation = Matrix33::Identity(3, 3);
  // Ego moves on straight line forward in x-direction -> compensate POS_X of object only
  if (fabs(egoMotion.meanOf(COMPONENT::VEL_ROT_Z).value()) < CTRV_EPS)
  {
    translation(0) = -deltaX;
  }
  else  // ego-object moves like a CTRV Model
  {
    double const cosDeltaPhi = std::cos(deltaPhi);
    double const sinDeltaPhi = std::sin(deltaPhi);
    Matrix const rotMatrix{ { cosDeltaPhi, sinDeltaPhi }, { -sinDeltaPhi, cosDeltaPhi } };
    rotation({ 0, 1 }, { 0, 1 }) = rotMatrix;

    translation(0) = -dv / dphi * sinDeltaPhi;
    translation(1) = -dv / dphi * (-cosDeltaPhi + 1);
  }
  auto trafo = transformation::transform(state, cov, comps, translation, rotation);
  return { .state = std::move(trafo.value().mean), .cov = std::move(trafo.value().cov) };
}

}  // namespace ttb