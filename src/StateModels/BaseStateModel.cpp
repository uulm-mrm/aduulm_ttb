#include "tracking_lib/StateModels/BaseStateModel.h"
// ######################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/Transformations/Transformation.h"

#include <stacktrace>

namespace ttb
{

template EgoCompensated<double> compensateEgoMotion(Vector<double> const& state,
                                           Matrix<double> const& cov,
                                           Components const& comps,
                                           Duration dt,
                                           EgoMotionDistribution const& egoMotion);
template EgoCompensated<AD> compensateEgoMotion(Vector<AD> const& state,
                                       Matrix<AD> const& cov,
                                       Components const& comps,
                                       Duration dt,
                                       EgoMotionDistribution const& egoMotion);
template <class UDT>
EgoCompensated<UDT> compensateEgoMotion(Vector<UDT> const& state,
                                        Matrix<UDT> const& cov,
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
    Matrix<double> const rotMatrix{ { cosDeltaPhi, sinDeltaPhi }, { -sinDeltaPhi, cosDeltaPhi } };
    rotation({ 0, 1 }, { 0, 1 }) = rotMatrix;

    translation(0) = -dv / dphi * sinDeltaPhi;
    translation(1) = -dv / dphi * (-cosDeltaPhi + 1);
  }
  SE3Trafo t = SE3Trafo::Identity();
  t.matrix()({ 0, 1, 2 }, { 0, 1, 2 }) = rotation;
  t.matrix()({ 0, 1, 2 }, { 3 }) = translation;
  auto trafo = transformation::transform(state, cov, comps, t);
  return { .state = std::move(trafo.mean), .cov = std::move(trafo.cov) };
}

template Components const& StateModel<double>::state_comps() const;
template Components const& StateModel<AD>::state_comps() const;
template <class UDT>
Components const& StateModel<UDT>::state_comps() const
{
  return _manager.state_comps(id());
}

template Components const& StateModel<double>::noise_comps() const;
template Components const& StateModel<AD>::noise_comps() const;
template <class UDT>
Components const& StateModel<UDT>::noise_comps() const
{
  return _manager.noise_comps(id());
}

template double StateModel<double>::std_noise(COMPONENT comp) const;
template double StateModel<AD>::std_noise(COMPONENT comp) const;
template <class UDT>
double StateModel<UDT>::std_noise(COMPONENT comp) const
{
  return _manager.params()->state_model_params(id()).model_noise_std_dev.at(to_string(comp));
}

template bool StateModel<double>::is_valid() const;
template bool StateModel<AD>::is_valid() const;
template <class UDT>
bool StateModel<UDT>::is_valid() const
{
  bool ok = true;
  for (COMPONENT const noise_comp : noise_comps()._comps)
  {
    if (not _manager.params()->state_model_params(id()).model_noise_std_dev.contains(to_string(noise_comp)))
    {
      LOG_ERR(std::format(
          "Parameters of state model {} do not contain required noise comp {}.", id().value_, to_string(noise_comp)));
      ok = false;
    }
  }
  return ok;
}

template GaussianDistribution<double>
StateModel<double>::compensateEgoMotion(Duration dt,
                                        GaussianDistribution<double> const& dist,
                                        EgoMotionDistribution const& egoMotion) const;
template GaussianDistribution<AD> StateModel<AD>::compensateEgoMotion(Duration dt,
                                                                      GaussianDistribution<AD> const& dist,
                                                                      EgoMotionDistribution const& egoMotion) const;
template <class UDT>
GaussianDistribution<UDT> StateModel<UDT>::compensateEgoMotion(Duration dt,
                                                               GaussianDistribution<UDT> const& dist,
                                                               EgoMotionDistribution const& egoMotion) const
{
  auto [state, cov] =
      ::ttb::compensateEgoMotion(dist.mean(), dist.covariance(), _manager.state_comps(id()), dt, egoMotion);
  return { std::move(state), std::move(cov), dist.ref_point() };
}

template double std_noise(TTBManager<double> const& manager, StateModelId id, COMPONENT comp);
template double std_noise(TTBManager<AD> const& manager, StateModelId id, COMPONENT comp);
template <class UDT>
double std_noise(TTBManager<UDT> const& manager, StateModelId id, COMPONENT comp)
{
  assert(manager.params()->state_model_params(id).model_noise_std_dev.contains(to_string(comp)));
  return manager.params()->state_model_params(id).model_noise_std_dev.at(to_string(comp));
}

template GaussianDistribution<double> predict_linear(Duration dt,
                                            GaussianDistribution<double> const& dist,
                                            EgoMotionDistribution const& egoDist,
                                            Components const& state_comps,
                                            ProcessMatrices<double> const& kinematic);
template GaussianDistribution<AD> predict_linear(Duration dt,
                                        GaussianDistribution<AD> const& dist,
                                        EgoMotionDistribution const& egoDist,
                                        Components const& state_comps,
                                        ProcessMatrices<AD> const& kinematic);
template <class UDT>
GaussianDistribution<UDT> predict_linear(Duration dt,
                                         GaussianDistribution<UDT> const& dist,
                                         EgoMotionDistribution const& egoDist,
                                         Components const& state_comps,
                                         ProcessMatrices<UDT> const& kinematic)
{
  // compensate Ego
  EgoCompensated ego_comp = ::ttb::compensateEgoMotion(dist.mean(), dist.covariance(), state_comps, dt, egoDist);

  // apply kinematic
  Vector<UDT> x = kinematic.F * ego_comp.state;
  if (state_comps.indexOf(COMPONENT::ROT_Z).has_value())
  {
    angles::normalizeAngle(x(state_comps.indexOf(COMPONENT::ROT_Z).value()));
  }
  // compute variance
  Matrix<UDT> P = kinematic.F * ego_comp.cov * kinematic.F.transpose() + kinematic.Q;

  return { std::move(x), std::move(P), dist.ref_point() };
}

}  // namespace ttb
