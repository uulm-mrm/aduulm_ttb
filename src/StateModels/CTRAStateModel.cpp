#include "tracking_lib/StateModels/CTRAStateModel.h"
// ######################################################################################################################
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/TTBManager/TTBManager.h"

namespace ttb
{

template class CTRAStateModel<double>;
template class CTRAStateModel<AD>;

template <class UDT>
GaussianDistribution<UDT> CTRAStateModel<UDT>::predict(Duration dt,
                                                       GaussianDistribution<UDT> const& dist,
                                                       EgoMotionDistribution const& ego) const
{
  EXTENT const extent = _manager.params()->state_model_params(_id).extent;
  auto const [state_comps, noise_comps] = state_and_noise_comps<STATE_MODEL_TYPE::CTRA>(extent);
  return predict_non_linear(_manager,
                            _id,
                            dt,
                            dist,
                            ego,
                            state_comps,
                            noise_comps,
                            [&](Duration dt, Vector<UDT> const& state, Vector<UDT> const& noise) {
                              return applyKinematicModel(dt, state, noise);
                            });
}

///  nonlinear CTRA process model including the noise parameters as jerk (derivative of acceleration)
///  and yaw acceleration
///      (generally, there are two versions of the CTRA model:
///          1) with acceleration increment as noise parameter
///          2) with jerk as noise parameter
///          --> this implementation follows the second variant because if we do not have a constant sampling time
///             (and we definitely do not) the noise parameter modelled as jerk is easier to understand and to choose)
///
///  CTRA model as in
///      Schubert, Robin, Eric Richter, and Gerd Wanielik.
///      "Comparison and evaluation of advanced motion models for vehicle tracking."
///      2008 11th international conference on information fusion. IEEE, 2008.
///      https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4632283
///  with noise parameter handling as, e.g.,
///  in CA model in
///      "Filter- und Trackingverfahren"-script (Version 3.1, October 2017, page 54-56)
///  or in Wiener-Process Acceleration Model in
///      Li, X. Rong, and Vesselin P. Jilkov. "Survey of maneuvering target tracking.
///      Part I. Dynamic models." IEEE Transactions on aerospace and electronic systems 39.4 (2003): 1333-1364.
///      https://ieeexplore.ieee.org/document/1261132
///      --> see Wiener-Process Acceleration Model (page 1336-1337)
///  or in Discrete Wiener Process Acceleration Model in
///      Bar-Shalom, Yaakov, X. Rong Li, and Thiagalingam Kirubarajan.
///      Estimation with applications to tracking and navigation: theory algorithms and software.
///      John Wiley & Sons, 2004.
template <class UDT>
Vector<UDT> CTRAStateModel<UDT>::applyKinematicModel(Duration dt, Vector<UDT> const& state, Vector<UDT> const& noise) const
{
  Vector<UDT> predicted_state = state;
  EXTENT const extent = _manager.params()->state_model_params(_id).extent;
  auto const [state_comps, noise_comps] = state_and_noise_comps<STATE_MODEL_TYPE::CTRA>(extent);

  UDT const x = state(state_comps.indexOf(COMPONENT::POS_X).value());
  UDT const y = state(state_comps.indexOf(COMPONENT::POS_Y).value());
  UDT const v = state(state_comps.indexOf(COMPONENT::VEL_ABS).value());
  UDT const a = state(state_comps.indexOf(COMPONENT::ACC_ABS).value());
  UDT const p = state(state_comps.indexOf(COMPONENT::ROT_Z).value());
  UDT const vp = state(state_comps.indexOf(COMPONENT::VEL_ROT_Z).value());
  // process noise from augmented state
  UDT const j = noise(noise_comps.indexOf(COMPONENT::JERK_ABS).value());
  UDT const ap = noise(noise_comps.indexOf(COMPONENT::ACC_ROT_Z).value());

  double const deltaT = to_seconds(dt);

  double const dt2div2 = 1.0 / 2.0 * deltaT * deltaT;
  double const dt3div6 = 1.0 / 6.0 * deltaT * deltaT * deltaT;
  /// modified TTB_EPS for case distinction in CTRA kinematic model, see
  /// https://mrm-git.e-technik.uni-ulm.de/aduulm/source/tracking/issues/56
  /// for detailed derivation and explanation
  static constexpr double CTRA_EPS = 10 * 10 * 10 * TTB_EPS;
  if (fabs(vp * vp) < CTRA_EPS)
  {
    predicted_state(state_comps.indexOf(COMPONENT::POS_X).value()) =
        x + cos(p) * (deltaT * v + dt2div2 * a) + dt3div6 * cos(p) * j;
    predicted_state(state_comps.indexOf(COMPONENT::POS_Y).value()) =
        y + sin(p) * (deltaT * v + dt2div2 * a) + dt3div6 * sin(p) * j;
    predicted_state(state_comps.indexOf(COMPONENT::ROT_Z).value()) = p * (deltaT * vp + dt2div2 * ap);
  }
  else
  {
    UDT const dphi = vp * deltaT;
    predicted_state(state_comps.indexOf(COMPONENT::POS_X).value()) =
        x +
        (1. / (vp * vp)) *
            ((v * vp + deltaT * a * vp) * sin(p + dphi) + a * cos(p + dphi) - v * vp * sin(p) - a * cos(p)) +
        dt3div6 * cos(p) * j;
    predicted_state(state_comps.indexOf(COMPONENT::POS_Y).value()) =
        y +
        (1. / (vp * vp)) *
            ((-v * vp - deltaT * a * vp) * cos(p + dphi) + a * sin(p + dphi) + v * vp * cos(p) - a * sin(p)) +
        dt3div6 * sin(p) * j;
    predicted_state(state_comps.indexOf(COMPONENT::ROT_Z).value()) = p + (dphi + dt2div2 * ap);
  }

  predicted_state(state_comps.indexOf(COMPONENT::VEL_ABS).value()) = v + deltaT * a + dt2div2 * j;
  predicted_state(state_comps.indexOf(COMPONENT::ACC_ABS).value()) = a + deltaT * j;
  predicted_state(state_comps.indexOf(COMPONENT::VEL_ROT_Z).value()) = vp + (deltaT * ap);

  auto add_extent_noise = [&](Components const& state_extent, Components const& noise_extent) {
    for (auto [state_comp, noise_comp] : std::views::zip(state_extent._comps, noise_extent._comps))
    {
      UDT const noise_val = noise(noise_comps.indexOf(noise_comp).value());
      predicted_state(state_comps.indexOf(state_comp).value()) += noise_val;
    }
  };
  switch (extent)
  {
    case EXTENT::NONE:
    {
      // Do nothing
      break;
    }
    case EXTENT::RECTANGULAR:
    {
      add_extent_noise(EXTENT_RECTANGULAR, EXTENT_RECTANGULAR_NOISE);
      break;
    }
    case EXTENT::CIRCULAR:
      add_extent_noise(EXTENT_CIRCULAR, EXTENT_CIRCULAR_NOISE);
      break;
    default:
      assert(false);
      std::unreachable();
  }

  angles::normalizeAngle(predicted_state(state_comps.indexOf(COMPONENT::ROT_Z).value()));
  return predicted_state;
}

template <class UDT>
STATE_MODEL_TYPE CTRAStateModel<UDT>::type() noexcept
{
  return STATE_MODEL_TYPE::CTRA;
}

}  // namespace ttb
