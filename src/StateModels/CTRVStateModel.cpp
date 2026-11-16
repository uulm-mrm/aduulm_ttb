#include "tracking_lib/StateModels/CTRVStateModel.h"
// ######################################################################################################################
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/TTBManager/TTBManager.h"

namespace ttb
{

template class CTRVStateModel<double>;
template class CTRVStateModel<AD>;

template <class UDT>
GaussianDistribution<UDT> CTRVStateModel<UDT>::predict(Duration dt,
                                                  GaussianDistribution<UDT> const& dist,
                                                  EgoMotionDistribution const& ego) const
{
  EXTENT const extent = _manager.params()->state_model_params(_id).extent;
  auto const [state_comps, noise_comps] = state_and_noise_comps<STATE_MODEL_TYPE::CTRV>(extent);
  return predict_non_linear(
      _manager,
      _id,
      dt,
      dist,
      ego,
      state_comps,
      noise_comps,
      [&](Duration dt, Vector<UDT> const& state, Vector <UDT>const& noise) {
        return applyKinematicModel(dt, state, noise);
      });
}

/// nonlinear CTRV process model including the noise parameters
/// CTRV model as in "Filter- und Trackingverfahren"-script (Version 3.1, October 2017, page 84-86)
template <class UDT>
Vector<UDT> CTRVStateModel<UDT>::applyKinematicModel(Duration dt, Vector<UDT> const& state, Vector<UDT> const& noise) const
{
  Vector<UDT> predicted_state = state;

  EXTENT const extent = _manager.params()->state_model_params(_id).extent;
  auto const [state_comps, noise_comps] = state_and_noise_comps<STATE_MODEL_TYPE::CTRV>(extent);

  UDT const x = state(state_comps.indexOf(COMPONENT::POS_X).value());
  UDT const y = state(state_comps.indexOf(COMPONENT::POS_Y).value());
  UDT const v = state(state_comps.indexOf(COMPONENT::VEL_ABS).value());
  UDT const p = state(state_comps.indexOf(COMPONENT::ROT_Z).value());
  UDT const vp = state(state_comps.indexOf(COMPONENT::VEL_ROT_Z).value());
  // process noise from augmented state
  UDT const a = noise(noise_comps.indexOf(COMPONENT::ACC_ABS).value());
  UDT const ap = noise(noise_comps.indexOf(COMPONENT::ACC_ROT_Z).value());

  double const deltaT = to_seconds(dt);
  double const dt2div2 = 0.5 * deltaT * deltaT;
  // modified TTB_EPS for case distinction in CTRV kinematic model, see
  // https://mrm-git.e-technik.uni-ulm.de/aduulm/source/tracking/issues/56
  // for detailed derivation and explanation
  static constexpr double CTRV_EPS = 3 * 10 * 10 * 10 * TTB_EPS;

  if (fabs(vp) < CTRV_EPS)
  {
    predicted_state(state_comps.indexOf(COMPONENT::POS_X).value()) = x + cos(p) * deltaT * v + dt2div2 * cos(p) * a;
    predicted_state(state_comps.indexOf(COMPONENT::POS_Y).value()) = y + sin(p) * deltaT * v + dt2div2 * sin(p) * a;
    predicted_state(state_comps.indexOf(COMPONENT::ROT_Z).value()) = p + vp * deltaT + dt2div2 * ap;
  }
  else
  {
    UDT const dphi = vp * deltaT;
    predicted_state(state_comps.indexOf(COMPONENT::POS_X).value()) =
        x + v * (sin(p + dphi) - sin(p)) / vp + dt2div2 * cos(p) * a;
    predicted_state(state_comps.indexOf(COMPONENT::POS_Y).value()) =
        y + v * (cos(p) - cos(p + dphi)) / vp + dt2div2 * sin(p) * a;
    predicted_state(state_comps.indexOf(COMPONENT::ROT_Z).value()) = p + dphi + dt2div2 * ap;
  }
  predicted_state(state_comps.indexOf(COMPONENT::VEL_ABS).value()) = v + a * deltaT;
  predicted_state(state_comps.indexOf(COMPONENT::VEL_ROT_Z).value()) = vp + ap * deltaT;

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
STATE_MODEL_TYPE CTRVStateModel<UDT>::type() noexcept
{
  return STATE_MODEL_TYPE::CTRV;
}

}  // namespace ttb
