#include "tracking_lib/StateModels/CTRVStateModel.h"
// ######################################################################################################################
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/TTBManager/TTBManager.h"

namespace ttb
{
CTRVStateModel::CTRVStateModel(TTBManager* manager, StateModelId id)
  : NonLinearStateModel(
        manager,
        id,
        [&] {
          switch (manager->state_model_params(id).extent)
          {
            case EXTENT::NONE:
              return CTRV_state_comps;
            case EXTENT::RECTANGULAR:
              return CTRV_state_comps.merge(Components(EXTENT_RECTANGULAR));
            case EXTENT::CIRCULAR:
              return CTRV_state_comps.merge(Components(EXTENT_CIRCULAR));
            default:
              assert(false);
              DEBUG_ASSERT_MARK_UNREACHABLE;
          }
        }(),
        [&] {
          switch (manager->state_model_params(id).extent)
          {
            case EXTENT::NONE:
              return CTRV_noise_comps;
            case EXTENT::RECTANGULAR:
              return CTRV_noise_comps.merge(Components(EXTENT_RECTANGULAR_NOISE));
            case EXTENT::CIRCULAR:
              return CTRV_noise_comps.merge(Components(EXTENT_CIRCULAR_NOISE));
            default:
              assert(false);
              DEBUG_ASSERT_MARK_UNREACHABLE;
          }
        }())
{
}

/// nonlinear CTRV process model including the noise parameters
/// CTRV model as in "Filter- und Trackingverfahren"-script (Version 3.1, October 2017, page 84-86)
Vector CTRVStateModel::applyKinematicModel(Duration dt, Vector state, Vector noise) const
{
  double const x = state(state_comps().indexOf(COMPONENT::POS_X).value());
  double const y = state(state_comps().indexOf(COMPONENT::POS_Y).value());
  double const v = state(state_comps().indexOf(COMPONENT::VEL_ABS).value());
  double const p = state(state_comps().indexOf(COMPONENT::ROT_Z).value());
  double const vp = state(state_comps().indexOf(COMPONENT::VEL_ROT_Z).value());
  // process noise from augmented state
  double const a = noise(noise_comps().indexOf(COMPONENT::ACC_ABS).value());
  double const ap = noise(noise_comps().indexOf(COMPONENT::ACC_ROT_Z).value());

  double const deltaT = to_seconds(dt);
  double const dt2div2 = 0.5 * deltaT * deltaT;
  // check that the yaw acceleration is not zero to prevent dividing by zero
  // for the derivation of CTRV_EPS see
  // https://mrm-git.e-technik.uni-ulm.de/aduulm/source/tracking/issues/56
  if (fabs(vp) < CTRV_EPS)
  {
    state(state_comps().indexOf(COMPONENT::POS_X).value()) = x + cos(p) * deltaT * v + dt2div2 * cos(p) * a;
    state(state_comps().indexOf(COMPONENT::POS_Y).value()) = y + sin(p) * deltaT * v + dt2div2 * sin(p) * a;
    state(state_comps().indexOf(COMPONENT::ROT_Z).value()) = p + vp * deltaT + dt2div2 * ap;
  }
  else
  {
    double const dphi = vp * deltaT;
    state(state_comps().indexOf(COMPONENT::POS_X).value()) =
        x + v * (sin(p + dphi) - sin(p)) / vp + dt2div2 * cos(p) * a;
    state(state_comps().indexOf(COMPONENT::POS_Y).value()) =
        y + v * (cos(p) - cos(p + dphi)) / vp + dt2div2 * sin(p) * a;
    state(state_comps().indexOf(COMPONENT::ROT_Z).value()) = p + dphi + dt2div2 * ap;
  }
  state(state_comps().indexOf(COMPONENT::VEL_ABS).value()) = v + a * deltaT;
  state(state_comps().indexOf(COMPONENT::VEL_ROT_Z).value()) = vp + ap * deltaT;

  auto add_extent_noise = [&](std::vector<COMPONENT> const& state_extent, std::vector<COMPONENT> const& noise_extent) {
    for (auto [state_comp, noise_comp] : std::views::zip(state_extent, noise_extent))
    {
      double const noise_val = noise(noise_comps().indexOf(noise_comp).value());
      state(state_comps().indexOf(state_comp).value()) += noise_val;
    }
  };
  switch (manager()->state_model_params(id()).extent)
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
      DEBUG_ASSERT_MARK_UNREACHABLE;
  }

  angles::normalizeAngle(state(state_comps().indexOf(COMPONENT::ROT_Z).value()));
  return state;
}

STATE_MODEL_TYPE CTRVStateModel::type() const noexcept
{
  return STATE_MODEL_TYPE::CTRV;
}

}  // namespace ttb
