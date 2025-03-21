#include "tracking_lib/StateModels/ISCATRStateModel.h"
// ######################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"

namespace ttb
{

ISCATRStateModel::ISCATRStateModel(TTBManager* manager, StateModelId id)
  : LinearStateModel(
        manager,
        id,
        [&] {
          switch (manager->state_model_params(id).extent)
          {
            case EXTENT::NONE:
              return ISCATR_state_comps;
            case EXTENT::RECTANGULAR:
              return ISCATR_state_comps.merge(Components(EXTENT_RECTANGULAR));
            case EXTENT::CIRCULAR:
              return ISCATR_state_comps.merge(Components(EXTENT_CIRCULAR));
            default:
              assert(false);
              DEBUG_ASSERT_MARK_UNREACHABLE;
          }
        }(),
        [&] {
          switch (manager->state_model_params(id).extent)
          {
            case EXTENT::NONE:
              return ISCATR_noise_comps;
            case EXTENT::RECTANGULAR:
              return ISCATR_noise_comps.merge(Components(EXTENT_RECTANGULAR_NOISE));
            case EXTENT::CIRCULAR:
              return ISCATR_noise_comps.merge(Components(EXTENT_CIRCULAR_NOISE));
            default:
              assert(false);
              DEBUG_ASSERT_MARK_UNREACHABLE;
          }
        }())
{
}

LinearStateModel::ProcessMatrices ISCATRStateModel::processMatrix(Duration dt) const
{
  auto IX = state_comps().indexOf(COMPONENT::POS_X).value();
  auto IVX = state_comps().indexOf(COMPONENT::VEL_X).value();
  auto IAX = state_comps().indexOf(COMPONENT::ACC_X).value();
  auto IY = state_comps().indexOf(COMPONENT::POS_Y).value();
  auto IVY = state_comps().indexOf(COMPONENT::VEL_Y).value();
  auto IAY = state_comps().indexOf(COMPONENT::ACC_Y).value();
  auto IZ = state_comps().indexOf(COMPONENT::POS_Z).value();
  auto IR = state_comps().indexOf(COMPONENT::ROT_Z).value();
  auto IVR = state_comps().indexOf(COMPONENT::VEL_ROT_Z).value();

  double const deltaT = to_seconds(dt);

  double const dt2div2 = 0.5 * deltaT * deltaT;
  double const dt3div6 = 1. / 6. * deltaT * deltaT * deltaT;

  Matrix F{ Matrix::Identity(state_comps()._comps.size(), state_comps()._comps.size()) };
  Matrix gamma{ Matrix::Zero(state_comps()._comps.size(), noise_comps()._comps.size()) };
  Vector sigma(noise_comps()._comps.size());
  for (std::size_t i = 0; i < noise_comps()._comps.size(); i++)
  {
    sigma(i) = std_noise(noise_comps()._comps.at(i)) * std_noise(noise_comps()._comps.at(i));
  }

  gamma(IX, noise_comps().indexOf(COMPONENT::JERK_X).value()) = dt3div6;
  gamma(IVX, noise_comps().indexOf(COMPONENT::JERK_X).value()) = dt2div2;
  gamma(IAX, noise_comps().indexOf(COMPONENT::JERK_X).value()) = deltaT;

  gamma(IY, noise_comps().indexOf(COMPONENT::JERK_Y).value()) = dt3div6;
  gamma(IVY, noise_comps().indexOf(COMPONENT::JERK_Y).value()) = dt2div2;
  gamma(IAY, noise_comps().indexOf(COMPONENT::JERK_Y).value()) = deltaT;

  gamma(IZ, noise_comps().indexOf(COMPONENT::VEL_Z).value()) = deltaT;

  gamma(IR, noise_comps().indexOf(COMPONENT::ACC_ROT_Z).value()) = dt2div2;
  gamma(IVR, noise_comps().indexOf(COMPONENT::ACC_ROT_Z).value()) = deltaT;

  auto fill_gamma_extent = [&](std::vector<COMPONENT> const& state_extent, std::vector<COMPONENT> const& noise_extent) {
    for (auto [state_comp, noise_comp] : std::views::zip(state_extent, noise_extent))
    {
      Index const state_ind = state_comps().indexOf(state_comp).value();
      Index const noise_ind = noise_comps().indexOf(noise_comp).value();
      gamma(state_ind, noise_ind) = deltaT;
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
      fill_gamma_extent(EXTENT_RECTANGULAR, EXTENT_RECTANGULAR_NOISE);
      break;
    }
    case EXTENT::CIRCULAR:
      fill_gamma_extent(EXTENT_CIRCULAR, EXTENT_CIRCULAR_NOISE);
      break;
    default:
      assert(false);
      DEBUG_ASSERT_MARK_UNREACHABLE;
  }

  Matrix Q = gamma * sigma.asDiagonal() * gamma.transpose();

  F(IX, IVX) = deltaT;
  F(IY, IVY) = deltaT;
  F(IX, IAX) = dt2div2;
  F(IY, IAY) = dt2div2;
  F(IVX, IAX) = deltaT;
  F(IVY, IAY) = deltaT;
  F(IR, IVR) = deltaT;

  return { .Gamma = std::move(gamma), .Q = std::move(Q), .F = std::move(F) };
}

STATE_MODEL_TYPE ISCATRStateModel::type() const noexcept
{
  return STATE_MODEL_TYPE::ISCATR;
}

}  // namespace ttb