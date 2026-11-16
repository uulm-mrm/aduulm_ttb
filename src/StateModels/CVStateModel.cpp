#include "tracking_lib/StateModels/CVStateModel.h"
// ######################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"

namespace ttb
{

template class CVStateModel<double>;
template class CVStateModel<AD>;

template <class UDT>
GaussianDistribution<UDT> CVStateModel<UDT>::predict(Duration dt,
                                                GaussianDistribution<UDT> const& dist,
                                                EgoMotionDistribution const& ego) const
{
  EXTENT const extent = _manager.params()->state_model_params(_id).extent;
  auto const [state_comps, noise_comps] = state_and_noise_comps<STATE_MODEL_TYPE::CV>(extent);
  return predict_linear(dt, dist, ego, state_comps, processMatrix(dt));
}

template <class UDT>
ProcessMatrices<UDT> CVStateModel<UDT>::processMatrix(Duration dt) const
{
  EXTENT const extent = _manager.params()->state_model_params(_id).extent;
  auto const [state_comps, noise_comps] = state_and_noise_comps<STATE_MODEL_TYPE::CV>(extent);

  auto IX = state_comps.indexOf(COMPONENT::POS_X).value();
  auto IVX = state_comps.indexOf(COMPONENT::VEL_X).value();
  auto IY = state_comps.indexOf(COMPONENT::POS_Y).value();
  auto IVY = state_comps.indexOf(COMPONENT::VEL_Y).value();
  auto IZ = state_comps.indexOf(COMPONENT::POS_Z).value();

  double const deltaT = to_seconds(dt);
  double dt2div2 = 0.5 * deltaT * deltaT;

  Matrix<UDT> F{ Matrix<UDT>::Identity(state_comps._comps.size(), state_comps._comps.size()) };
  Matrix<UDT> gamma{ Matrix<UDT>::Zero(state_comps._comps.size(), noise_comps._comps.size()) };
  Vector<UDT> sigma(noise_comps._comps.size());
  for (std::size_t i = 0; i < noise_comps._comps.size(); i++)
  {
    sigma(i) = std_noise(_manager, _id, noise_comps._comps.at(i)) * std_noise(_manager, _id, noise_comps._comps.at(i));
  }

  gamma(IX, noise_comps.indexOf(COMPONENT::ACC_X).value()) = dt2div2;
  gamma(IVX, noise_comps.indexOf(COMPONENT::ACC_X).value()) = deltaT;
  gamma(IY, noise_comps.indexOf(COMPONENT::ACC_Y).value()) = dt2div2;
  gamma(IVY, noise_comps.indexOf(COMPONENT::ACC_Y).value()) = deltaT;
  gamma(IZ, noise_comps.indexOf(COMPONENT::VEL_Z).value()) = deltaT;

  auto fill_gamma_extent = [&](Components const& state_extent, Components const& noise_extent) {
    for (auto [state_comp, noise_comp] : std::views::zip(state_extent._comps, noise_extent._comps))
    {
      Index const state_ind = state_comps.indexOf(state_comp).value();
      Index const noise_ind = noise_comps.indexOf(noise_comp).value();
      gamma(state_ind, noise_ind) = deltaT;
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
      fill_gamma_extent(EXTENT_RECTANGULAR, EXTENT_RECTANGULAR_NOISE);
      break;
    }
    case EXTENT::CIRCULAR:
      fill_gamma_extent(EXTENT_CIRCULAR, EXTENT_CIRCULAR_NOISE);
      break;
    default:
      assert(false);
      std::unreachable();
  }

  Matrix<UDT> Q = gamma * sigma.asDiagonal() * gamma.transpose();

  F(IX, IVX) = deltaT;
  F(IY, IVY) = deltaT;
  return { .Q = std::move(Q), .F = std::move(F) };
}

template <class UDT>
STATE_MODEL_TYPE CVStateModel<UDT>::type() noexcept
{
  return STATE_MODEL_TYPE::CV;
}

}  // namespace ttb
