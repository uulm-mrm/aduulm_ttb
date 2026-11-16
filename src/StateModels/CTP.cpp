#include "tracking_lib/StateModels/CTP.h"
// ######################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"

namespace ttb
{

template class CTP<double>;
template class CTP<AD>;

template <class UDT>
GaussianDistribution<UDT> CTP<UDT>::predict(Duration dt,
                                       GaussianDistribution<UDT> const& dist,
                                       EgoMotionDistribution const& ego) const
{
  EXTENT const extent = _manager.params()->state_model_params(_id).extent;
  auto const [state_comps, noise_comps] = state_and_noise_comps<STATE_MODEL_TYPE::CTP>(extent);
  return predict_linear(dt, dist, ego, state_comps, processMatrix(dt));
}

template <class UDT>
ProcessMatrices<UDT> CTP<UDT>::processMatrix(Duration dt) const
{
  EXTENT const extent = _manager.params()->state_model_params(_id).extent;
  auto const [state_comps, noise_comps] = state_and_noise_comps<STATE_MODEL_TYPE::CTP>(extent);

  double const fac = to_seconds(dt);

  Matrix<UDT> F{ Matrix<UDT>::Identity(state_comps._comps.size(), state_comps._comps.size()) };
  Matrix<UDT> gamma{ Matrix<UDT>::Zero(state_comps._comps.size(), noise_comps._comps.size()) };
  Vector<UDT> sigma(noise_comps._comps.size());

  for (std::size_t i = 0; i < noise_comps._comps.size(); i++)
  {
    sigma(i) = std_noise(_manager, _id, noise_comps._comps.at(i)) * std_noise(_manager, _id, noise_comps._comps.at(i));
  }

  gamma(state_comps.indexOf(COMPONENT::POS_X).value(), noise_comps.indexOf(COMPONENT::VEL_X).value()) = fac;
  gamma(state_comps.indexOf(COMPONENT::POS_Y).value(), noise_comps.indexOf(COMPONENT::VEL_Y).value()) = fac;
  gamma(state_comps.indexOf(COMPONENT::POS_Z).value(), noise_comps.indexOf(COMPONENT::VEL_Z).value()) = fac;
  gamma(state_comps.indexOf(COMPONENT::ROT_Z).value(), noise_comps.indexOf(COMPONENT::VEL_ROT_Z).value()) = fac;

  auto fill_gamma_extent = [&](Components const& state_extent, Components const& noise_extent) {
    for (auto [state_comp, noise_comp] : std::views::zip(state_extent._comps, noise_extent._comps))
    {
      Index const state_ind = state_comps.indexOf(state_comp).value();
      Index const noise_ind = noise_comps.indexOf(noise_comp).value();
      gamma(state_ind, noise_ind) = fac;
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
  return { .Q = std::move(Q), .F = std::move(F) };
}

template <class UDT>
STATE_MODEL_TYPE CTP<UDT>::type() noexcept
{
  return STATE_MODEL_TYPE::CTP;
}

}  // namespace ttb
