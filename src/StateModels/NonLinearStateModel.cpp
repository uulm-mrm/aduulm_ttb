#include "tracking_lib/StateModels/NonLinearStateModel.h"
// #####################################################################################################################
#include "tracking_lib/Distributions/BaseDistribution.h"
#include "tracking_lib/Transformations/Transformation.h"
#include "tracking_lib/TTBManager/TTBManager.h"
// #####################################################################################################################
#include <tracy/tracy/Tracy.hpp>

namespace ttb
{

constexpr auto tracy_color = tracy::Color::DarkGoldenrod;

NonLinearStateModel::NonLinearStateModel(TTBManager* manager,
                                         StateModelId id,
                                         Components state_comps,
                                         Components noise_comps)
  : _manager{ manager }, _id{ id }, _state_comps{ std::move(state_comps) }, _noise_comps{ std::move(noise_comps) }
{
}

void NonLinearStateModel::predict(Duration dt, BaseDistribution& dist, EgoMotionDistribution const& egoDist) const
{
  ZoneScopedNC("NonLinearStateModel::predict", tracy_color);
  LOG_DEB("NonlinearModel predict");
  Vector const x = dist.mean();
  Matrix const P = dist.covariance();
  Vector sigma(noise_comps()._comps.size());
  for (auto const& [i, comp] : std::views::enumerate(noise_comps()._comps))
  {
    sigma(i) = std::pow(std_noise(comp) * to_seconds(dt), 2);
  }
  Matrix Q = sigma.asDiagonal();
  std::optional<transformation::Transformed> trafo = transformation::unscentTransform(
      x,
      P,
      Q,
      [&](Vector state, Vector noise) -> std::optional<Vector> {
        state =
            ::ttb::compensateEgoMotion(state, Matrix::Identity(state.rows(), state.rows()), state_comps(), dt, egoDist)
                .state;
        state = applyKinematicModel(dt, std::move(state), std::move(noise));
        return state;
      },
      state_comps().indexOf(COMPONENT::ROT_Z));
  dist.set(std::move(trafo.value().mean));
  dist.set(std::move(trafo.value().cov));
}

void NonLinearStateModel::compensateEgoMotion(Duration dt,
                                              BaseDistribution& dist,
                                              EgoMotionDistribution const& egoMotion) const
{
  ZoneScopedNC("NonLinearStateModel::compensateEgoMotion", tracy_color);
  if (dist.type() != DISTRIBUTION_TYPE::GAUSSIAN)
  {
    LOG_FATAL("EgoCompensate non-Gaussian distribution - BUG ALERT");
    LOG_FATAL("Distribution " + dist.toString());
    throw std::runtime_error("EgoCompensate non-Gaussian distribution - BUG ALERT");
  }
  // compensate Ego
  EgoCompensated ego_comp = ::ttb::compensateEgoMotion(dist.mean(), dist.covariance(), state_comps(), dt, egoMotion);

  dist.set(std::move(ego_comp.state));
  dist.set(std::move(ego_comp.cov));
}

double NonLinearStateModel::std_noise(COMPONENT comp) const
{
  if (not manager()->state_model_params(id()).model_noise_std_dev.contains(to_string(comp)))
  {
    throw std::runtime_error("noise comp " + to_string(comp) + " not in the noise map for model " +
                             std::to_string(id().value_));
  }
  return manager()->state_model_params(id()).model_noise_std_dev.at(to_string(comp));
}

std::string NonLinearStateModel::toString() const
{
  return "State Model: " + std::to_string(_id.value_);
}

TTBManager* NonLinearStateModel::manager() const
{
  return _manager;
}

Components const& NonLinearStateModel::state_comps() const
{
  return _state_comps;
}

Components const& NonLinearStateModel::noise_comps() const
{
  return _noise_comps;
}

StateModelId NonLinearStateModel::id() const
{
  return _id;
}

}  // namespace ttb
