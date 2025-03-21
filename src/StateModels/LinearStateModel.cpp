#include "tracking_lib/StateModels/LinearStateModel.h"
// ######################################################################################################################
#include "tracking_lib/Distributions/BaseDistribution.h"
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/TTBManager/TTBManager.h"

#include <tracy/tracy/Tracy.hpp>

namespace ttb
{

constexpr auto tracy_color = tracy::Color::LightGoldenrod;

LinearStateModel::LinearStateModel(TTBManager* manager, StateModelId id, Components state_comps, Components noise_comps)
  : _manager{ manager }, _id{ id }, _state_comps{ std::move(state_comps) }, _noise_comps{ std::move(noise_comps) }
{
}

void LinearStateModel::predict(Duration dt, BaseDistribution& dist, EgoMotionDistribution const& egoDist) const
{
  ZoneScopedNC("LinearStateModel::predict", tracy_color);
  if (dist.type() != DISTRIBUTION_TYPE::GAUSSIAN)
  {
    LOG_FATAL("Predicting non-Gaussian distribution - BUG ALERT");
    LOG_FATAL("Distribution " + dist.toString());
    throw std::runtime_error("Predicting non-Gaussian distribution - BUG ALERT");
  }

  // compensate Ego
  EgoCompensated ego_comp = ::ttb::compensateEgoMotion(dist.mean(), dist.covariance(), state_comps(), dt, egoDist);

  auto [gamma, Q, F] = processMatrix(dt);

  // apply kinematic
  Vector x = F * ego_comp.state;
  if (state_comps().indexOf(COMPONENT::ROT_Z).has_value())
  {
    angles::normalizeAngle(x(state_comps().indexOf(COMPONENT::ROT_Z).value()));
  }
  // compute variance
  Matrix P = F * ego_comp.cov * F.transpose() + Q;

  dist.set(std::move(x));
  dist.set(std::move(P));
}

void LinearStateModel::compensateEgoMotion(Duration dt,
                                           BaseDistribution& dist,
                                           EgoMotionDistribution const& egoMotion) const
{
  ZoneScopedNC("LinearStateModel::predict", tracy_color);
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

double LinearStateModel::std_noise(COMPONENT comp) const
{
  if (not manager()->state_model_params(id()).model_noise_std_dev.contains(to_string(comp)))
  {
    throw std::runtime_error("noise comp " + to_string(comp) + " not in the noise map for model " +
                             std::to_string(id().value_));
  }
  return manager()->state_model_params(id()).model_noise_std_dev.at(to_string(comp));
}

std::string LinearStateModel::toString() const
{
  return "State Model: " + std::to_string(_id.value_);
}

TTBManager* LinearStateModel::manager() const
{
  return _manager;
}

Components const& LinearStateModel::state_comps() const
{
  return _state_comps;
}

Components const& LinearStateModel::noise_comps() const
{
  return _noise_comps;
}

StateModelId LinearStateModel::id() const
{
  return _id;
}

}  // namespace ttb
