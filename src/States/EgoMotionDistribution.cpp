#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"

namespace ttb
{

EgoMotionDistribution::EgoMotionDistribution(std::unique_ptr<BaseDistribution> dist, Components stateComps)
  : _dist{ std::move(dist) }, _comps{ std::move(stateComps) }
{
}

EgoMotionDistribution::EgoMotionDistribution(GaussianDistribution const& dist, Components stateComps)
  : EgoMotionDistribution(std::make_unique<GaussianDistribution>(dist), std::move(stateComps))
{
}

std::string EgoMotionDistribution::toString(std::string const& prefix) const
{
  return prefix + "EgoMotion\n" + _dist->toString(prefix + "|\t") + _comps.toString(prefix + "|\t");
}

std::optional<double> EgoMotionDistribution::meanOf(ttb::COMPONENT sc) const
{
  auto ind = _comps.indexOf(sc);
  if (ind.has_value())
  {
    return _dist->mean()(ind.value());
  }
  // compute vel_abs based on vel_x and vel_y
  if (sc == COMPONENT::VEL_ABS and _comps.indexOf(COMPONENT::VEL_X).has_value() and
      _comps.indexOf(COMPONENT::VEL_Y).has_value())
  {
    return std::hypot(_dist->mean()(_comps.indexOf(COMPONENT::VEL_X).value()),
                      _dist->mean()(_comps.indexOf(COMPONENT::VEL_Y).value()));
  }
  // compute orientation based on vel_x and vel_y assuming driving forward
  if (sc == COMPONENT::VEL_ABS and _comps.indexOf(COMPONENT::VEL_X).has_value() and
      _comps.indexOf(COMPONENT::VEL_Y).has_value())
  {
    return std::atan2(_dist->mean()(_comps.indexOf(COMPONENT::VEL_Y).value()),
                      _dist->mean()(_comps.indexOf(COMPONENT::VEL_X).value()));
  }
  // compute vel_x based on vel_abs
  if (sc == COMPONENT::VEL_X and _comps.indexOf(COMPONENT::VEL_ABS).has_value())
  {
    return _dist->mean()(_comps.indexOf(COMPONENT::VEL_ABS).value());
  }
  // compute vel_y based on vel_abs
  if (sc == COMPONENT::VEL_Y and _comps.indexOf(COMPONENT::VEL_ABS).has_value())
  {
    return 0;
  }
  return {};
}

EgoMotionDistribution::EgoMotionDistribution(const EgoMotionDistribution& other)
  : _dist{ other._dist->clone() }, _comps{ other._comps }
{
}

EgoMotionDistribution& EgoMotionDistribution::operator=(const EgoMotionDistribution& other)
{
  EgoMotionDistribution tmp(other);
  *this = std::move(tmp);
  return *this;
}

EgoMotionDistribution& EgoMotionDistribution::operator=(EgoMotionDistribution&& other) noexcept = default;

EgoMotionDistribution::EgoMotionDistribution(EgoMotionDistribution&& other) noexcept = default;

EgoMotionDistribution::~EgoMotionDistribution() = default;

EgoMotionDistribution EgoMotionDistribution::zero()
{
  return { std::make_unique<GaussianDistribution>(
               Vector::Zero(REQUIRED_EGO_MOTION_COMPONENTS.size()),
               Matrix::Identity(REQUIRED_EGO_MOTION_COMPONENTS.size(), REQUIRED_EGO_MOTION_COMPONENTS.size())),
           Components(REQUIRED_EGO_MOTION_COMPONENTS) };
}

}  // namespace ttb
