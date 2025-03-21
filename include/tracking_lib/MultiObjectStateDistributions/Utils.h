#pragma once

#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBManager/TTBManager.h"

namespace ttb::utils
{

/// predict all states inside the state_container
/// proj: should map from a single element of the container to a State&
/// trans: additional steps after the prediction
template <class Ts, class Trans = std::identity, class Proj = std::identity>
void predict(TTBManager* manager,
             Ts& state_container,
             Duration dt,
             EgoMotionDistribution const& ego_motion_distribution,
             Proj proj = {},
             Trans trans = {})
{
  auto work = [&](auto& track) {
    std::invoke(proj, track).predict(dt, ego_motion_distribution);
    trans(proj(track));
  };
  if (manager->params().thread_pool_size > 0)
  {
    for (auto& T : state_container)
    {
      manager->thread_pool().detach_task([&] { work(T); });
    }
    manager->thread_pool().wait();
  }
  else
  {
    std::ranges::for_each(state_container, work);
  }
}

/// innovate all states inside the state_container
/// proj: should map from a single element of the container to a State&
template <class Ts, class Proj = std::identity>
void innovate(TTBManager* manager,
              Ts& state_container,
              MeasurementContainer const& measurement_container,
              Proj proj = {})
{
  auto work = [&](auto& T) { std::invoke(proj, T).innovate(measurement_container); };
  if (manager->params().thread_pool_size > 0)
  {
    for (auto& T : state_container)
    {
      manager->thread_pool().detach_task([&] { work(T); });
    }
    manager->thread_pool().wait();
  }
  else
  {
    std::ranges::for_each(state_container, work);
  }
}

}  // namespace ttb::utils
