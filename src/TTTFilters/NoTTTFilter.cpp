#include "tracking_lib/TTTFilters/NoTTTFilter.h"

#include "tracking_lib/Transformations/Transformation.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/StateModels/BaseStateModel.h"

namespace ttb
{

NoTTTFilter::NoTTTFilter(TTBManager* manager) : _manager{ manager }
{
  LOG_DEB("NoTTTFilter constructed");
}

TTBManager* NoTTTFilter::manager() const
{
  return _manager;
}

Time NoTTTFilter::time() const
{
  return Time::min();
}

TTT_FILTER_TYPE NoTTTFilter::type() const
{
  return TTT_FILTER_TYPE::NO;
}

std::vector<State> NoTTTFilter::getEstimate() const
{
  return _tracks;
}

void NoTTTFilter::reset()
{
}

void NoTTTFilter::cycle(Time time, std::vector<StateContainer> trackContainers)
{
  LOG_DEB("NoTTTFilter::cycle");
  _tracks.clear();
  for (StateContainer const& trackContainer : trackContainers)
  {
    for (State const& track : trackContainer._data)
    {
      State estimatedTrack = track;
      auto [stateModelId, state] = track.getEstimate();
      estimatedTrack._state_dist.clear();
      estimatedTrack._state_dist[stateModelId] = std::move(state);
      _tracks.push_back(std::move(estimatedTrack));
    }
  }
  LOG_DEB("NoTTTFilter::cycle done");
}

}  // namespace ttb