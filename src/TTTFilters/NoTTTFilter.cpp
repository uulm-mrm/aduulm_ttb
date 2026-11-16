#include "tracking_lib/TTTFilters/NoTTTFilter.h"

#include "tracking_lib/Transformations/Transformation.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/States/State.h"

namespace ttb
{

template class NoTTTFilter<double>;
template class NoTTTFilter<AD>;

template <class UDT>
NoTTTFilter<UDT>::NoTTTFilter(TTBManager<UDT>* manager) : _manager{ manager }
{
  LOG_DEB("NoTTTFilter constructed");
}

template <class UDT>
TTBManager<UDT>* NoTTTFilter<UDT>::manager() const
{
  return _manager;
}

template <class UDT>
Time NoTTTFilter<UDT>::time() const
{
  return Time::min();
}

template <class UDT>
TTT_FILTER_TYPE NoTTTFilter<UDT>::type() const
{
  return TTT_FILTER_TYPE::NO;
}

template <class UDT>
std::vector<BernoulliState<UDT>> NoTTTFilter<UDT>::get_estimate() const
{
  return _tracks | std::views::transform([](State<UDT> const& state) { return BernoulliState<UDT>{ state, 1 }; }) |
         std::ranges::to<std::vector<BernoulliState<UDT>>>();
}

template <class UDT>
std::vector<StateId> NoTTTFilter<UDT>::all_tracks() const
{
  return {};
}

template <class UDT>
void NoTTTFilter<UDT>::reset()
{
}

template <class UDT>
void NoTTTFilter<UDT>::cycle(std::vector<DataSourceContainer<UDT>> trackContainers)
{
  LOG_DEB("NoTTTFilter::cycle");
  _tracks.clear();
  for (DataSourceContainer<UDT> const& trackContainer : trackContainers)
  {
    for (State<UDT> const& track : trackContainer._tracks | member<&BernoulliState<UDT>::state>)
    {
      State estimatedTrack = track;
      auto [stateModelId, state] = track.getEstimate();
      estimatedTrack._state_dist.clear();
      estimatedTrack._state_dist.emplace(stateModelId, std::pair{ UDT{ 1.0 }, Distribution<UDT>{ std::move(state) } });
      _tracks.push_back(std::move(estimatedTrack));
    }
  }
  LOG_DEB("NoTTTFilter::cycle done");
}

}  // namespace ttb
