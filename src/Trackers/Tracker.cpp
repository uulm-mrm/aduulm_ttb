#include "tracking_lib/Trackers/Tracker.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include <tracy/tracy/TracyC.h>

namespace ttb
{

template class Tracker<double>;
template class Tracker<AD>;

template <class UDT>
Tracker<UDT>::Tracker(TTBManager<UDT>* manager)
  : _manager{ manager }, _tracker{ [&] -> AllFilters {
    if (manager->params()->filter.enable)
    {
      switch (manager->params()->filter.type)
      {
        case FILTER_TYPE::LMB_IC:
          return ICTracker<UDT, LMBDistribution<UDT>, TTBManager<UDT>>{ manager };
        case FILTER_TYPE::LMB_MS:
          return MSTracker<UDT, LMBDistribution<UDT>, TTBManager<UDT>>{ manager };
        case FILTER_TYPE::NO:
          return NO_Tracker(manager);
        case FILTER_TYPE::NN:
          return NN_Tracker(manager);
        case FILTER_TYPE::GNN:
          return GNN_Tracker(manager);
        case FILTER_TYPE::GLMB_IC:
          return ICTracker<UDT, GLMBDistribution<UDT>, TTBManager<UDT>>{ manager };
        case FILTER_TYPE::GLMB_MS:
          return MSTracker<UDT, GLMBDistribution<UDT>, TTBManager<UDT>>{ manager };
        case FILTER_TYPE::ID_TRACKER:
          return Id_Tracker(manager);
        case FILTER_TYPE::PMBM_IC:
          return ICTracker<UDT, PMBMDistribution<UDT>, TTBManager<UDT>>{ manager };
      }
    }
    if (manager->params()->ttt_filter.enable)
    {
      switch (manager->params()->ttt_filter.type)
      {
        case TTT_FILTER_TYPE::NO:
          return NoTTTFilter(manager);
        case TTT_FILTER_TYPE::TRANS:
          return TransTTTFilter(manager);
      }
    }
    LOG_FATAL("Neither Filter nor Track-to-track Filter enabled. Doing nothing.");
    return Nothing_Tracker(manager);
  }() }
{
}

template <class UDT>
void Tracker<UDT>::cycle(std::vector<DataSourceContainer<UDT>> measContainers)
{
  TracyFiberEnter("Tracker::cycle");
  TracyCZoneN(ctx, "Tracker::cycle", 1);
  std::visit([&](auto& tracker) { tracker.cycle(std::move(measContainers)); }, _tracker);
  TracyFiberEnter("Tracker::cycle");
  TracyCZoneEnd(ctx);
}

template <class UDT>
void Tracker<UDT>::reset()
{
  std::visit([&](auto& tracker) { tracker.reset(); }, _tracker);
}

template <class UDT>
Time Tracker<UDT>::time() const
{
  return std::visit([&](auto const& tracker) { return tracker.time(); }, _tracker);
}

template <class UDT>
std::vector<BernoulliState<UDT>> Tracker<UDT>::get_estimate() const
{
  return std::visit([&](auto const& tracker) { return tracker.get_estimate(); }, _tracker);
}

template <class UDT>
std::vector<StateId> Tracker<UDT>::all_tracks() const
{
  return std::visit([&](auto const& tracker) { return tracker.all_tracks(); }, _tracker);
}

}  // namespace ttb
