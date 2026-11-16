#include "tracking_lib/Trackers/Nothing_Tracker.h"
#include "tracking_lib/DataContainer/DataSourceContainer.h"

namespace ttb
{


template class Nothing_Tracker<double>;
template class Nothing_Tracker<AD>;

template <class UDT>
Nothing_Tracker<UDT>::Nothing_Tracker(TTBManager<UDT>* manager) : _manager{ manager }
{
}

template <class UDT>
Time Nothing_Tracker<UDT>::time() const
{
  return _time;
}

template <class UDT>
void Nothing_Tracker<UDT>::cycle(std::vector<DataSourceContainer<UDT>> measContainerList)
{
  if (not measContainerList.empty())
  {
    _time = measContainerList.back()._time;
  }
}

template <class UDT>
std::vector<BernoulliState<UDT>> Nothing_Tracker<UDT>::get_estimate() const
{
  return {};
}

template <class UDT>
std::vector<StateId> Nothing_Tracker<UDT>::all_tracks() const
{
  return {};
}

template <class UDT>
void Nothing_Tracker<UDT>::reset()
{
  _time = Time{ 0s };
}

}  // namespace ttb
