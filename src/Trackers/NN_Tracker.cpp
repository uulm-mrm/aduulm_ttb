#include "tracking_lib/Trackers/NN_Tracker.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"

namespace ttb
{

template class NN_Tracker<double>;
template class NN_Tracker<AD>;

template <class UDT>
NN_Tracker<UDT>::NN_Tracker(TTBManager<UDT>* manager) : _manager{ manager }
{
}

template <class UDT>
FILTER_TYPE NN_Tracker<UDT>::type() const
{
  return FILTER_TYPE::NN;
}

template <class UDT>
TTBManager<UDT>* NN_Tracker<UDT>::manager() const
{
  return _manager;
}

template <class UDT>
Time NN_Tracker<UDT>::time() const
{
  return _time;
}

template <class UDT>
void NN_Tracker<UDT>::cycle(std::vector<DataSourceContainer<UDT>> measContainerList)
{
  LOG_DEB("Update " + std::to_string(_tracks.size()) + " current Tracks with " << measContainerList.size());
}

template <class UDT>
std::vector<BernoulliState<UDT>> NN_Tracker<UDT>::get_estimate() const
{
  return {};
}

template <class UDT>
std::vector<StateId> NN_Tracker<UDT>::all_tracks() const
{
  return {};
}

template <class UDT>
void NN_Tracker<UDT>::reset()
{
}

}  // namespace ttb
