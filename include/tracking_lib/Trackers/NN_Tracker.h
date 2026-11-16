#pragma once

#include "tracking_lib/TTBTypes/Concepts.h"

namespace ttb
{

/// This is a very simple nearest neighbor IC filter
/// For convenience it uses a LMB Distribution to handle the tracks
/// It may uses measurements multiple times to update separate tracks, i.e., is no real filter ....
template <class UDT>
class NN_Tracker
{
public:
  explicit NN_Tracker(TTBManager<UDT>* manager);
  void cycle(std::vector<DataSourceContainer<UDT>> measContainerList);
  [[nodiscard]] std::vector<BernoulliState<UDT>> get_estimate() const;
  [[nodiscard]] std::vector<StateId> all_tracks() const;
  void reset();
  [[nodiscard]] TTBManager<UDT>* manager() const;
  [[nodiscard]] Time time() const;
  [[nodiscard]] FILTER_TYPE type() const;
  TTBManager<UDT>* _manager;
  /// The current time of the Filter
  Time _time{ 0s };
  std::vector<State<UDT>> _tracks;
};

static_assert(CPTTracker<NN_Tracker<double>, double>);

}  // namespace ttb