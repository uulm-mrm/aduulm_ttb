#pragma once

#include "tracking_lib/TTBTypes/Concepts.h"

namespace ttb
{

/// doing nothing at all
template <class UDT>
class Nothing_Tracker
{
public:
  explicit Nothing_Tracker(TTBManager<UDT>* manager);
  void cycle(std::vector<DataSourceContainer<UDT>> measContainerList);
  [[nodiscard]] std::vector<BernoulliState<UDT>> get_estimate() const;
  [[nodiscard]] std::vector<StateId> all_tracks() const;
  void reset();
  [[nodiscard]] Time time() const;
  TTBManager<UDT>* _manager;
  Time _time{ 0s };
};

static_assert(CPTTracker<Nothing_Tracker<double>, double>);
}  // namespace ttb