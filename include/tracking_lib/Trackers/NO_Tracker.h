#pragma once

#include "tracking_lib/TTBTypes/Concepts.h"

namespace ttb
{

/// This filter just forwards the received measurements
/// It performs NO filter/tracking functionality and can be used as a baseline to evaluate the effect of the filter and
/// evaluate the runtime overhead of a specific filter without common boilerplate code
template <class UDT>
class NO_Tracker final
{
public:
  explicit NO_Tracker(TTBManager<UDT>* manager);
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
  std::map<SourceId, std::vector<State<UDT>>> _tracks;
};

static_assert(CPTTracker<NO_Tracker<double>, double>);

}  // namespace ttb