#pragma once

#include "tracking_lib/TTBTypes/Concepts.h"

namespace ttb
{

/// This is a very simple Tracker that uses only the ObjectIds of a measurement for the data association
/// This can, e.g., be used to track some Aruco marker with their Id
template <class UDT>
class Id_Tracker
{
public:
  explicit Id_Tracker(TTBManager<UDT>* manager);
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
  std::unordered_map<ObjectId, State<UDT>> _tracks;
};

static_assert(CPTTracker<Id_Tracker<double>, double>);

}  // namespace ttb