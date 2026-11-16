#pragma once

#include "tracking_lib/TTBTypes/Concepts.h"

namespace ttb
{

/// Very simple transparent track to track filter.
/// It merges the tracks from all data sources, i.e., performs a OR merging

template<class UDT>
class TransTTTFilter
{
public:
  explicit TransTTTFilter(TTBManager<UDT>* manager);

  void cycle(std::vector<DataSourceContainer<UDT>> trackContainers);

  [[nodiscard]] std::vector<BernoulliState<UDT>> get_estimate() const;

  [[nodiscard]] std::vector<StateId> all_tracks() const;

  [[nodiscard]] TTBManager<UDT>* manager() const;

  [[nodiscard]] Time time() const;

  [[nodiscard]] TTT_FILTER_TYPE type() const;

  void reset();

  TTBManager<UDT>* _manager;
  Time _time = Time{ 0s };
  std::vector<BernoulliState<UDT>> _tracks;
  std::map<SourceId, Label> _cam_vam_source_to_label;  /// always use the same label for tracks associated with a cam or
                                                       /// vam
};

static_assert(CPTTracker<TransTTTFilter<double>, double>);

}  // namespace ttb