#pragma once

#include "tracking_lib/TTBTypes/Concepts.h"

namespace ttb
{

template <class UDT>
class NoTTTFilter
{
public:
  explicit NoTTTFilter(TTBManager<UDT>* manager);

  void cycle(std::vector<DataSourceContainer<UDT>> trackContainers);

  [[nodiscard]] std::vector<BernoulliState<UDT>> get_estimate() const;

  [[nodiscard]] std::vector<StateId> all_tracks() const;

  [[nodiscard]] TTBManager<UDT>* manager() const;

  [[nodiscard]] Time time() const;

  [[nodiscard]] TTT_FILTER_TYPE type() const;

  void reset();

  TTBManager<UDT>* _manager;
  std::vector<State<UDT>> _tracks;
};

static_assert(CPTTracker<NoTTTFilter<double>, double>);

}  // namespace ttb