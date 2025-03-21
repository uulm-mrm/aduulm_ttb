#pragma once

#include "tracking_lib/Trackers/BaseTracker.h"
// #####################################################################################################################
#include "tracking_lib/TTBTypes/Params.h"

namespace ttb
{

/// This filter just forwards the received measurements
/// It performs NO filter/tracking functionality and can be used as a baseline to evaluate the effect of the filter and
/// evaluate the runtime overhead of a specific filter without common boilerplate code
class NO_Tracker final : public BaseTracker
{
public:
  NO_Tracker(TTBManager* manager);
  void cycle(Time time, std::vector<MeasurementContainer>&& measContainerList) override;
  [[nodiscard]] std::vector<State> getEstimate() const override;
  void reset() override;
  [[nodiscard]] TTBManager* manager() const override;
  [[nodiscard]] Time time() const override;
  [[nodiscard]] FILTER_TYPE type() const override;
  TTBManager* _manager;
  /// The current time of the Filter
  Time _time{ 0s };
  std::map<MeasModelId, std::vector<State>> _tracks;
};

}  // namespace ttb