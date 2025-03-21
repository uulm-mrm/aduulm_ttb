#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"

namespace ttb
{

class State;
class MeasurementContainer;
class TTBManager;

/// This is the interface for all our trackers.
class BaseTracker
{
public:
  virtual ~BaseTracker() = default;
  /// Performs one complete cycle of the filter using the given measurements.
  /// If the measContainerList is empty, perform a prediction to given time.
  /// Otherwise, the filter usually stays at the time of the newest measurement container, i.e., does not perform an
  /// additional prediction after the last update.
  virtual void cycle(Time time, std::vector<MeasurementContainer>&& measContainerList) = 0;
  /// return the current estimation of the filter
  [[nodiscard]] virtual std::vector<State> getEstimate() const = 0;
  /// access to the manager
  [[nodiscard]] virtual TTBManager* manager() const = 0;
  /// the current time of the filter, usually the filters are not able to process data out-of-sequence, i.e., with time
  /// previous to this time
  [[nodiscard]] virtual Time time() const = 0;
  /// the type of the filter
  [[nodiscard]] virtual FILTER_TYPE type() const = 0;
  /// reset the filter
  virtual void reset() = 0;
};

}  // namespace ttb