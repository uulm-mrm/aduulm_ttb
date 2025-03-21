#pragma once

#include "tracking_lib/Trackers/BaseTracker.h"
#include "tracking_lib/MultiObjectStateDistributions/PHDDistribution.h"

namespace ttb
{

/// Probability Hypothesis Density Filter, https://ieeexplore.ieee.org/document/1710358
class PHD_Tracker final : public BaseTracker
{
public:
  explicit PHD_Tracker(TTBManager* manager);
  void cycle(Time time, std::vector<MeasurementContainer>&& measContainers) override;
  [[nodiscard]] std::vector<State> getEstimate() const override;
  void reset() override;
  [[nodiscard]] TTBManager* manager() const override;
  [[nodiscard]] Time time() const override;
  [[nodiscard]] FILTER_TYPE type() const override;
  TTBManager* _manager;
  Time _time{ 0s };
  PHDDistribution _distribution;
};

}  // namespace ttb
