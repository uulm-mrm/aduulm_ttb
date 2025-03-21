#pragma once

#include "tracking_lib/Trackers/BaseTracker.h"
// #####################################################################################################################
#include "tracking_lib/TTBTypes/Params.h"

namespace ttb
{

/// This is a very simple nearest neighbor IC filter
/// For convenience it uses a LMB Distribution to handle the tracks
/// It may uses measurements multiple times to update separate tracks, i.e., is no real filter ....
class NN_Tracker final : public BaseTracker
{
public:
  NN_Tracker(TTBManager* manager);
  void cycle(Time time, std::vector<MeasurementContainer>&& measContainerList) override;
  [[nodiscard]] std::vector<State> getEstimate() const override;
  void reset() override;
  [[nodiscard]] TTBManager* manager() const override;
  [[nodiscard]] Time time() const override;
  [[nodiscard]] FILTER_TYPE type() const override;
  TTBManager* _manager;
  /// The current time of the Filter
  Time _time{ 0s };
  std::vector<State> _tracks;
};

}  // namespace ttb