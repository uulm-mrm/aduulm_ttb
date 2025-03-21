#pragma once

#include "tracking_lib/Trackers/BaseTracker.h"
// #####################################################################################################################
#include "tracking_lib/TTBTypes/Params.h"
#include "tracking_lib/States/State.h"

namespace ttb
{

/// This is a very simple Tracker that uses only the ObjectIds of a measurement for the data association
/// This can, e.g., be used to track some Aruco marker with their Id
class Id_Tracker final : public BaseTracker
{
public:
  explicit Id_Tracker(TTBManager* manager);
  void cycle(Time time, std::vector<MeasurementContainer>&& measContainerList) override;
  [[nodiscard]] std::vector<State> getEstimate() const override;
  void reset() override;
  [[nodiscard]] TTBManager* manager() const override;
  [[nodiscard]] Time time() const override;
  [[nodiscard]] FILTER_TYPE type() const override;
  TTBManager* _manager;
  /// The current time of the Filter
  Time _time{ 0s };
  std::unordered_map<ObjectId, State> _tracks;
};

}  // namespace ttb