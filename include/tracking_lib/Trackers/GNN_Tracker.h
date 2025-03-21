#pragma once

#include "tracking_lib/Trackers/BaseTracker.h"
// #####################################################################################################################
#include "tracking_lib/TTBTypes/Params.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/States/State.h"

namespace ttb
{

/// This is a simple global nearest neighbor (GNN) iterator corrector (sequential updating) tracker.
class GNN_Tracker final : public BaseTracker
{
public:
  explicit GNN_Tracker(TTBManager* manager);
  void cycle(Time time, std::vector<MeasurementContainer>&& measContainerList) override;
  [[nodiscard]] std::vector<State> getEstimate() const override;
  void reset() override;
  [[nodiscard]] TTBManager* manager() const override;
  [[nodiscard]] Time time() const override;
  /// check if all tracks are valid
  [[nodiscard]] bool hasValidTracks() const;
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
  [[nodiscard]] FILTER_TYPE type() const override;

  /// builds the GNN-specific cost matrix for the association (size: # measurements x (# measurements + # tracks))
  [[nodiscard]] Matrix buildCostMatrix(std::vector<State> const& tracks,
                                       MeasurementContainer const& measurements,
                                       const MeasModelId& measContainerID);
  /// add new birth tracks (static or dynamic depending on the birth model type)
  void addBirthTracks(MeasurementContainer const& measurementContainer, std::map<MeasurementId, double> const& rzMap);
  /// performs track history logic for track maintenance
  void performHistoryTrackLogic(std::size_t num_sensors);
  /// performs the prediction step
  void performPrediction(Duration deltaT, EgoMotionDistribution const& egoDist);
  /// performs the innovation step
  void performInnovation(MeasurementContainer const& Z);
  /// performs the GNN-specific update
  std::map<MeasurementId, double> performUpdate(MeasurementContainer const& measurementContainer,
                                                std::size_t num_sensors);
  /// performs a GNN-specific post processing after the update step - mainly track maintenance tasks
  void postProcessUpdate(std::size_t num_sensors);

  TTBManager* _manager;
  /// current time of the tracker
  Time _time{ 0s };
  /// all GNN tracks
  std::vector<State> _gnn_tracks;
  double birth_track_density = 0;
};

}  // namespace ttb
