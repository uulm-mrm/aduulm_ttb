#pragma once

#include "tracking_lib/TTBTypes/Concepts.h"

namespace ttb
{

/// This is a simple global nearest neighbor (GNN) iterator corrector (sequential updating) tracker.
template <class UDT>
class GNN_Tracker
{
public:
  explicit GNN_Tracker(TTBManager<UDT>* manager);
  void cycle(std::vector<DataSourceContainer<UDT>> measContainerList);
  [[nodiscard]] std::vector<StateId> all_tracks() const;
  [[nodiscard]] std::vector<BernoulliState<UDT>> get_estimate() const;
  void reset();
  [[nodiscard]] TTBManager<UDT>* manager() const;
  [[nodiscard]] Time time() const;
  /// check if all tracks are valid
  [[nodiscard]] bool hasValidTracks() const;
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
  [[nodiscard]] FILTER_TYPE type() const;

  /// builds the GNN-specific cost matrix for the association (size: # measurements x (# measurements + # tracks))
  [[nodiscard]] Matrix<double> buildCostMatrix(std::vector<State<UDT>> const& tracks,
                                               DataSourceContainer<UDT> const& measurements);
  /// add new birth tracks (static or dynamic depending on the birth model type)
  void addBirthTracks(DataSourceContainer<UDT> const& measurementContainer,
                      std::map<MeasurementId, UDT> const& rzMap);
  /// performs track history logic for track maintenance
  void performHistoryTrackLogic(std::size_t num_sensors);
  /// performs the prediction step
  void performPrediction(Duration deltaT, PredictionInformation const& prediction_info);
  /// performs the innovation step
  void performInnovation(DataSourceContainer<UDT> const& Z);
  /// performs the GNN-specific update
  std::map<MeasurementId, UDT> performUpdate(DataSourceContainer<UDT> const& measurementContainer,
                                                std::size_t num_sensors);
  /// performs a GNN-specific post processing after the update step - mainly track maintenance tasks
  void postProcessUpdate(std::size_t num_sensors);

  TTBManager<UDT>* _manager;
  /// current time of the tracker
  Time _time{ 0s };
  /// all GNN tracks
  std::vector<State<UDT>> _gnn_tracks;
  double birth_track_density = 0;
};

static_assert(CPTTracker<GNN_Tracker<double>, double>);

}  // namespace ttb
