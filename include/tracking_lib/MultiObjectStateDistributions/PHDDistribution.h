#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/States/EgoMotionDistribution.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"

namespace ttb
{

/// Probability Hypothesis Density, https://ieeexplore.ieee.org/document/1710358
class PHDDistribution
{
public:
  explicit PHDDistribution(TTBManager* manager);
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
  /// check if all tracks are valid
  [[nodiscard]] bool isValid() const;
  /// Post-processing of the prediction step.
  void postProcessPrediction();
  /// Post-processing of the update step.
  void postProcessUpdate();
  /// Return an estimate of this distribution.
  [[nodiscard]] std::vector<State> getEstimate() const;
  /// Predict this distribution deltaT time into the future.
  void predict(Duration deltaT, EgoMotionDistribution const& egoDist);
  /// Update this distribution with the given MeasurementContainer.
  /// The update method is selected depending on the parameters.
  void update(MeasurementContainer const& measurementContainer);
  /// add a track to this distribution
  void addTracks(std::vector<State> tracks);
  /// calculate the state Innovation for all Tracks for all Measurements in the MeasurementContainer
  void calcInnovation(MeasurementContainer const& Z);
  /// the probability/weight a measurement contributed to the update of any track in the last update
  std::map<MeasurementId, Probability> const& meas_assignment_prob() const;
  /// sum the weights of all tracks
  [[nodiscard]] double sum_weights() const;
  void merge(double max_merge_distance);
  TTBManager* _manager;
  /// ID of this distribution, you should not set this manually
  MODistributionId _id{ _idGenerator.getID() };
  static IDGenerator<MODistributionId> _idGenerator;
  /// all tracks
  std::vector<State> _tracks;

private:
  /// weight, in the sense of https://ieeexplore.ieee.org/document/1710358, of a state
  static double& weight(State& state);
  static double weight(State const& state);
  std::map<MeasurementId, Probability> mutable _meas_assignment_prob;
};

}  // namespace ttb
