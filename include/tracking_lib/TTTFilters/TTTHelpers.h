#pragma once

#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include <tracking_lib/Graph/Graph.h>

namespace ttb::tttfusion
{
/// @param _id id of sensor
/// @param _trackMapSensor Map of tracks of this sensors
/// @param _pD detection probability of this sensor
/// @param _sensorInfo contains for example the FOV
struct TracksSensor
{
  MeasModelId _id;
  std::map<StateId, State> _trackMapSensor;
  Probability _pD;
  SensorInformation _sensorInfo{};
};

struct Cluster
{
  std::vector<StateId> _cluster;
};

struct Associations
{
  std::map<StateId, std::size_t> _trackID2sensor;
  std::vector<Cluster> _clusters;
};

[[nodiscard]] std::size_t calculateHash(Vector const& vec);

[[nodiscard]] double gaussian_likelihood(const Vector& mean, const Matrix& cov, const Vector& x);

[[nodiscard]] double cluster_lik(const std::vector<StateId>& cluster,
                                 const std::map<StateId, Vector>& id2tracks,
                                 const std::map<StateId, std::size_t>& trackID2sensor,
                                 std::map<std::size_t, double> sensor2detectProb);

[[nodiscard]] std::vector<Cluster>
get_best_solution(const std::vector<TracksSensor>&& tracksVec,
                  const std::map<std::size_t, std::map<std::size_t, std::vector<StateId>>>&& samples,
                  std::map<StateId, std::size_t>& trackID2sensor,
                  std::map<std::size_t, double>&& sensor2detectProb,
                  const StateModelId sm_id);

/// stochastic t2t association published in
/// L. M. Wolf, S. Steuernagel, K. Thormann and M. Baum, "Track-to-track Association based on Stochastic Optimization,"
/// 2023 26th International Conference on Information Fusion (FUSION), Charleston, SC, USA, 2023, pp. 1-7,
/// doi: 10.23919/FUSION52260.2023.10224113. https://ieeexplore.ieee.org/document/10224113
[[nodiscard]] Associations t2ta_stochastic_optimization(std::vector<TracksSensor>&& tracksVec,
                                                        const std::size_t num_samples,
                                                        const StateModelId sm_id);

using Metric = std::function<double(State const&, State const&)>;

/// build the graph with the given metric and maximal distance
[[nodiscard]] graph::Graph<StateId, std::size_t, double>
graph(std::vector<State> const& first_tracks,
      std::vector<State> const& second_tracks,
      std::vector<std::pair<Metric, double>> const& metric_max_dist);

struct AssignmentSol
{
  std::vector<std::tuple<State, State>> assigned;
  std::vector<State> non_assigned_first;
  std::vector<State> non_assigned_second;
};
/// Compute an assigment between the two sets of States which minimizes the total distances between the states given by
/// the sum of all metrics
[[nodiscard]] AssignmentSol compute_assignment(std::vector<State> const& first,
                                               std::vector<State> const& second,
                                               std::vector<std::pair<Metric, double>> const& metric_max_dist);

/// Compute the cross-covariance intersection of the two States
[[nodiscard]] std::optional<State> cross_covariance_intersection(State const& first,
                                                                 State const& second,
                                                                 TTBManager* manager);

}  // namespace ttb::tttfusion
