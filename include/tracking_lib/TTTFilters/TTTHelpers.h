#pragma once

#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Graph/Graph.h"

namespace ttb::tttfusion
{

template <class UDT>
using Metric = std::function<UDT(BernoulliState<UDT> const&, BernoulliState<UDT> const&)>;

/// build the graph with the given metric and maximal distance
template <class UDT>
[[nodiscard]] graph::DiGraph<StateId, std::size_t, double>
graph(std::vector<State<UDT>> const& first_tracks,
      std::vector<State<UDT>> const& second_tracks,
      std::vector<std::pair<Metric<UDT>, UDT>> const& metric_max_dist);

template <class UDT>
struct AssignmentSol
{
  std::vector<std::tuple<BernoulliState<UDT>, BernoulliState<UDT>>> assigned;
  std::vector<BernoulliState<UDT>> non_assigned_first;
  std::vector<BernoulliState<UDT>> non_assigned_second;
};
/// Compute an assigment between the two sets of States which minimizes the total distances between the states given by
/// the sum of all metrics
template <class UDT>
[[nodiscard]] AssignmentSol<UDT> compute_assignment(std::vector<BernoulliState<UDT>> const& first,
                                                    std::vector<BernoulliState<UDT>> const& second,
                                                    std::vector<std::pair<Metric<UDT>, UDT>> const& metric_max_dist,
                                                    bool and_metric  // true: all metrics must be met to merge, false:
                                                                     // one metric must be met to merge;
);

/// Compute the cross-covariance intersection of the two States
template <class UDT>
[[nodiscard]] std::optional<BernoulliState<UDT>> cross_covariance_intersection(BernoulliState<UDT> const& first,
                                                                               BernoulliState<UDT> const& second,
                                                                               TTBManager<UDT>* manager);

}  // namespace ttb::tttfusion
