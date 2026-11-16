#include "tracking_lib/TTTFilters/TTTHelpers.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/Misc/MurtyAlgorithm.h"
#include "tracking_lib/Transformations/Transformation.h"

namespace ttb::tttfusion
{

template std::optional<BernoulliState<double>> cross_covariance_intersection(BernoulliState<double> const& first,
                                                                             BernoulliState<double> const& second,
                                                                             TTBManager<double>* manager);
template std::optional<BernoulliState<AD>> cross_covariance_intersection(BernoulliState<AD> const& first,
                                                                         BernoulliState<AD> const& second,
                                                                         TTBManager<AD>* manager);
template <class UDT>
std::optional<BernoulliState<UDT>> cross_covariance_intersection(BernoulliState<UDT> const& first,
                                                                 BernoulliState<UDT> const& second,
                                                                 TTBManager<UDT>* manager)
{
  ZoneScopedN("cross_covariance_intersection");
  // find common state model
  StateModel<UDT> const& target_state_model =
      manager->getStateModel(manager->params()->state.estimation.output_state_model);
  auto const& [first_id, first_dist] = first.state.getEstimate();
  auto first_trans = transformation::transform(first_dist.mean(),
                                               first_dist.covariance(),
                                               manager->state_comps(first_id),
                                               manager->state_comps(target_state_model.id()));
  if (not first_trans.has_value())
  {
    LOG_WARN("Can not transform Track " + first.state.toString() + " to desired output state model");
    return {};
  }
  auto const& [second_id, second_dist] = second.state.getEstimate();
  auto second_trans = transformation::transform(second_dist.mean(),
                                                second_dist.covariance(),
                                                manager->state_comps(second_id),
                                                manager->state_comps(target_state_model.id()));
  if (not second_trans.has_value())
  {
    LOG_WARN("Can not transform Track " + second.state.toString() + " to desired output state model");
    return {};
  }
  auto const make_vel_abs_positive = [&](Vector<UDT>& mean) {
    if (auto const vel_abs_ind = manager->state_comps(target_state_model.id()).indexOf(COMPONENT::VEL_ABS);
        vel_abs_ind.has_value())
    {
      if (mean(vel_abs_ind.value()) < 0)
      {
        mean(vel_abs_ind.value()) *= -1;
        if (auto const rot_ind = manager->state_comps(target_state_model.id()).indexOf(COMPONENT::ROT_Z);
            rot_ind.has_value())
        {
          mean(rot_ind.value()) += std::numbers::pi;
          angles::normalizeAngle(mean(rot_ind.value()));
        }
      }
    }
  };
  // assume positive velocity
  make_vel_abs_positive(first_trans.value().mean);
  make_vel_abs_positive(second_trans.value().mean);

  Components const euclidean_comps =
      manager->state_comps(target_state_model.id()).diff(Components({ COMPONENT::ROT_Z }));
  Indices const euclidean_inds = manager->state_comps(target_state_model.id()).indexOf(euclidean_comps).value();
  // perform cross covariance intersection
  // choosing the weights as in
  //  W. Niehsen, "Information fusion based on fast covariance intersection filtering," Proceedings of the Fifth
  //  International Conference on Information Fusion. FUSION 2002. (IEEE Cat.No.02EX5997), Annapolis, MD, USA, 2002, pp.
  //  901-904 vol.2, https://doi.org/10.1109/ICIF.2002.1020907
  UDT const w1 = second_trans.value().cov.diagonal().sum() /
                 (first_trans.value().cov.diagonal().sum() + second_trans.value().cov.diagonal().sum());
  UDT const w2 = first_trans.value().cov.diagonal().sum() /
                 (first_trans.value().cov.diagonal().sum() + second_trans.value().cov.diagonal().sum());
  assert(std::abs(1.0 - (w1 + w2)) < 1e-7);
  Matrix<UDT> const cov1_inv = first_trans.value().cov(euclidean_inds, euclidean_inds).inverse();
  Matrix<UDT> const cov2_inv = second_trans.value().cov(euclidean_inds, euclidean_inds).inverse();
  Matrix<UDT> euclidian_cov = (w1 * cov1_inv + w2 * cov2_inv).inverse();
  Vector<UDT> euclidian_mean = euclidian_cov * (w1 * cov1_inv * first_trans.value().mean(euclidean_inds) +
                                                w2 * cov2_inv * second_trans.value().mean(euclidean_inds));
  Vector<UDT> mean = Vector<UDT>::Zero(manager->state_comps(target_state_model.id())._comps.size());
  Matrix<UDT> cov = Matrix<UDT>::Identity(manager->state_comps(target_state_model.id())._comps.size(),
                                          manager->state_comps(target_state_model.id())._comps.size());
  mean(euclidean_inds) = euclidian_mean;
  cov(euclidean_inds, euclidean_inds) = euclidian_cov;
  if (auto const rot_ind = manager->state_comps(target_state_model.id()).indexOf(COMPONENT::ROT_Z); rot_ind.has_value())
  {
    auto const ind = rot_ind.value();
    Vector<UDT> weights{ { second_trans.value().cov(ind, ind), first_trans.value().cov(ind, ind) } };
    weights /= weights.sum();
    std::vector<std::pair<UDT, UDT>> weight_angles{ { weights(0), first_trans.value().mean(ind) },
                                                    { weights(1), second_trans.value().mean(ind) } };
    mean(ind) = angles::weighted_mean(weight_angles);
  }

  State<UDT> fused_state(
      first.state._manager,
      std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>>{
          { manager->params()->state.estimation.output_state_model,
            std::vector{ std::pair{ UDT{ 1.0 }, GaussianDistribution<UDT>(std::move(mean), std::move(cov)) } } } });
  UDT const fused_state_existenceProbability =
      std::pow(first.existence_probability, w1) * std::pow(second.existence_probability, w2) /
      (std::pow(first.existence_probability, w1) * std::pow(second.existence_probability, w2) +
       std::pow(1 - first.existence_probability, w1) * std::pow(1 - second.existence_probability, w2));
  fused_state._time = first.state._time;
  fused_state._label = first.state._label;
  fused_state._meta_data._durationSinceLastAssociation = 0s;
  fused_state._meta_data._numUpdates += 1;
  fused_state._classification = first.state._classification;
  fused_state._misc = first.state._misc;
  fused_state._misc.insert(second.state._misc.begin(), second.state._misc.end());

  fused_state._classification.merge(second.state._classification);

  assert([&] {  // NOLINT
    if (not fused_state.isValid())
    {
      LOG_FATAL("fused state with cross covariance intersection is not valid.");
      LOG_FATAL(fused_state.toString());
      LOG_FATAL("State 1: " << first.state.toString());
      LOG_FATAL("State 2: " << second.state.toString());
      return false;
    }
    return true;
  }());
  return BernoulliState{ fused_state, fused_state_existenceProbability };
}

template <class UDT>
[[nodiscard]] graph::DiGraph<StateId, std::size_t, double>
graph(std::vector<BernoulliState<UDT>> const& first_tracks,
      std::vector<BernoulliState<UDT>> const& second_tracks,
      std::vector<std::pair<Metric<UDT>, UDT>> const& metric_max_dist,
      bool and_metric)
{
  std::vector<StateId> nodes;
  std::vector<graph::Edge<std::size_t, StateId, double>> edges;
  for (BernoulliState<UDT> const& first : first_tracks)
  {
    nodes.push_back(first.state._id);
  }
  for (BernoulliState<UDT> const& second : second_tracks)
  {
    nodes.push_back(second.state._id);
  }
  std::size_t edge_ctr = 0;
  for (BernoulliState<UDT> const& first : first_tracks)
  {
    for (BernoulliState<UDT> const& second : second_tracks)
    {
      bool const ok = [&] {
        if (and_metric)
        {
          return std::ranges::all_of(metric_max_dist, [&](std::pair<Metric<UDT>, UDT> const& m_d) {
            UDT const dist = m_d.first(first, second);
            return dist < m_d.second;
          });
        }
        return std::ranges::any_of(metric_max_dist, [&](std::pair<Metric<UDT>, UDT> const& m_d) {
          UDT const dist = m_d.first(first, second);
          return dist < m_d.second;
        });
      }();
      if (ok)
      {
        edges.emplace_back(edge_ctr, first.state._id, second.state._id, 1);
        edge_ctr++;
      }
    }
  }
  return { std::move(nodes), std::move(edges) };
}

template AssignmentSol<double> compute_assignment(std::vector<BernoulliState<double>> const& first,
                                                  std::vector<BernoulliState<double>> const& second,
                                                  std::vector<std::pair<Metric<double>, double>> const& metric_max_dist,
                                                  bool and_metric);
template AssignmentSol<AD> compute_assignment(std::vector<BernoulliState<AD>> const& first,
                                              std::vector<BernoulliState<AD>> const& second,
                                              std::vector<std::pair<Metric<AD>, AD>> const& metric_max_dist,
                                              bool and_metric);
template <class UDT>
[[nodiscard]] AssignmentSol<UDT> compute_assignment(std::vector<BernoulliState<UDT>> const& first,
                                                    std::vector<BernoulliState<UDT>> const& second,
                                                    std::vector<std::pair<Metric<UDT>, UDT>> const& metric_max_dist,
                                                    bool and_metric)
{
  ZoneScopedN("compute_assignment");
  std::vector<std::vector<StateId>> comps = graph(first, second, metric_max_dist, and_metric).components();
  AssignmentSol<UDT> solution;
  for (std::vector<StateId> const& comp : comps)
  {
    std::vector<std::size_t> first_tracks;
    std::vector<std::size_t> second_tracks;
    for (StateId id : comp)
    {
      if (auto it =
              std::ranges::find_if(first, [id](BernoulliState<UDT> const& track) { return track.state._id == id; });
          it != first.end())  // node of component belong to first
      {
        first_tracks.push_back(std::distance(first.begin(), it));
      }
      else  // node belong to second
      {
        second_tracks.push_back(std::distance(
            second.begin(),
            std::ranges::find_if(second, [id](BernoulliState<UDT> const& track) { return track.state._id == id; })));
      }
    }
    if (first_tracks.empty())  // we do not have first tracks in this group -> second are not assigned
    {
      for (std::size_t second_ind : second_tracks)
      {
        solution.non_assigned_second.push_back(second.at(second_ind));
      }
      continue;
    }
    if (second_tracks.empty())  // first are not assigned
    {
      for (std::size_t first_ind : first_tracks)
      {
        solution.non_assigned_first.push_back(first.at(first_ind));
      }
      continue;
    }
    Matrix<double> costMatrix = Matrix<double>::Zero(first_tracks.size(), second_tracks.size());
    for (std::size_t i = 0; i < first_tracks.size(); ++i)
    {
      for (std::size_t j = 0; j < second_tracks.size(); ++j)
      {
        for (auto const& [metric, _] : metric_max_dist)
        {
          costMatrix(i, j) += to_val(metric(first.at(first_tracks.at(i)), second.at(second_tracks.at(j))));
        }
      }
    }
    bool transpose = false;
    if (costMatrix.rows() > costMatrix.cols())
    {
      transpose = true;
      costMatrix.transposeInPlace();
    }
    auto [assignment, cost] = murty::getAssignments(costMatrix, 1);
    std::vector<std::size_t> used_first;
    std::vector<std::size_t> used_second;
    for (std::size_t i = 0; i < static_cast<std::size_t>(assignment.rows()); ++i)
    {
      auto const& [first_ind, second_ind] = [&]() -> std::pair<std::size_t, std::size_t> {
        if (not transpose)
        {
          return { first_tracks.at(i), second_tracks.at(assignment(i)) };
        }
        return { first_tracks.at(assignment(i)), second_tracks.at(i) };
      }();
      used_first.push_back(first_ind);
      used_second.push_back(second_ind);
      solution.assigned.emplace_back(first.at(first_ind), second.at(second_ind));
    }
    for (std::size_t first_ind : first_tracks)
    {
      if (std::ranges::find(used_first, first_ind) == used_first.end())
      {
        solution.non_assigned_first.push_back(first.at(first_ind));
      }
    }
    for (std::size_t second_ind : second_tracks)
    {
      if (std::ranges::find(used_second, second_ind) == used_second.end())
      {
        solution.non_assigned_second.push_back(second.at(second_ind));
      }
    }
  }
  assert(2 * solution.assigned.size() + solution.non_assigned_first.size() + solution.non_assigned_second.size() ==
         first.size() + second.size());
  return solution;
}

}  // namespace ttb::tttfusion
