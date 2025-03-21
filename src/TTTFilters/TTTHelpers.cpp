#include "tracking_lib/TTTFilters/TTTHelpers.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/Misc/MurtyAlgorithm.h"

#include <tracy/tracy/Tracy.hpp>

auto constexpr tracy_color = tracy::Color::Tomato;

namespace ttb::tttfusion
{
std::size_t calculateHash(Vector const& value_vec)
{
  std::size_t hash = value_vec.size();
  for (const auto value : value_vec)
  {
    auto temp = static_cast<uint32_t>(value);
    temp = ((temp >> 16) ^ temp) * 0x45d9f3b;
    temp = ((temp >> 16) ^ temp) * 0x45d9f3b;
    temp = (temp >> 16) ^ temp;
    hash ^= temp + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  return hash;
}

double gaussian_likelihood(const Vector& mean, const Matrix& cov, const Vector& x)
{
  Matrix cov_inv = cov.inverse();
  Vector diff = x.head(2) - mean;
  double distance = diff.transpose() * cov_inv * diff;
  return exp((-0.5) * distance) / std::sqrt(2 * M_PI * (cov).determinant());
}

double cluster_lik(const Cluster& cluster,
                   const std::map<StateId, Vector>& id2tracks,
                   const std::map<StateId, std::size_t>& trackID2sensor,
                   std::map<std::size_t, double> sensor2detectProb)
{
  if (cluster._cluster.empty())
  {
    return 1.0;
  }
  // spatial likelihood
  std::size_t cluster_len = cluster._cluster.size();
  Vector mean = Vector::Zero(2);
  double prob = 1.0;
  for (auto const& member : cluster._cluster)
  {
    std::size_t sensor = trackID2sensor.at(member);
    if (!sensor2detectProb.contains(sensor))
    {
      LOG_FATAL("sensor is multiple times in cluster!! Should not happen -> BUG!");
      throw std::runtime_error("sensor is multiple times in cluster!! Should not happen -> BUG!");
    }
    prob *= sensor2detectProb.at(sensor);
    sensor2detectProb.erase(sensor);
    mean += id2tracks.at(member).head(2);
  }
  mean /= cluster._cluster.size();
  Matrix cov_cluster(2, 2);
  cov_cluster.setIdentity();  // todo(hermann): Set cov as param!
  Matrix cov = (1 + 1 / cluster_len) * cov_cluster;

  double likelihood = 1.0;
  for (auto const& member : cluster._cluster)
  {
    Vector x = id2tracks.at(member);
    likelihood *= gaussian_likelihood(mean, cov, x);
  }

  for (auto const& [_, pD] : sensor2detectProb)
  {
    prob *= (1 - pD);
  }
  // spatial likelihood and detection prob
  return prob * likelihood;
}

std::vector<Cluster> get_best_solution(const std::vector<TracksSensor>&& tracksVec,
                                       const std::map<std::size_t, std::map<std::size_t, Cluster>>&& samples,
                                       std::map<StateId, std::size_t>& trackID2sensor,
                                       std::map<std::size_t, double>&& sensor2detectProb,
                                       const StateModelId sm_id)
{
  std::vector<Cluster> res;
  double highest_weight = 0.0;
  for (auto const& [_, sample] : samples)  // todo( hermann): parallelize
  {
    double prod_weight = 1.0;
    std::vector<Cluster> final_selections;
    for (auto const& [cluster_id, cluster] : sample)
    {
      std::map<StateId, Vector> id2tracks;
      for (auto const& member : cluster._cluster)
      {
        Vector mean = tracksVec.at(trackID2sensor.at(member))._trackMapSensor.at(member)._state_dist.at(sm_id)->mean();
        id2tracks.emplace(member, mean);
      }
      prod_weight *= cluster_lik(cluster, id2tracks, trackID2sensor, sensor2detectProb);
      final_selections.push_back({ ._cluster = std::move(cluster._cluster) });
    }
    if (highest_weight < prod_weight)
    {
      highest_weight = prod_weight;
      res = std::move(final_selections);
    }
  }
  return res;
}

Associations t2ta_stochastic_optimization(std::vector<TracksSensor>&& tracksVec,
                                          const std::size_t num_samples,
                                          const StateModelId sm_id)
{
  std::vector<StateId> trackIDs_all;
  std::map<StateId, std::size_t> trackID2t;
  std::size_t num_tracks_tot = 0;
  for (auto const& tracks : tracksVec)
  {
    num_tracks_tot += tracks._trackMapSensor.size();
  }
  std::size_t next_cluster_key = 0;

  std::map<StateId, std::size_t> trackID2sensor;
  std::map<std::size_t, double> sensor2detectProb;
  std::map<std::size_t, Cluster> clusters;
  Vector curr_sample(num_tracks_tot);
  std::size_t ctr = 0;
  std::size_t ctr2 = 0;
  for (auto& tracks : tracksVec)
  {
    if (tracks._pD > 0.97)
    {
      tracks._pD = 0.97;
    }
    sensor2detectProb.emplace(ctr, tracks._pD);
    for (auto const& track : tracks._trackMapSensor)
    {
      //
      trackID2sensor.emplace(track.first, ctr);
      Cluster cluster;
      trackIDs_all.push_back(track.first);
      trackID2t.emplace(ctr2, track.first);
      cluster._cluster.push_back(track.first);
      clusters.emplace(next_cluster_key, cluster);
      next_cluster_key++;
      curr_sample(ctr2) = ctr2;
      ctr2++;
    }
    ctr++;
  }
  LOG_DEB("Total # of Tracks: " << num_tracks_tot);
  std::map<std::size_t, std::map<std::size_t, Cluster>> samples;
  samples.emplace(calculateHash(curr_sample), clusters);

  std::vector<std::size_t> del_cluster_list;
  for (std::size_t n = 0; n < num_samples; n++)
  {
    for (std::size_t t = 0; t < num_tracks_tot; t++)
    {
      StateId track_id = trackIDs_all.at(t);
      std::size_t num_clusters = clusters.size();
      Cluster current_cluster = clusters[curr_sample(t)];
      std::map<StateId, Vector> id2tracks;
      std::map<StateId, Vector> id2tracks_singleton;
      bool t_occurs = false;
      for (auto const& member : current_cluster._cluster)
      {
        Vector mean = tracksVec.at(trackID2sensor.at(member))._trackMapSensor.at(member)._state_dist.at(sm_id)->mean();
        id2tracks.emplace(member, mean);
        if (member == track_id)
        {
          t_occurs = true;
          id2tracks_singleton.emplace(member, mean);
        }
      }
      if (!t_occurs)
      {
        LOG_ERR("track is not contained in cluster. Bug!!!");
      }
      double weight_curr_cluster = cluster_lik(current_cluster, id2tracks, trackID2sensor, sensor2detectProb);
      // remove track t from current_cluster
      auto it = std::find(current_cluster._cluster.begin(), current_cluster._cluster.end(), track_id);
      if (it != current_cluster._cluster.end())
      {
        current_cluster._cluster.erase(it);
      }
      else
      {
        LOG_ERR("Track is not in cluster -> BUG!!!");
      }
      if (id2tracks.contains(track_id))
      {
        id2tracks.erase(track_id);
      }
      else
      {
        LOG_ERR("Can this happen and if yes, what is to do?");
      }
      double weight_curr_cluster_minus_t = cluster_lik(current_cluster, id2tracks, trackID2sensor, sensor2detectProb);
      Array sample_lik = Array::Zero(2 * num_clusters + 2, 1);
      // remain in currrent cluster
      sample_lik(0) = 1.0;
      // calculate likelihood of creating a singleton (eq. (7))
      Cluster singleton_cluster;
      singleton_cluster._cluster.push_back(track_id);
      sample_lik(1) = cluster_lik(singleton_cluster, id2tracks_singleton, trackID2sensor, sensor2detectProb) *
                      weight_curr_cluster_minus_t / weight_curr_cluster;
      // calculate probability of moving track t to Cluster Cc (eq. (8))
      std::size_t sensor_t = trackID2sensor.at(track_id);
      std::size_t i = 0;
      std::map<std::size_t, std::size_t> i2cluster_id;
      for (auto& [c_id, cluster] : clusters)
      {
        i2cluster_id.emplace(i, c_id);
        if (c_id == curr_sample[t])
        {
          // current cluster
          i++;
          continue;
        }
        if (clusters[c_id]._cluster.empty())
        {
          i++;
          del_cluster_list.push_back(c_id);
          continue;
        }
        // Check if track from same sensor is in the cluster
        bool sensor_already_inside = false;
        for (auto const& member : cluster._cluster)
        {
          if (trackID2sensor.at(member) == sensor_t)
          {
            sensor_already_inside = true;
            break;
          }
        }
        if (sensor_already_inside)
        {
          i++;
          continue;
        }
        std::map<StateId, Vector> id2tracks_cluster;
        for (auto const& member : cluster._cluster)
        {
          Vector mean =
              tracksVec.at(trackID2sensor.at(member))._trackMapSensor.at(member)._state_dist.at(sm_id)->mean();
          id2tracks_cluster.emplace(member, mean);
        }
        double weight_cluster_c = cluster_lik(cluster, id2tracks_cluster, trackID2sensor, sensor2detectProb);
        // add current track to cluster
        Cluster current_cluster_moved = cluster;
        Vector mean_track =
            tracksVec.at(trackID2sensor.at(track_id))._trackMapSensor.at(track_id)._state_dist.at(sm_id)->mean();
        id2tracks_cluster.emplace(track_id, mean_track);
        current_cluster_moved._cluster.push_back(track_id);
        double weight_cluster_c_with_t =
            cluster_lik(current_cluster_moved, id2tracks_cluster, trackID2sensor, sensor2detectProb);
        sample_lik(i + 2) =
            weight_cluster_c_with_t * weight_curr_cluster_minus_t / (weight_curr_cluster * weight_cluster_c);

        // calc probability of merging cluster theta_t and C_c (eq. 9)
        // merge clusters if current cluster is >1 and sensors are disjoint
        bool is_disjoint = true;
        if (clusters[curr_sample[t]]._cluster.size() > 1)
        {
          for (auto const& member_current_c : clusters[curr_sample(t)]._cluster)
          {
            std::size_t sensor_current = trackID2sensor.at(member_current_c);
            for (auto const& member_cluster_c : cluster._cluster)
            {
              if (sensor_current == trackID2sensor.at(member_cluster_c))
              {
                is_disjoint = false;
                break;
              }
            }
            if (!is_disjoint)
            {
              break;
            }
          }
        }

        // calc prob
        if (clusters[curr_sample[t]]._cluster.size() > 1 && is_disjoint)
        {
          Cluster combined_clusters = cluster;
          combined_clusters._cluster.insert(combined_clusters._cluster.end(),
                                            clusters[curr_sample(t)]._cluster.begin(),
                                            clusters[curr_sample(t)]._cluster.end());
          id2tracks_cluster.insert(id2tracks.begin(), id2tracks.end());
          sample_lik(num_clusters + i + 2) =
              cluster_lik(combined_clusters, id2tracks_cluster, trackID2sensor, sensor2detectProb);
        }
        i++;
      }
      // normalize
      sample_lik /= sample_lik.sum();
      // sample
      Array random_samples = Array::Random(sample_lik.size(), 1);
      random_samples += 1;
      random_samples /= 2;
      Array scaled_lik = random_samples * sample_lik;
      Vector scaled_lik_vec = scaled_lik.matrix();
      std::size_t assign;
      scaled_lik_vec.maxCoeff(&assign);

      if (assign == 1 && clusters[curr_sample(t)]._cluster.size() > 1)
      {
        // singleton and current cluster>1
        auto tIt =
            std::find(clusters[curr_sample(t)]._cluster.begin(), clusters[curr_sample(t)]._cluster.end(), track_id);
        if (tIt == clusters[curr_sample(t)]._cluster.end())
        {
          LOG_FATAL("track id is not in cluster!?!");
          throw std::runtime_error("track id is not in cluster!?!");
        }
        clusters[curr_sample(t)]._cluster.erase(tIt);
        Cluster new_cluster;
        new_cluster._cluster.push_back(track_id);
        clusters.emplace(next_cluster_key, std::move(new_cluster));
        curr_sample(t) = next_cluster_key;
        next_cluster_key++;
      }
      else if (1 < assign and assign <= num_clusters + 1)
      {
        // move track
        auto it2 = std::find(
            clusters.at(curr_sample(t))._cluster.begin(), clusters.at(curr_sample(t))._cluster.end(), track_id);
        if (it2 != clusters.at(curr_sample(t))._cluster.end())
        {
          // remove track t from old cluster
          clusters.at(curr_sample(t))._cluster.erase(it2);
          if (clusters.at(curr_sample(t))._cluster.empty())
          {
            del_cluster_list.push_back(curr_sample(t));
          }
        }
        else
        {
          LOG_FATAL("Track is not in cluster -> BUG!");
          throw std::runtime_error("Track is not in cluster -> BUG!");
        }
        // add track t to cluster c
        std::size_t cluster_key = i2cluster_id.at(assign - 2);
        clusters.at(cluster_key)._cluster.push_back(track_id);
        curr_sample(t) = cluster_key;
      }
      else if (assign > num_clusters + 1)
      {
        // merge clusters
        std::size_t cluster_key = i2cluster_id.at(assign - 2 - num_clusters);
        std::size_t old_cluster_key = curr_sample(t);
        for (auto const& old_member : clusters.at(old_cluster_key)._cluster)
        {
          curr_sample(trackID2t.at(old_member)) = cluster_key;
          clusters.at(cluster_key)._cluster.push_back(old_member);
        }
        clusters.erase(old_cluster_key);  // delete empty cluster
      }
      for (auto const& del_key : del_cluster_list)
      {
        if (clusters.contains(del_key))
        {
          clusters.erase(del_key);
        }
      }
      // save new sample
      std::size_t hash = calculateHash(curr_sample);
      if (!samples.contains(hash))
      {
        samples.emplace(hash, clusters);
      }
      // todo(hermann): Check if next_cluster_key becomes to big, if yes the cluster ids have to be adapted
    }
  }
  std::vector<Cluster> clusters_final =
      get_best_solution(std::move(tracksVec), std::move(samples), trackID2sensor, std::move(sensor2detectProb), sm_id);
  return { ._trackID2sensor = std::move(trackID2sensor), ._clusters = std::move(clusters_final) };
}

std::optional<State> cross_covariance_intersection(State const& first, State const& second, TTBManager* manager)
{
  ZoneScopedNC("cross_covariance_intersection", tracy_color);
  // find common state model
  BaseStateModel const& target_state_model =
      manager->getStateModel(manager->params().state.estimation.output_state_model);
  auto const& [first_id, first_dist] = first.getEstimate();
  auto first_trans = transformation::transform(first_dist->mean(),
                                               first_dist->covariance(),
                                               manager->getStateModel(first_id).state_comps(),
                                               target_state_model.state_comps());
  if (not first_trans.has_value())
  {
    LOG_WARN("Can not transform Track " + first.toString() + " to desired output state model");
    return {};
  }
  auto const& [second_id, second_dist] = second.getEstimate();
  auto second_trans = transformation::transform(second_dist->mean(),
                                                second_dist->covariance(),
                                                manager->getStateModel(second_id).state_comps(),
                                                target_state_model.state_comps());
  if (not second_trans.has_value())
  {
    LOG_WARN("Can not transform Track " + second.toString() + " to desired output state model");
    return {};
  }
  auto const make_vel_abs_positive = [&](Vector& mean) {
    if (auto const vel_abs_ind = target_state_model.state_comps().indexOf(COMPONENT::VEL_ABS); vel_abs_ind.has_value())
    {
      if (mean(vel_abs_ind.value()) < 0)
      {
        mean(vel_abs_ind.value()) *= -1;
        if (auto const rot_ind = target_state_model.state_comps().indexOf(COMPONENT::ROT_Z); rot_ind.has_value())
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

  Components const euclidean_comps = target_state_model.state_comps().diff(Components({ COMPONENT::ROT_Z }));
  Indices const euclidean_inds = target_state_model.state_comps().indexOf(euclidean_comps._comps).value();
  // perform cross covariance intersection
  // choosing the weights as in
  //  W. Niehsen, "Information fusion based on fast covariance intersection filtering," Proceedings of the Fifth
  //  International Conference on Information Fusion. FUSION 2002. (IEEE Cat.No.02EX5997), Annapolis, MD, USA, 2002, pp.
  //  901-904 vol.2, https://doi.org/10.1109/ICIF.2002.1020907
  double const w1 = second_trans.value().cov.diagonal().sum() /
                    (first_trans.value().cov.diagonal().sum() + second_trans.value().cov.diagonal().sum());
  double const w2 = first_trans.value().cov.diagonal().sum() /
                    (first_trans.value().cov.diagonal().sum() + second_trans.value().cov.diagonal().sum());
  assert(std::abs(1.0 - (w1 + w2)) < 1e-7);
  Matrix const cov1_inv = first_trans.value().cov(euclidean_inds, euclidean_inds).inverse();
  Matrix const cov2_inv = second_trans.value().cov(euclidean_inds, euclidean_inds).inverse();
  Matrix euclidian_cov = (w1 * cov1_inv + w2 * cov2_inv).inverse();
  Vector euclidian_mean = euclidian_cov * (w1 * cov1_inv * first_trans.value().mean(euclidean_inds) +
                                           w2 * cov2_inv * second_trans.value().mean(euclidean_inds));
  Vector mean = Vector::Zero(target_state_model.state_comps()._comps.size());
  Matrix cov =
      Matrix::Identity(target_state_model.state_comps()._comps.size(), target_state_model.state_comps()._comps.size());
  mean(euclidean_inds) = euclidian_mean;
  cov(euclidean_inds, euclidean_inds) = euclidian_cov;
  if (auto const rot_ind = target_state_model.state_comps().indexOf(COMPONENT::ROT_Z); rot_ind.has_value())
  {
    auto const ind = rot_ind.value();
    Vector weights{ { second_trans.value().cov(ind, ind), first_trans.value().cov(ind, ind) } };
    weights /= weights.sum();
    RowVector angles{ { first_trans.value().mean(ind), second_trans.value().mean(ind) } };
    mean(ind) = angles::weightedMean(angles, weights);
  }
  State fused_state = manager->createState();
  fused_state._state_dist[manager->params().state.estimation.output_state_model] =
      std::make_unique<GaussianDistribution>(std::move(mean), std::move(cov));

  fused_state._existenceProbability =
      std::pow(first._existenceProbability, w1) * std::pow(second._existenceProbability, w2) /
      (std::pow(first._existenceProbability, w1) * std::pow(second._existenceProbability, w2) +
       std::pow(1 - first._existenceProbability, w1) * std::pow(1 - second._existenceProbability, w2));
  fused_state._time = first._time;
  fused_state._label = first._label;
  fused_state._meta_data._durationSinceLastAssociation = 0s;
  fused_state._meta_data._numUpdates += 1;
  fused_state._classification = first._classification;
  fused_state._misc = first._misc;
  fused_state._misc.insert(second._misc.begin(), second._misc.end());

  fused_state._classification.merge(second._classification);

  assert([&] {  // NOLINT
    if (not fused_state.isValid())
    {
      LOG_FATAL("fused state with cross covariance intersection is not valid.");
      LOG_FATAL(fused_state.toString());
      LOG_FATAL("State 1: " << first.toString());
      LOG_FATAL("State 2: " << second.toString());
      return false;
    }
    return true;
  }());
  return fused_state;
}

[[nodiscard]] graph::Graph<StateId, std::size_t, double>
graph(std::vector<State> const& first_tracks,
      std::vector<State> const& second_tracks,
      std::vector<std::pair<Metric, double>> const& metric_max_dist)
{
  std::vector<StateId> nodes;
  std::vector<graph::Edge<std::size_t, StateId, double>> edges;
  for (State const& first : first_tracks)
  {
    nodes.push_back(first._id);
  }
  for (State const& second : second_tracks)
  {
    nodes.push_back(second._id);
  }
  std::size_t edge_ctr = 0;
  for (State const& first : first_tracks)
  {
    for (State const& second : second_tracks)
    {
      bool ok = true;
      for (auto const& [metric, max_dist] : metric_max_dist)
      {
        auto dist = metric(first, second);
        if (dist > max_dist)
        {
          ok = false;
          break;
        }
      }
      if (ok)
      {
        edges.emplace_back(edge_ctr, first._id, second._id, 1);
        edge_ctr++;
      }
    }
  }
  return { std::move(nodes), std::move(edges) };
}

[[nodiscard]] AssignmentSol compute_assignment(std::vector<State> const& first,
                                               std::vector<State> const& second,
                                               std::vector<std::pair<Metric, double>> const& metric_max_dist)
{
  ZoneScopedNC("compute_assignment", tracy_color);
  std::vector<std::vector<StateId>> comps = graph(first, second, metric_max_dist).components();
  AssignmentSol solution;
  for (std::vector<StateId> comp : comps)
  {
    std::vector<std::size_t> first_tracks;
    std::vector<std::size_t> second_tracks;
    for (StateId id : comp)
    {
      if (auto it = std::ranges::find_if(first, [id](State const& track) { return track._id == id; });
          it != first.end())  // node of component belong to first
      {
        first_tracks.push_back(std::distance(first.begin(), it));
      }
      else  // node belong to second
      {
        second_tracks.push_back(std::distance(
            second.begin(), std::ranges::find_if(second, [id](State const& track) { return track._id == id; })));
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
    Matrix costMatrix = Matrix::Zero(first_tracks.size(), second_tracks.size());
    for (std::size_t i = 0; i < first_tracks.size(); ++i)
    {
      for (std::size_t j = 0; j < second_tracks.size(); ++j)
      {
        for (auto const& [metric, _] : metric_max_dist)
        {
          costMatrix(i, j) += metric(first.at(first_tracks.at(i)), second.at(second_tracks.at(j)));
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
    for (std::size_t i = 0; i < assignment.rows(); ++i)
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
      if (std::find(used_first.begin(), used_first.end(), first_ind) == used_first.end())
      {
        solution.non_assigned_first.push_back(first.at(first_ind));
      }
    }
    for (std::size_t second_ind : second_tracks)
    {
      if (std::find(used_second.begin(), used_second.end(), second_ind) == used_second.end())
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
