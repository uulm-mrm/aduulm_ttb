#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Concepts.h"
#include "tracking_lib/Graph/Graph.h"
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"

namespace ttb
{

/// This represents a mixture distribution of other Distributions
template <CPTDistribution Dist, class UDT>
class MixtureDistribution
{
public:
  using MixtureComponentType = Dist;
  static_assert(not std::is_same_v<Dist, MixtureDistribution>);
  MixtureDistribution() = default;
  explicit MixtureDistribution(Dist comp)
    : _dists{ [&] {
      std::vector<std::pair<UDT, Dist>> dists;
      dists.emplace_back(1, std::move(comp));
      return dists;
    }() }
  {
  }

  explicit MixtureDistribution(std::vector<std::pair<UDT, Dist>> comps) : _dists{ std::move(comps) }
  {
    UDT const sum = sum_weights();
    if (sum < 1e-10)
    {
      LOG_WARN(std::format("build a Mixture distribution with very small weight {}. We probably refuse to do this, "
                           "because it can cause trouble in normalization (division by 0...)",
                           sum));
    }
    multiply_weights(1 / sum);
  }

  [[nodiscard]] DistributionId id() const
  {
    return _id;
  }

  [[nodiscard]] bool is_empty() const noexcept
  {
    return _dists.empty();
  }

  [[nodiscard]] Vector<UDT> mean(Components const& comps) const
  {
    Vector<UDT> init = _dists.at(0).second.mean() * _dists.at(0).first;
    Vector<UDT> mean = std::accumulate(std::next(_dists.begin()),
                                       _dists.end(),
                                       init,
                                       [](Vector<UDT> const& old, auto const& comp) -> Vector<UDT> {
                                         return old + comp.second.mean() * comp.first;
                                       }) /
                       sum_weights();
    if (auto const rot_z = comps.indexOf(COMPONENT::ROT_Z); rot_z.has_value())
    {
      mean(rot_z.value()) = angles::weighted_mean<UDT>([&] {
        std::vector<std::pair<UDT, UDT>> wa;
        for (auto const& [weight, dist] : _dists)
        {
          wa.emplace_back(weight, dist.mean()(rot_z.value()));
        }
        return wa;
      }());
    }
    return mean;
  }

  [[nodiscard]] Matrix<UDT> covariance(Components const& comps) const
  {
    Matrix<UDT> init = _dists.at(0).second.covariance() * _dists.at(0).first;
    init = std::accumulate(
        std::next(_dists.begin()), _dists.end(), init, [](Matrix<UDT> const& old, auto const& comp) -> Matrix<UDT> {
          return old + comp.second.covariance() * comp.first;
        });
    Vector<UDT> const mu = mean(comps);
    return std::accumulate(_dists.begin(),
                           _dists.end(),
                           init,
                           [&](Matrix<UDT> const& old, auto const& comp) -> Matrix<UDT> {
                             return old +
                                    (comp.second.mean() - mu) * (comp.second.mean() - mu).transpose() * comp.first;
                           }) /
           sum_weights();
  }

  [[nodiscard]] std::span<std::pair<UDT, Dist>> dists()
  {
    _id = _dist_id_Generator.getID();
    return _dists;
  }

  [[nodiscard]] std::span<std::pair<UDT, Dist> const> dists() const
  {
    return _dists;
  }

  [[nodiscard]] std::string to_string(std::string const& prefix = "") const
  {
    std::string out = prefix + "Mixture with " + std::to_string(_dists.size()) + " Components\n" + prefix +
                      "|\tID: " + std::to_string(_id.value_) + "\n";
    for (auto const& [weight, dist] : _dists)
    {
      out += prefix + "|\tWeight: " + std::to_string(weight) + "\n";
      out += dist.to_string(prefix + "|\t");
    }
    if (not _misc.empty())
    {
      for (auto const& val : _misc)
      {
        out += prefix + "|\tMisc given for: " + val.first + '\n';
      }
    }
    out += prefix + "|\tSum weights is: " + std::to_string(sum_weights()) + '\n';
    return out;
  }

  [[nodiscard]] bool is_valid() const
  {
    if (_dists.empty())
    {
      return false;
    }
    Index const mean_size = _dists.at(0).second.mean().rows();
    for (auto const& [weight, dist] : _dists)
    {
      if (dist.mean().rows() != mean_size)
      {
        LOG_FATAL("Dimension of Mean does not fit for Components");
        LOG_FATAL(to_string());
        return false;
      }
      if (weight < 0)
      {
        LOG_FATAL("Weight of Component is negative " + std::to_string(weight));
        LOG_FATAL(to_string());
        return false;
      }
      if (not dist.is_valid())
      {
        LOG_ERR("Component " + dist.to_string() + " is invalid");
        LOG_FATAL(to_string());
        return false;
      }
    }
    if (std::isinf(sum_weights()) or std::abs(sum_weights() - 1) > 1e-7)
    {
      LOG_FATAL(std::format("Total weight is not finite or not near 1: {}", sum_weights()));
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }

  [[nodiscard]] DistributionId prior_id() const
  {
    return _prior_id;
  }
  [[nodiscard]] DistributionId& prior_id()
  {
    return _prior_id;
  }

  std::size_t prune_weight(double weightThreshold)
  {
    assert(is_valid());
    // prune/erase all components with weight smaller the threshold,
    const std::size_t before = _dists.size();
    std::erase_if(_dists, [weightThreshold](auto const& comp) { return comp.first < weightThreshold; });

    const auto removed = before - _dists.size();
    if (removed > 0)
    {
      if (sum_weights() > 0)
      {
        multiply_weights(1 / sum_weights());
      }
      _id = _dist_id_Generator.getID();
    }
    assert(is_valid() or sum_weights() == 0);
    return removed;
  }

  std::size_t prune_var(double varThreshold)
  {
    // prune/erase all components with weight smaller the threshold,
    const std::size_t before = _dists.size();
    std::erase_if(_dists, [varThreshold](auto const& comp) {
      return comp.second.covariance().array().abs().maxCoeff() > varThreshold;
    });
    const auto removed = before - _dists.size();
    if (removed > 0)
    {
      if (sum_weights() > 1e-10)
      {
        multiply_weights(1 / sum_weights());
      }
      else
      {
        _dists.clear();
      }
      _id = _dist_id_Generator.getID();
    }
    assert(is_valid() or is_empty());
    return removed;
  }

  std::size_t truncate(std::size_t maxComponents)
  {
    if (_dists.size() > maxComponents)
    {
      const std::size_t before = _dists.size();
      std::ranges::sort(_dists, [](auto const& a, auto const& b) { return a.first > b.first; });
      _dists.erase(std::next(_dists.begin(), maxComponents), _dists.end());
      if (sum_weights() > 1e-10)
      {
        multiply_weights(1 / sum_weights());
      }
      else
      {
        _dists.clear();
      }
      _id = _dist_id_Generator.getID();
      return before - _dists.size();
    }
    assert(is_valid() or is_empty());
    return 0;
  }

  std::size_t merge_components(double max_dist, Components const& comps)
  {
    if (max_dist < TTB_EPS or _dists.size() <= 1 or sum_weights() < 1e-10)
    {
      return 0;
    }
    REFERENCE_POINT const reference_point = _dists.front().second.ref_point();
    assert([&] {
      return std::ranges::all_of(_dists, [&](auto const& dist) { return dist.second.ref_point() == reference_point; });
    }());
    LOG_DEB("MixtureDistribution::merge_components");
    std::size_t const num_comps_before = _dists.size();
    using NodeId = std::size_t;
    using EdgeId = std::size_t;
    std::optional<Index> const angle_ind = comps.indexOf(COMPONENT::ROT_Z);
    std::vector<graph::Edge<EdgeId, NodeId, double>> edges;
    std::vector<NodeId> nodes;
    EdgeId edge_ctr = 0;
    for (auto const& [i, dist] : std::views::enumerate(_dists))
    {
      nodes.push_back(i);
      for (std::size_t j = 0; j < _dists.size(); ++j)
      {
        if (i == j)
        {
          continue;
        }
        Vector<UDT> diff = dist.second.mean() - _dists.at(j).second.mean();
        if (angle_ind.has_value())
        {
          angles::normalizeAngle(diff(angle_ind.value()));
        }
        Matrix<UDT> const C_merged = dist.second.covariance() + _dists.at(j).second.covariance();
        UDT const mhd2 = diff.transpose() * C_merged.llt().solve(diff);
        if (mhd2 < max_dist)
        {
          edges.emplace_back(edge_ctr, i, j, 1);
          ++edge_ctr;
        }
      }
    }
    graph::DiGraph graph(std::move(nodes), std::move(edges));
    auto const graph_components = graph.components();
    LOG_DEB(std::format("previous comps {} merged comps {}", _dists.size(), graph_components.size()));
    assert([&] {  // NOLINT
      std::size_t const before = _dists.size();
      std::size_t after = 0;
      for (std::vector<NodeId> const& merge_inds : graph_components)
      {
        after += merge_inds.size();
      }
      return before == after;
    }());
    std::vector<std::pair<UDT, Dist>> merged_components;
    for (std::vector<NodeId> const& merge_inds : graph_components)
    {
      if (merge_inds.size() == 1)
      {
        assert([&] {  // NOLINT
          if (merge_inds.front() >= _dists.size())
          {
            LOG_FATAL(to_string());
            LOG_FATAL(std::stacktrace::current());
            return false;
          }
          return true;
        }());
        merged_components.push_back(std::move(_dists.at(merge_inds.front())));
        continue;
      }
      std::vector<std::pair<UDT, Dist>> merge_dist;
      UDT weight = 0;
      for (NodeId id : merge_inds)
      {
        assert([&] {  // NOLINT
          if (id >= _dists.size())
          {
            LOG_FATAL(to_string());
            LOG_FATAL(std::stacktrace::current());
            return false;
          }
          return true;
        }());
        weight += _dists.at(id).first;
        merge_dist.push_back(std::move(_dists.at(id)));
      }
      assert(not merge_dist.empty());

      MixtureDistribution const mixture(std::move(merge_dist));
      if (not mixture.is_empty())
      {
        assert(mixture.is_valid());
        GaussianDistribution merged(mixture.mean(comps), mixture.covariance(comps), reference_point);
        merged_components.emplace_back(weight, std::move(merged));
      }
    }
    _dists = std::move(merged_components);
    assert(is_valid());
    return num_comps_before - _dists.size();
  }

  [[nodiscard]] Dist const& best_component() const
  {
    if (_dists.empty())
    {
      LOG_FATAL("Best Component of empty Mixture requested: " << to_string());
      throw std::runtime_error("No Component in Mixture");
    }
    auto const& best_comp =
        std::ranges::max_element(_dists, [](auto const& a, auto const& b) { return a.first < b.first; });
    return best_comp->second;
  }

private:
  [[nodiscard]] UDT sum_weights() const
  {
    return std::accumulate(
        _dists.begin(), _dists.end(), UDT{ 0.0 }, [](UDT old, auto const& b) { return old + b.first; });
  }
  void multiply_weights(UDT factor)
  {
    std::ranges::for_each(_dists, [factor](auto& comp) { comp.first *= factor; });
    _id = _dist_id_Generator.getID();
  }

public:
  [[nodiscard]] std::map<std::string, std::any>& misc()
  {
    return _misc;
  }
  void give_new_id()
  {
    for (auto [weight, dist] : _dists)
    {
      dist.give_new_id();
    }
    _id = _dist_id_Generator.getID();
  }

private:
  std::vector<std::pair<UDT, Dist>> _dists;
  DistributionId _id{ _dist_id_Generator.getID() };
  DistributionId _prior_id{ NO_DISTRIBUTION_ID_HISTORY };
  std::map<std::string, std::any> _misc;
};

[[nodiscard]] inline double kl_divergence(MixtureDistribution<GaussianDistribution<double>, double> const& first,
                                          MixtureDistribution<GaussianDistribution<double>, double> const& second)
{
  double d_var = 0;
  double d_prod = 0;
  for (auto const& [first_weight, first_gauss_comp] : first.dists())
  {
    d_var += first_weight * std::log([&] {
               double num = 0;
               for (auto const& [weight, comp] : first.dists())
               {
                 num += weight * std::exp(-kl_divergence(first_gauss_comp, comp));
               }
               return num;
             }() /
                                     [&] {
                                       double num = 0;
                                       for (auto const& [weight, comp] : second.dists())
                                       {
                                         num += weight * std::exp(-kl_divergence(first_gauss_comp, comp));
                                       }
                                       return num;
                                     }());
    d_prod += first_weight *
              std::log([&] {
                double num = 0;
                for (auto const& [weight, comp] : first.dists())
                {
                  auto const llt = (first_gauss_comp.covariance() + comp.covariance()).llt();
                  Vector<double> const res = comp.mean() - first_gauss_comp.mean();
                  num += weight * std::exp(-0.5 * (comp.mean().rows() * std::log(2 * std::numbers::pi) +
                                                   llt.matrixL().determinant() * llt.matrixL().determinant() +
                                                   res.transpose() * llt.solve(res)));
                }
                return num;
              }() /
                       [&] {
                         double num = 0;
                         for (auto const& [weight, comp] : second.dists())
                         {
                           auto const llt = (first_gauss_comp.covariance() + comp.covariance()).llt();
                           Vector<double> const res = comp.mean() - first_gauss_comp.mean();
                           num += weight * std::exp(-0.5 * (comp.mean().rows() * std::log(2 * std::numbers::pi) +
                                                            llt.matrixL().determinant() * llt.matrixL().determinant() +
                                                            res.transpose() * llt.solve(res)));
                         }
                         return num;
                       }());
  }
  if (not std::isfinite(d_var) and not std::isfinite(d_prod))
  {
    LOG_WARN("non finite kl divergence ... for ");
    LOG_WARN(first.to_string());
    LOG_WARN(second.to_string());
    return 0;
  }
  if (std::isfinite(d_var) and not std::isfinite(d_prod))
  {
    return d_var;
  }
  if (not std::isfinite(d_var) and std::isfinite(d_prod))
  {
    return d_prod;
  }

  return 0.5 * (d_var + d_prod);
}

}  // namespace ttb
