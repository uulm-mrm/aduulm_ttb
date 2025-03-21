#include "tracking_lib/Distributions/MixtureDistribution.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/Misc/AngleNormalization.h"

#include <tracy/tracy/Tracy.hpp>
#include <ranges>

namespace ttb
{

constexpr auto tracy_color = tracy::Color::Crimson;

MixtureDistribution::MixtureDistribution() : _id{ _id_Generator.getID() }
{
}

MixtureDistribution::MixtureDistribution(std::unique_ptr<BaseDistribution> comp)
  : _dists{ [&]() {
    std::vector<std::unique_ptr<BaseDistribution>> out;
    out.emplace_back(std::move(comp));
    return out;
  }() }
  , _id{ _id_Generator.getID() }
{
}

MixtureDistribution::MixtureDistribution(std::vector<std::unique_ptr<BaseDistribution>> comps)
  : _dists{ std::move(comps) }, _id{ _id_Generator.getID() }
{
}

DISTRIBUTION_TYPE MixtureDistribution::type() const
{
  return DISTRIBUTION_TYPE::MIXTURE;
}

DistributionId MixtureDistribution::id() const
{
  return _id;
}

REFERENCE_POINT MixtureDistribution::refPoint() const
{
  LOG_FATAL("refPoint of Mixture Dist called - this is most probably a serious bug");
  throw std::runtime_error("Reference Point of Mixture Dist");
}

std::vector<BaseDistribution*> MixtureDistribution::dists()
{
  std::vector<BaseDistribution*> out;
  for (auto& dist : _dists)
  {
    out.push_back(dist.get());
  }
  return out;
}

std::vector<BaseDistribution const*> MixtureDistribution::dists() const
{
  std::vector<BaseDistribution const*> out;
  for (auto const& dist : _dists)
  {
    out.push_back(dist.get());
  }
  return out;
}

Matrix const& MixtureDistribution::covariance() const
{
  if (_dists.empty())
  {
    LOG_FATAL("Covariance of empty mixture called - this is probably a serious bug");
    throw std::runtime_error("Covariance of empty Mixture");
  }
  if (_covCache.has_value())
  {
    return _covCache.value();
  }
  Matrix init = _dists.at(0)->covariance() * _dists.at(0)->sumWeights();
  init =
      std::accumulate(std::next(_dists.begin()), _dists.end(), init, [](Matrix const& old, auto const& comp) -> Matrix {
        return old + comp->covariance() * comp->sumWeights();
      });
  Vector const& mu = mean();
  _covCache = std::accumulate(_dists.begin(),
                              _dists.end(),
                              init,
                              [&](Matrix const& old, auto const& comp) -> Matrix {
                                Matrix add = (comp->mean() - mu) * (comp->mean() - mu).transpose() * comp->sumWeights();
                                return old + add;
                              }) /
              sumWeights();
  return _covCache.value();
}

Vector const& MixtureDistribution::mean() const
{
  if (_dists.empty())
  {
    LOG_FATAL("Mean of empty mixture");
    throw std::runtime_error("Mean of empty Mixture");
  }
  if (_meanCache.has_value())
  {
    return _meanCache.value();
  }
  Vector init = _dists.at(0)->mean() * _dists.at(0)->sumWeights();
  _meanCache = std::accumulate(std::next(_dists.begin()),
                               _dists.end(),
                               init,
                               [](Vector const& old, auto const& comp) -> Vector {
                                 return old + comp->mean() * comp->sumWeights();
                               }) /
               sumWeights();
  return _meanCache.value();
}

double MixtureDistribution::pdf(Vector const& x) const
{
  double init = _dists[0]->pdf(x) * _dists[0]->sumWeights();
  return std::accumulate(std::next(_dists.begin()), _dists.end(), init, [&](double old, auto const& comp) {
    return old + comp->sumWeights() * comp->pdf(x);
  });
}

std::string MixtureDistribution::toString(std::string const& prefix) const
{
  std::string out = prefix + "Mixture with " + std::to_string(_dists.size()) + " Components\n" + prefix +
                    "|\tID: " + std::to_string(_id.value_) + "\n";
  for (auto const& dist : _dists)
  {
    out += dist->toString(prefix + "|\t");
  }
  if (not _misc.empty())
  {
    for (auto const& val : _misc)
    {
      out += prefix + "|\tMisc given for: " + val.first + '\n';
    }
  }
  out += prefix + "|\tSum weights is: " + std::to_string(sumWeights()) + '\n';
  return out;
}

bool MixtureDistribution::isValid() const
{
  if (_dists.empty())
  {
    return true;
  }
  double wSum = 0;
  Index const mean_size = _dists.at(0)->mean().rows();
  for (auto const& dist : _dists)
  {
    if (dist->type() != DISTRIBUTION_TYPE::GAUSSIAN)
    {
      LOG_FATAL("Mixture of non Gaussian Distribution - do not do that");
      LOG_FATAL(toString());
      return false;
    }
    if (dist->mean().rows() != mean_size)
    {
      LOG_FATAL("Dimension of Mean does not fit for Components");
      LOG_FATAL(toString());
      return false;
    }
    if (dist->sumWeights() < 0)
    {
      LOG_FATAL("Weight of Component is negative " + std::to_string(dist->sumWeights()));
      LOG_FATAL(toString());
      return false;
    }
    wSum += dist->sumWeights();
    if (not dist->isValid())
    {
      LOG_ERR("Component " + dist->toString() + " is invalid");
      LOG_FATAL(toString());
      return false;
    }
  }
  if (std::isinf(wSum) or (wSum < 0))
  {
    LOG_FATAL("Total weight is not finite or negative: " + std::to_string(wSum));
    LOG_FATAL(toString());
    return false;
  }
  return true;
}

std::size_t MixtureDistribution::pruneWeight(double weightThreshold)
{
  // prune/erase all components with weight smaller the threshold,
  // in case all components would be erased, keep the best component
  if (_dists.empty())
  {
    return 0;
  }
  const std::size_t before = _dists.size();
  std::erase_if(_dists, [weightThreshold](auto const& comp) { return comp->sumWeights() < weightThreshold; });

  const auto removed = before - _dists.size();
  if (removed > 0)
  {
    _meanCache.reset();
    _covCache.reset();
    _id = _id_Generator.getID();
  }
  return removed;
}

std::size_t MixtureDistribution::pruneVar(double varThreshold)
{
  // prune/erase all components with weight smaller the threshold,
  // in case all components would be erased, keep the best component
  if (_dists.size() <= 1)
  {
    return 0;
  }
  const std::size_t before = _dists.size();
  std::erase_if(
      _dists, [varThreshold](auto const& comp) { return comp->covariance().array().abs().maxCoeff() > varThreshold; });
  const auto removed = before - _dists.size();
  if (removed > 0)
  {
    _meanCache.reset();
    _covCache.reset();
    _id = _id_Generator.getID();
  }
  return removed;
}

std::size_t MixtureDistribution::truncate(std::size_t maxComponents)
{
  if (_dists.size() > maxComponents)
  {
    const std::size_t before = _dists.size();
    std::ranges::sort(_dists, [](auto const& a, auto const& b) { return a->sumWeights() < b->sumWeights(); });
    _dists.erase(std::next(_dists.begin(), maxComponents), _dists.end());
    _meanCache.reset();
    _covCache.reset();
    _id = _id_Generator.getID();
    return before - _dists.size();
  }
  return 0;
}

BaseDistribution const& MixtureDistribution::bestComponent() const
{
  if (_dists.empty())
  {
    LOG_FATAL("Best Component of empty Mixture requested: " << toString());
    throw std::runtime_error("No Component in Mixture");
  }
  auto const& best_comp =
      std::ranges::max_element(_dists, [](auto const& a, auto const& b) { return a->sumWeights() < b->sumWeights(); });
  return **best_comp;
}

double MixtureDistribution::sumWeights() const
{
  return std::accumulate(
      _dists.begin(), _dists.end(), 0.0, [](double old, auto const& b) { return old + b->sumWeights(); });
}

void MixtureDistribution::multiplyWeights(double factor)
{
  std::ranges::for_each(_dists, [factor](auto& comp) { comp->multiplyWeights(factor); });
  _meanCache.reset();
  _covCache.reset();
  _id = _id_Generator.getID();
}

void MixtureDistribution::merge(std::unique_ptr<BaseDistribution> dist)
{
  /// releases the memory of the other dist, and grab for myself
  /// should be safe ... :)
  BaseDistribution* other_dist = dist.release();
  if (other_dist->type() == DISTRIBUTION_TYPE::MIXTURE)
  {
    for (BaseDistribution* comp : other_dist->dists())
    {
      _dists.emplace_back(std::unique_ptr<BaseDistribution>(comp));
    }
  }
  else if (other_dist->type() == DISTRIBUTION_TYPE::GAUSSIAN)
  {
    _dists.emplace_back(std::unique_ptr<BaseDistribution>(other_dist));
  }
  _meanCache.reset();
  _covCache.reset();
  _id = _id_Generator.getID();
}

void MixtureDistribution::merge(std::vector<std::unique_ptr<BaseDistribution>> others)
{
  for (std::unique_ptr<BaseDistribution>& other : others)
  {
    LOG_DEB("merge dist with id " << other->id());
    merge(std::move(other));
  }
  _id = _id_Generator.getID();
}

std::size_t MixtureDistribution::mergeComponents(double max_dist, Components const& comps)
{
  ZoneScopedNC("MixtureDistribution::mergeComponents", tracy_color);
  std::string info_str =
      "DistributionId: " + std::to_string(id().value_) + "\n#Components before: " + std::to_string(_dists.size());
  ZoneText(info_str.c_str(), info_str.size());
  if (max_dist < TTB_EPS or _dists.size() <= 1 or sumWeights() < TTB_EPS)
  {
    return 0;
  }
  assert([&]() {
    return std::all_of(
        _dists.begin(), _dists.end(), [](auto const& dist) { return dist->type() == DISTRIBUTION_TYPE::GAUSSIAN; });
  }());
  assert([&]() {
    REFERENCE_POINT first_rp = _dists.front()->refPoint();
    return std::all_of(_dists.begin(), _dists.end(), [&](auto const& dist) { return dist->refPoint() == first_rp; });
  }());
  /// Greedy hierachical merging
  /// 1 Repeat until stop
  ///     Merge the two components with smallest mhd distance
  ///     if (smallest dist > max_dist) --> stop
  /// Problem: this is really, and i mean mean really, slow
  //  while (true)
  //  {
  //    /// Min diff
  //    std::vector<std::unique_ptr<BaseDistribution>> mergedDists;
  //    double minDiff = std::numeric_limits<double>::max();
  //    std::pair<std::size_t, std::size_t> nearest;
  //    for (std::size_t i = 0; i < _dists.size() - 1; ++i)
  //    {
  //      if (_dists.at(i)->sumWeights() < TTB_EPS)
  //      {
  //        continue;
  //      }
  //      for (std::size_t j = i + 1; j < _dists.size(); ++j)
  //      {
  //        if (_dists.at(j)->sumWeights() < TTB_EPS)
  //        {
  //          continue;
  //        }
  //        Eigen::VectorXd delta_eigen = _dists.at(i)->mean() - dists().at(j)->mean();
  //        Eigen::MatrixXd C_eigen = _dists.at(i)->covariance() + dists().at(j)->covariance();
  //        double mhd2_eigen = delta_eigen.transpose() * C_eigen.llt().solve(delta_eigen);
  //        if (mhd2_eigen < minDiff)
  //        {
  //          minDiff = mhd2_eigen;
  //          nearest = { i, j };
  //        }
  //      }
  //    }
  //    if (minDiff > max_dist * max_dist)
  //    {
  //      LOG_DEB("Merged #Gaussian Dists: " << old_size - _dists.size());
  //      _id = _id_Generator.getID();
  //      _meanCache.reset();
  //      _covCache.reset();
  //      info_str = "#Components after: " + std::to_string(_dists.size());
  //      ZoneText(info_str.c_str(), info_str.size());
  //      assert([&]() {
  //        if (not isValid())
  //        {
  //          LOG_FATAL("Invalid after merge components");
  //          LOG_FATAL(toString());
  //          return false;
  //        }
  //        return true;
  //      }());
  //      return old_size - _dists.size();
  //    }
  //    MixtureDistribution merge(std::move(_dists.at(nearest.first)));
  //    merge.merge(std::move(_dists.at(nearest.second)));
  //    mergedDists.emplace_back(std::make_unique<GaussianDistribution>(
  //        merge.mean(), merge.covariance(), merge.sumWeights(), merge._dists.front()->refPoint()));
  //    for (std::size_t i = 0; i < _dists.size(); ++i)
  //    {
  //      if (i == nearest.first or i == nearest.second)
  //      {
  //        continue;
  //      }
  //      mergedDists.emplace_back(std::move(_dists.at(i)));
  //    }
  //    _dists = std::move(mergedDists);
  //  }

  /// greedy, weight based single shot merging
  /// 1. sort components by weight from biggest to lowest
  /// 2. from biggest to smallest weight:
  ///     1. if distance form current to next component < threshold -> merge
  std::ranges::sort(_dists,
                    [](auto const& first, auto const& second) { return first->sumWeights() > second->sumWeights(); });
  std::size_t num_merged = 0;
  std::vector<std::unique_ptr<BaseDistribution>> merged;
  merged.push_back(std::move(_dists.front()));
  for (std::unique_ptr<BaseDistribution>& dist : _dists | std::views::drop(1))
  {
    Eigen::VectorXd delta_eigen = dist->mean() - merged.back()->mean();
    auto angle_ind = comps.indexOf(COMPONENT::ROT_Z);
    if (angle_ind.has_value())
    {
      angles::normalizeAngle(delta_eigen(angle_ind.value()));
    }
    Eigen::MatrixXd C_eigen = dist->covariance() + merged.back()->covariance();
    double mhd2_eigen = delta_eigen.transpose() * C_eigen.llt().solve(delta_eigen);
    if (mhd2_eigen < max_dist)  // merge back and current dist
    {
      double sum_weights = dist->sumWeights() + merged.back()->sumWeights();
      Vector merged_mean =
          (dist->mean() * dist->sumWeights() + merged.back()->mean() * merged.back()->sumWeights()) / sum_weights;
      if (angle_ind.has_value())
      {
        merged_mean(angle_ind.value()) = angles::weightedMean(
            RowVector{ { dist->mean()(angle_ind.value()), merged.back()->mean()(angle_ind.value()) } },
            Vector{ { dist->sumWeights() / sum_weights, merged.back()->sumWeights() / sum_weights } });
      }
      Matrix merged_var =
          (dist->covariance() * dist->sumWeights() + merged.back()->covariance() * merged.back()->sumWeights() +
           (dist->mean() - merged_mean) * (dist->mean() - merged_mean).transpose() * dist->sumWeights() +
           (merged.back()->mean() - merged_mean) * (merged.back()->mean() - merged_mean).transpose() *
               merged.back()->sumWeights()) /
          sum_weights;
      merged.back() = std::make_unique<GaussianDistribution>(merged_mean, merged_var, sum_weights, dist->refPoint());
      num_merged++;
    }
    else
    {
      merged.push_back(std::move(dist));
    }
  }
  _dists = std::move(merged);
  info_str = "#Components after: " + std::to_string(_dists.size());
  ZoneText(info_str.c_str(), info_str.size());
  return num_merged;
}

std::unique_ptr<BaseDistribution> MixtureDistribution::clone() const
{
  auto mixture = std::make_unique<MixtureDistribution>([&] {
    std::vector<std::unique_ptr<BaseDistribution>> out;
    for (auto const& dist : _dists)
    {
      out.emplace_back(dist->clone());
    }
    return out;
  }());
  mixture->_id = _id;
  return mixture;
}

void MixtureDistribution::set(Vector mean)
{
  LOG_FATAL("It is forbidden to set the mean for a mixture distribution - BUG ALERT");
  throw std::runtime_error("It is forbidden to set the mean for a mixture distribution");
}

void MixtureDistribution::set(Matrix cov)
{
  LOG_FATAL("It is forbidden to set the covariance for a mixture distribution - BUG ALERT");
  throw std::runtime_error("It is forbidden to set the covariance for a mixture distribution");
}

void MixtureDistribution::set(double weight)
{
  LOG_FATAL("It is forbidden to set the weight for a mixture distribution - BUG ALERT");
  throw std::runtime_error("It is forbidden to set the weight for a mixture distribution");
}

void MixtureDistribution::set(REFERENCE_POINT ref)
{
  LOG_FATAL("It is forbidden to set the reference point for a mixture distribution - BUG ALERT");
  throw std::runtime_error("It is forbidden to set the reference point for a mixture distribution");
}

void MixtureDistribution::set(Vector mean, Matrix cov)
{
  LOG_FATAL("It is forbidden to set the mean/cov for a mixture distribution - BUG ALERT");
  throw std::runtime_error("It is forbidden to set the mean/cov for a mixture distribution");
}

void MixtureDistribution::setPriorId(DistributionId new_id)
{
  std::ranges::for_each(_dists, [new_id](auto& comp) { comp->setPriorId(new_id); });
}

DistributionId MixtureDistribution::priorId() const
{
  LOG_FATAL("Can not return a prior id for a mixture distribution - BUG ALERT");
  throw std::runtime_error("Can not return a prior id for a mixture distribution");
}
void MixtureDistribution::resetPriorId()
{
  std::ranges::for_each(_dists, [](auto& comp) { comp->resetPriorId(); });
}

}  // namespace ttb