#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/Distributions/MixtureDistribution.h"
#include "tracking_lib/TTBTypes/Concepts.h"

namespace ttb
{

template <class UDT>
class EmptyDistribution
{
public:
  [[nodiscard]] bool static is_valid() noexcept
  {
    return false;
  }

  [[nodiscard]] std::string static to_string(std::string const& prefix = "") noexcept
  {
    return prefix + "EMPTY_DISTRIBUTION";
  }

  [[nodiscard]] bool static is_empty() noexcept
  {
    return true;
  }

  [[nodiscard]] Vector<UDT> static mean() noexcept
  {
    return Vector<UDT>{ { 666 } };
  }

  [[nodiscard]] Vector<UDT> static covariance() noexcept
  {
    return Matrix<UDT>{ { 666 } };
  }

  [[nodiscard]] DistributionId static id() noexcept
  {
    return NO_DISTRIBUTION;
  }

  [[nodiscard]] DistributionId static prior_id() noexcept
  {
    return NO_DISTRIBUTION;
  }

  [[nodiscard]] std::map<std::string, std::any> static misc() noexcept
  {
    return {};
  }

  void static give_new_id() noexcept
  {
  }
};

static_assert(CPTDistribution<EmptyDistribution<double>>);
static_assert(CPTDistribution<EmptyDistribution<AD>>);

template <class UDT>
class Distribution
{
public:
  using AllDistributions = std::
      variant<EmptyDistribution<UDT>, GaussianDistribution<UDT>, MixtureDistribution<GaussianDistribution<UDT>, UDT>>;
  AllDistributions _distribution;

  [[nodiscard]] bool is_valid() const
  {
    return std::visit([&](CPTDistribution auto const& dist) { return dist.is_valid(); }, _distribution);
  }

  [[nodiscard]] bool is_empty() const
  {
    return std::visit([&](auto const& dist) { return dist.is_empty(); }, _distribution);
  }

  [[nodiscard]] std::string to_string(std::string const& prefix) const
  {
    return std::visit([&](CPTDistribution auto const& dist) { return dist.to_string(prefix); }, _distribution);
  }

  void merge_components(UDT max_dist, Components const& comps)
  {
    if (std::holds_alternative<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution))
    {
      auto& mixture = std::get<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution);
      mixture.merge_components(to_val(max_dist), comps);
    }
  }

  void prune_weight(UDT min_weight)
  {
    if (std::holds_alternative<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution))
    {
      auto& mixture = std::get<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution);
      mixture.prune_weight(to_val(min_weight));
    }
  }

  void prune_var(UDT max_var)
  {
    if (std::holds_alternative<GaussianDistribution<UDT>>(_distribution))
    {
      auto& gauss = std::get<GaussianDistribution<UDT>>(_distribution);
      gauss.prune_var(to_val(max_var));
    }
    if (std::holds_alternative<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution))
    {
      auto& mixture = std::get<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution);
      mixture.prune_var(to_val(max_var));
    }
  }

  void truncate(std::size_t max_comps)
  {
    if (std::holds_alternative<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution))
    {
      auto& mixture = std::get<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution);
      mixture.truncate(max_comps);
    }
  }

  void give_new_id()
  {
    std::visit([&](CPTDistribution auto& dist) { return dist.give_new_id(); }, _distribution);
  }

  std::vector<std::pair<UDT&, GaussianDistribution<UDT>&>> to_gaussian_components()
  {
    if (std::holds_alternative<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution))
    {
      auto& mixture = std::get<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution);
      std::vector<std::pair<UDT&, GaussianDistribution<UDT>&>> tmp;
      tmp.assign(mixture.dists().begin(), mixture.dists().end());
      return tmp;
    }
    if (std::holds_alternative<GaussianDistribution<UDT>>(_distribution))
    {
      auto& gauss = std::get<GaussianDistribution<UDT>>(_distribution);
      static UDT one{ 1 };
      return { { one, gauss } };
    }
    LOG_FATAL("unsupported distribution type.");
    assert(false);
    std::unreachable();
  }

  [[nodiscard]] std::vector<std::pair<UDT, GaussianDistribution<UDT> const&>> to_gaussian_components() const
  {
    if (std::holds_alternative<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution))
    {
      auto const& mixture = std::get<MixtureDistribution<GaussianDistribution<UDT>, UDT>>(_distribution);
      std::vector<std::pair<UDT, GaussianDistribution<UDT> const&>> tmp;
      for (auto const& dist_elem : mixture.dists())
      {
        tmp.emplace_back(dist_elem);
      }
      return tmp;
    }
    if (std::holds_alternative<GaussianDistribution<UDT>>(_distribution))
    {
      auto const& gauss = std::get<GaussianDistribution<UDT>>(_distribution);
      static const UDT one{ 1 };
      return { { one, gauss } };
    }
    assert(false);
    std::unreachable();
  }
};

static_assert(CPTDistribution<GaussianDistribution<double>>);
static_assert(CPTDistribution<GaussianDistribution<AD>>);
static_assert(CPTDistribution<MixtureDistribution<GaussianDistribution<double>, double>>);
static_assert(CPTDistribution<MixtureDistribution<GaussianDistribution<AD>, AD>>);

}  // namespace ttb