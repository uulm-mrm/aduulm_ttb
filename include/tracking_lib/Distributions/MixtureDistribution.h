#pragma once

#include "tracking_lib/Distributions/BaseDistribution.h"
// ######################################################################################################################
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// This represents a mixture distribution of other arbitrary Distributions
class MixtureDistribution final : public BaseDistribution
{
public:
  MixtureDistribution();
  explicit MixtureDistribution(std::unique_ptr<BaseDistribution> comp);
  explicit MixtureDistribution(std::vector<std::unique_ptr<BaseDistribution>> components);
  [[nodiscard]] DISTRIBUTION_TYPE type() const override;
  [[nodiscard]] DistributionId id() const override;
  [[nodiscard]] REFERENCE_POINT refPoint() const override;
  [[nodiscard]] std::vector<BaseDistribution*> dists() override;
  [[nodiscard]] std::vector<BaseDistribution const*> dists() const override;
  [[nodiscard]] std::string toString(std::string const& prefix = "") const override;  // NOLINT
  [[nodiscard]] bool isValid() const override;
  [[nodiscard]] Matrix const& covariance() const override;
  [[nodiscard]] Vector const& mean() const override;
  [[nodiscard]] double pdf(Vector const& x) const override;
  void set(Vector mean) override;
  void set(Matrix cov) override;
  void set(double weight) override;
  void set(REFERENCE_POINT ref) override;
  void set(Vector mean, Matrix cov) override;
  void setPriorId(DistributionId new_id) override;
  [[nodiscard]] DistributionId priorId() const override;
  void resetPriorId() override;
  std::size_t pruneWeight(double weightThreshold) override;
  std::size_t pruneVar(double varThreshold) override;
  std::size_t truncate(std::size_t maxComponents) override;
  std::size_t mergeComponents(double max_dist, Components const& comps) override;
  void merge(std::unique_ptr<BaseDistribution> dist) override;
  void merge(std::vector<std::unique_ptr<BaseDistribution>> others) override;
  [[nodiscard]] BaseDistribution const& bestComponent() const override;
  [[nodiscard]] std::unique_ptr<BaseDistribution> clone() const override;
  [[nodiscard]] double sumWeights() const override;
  void multiplyWeights(double factor) override;

private:
  mutable std::optional<Vector> _meanCache;
  mutable std::optional<Matrix> _covCache;
  std::vector<std::unique_ptr<BaseDistribution>> _dists;
  DistributionId _id;
};

}  // namespace ttb