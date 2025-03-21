#pragma once

#include "tracking_lib/Distributions/BaseDistribution.h"
// ######################################################################################################################
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// This is a simple representation of a Gaussian Distribution, see
/// https://en.wikipedia.org/wiki/Multivariate_normal_distribution
class GaussianDistribution final : public BaseDistribution
{
public:
  GaussianDistribution();
  GaussianDistribution(Vector mean, Matrix cov);
  GaussianDistribution(Vector mean, Matrix cov, double weight, REFERENCE_POINT ref);
  ~GaussianDistribution() override = default;
  [[nodiscard]] DISTRIBUTION_TYPE type() const override;
  [[nodiscard]] DistributionId id() const override;
  [[nodiscard]] REFERENCE_POINT refPoint() const override;
  [[nodiscard]] bool isValid() const override;
  [[nodiscard]] double pdf(Vector const& x) const override;
  [[nodiscard]] Matrix const& covariance() const override;
  [[nodiscard]] Vector const& mean() const override;
  [[nodiscard]] std::string toString(std::string const& prefix = "") const override;  // NOLINT
  std::size_t pruneWeight(double weightThreshold) override;
  std::size_t pruneVar(double varThreshold) override;
  std::size_t truncate(std::size_t maxNComponents) override;
  std::size_t mergeComponents(double maxDist, Components const& com) override;
  void merge(std::unique_ptr<BaseDistribution> other) override;
  void merge(std::vector<std::unique_ptr<BaseDistribution>> other) override;
  [[nodiscard]] BaseDistribution const& bestComponent() const override;
  [[nodiscard]] double sumWeights() const override;
  void multiplyWeights(double fac) override;
  void set(Vector mean) override;
  void set(Matrix cov) override;
  void set(double weight) override;
  void set(REFERENCE_POINT refPoint) override;
  void setPriorId(DistributionId new_id) override;
  [[nodiscard]] DistributionId priorId() const override;
  void resetPriorId() override;
  void set(Vector mean, Matrix cov) override;
  [[nodiscard]] std::unique_ptr<BaseDistribution> clone() const override;
  [[nodiscard]] std::vector<BaseDistribution*> dists() override;
  [[nodiscard]] std::vector<BaseDistribution const*> dists() const override;
  /// returns the information matrix Y := covariance()⁻¹
  [[nodiscard]] Matrix const& getInfoMat() const;
  /// Returns the information matrix vector y := Y * mean()
  [[nodiscard]] Vector const& getInfoVec() const;

private:
  Vector _mean{};
  Matrix _cov{};
  /// Cached information matrix / vector
  mutable std::optional<Matrix> m_infoMat;
  mutable std::optional<Vector> m_infoVec;
  double _weight = 1;
  REFERENCE_POINT _ref_point = REFERENCE_POINT::CENTER;
  DistributionId _id;
  DistributionId _prior_id = NO_DISTRIBUTION_ID_HISTORY;
};

}  // namespace ttb