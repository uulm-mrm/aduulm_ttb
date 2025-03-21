#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

#include <any>

namespace ttb
{

class Components;

/// The Base Distribution interface.
/// This class represents a distribution in a stochastic sense with a reference point
class BaseDistribution
{
public:
  virtual ~BaseDistribution() = default;
  /// get the type of the Distribution
  [[nodiscard]] virtual DISTRIBUTION_TYPE type() const = 0;
  /// check whether this is a valid distribution
  [[nodiscard]] virtual bool isValid() const = 0;
  /// get the unique id of the Distribution
  [[nodiscard]] virtual DistributionId id() const = 0;
  /// get the referencePoint of the Distribution
  [[nodiscard]] virtual REFERENCE_POINT refPoint() const = 0;
  /// string representation
  [[nodiscard]] virtual std::string toString(std::string const& prefix = "") const = 0;  // NOLINT
  /// the covariance of the Distribution
  [[nodiscard]] virtual Matrix const& covariance() const = 0;
  /// the mean of the Distribution
  [[nodiscard]] virtual Vector const& mean() const = 0;
  /// the Probability Density Function
  [[nodiscard]] virtual double pdf(Vector const& x) const = 0;
  /// Set the Mean
  virtual void set(Vector mean) = 0;
  /// Set the Covariance
  virtual void set(Matrix cov) = 0;
  /// Set the weight
  virtual void set(double weight) = 0;
  /// Set the reference Point
  virtual void set(REFERENCE_POINT ref) = 0;
  /// set mean and cov simultaneously, this allows to change the dimension ...
  virtual void set(Vector mean, Matrix cov) = 0;
  /// sets prior_id to wished value (needed for fpm_lmb and pm_lmb)
  virtual void setPriorId(DistributionId new_id) = 0;
  /// get the id of the prior component (needed for fpm_lmb and pm_lmb)
  [[nodiscard]] virtual DistributionId priorId() const = 0;
  /// reset the prior id
  virtual void resetPriorId() = 0;
  /// Prune the Distribution, i.e, delete all Mixture Components with weight below the threshold
  /// Attention: At least the component with the biggest weight will remain, regardless of the value
  /// Return #removed Components
  virtual std::size_t pruneWeight(double weightThreshold) = 0;
  /// Prune the Distribution, i.e, delete all Components with variance greater the threshold
  /// Attention: At least one component remain
  /// Return #removed Components
  virtual std::size_t pruneVar(double varThreshold) = 0;
  /// Delete the n smallest Components so that most maxNComponents remain and sort distribution components by weight
  /// Return #removed Components
  virtual std::size_t truncate(std::size_t maxNComponents) = 0;
  /// Merge similar Components of this Distribution
  /// need information about the components because of the angle
  /// Return #merged Components
  virtual std::size_t mergeComponents(double max_dist, Components const& comps) = 0;
  /// Merge the other Distribution into myself
  virtual void merge(std::unique_ptr<BaseDistribution> other) = 0;
  /// Merge all other Distributions into myself
  virtual void merge(std::vector<std::unique_ptr<BaseDistribution>> other) = 0;
  /// Return the best/Component with biggest Weight of that Distribution
  [[nodiscard]] virtual BaseDistribution const& bestComponent() const = 0;
  /// Sum of weights of all Components
  [[nodiscard]] virtual double sumWeights() const = 0;
  /// Multiply the weights of all Components
  virtual void multiplyWeights(double fac) = 0;
  /// get a copy of this Distribution
  [[nodiscard]] virtual std::unique_ptr<BaseDistribution> clone() const = 0;
  /// get all underlying Base Distributions
  [[nodiscard]] virtual std::vector<BaseDistribution*> dists() = 0;
  [[nodiscard]] virtual std::vector<BaseDistribution const*> dists() const = 0;
  static IDGenerator<DistributionId> _id_Generator;
  /// store arbitrary additional information
  std::map<std::string, std::any> _misc{};
};

}  // namespace ttb