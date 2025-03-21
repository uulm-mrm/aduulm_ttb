#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

// ######################################################################################################################

namespace ttb
{

/// This is a simple representation of a Poisson Distribution, see
/// https://en.wikipedia.org/wiki/Poisson_distribution
class PoissonDistribution
{
public:
  explicit PoissonDistribution(double lambda);
  [[nodiscard]] double pmf(std::size_t x) const;
  [[nodiscard]] double variance() const;
  [[nodiscard]] double mean() const;
  [[nodiscard]] std::string toString() const;
  double _lambda;
};

}  // namespace ttb