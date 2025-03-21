#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// This is a simple representation of a Dirichlet Distribution, see
/// https://en.wikipedia.org/wiki/Categorical_distribution
class DirichletDistribution
{
public:
  explicit DirichletDistribution(Vector alpha);
  [[nodiscard]] Matrix variance() const;
  [[nodiscard]] Vector mean() const;
  [[nodiscard]] std::string toString() const;
  Vector _alpha;
};

}  // namespace ttb