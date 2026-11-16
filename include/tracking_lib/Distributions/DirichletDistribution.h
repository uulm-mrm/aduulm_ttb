#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// This is a simple representation of a Dirichlet Distribution, see
/// https://en.wikipedia.org/wiki/Categorical_distribution

class DirichletDistribution
{
public:
  explicit DirichletDistribution(Vector<double> alpha);
  [[nodiscard]] Matrix<double> variance() const;
  [[nodiscard]] Vector<double> mean() const;
  [[nodiscard]] std::string toString() const;
  Vector<double> _alpha;
};



namespace BetaDistribution
{
struct Params
{
  double p;
  double q;
};
[[nodiscard]] Params from_mean_var(double mean, double var);

}

}  // namespace ttb