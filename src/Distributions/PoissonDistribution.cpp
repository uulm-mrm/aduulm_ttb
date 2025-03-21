#include "tracking_lib/Distributions/PoissonDistribution.h"

namespace ttb
{

PoissonDistribution::PoissonDistribution(double lambda) : _lambda{ lambda }
{
  if (lambda <= 0)
  {
    throw std::runtime_error("Invalid lambda < 0: " + std::to_string(lambda));
  }
}

double PoissonDistribution::pmf(std::size_t k) const
{
  return std::pow(_lambda, k) * std::exp(-_lambda) / std::tgamma(k);
}

double PoissonDistribution::mean() const
{
  return _lambda;
}

double PoissonDistribution::variance() const
{
  return _lambda;
}

std::string PoissonDistribution::toString() const
{
  return "Poisson Distribution: lambda=" + std::to_string(_lambda);
}

}  // namespace ttb