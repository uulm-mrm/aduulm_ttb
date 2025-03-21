#include "tracking_lib/Distributions/GammaDistribution.h"

namespace ttb
{

GammaDistribution::GammaDistribution(double alpha, double beta) : _alpha{ alpha }, _beta{ beta }
{
  if (alpha <= 0 or beta <= 0)
  {
    throw std::runtime_error("invalid alpha or beta parameter: " + std::to_string(_alpha) + " " +
                             std::to_string(_beta));
  }
}

double GammaDistribution::mean() const
{
  return _alpha / _beta;
}

double GammaDistribution::variance() const
{
  return _alpha / (_beta * _beta);
}

double GammaDistribution::pdf(double x) const
{
  return std::pow(_beta, _alpha) / std::tgamma(_alpha) * std::pow(x, _alpha - 1) * std::exp(-_beta * x);
}

std::string GammaDistribution::toString() const
{
  return "Gamma Distribution alpha=" + std::to_string(_alpha) + " beta=" + std::to_string(_beta);
}

}  // namespace ttb