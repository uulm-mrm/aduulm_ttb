#include "tracking_lib/Distributions/GammaDistribution.h"

#include <stacktrace>

namespace ttb
{

GammaDistribution::GammaDistribution(double alpha, double beta) : _alpha{ alpha }, _beta{ beta }
{
  if (alpha <= 0 or beta <= 0 or not std::isfinite(alpha) or not std::isfinite(beta))
  {
    LOG_FATAL(std::stacktrace::current());
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

GammaDistributionParams GammaDistribution::from_mean_var(double mean, double var)
{
  GammaDistributionParams const params{ .alpha = mean * mean / var, .beta = mean / var };
  assert([&] {
    if (not std::isfinite(params.alpha) or not std::isfinite(params.beta) or params.alpha < 0 or params.beta < 0)
    {
      LOG_FATAL("Invalid gamma distribution paramters");
      return false;
    }
    GammaDistribution gamma(params.alpha, params.beta);
    if (std::abs(gamma.mean() - mean) > 1e-7 or std::abs(gamma.variance() - var) > 1e-7)
    {
      LOG_FATAL("mean and variance are not correct");
      return false;
    }
    return true;
  }());
  if (not std::isfinite(params.alpha) or not std::isfinite(params.beta) or params.alpha < 0 or params.beta < 0)
  {
    throw std::runtime_error("invalid params");
  }
  return params;
}

}  // namespace ttb