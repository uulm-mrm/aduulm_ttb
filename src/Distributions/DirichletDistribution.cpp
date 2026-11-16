#include "tracking_lib/Distributions/DirichletDistribution.h"

namespace ttb
{

DirichletDistribution::DirichletDistribution(Vector<double> alpha) : _alpha{ std::move(alpha) }
{
  if ((_alpha.array() <= 0).any())
  {
    throw std::runtime_error("Invalid Parameter Vector for a Dirichlet Distribution. Minimal value of alpha=" +
                             std::to_string(_alpha.minCoeff()));
  }
}

Vector<double> DirichletDistribution::mean() const
{
  return _alpha / _alpha.sum();
}

Matrix<double> DirichletDistribution::variance() const
{
  Vector<double> an = _alpha / _alpha.sum();

  Matrix<double> var = (an.array() * (1 - an.array()) / (_alpha.sum() + 1)).matrix().asDiagonal();

  for (Index i = 0; i < _alpha.size(); ++i)
  {
    for (Index j = 0; j < _alpha.size(); ++j)
    {
      if (i == j)
      {
        continue;
      }
      var(i, j) = -an(i) * an(j) / (_alpha.sum() + 1);
    }
  }
  return var;
}

std::string DirichletDistribution::toString() const
{
  std::stringstream ss{ "\"Dirichlet Distribution with alpha= " };
  ss << _alpha;
  return ss.str();
}

namespace BetaDistribution
{
Params from_mean_var(double mean, double var)
{
  Params const params{ .p = mean * (mean - var - mean * mean) / var,
                       .q = (mean - 1) * (var + mean * mean - mean) / var };
  assert([&] {
    if (not std::isfinite(params.p) or not std::isfinite(params.q) or params.p < 0 or params.q < 0)
    {
      LOG_FATAL("Invalid beta distribution paramters");
      LOG_FATAL(params.p);
      LOG_FATAL(params.q);
      LOG_FATAL(mean);
      LOG_FATAL(var);
      return false;
    }
    DirichletDistribution dir(Vector<double>{ { params.p, params.q } });
    if (std::abs(dir.mean()(0) - mean) > 1e-7 or std::abs(dir.variance()(0, 0) - var) > 1e-7)
    {
      LOG_FATAL("mean and variance are not correct");
      LOG_FATAL(params.p);
      LOG_FATAL(params.q);
      LOG_FATAL(mean);
      LOG_FATAL(var);
      return false;
    }
    return true;
  }());
  return params;
}
}  // namespace BetaDistribution

}  // namespace ttb
