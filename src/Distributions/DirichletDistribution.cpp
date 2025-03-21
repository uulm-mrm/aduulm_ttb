#include "tracking_lib/Distributions/DirichletDistribution.h"

namespace ttb
{

DirichletDistribution::DirichletDistribution(Vector alpha) : _alpha{ std::move(alpha) }
{
  if ((_alpha.array() <= 0).any())
  {
    throw std::runtime_error("Invalid Parameter Vector for a Dirichlet Distribution. Minimal value of alpha=" +
                             std::to_string(_alpha.minCoeff()));
  }
}

Vector DirichletDistribution::mean() const
{
  return _alpha / _alpha.sum();
}

Matrix DirichletDistribution::variance() const
{
  Vector an = _alpha / _alpha.sum();

  Matrix var = (an.array() * (1 - an.array()) / (_alpha.sum() + 1)).matrix().asDiagonal();

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

}  // namespace ttb
