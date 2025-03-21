#include "tracking_lib/Misc/Numeric.h"

namespace ttb::numeric
{

Vector fold(Vector const& first, Vector const& second)
{
  Vector merged = Vector::Zero(first.rows() + second.rows() - 1);
  for (Index i = 0; i < first.rows(); ++i)
  {
    merged(Eigen::seq(i, i - 1 + second.rows())) += first(i) * second;
  }
  return merged;
}

Matrix fold(Matrix const& first, Matrix const& second)
{
  Matrix merged = Matrix::Zero(first.rows() + second.rows() - 1, first.cols() + second.cols() - 1);
  for (Index i = 0; i < first.rows(); ++i)
  {
    for (Index j = 0; j < first.cols(); ++j)
    {
      auto ind_i = Eigen::seq(i, i - 1 + second.rows());
      auto ind_j = Eigen::seq(j, j - 1 + second.cols());
      merged(ind_i, ind_j) += first(i, j) * second;
    }
  }
  return merged;
}

double logsumexp(Vector const& v)
{
  double val = v.maxCoeff();
  if (std::isinf(-val))
  {
    return -std::numeric_limits<double>::infinity();
  }
  return std::log((v.array() - val).exp().sum()) + val;
}

}  // namespace ttb::numeric