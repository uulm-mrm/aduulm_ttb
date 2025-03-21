#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb::numeric
{

/// fold the two vector together, i.e., result(k) = \sum_{i+j=k} first(i)second(j)
[[nodiscard]] Vector fold(Vector const& first, Vector const& second);
/// fold the two matrices together, i.e., result(i, j) = \sum{f1+f2=i, s1+s2=j} first(f1, s1)second(f2, s2)
[[nodiscard]] Matrix fold(Matrix const& first, Matrix const& second);
/// compute the log of the sum of exponentiated log_values
[[nodiscard]] double logsumexp(Vector const& log_values);
/// compute the log of the sum of exponentiated log_values
template <class T, class Proj>
[[nodiscard]] double logsumexp(T const& log_vals, Proj proj)
{
  auto const max_elem = std::ranges::max_element(log_vals, {}, proj);
  if (max_elem == log_vals.end())
  {
    return -std::numeric_limits<double>::infinity();
  }
  auto const max_val = proj(*max_elem);
  double sum = 0;
  for (auto const& log_val : log_vals)
  {
    sum += std::exp(proj(log_val) - max_val);
  }
  return max_val + std::log(sum);
}

}  // namespace ttb::numeric