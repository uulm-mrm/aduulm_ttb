#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

#include <tracy/tracy/Tracy.hpp>

namespace ttb::numeric
{

/// compute the log of the sum of exponentiated log_values
template <std::ranges::range T, class Proj = std::identity>
[[nodiscard]] auto logsumexp(T const& log_vals, Proj proj = {})
    -> std::remove_cvref_t<std::invoke_result_t<Proj&, std::ranges::range_reference_t<T const&>>>
{
  using value_type = std::remove_cvref_t<std::invoke_result_t<Proj&, std::ranges::range_reference_t<T const&>>>;
  auto const max_elem = std::ranges::max_element(log_vals, {}, proj);
  if (max_elem == log_vals.end())
  {
    return -std::numeric_limits<double>::infinity();
  }
  value_type const max_val = std::invoke(proj, *max_elem);
  if (max_val == -std::numeric_limits<double>::infinity())
  {
    return -std::numeric_limits<double>::infinity();
  }
  value_type sum = 0;
  for (auto const& log_val : log_vals)
  {
    sum += exp(std::invoke(proj, log_val) - max_val);
  }
  return max_val + log(sum);
}

struct DefaultOpAdd
{
  auto operator()(auto const& first, auto const& second)
  {
    return first + second;
  }
};
struct DefaultCombineFirstSecond
{
  auto operator()(auto const& first, auto const& second)
  {
    return first * second;
  }
};

/// fold the two vector together, i.e., result(k) = \sum_{i+j=k} first(i)second(j)
template <class UDT>
[[nodiscard]] Vector<UDT> fold(Vector<UDT> const& first, Vector<UDT> const& second);

/// fold two vectors or matrices together
template <class UDT>
[[nodiscard]]
Matrix<UDT> fold(Matrix<UDT> const& first, Matrix<UDT> const& second)
{
  ZoneScopedNS("Numeric::fold_matrix", 10);
  assert([&] {  // NOLINT
    if (first.size() == 0 or second.size() == 0)
    {
      LOG_FATAL("empty input");
      return false;
    }
    return true;
  }());
  Matrix<UDT> merged = Matrix<UDT>::Zero(first.rows() + second.rows() - 1, first.cols() + second.cols() - 1);
  for (Index i = 0; i < first.rows(); ++i)
  {
    for (Index j = 0; j < first.cols(); ++j)
    {
      auto ind_i = Eigen::seq(i, i - 1 + second.rows());
      auto ind_j = Eigen::seq(j, j - 1 + second.cols());
      merged(ind_i, ind_j) += first(i, j) * second;
    }
  }
  Index max_rows = 0;
  Index max_cols = 0;
  for (Index i = 0; i < merged.rows(); ++i)
  {
    for (Index j = 0; j < merged.cols(); ++j)
    {
      if (merged(i, j) > 0)
      {
        max_rows = std::max(max_rows, i);
        max_cols = std::max(max_cols, j);
      }
    }
  }
  return merged(Eigen::seq(0, max_rows), Eigen::seq(0, max_cols));
}

/// fold two vectors or matrices together
template <class UDT, class T = Matrix<UDT>, class Op_fs = DefaultCombineFirstSecond, class Op_add = DefaultOpAdd>
  requires requires(Op_fs combine_first_second, Op_add op_add, double first, Matrix<UDT> second) {
    { combine_first_second(first, second) } -> std::convertible_to<Matrix<UDT>>;
    { op_add(second, second) } -> std::convertible_to<Matrix<UDT>>;
  }
[[nodiscard]] T
fold(T const& first, T const& second, double init_val = 0, Op_fs combine_first_second = {}, Op_add op_add = {})
{
  ZoneScopedNS("Numeric::fold_matrix", 10);
  assert([&] {  // NOLINT
    if (first.size() == 0 or second.size() == 0)
    {
      LOG_FATAL("empty input");
      return false;
    }
    return true;
  }());
  T merged = init_val * T::Ones(first.rows() + second.rows() - 1, first.cols() + second.cols() - 1);
  for (Index i = 0; i < first.rows(); ++i)
  {
    for (Index j = 0; j < first.cols(); ++j)
    {
      auto ind_i = Eigen::seq(i, i - 1 + second.rows());
      auto ind_j = Eigen::seq(j, j - 1 + second.cols());
      merged(ind_i, ind_j) = op_add(merged(ind_i, ind_j), combine_first_second(first(i, j), second));
    }
  }
  Index max_rows = 0;
  Index max_cols = 0;
  for (Index i = 0; i < merged.rows(); ++i)
  {
    for (Index j = 0; j < merged.cols(); ++j)
    {
      if (merged(i, j) > init_val)
      {
        max_rows = std::max(max_rows, i);
        max_cols = std::max(max_cols, j);
      }
    }
  }
  if constexpr (std::is_same_v<T, Matrix<UDT>>)
  {
    return merged(Eigen::seq(0, max_rows), Eigen::seq(0, max_cols));
  }
  if constexpr (std::is_same_v<T, Vector<UDT>>)
  {
    return merged(Eigen::seq(0, max_rows));
  }
  assert(false);
  std::unreachable();
}

template <class UDT, class T = Matrix<UDT>>
  requires std::is_same_v<T, Matrix<UDT>> or std::is_same_v<T, Vector<UDT>>
[[nodiscard]] T fold_log_values(T const& a, T const& b)
{
  return numeric::fold<T>(
      a,
      b,
      -std::numeric_limits<double>::infinity(),
      [](double first, T const& second) -> T { return first + second.array(); },
      [](T const& first, T const& second) {
        assert(first.rows() == second.rows());
        T result = [&] {
          if constexpr (std::is_same_v<T, Matrix<UDT>>)
          {
            return Matrix<UDT>(first.rows(), first.cols());
          }
          else
          {
            return Vector<UDT>(first.rows());
          }
        }();
        for (Index i = 0; i < first.rows(); ++i)
        {
          if constexpr (std::same_as<T, Matrix<UDT>>)
          {
            for (Index j = 0; j < first.cols(); ++j)
            {
              result(i, j) = numeric::logsumexp(std::vector{ first(i, j), second(i, j) });
            }
          }
          else
          {
            result(i) = numeric::logsumexp(std::vector{ first(i), second(i) });
          }
        }
        return result;
      });
}

}  // namespace ttb::numeric
