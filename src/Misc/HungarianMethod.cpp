#include "tracking_lib/Misc/HungarianMethod.h"

#include <tracy/tracy/Tracy.hpp>

namespace ttb::hungarian
{

constexpr auto tracy_color = tracy::Color::PaleTurquoise;

using MatrixXb = Eigen::Matrix<bool, -1, -1>;
using VectorXb = Eigen::Vector<bool, -1>;

Solution hungarian(Matrix const& cost_in);

Solution solve(Matrix const& m)
{
  ZoneScopedNC("Hungarian", tracy_color);
  return hungarian(m);
}

/// find true indices
Indices find(VectorXb const& bools)
{
  Indices idx =
      (bools).select(Indices::LinSpaced(bools.size(), 0, bools.size() - 1), std::numeric_limits<Index>::lowest());
  std::ranges::sort(idx, std::greater{});
  idx.conservativeResize(bools.cast<Index>().sum());
  return idx.reverse();
}

void computeassignmentcost(Matrix const& cost, Solution& solution);

void step2a(Matrix& cost,
            Solution& solution,
            MatrixXb& star,
            MatrixXb& prime,
            VectorXb& covered_cols,
            VectorXb& covered_rows);

void step2b(Matrix& cost,
            Solution& solution,
            MatrixXb& star,
            MatrixXb& prime,
            VectorXb& covered_cols,
            VectorXb& covered_rows);

void step3(Matrix& cost,
           Solution& solution,
           MatrixXb& star,
           MatrixXb& prime,
           VectorXb& covered_cols,
           VectorXb& covered_rows);

void step4(Index row,
           Index col,
           Matrix& cost,
           Solution& solution,
           MatrixXb& star,
           MatrixXb& prime,
           VectorXb& covered_cols,
           VectorXb& covered_rows);

void step5(Matrix& cost,
           Solution& solution,
           MatrixXb& star,
           MatrixXb& prime,
           VectorXb& covered_cols,
           VectorXb& covered_rows);

Solution hungarian(Matrix const& cost_in)
{
  Solution solution{ .assignment = Indices::Constant(cost_in.rows(), -1),
                     .cost = std::numeric_limits<double>::infinity() };
  if (cost_in.array().isInf().all() or cost_in.rows() > cost_in.cols())
  {
    return solution;
  }
  Matrix cost = cost_in;
  // make positive
  double const minVal = cost.minCoeff();
  cost.array() -= minVal;
  assert((cost.array() >= 0).all());
  // replace infinities
  double const _max_finite_value = cost_in.reshaped()(find(cost_in.reshaped().array().isFinite())).maxCoeff();
  double const inf_value = std::max(_max_finite_value + 42.0, 42.0);
  cost.reshaped().array()(find(cost.reshaped().array().isInf())) = inf_value;

  VectorXb covered_cols = VectorXb::Zero(cost_in.cols());
  VectorXb covered_rows = VectorXb::Zero(cost_in.rows());
  MatrixXb star = MatrixXb::Zero(cost_in.rows(), cost_in.cols());
  MatrixXb prime = MatrixXb::Zero(cost_in.rows(), cost_in.cols());

  cost.colwise() -= cost.rowwise().minCoeff();

  for (Index row = 0; row < cost.rows(); ++row)
  {
    for (Index col = 0; col < cost.cols(); ++col)
    {
      if (cost(row, col) == 0)
      {
        if (not covered_cols(col))
        {
          star(row, col) = true;
          covered_cols(col) = true;
          break;
        }
      }
    }
  }
  step2b(cost, solution, star, prime, covered_cols, covered_rows);
  computeassignmentcost(cost_in, solution);
  return solution;
}

/********************************************************/
void buildassignmentvector(MatrixXb const& _starMatrix, Solution& solution)
{
  for (Index row = 0; row < _starMatrix.rows(); ++row)
  {
    Index col;
    bool found = _starMatrix.row(row).maxCoeff(&col);
    if (found)
    {
      solution.assignment(row) = col;
    }
  }
}

/********************************************************/
void computeassignmentcost(Matrix const& cost, Solution& solution)
{
  if ((solution.assignment.array() < 0).any())
  {
    solution.assignment = Indices::Constant(solution.assignment.size(), -1);
    solution.cost = std::numeric_limits<double>::infinity();
    return;
  }
  solution.cost = 0;
  for (Index row = 0; row < cost.rows(); ++row)
  {
    Index const col = solution.assignment(row);
    if (not std::isfinite(cost(row, col)))
    {
      solution.assignment = Indices::Constant(solution.assignment.size(), -1);
      solution.cost = std::numeric_limits<double>::infinity();
      return;
    }
    solution.cost += cost(row, col);
  }
}

/********************************************************/
void step2a(Matrix& cost,
            Solution& solution,
            MatrixXb& star,
            MatrixXb& prime,
            VectorXb& covered_cols,
            VectorXb& covered_rows)
{
  for (Index col = 0; col < cost.cols(); ++col)
  {
    if (star.col(col).array().any())
    {
      covered_cols(col) = true;
    }
  }
  step2b(cost, solution, star, prime, covered_cols, covered_rows);
}

/********************************************************/
void step2b(Matrix& cost,
            Solution& solution,
            MatrixXb& star,
            MatrixXb& prime,
            VectorXb& covered_cols,
            VectorXb& covered_rows)
{
  if (covered_cols.cast<Index>().sum() == std::min(cost.rows(), cost.cols()))
  {
    buildassignmentvector(star, solution);
  }
  else
  {
    step3(cost, solution, star, prime, covered_cols, covered_rows);
  }
}

/********************************************************/
void step3(Matrix& cost,
           Solution& solution,
           MatrixXb& star,
           MatrixXb& prime,
           VectorXb& covered_cols,
           VectorXb& covered_rows)
{
  bool zerosFound = true;
  while (zerosFound)
  {
    zerosFound = false;
    for (Index col = 0; col < cost.cols(); ++col)
    {
      if (not covered_cols(col))
      {
        for (Index row = 0; row < cost.rows(); ++row)
        {
          if (not covered_rows(row) and cost(row, col) == 0)
          {
            prime(row, col) = true;
            Index star_col;
            bool found = star.row(row).maxCoeff(&star_col);
            if (not found) /* no starred zero found */
            {
              step4(row, col, cost, solution, star, prime, covered_cols, covered_rows);
              return;
            }
            covered_rows(row) = true;
            covered_cols(star_col) = false;
            zerosFound = true;
            break;
          }
        }
      }
    }
  }
  step5(cost, solution, star, prime, covered_cols, covered_rows);
}

/********************************************************/
void step4(Index row,
           Index col,
           Matrix& cost,
           Solution& solution,
           MatrixXb& star,
           MatrixXb& prime,
           VectorXb& covered_cols,
           VectorXb& covered_rows)
{
  MatrixXb new_star = star;
  new_star(row, col) = true;
  Index star_row;
  Index star_col = col;
  bool found = star.col(star_col).maxCoeff(&star_row);
  while (found)
  {
    new_star(star_row, star_col) = false;
    Index prime_row = star_row;
    Index prime_col;
    prime.row(prime_row).maxCoeff(&prime_col);
    assert(prime.row(prime_row).maxCoeff(&prime_col));  // assert found_col = prime.row(prime_row).maxCoeff(&prime_col)
    new_star(prime_row, prime_col) = true;
    star_col = prime_col;
    found = star.col(star_col).maxCoeff(&star_row);
  }
  prime.setConstant(false);
  star = new_star;
  covered_rows.setConstant(false);
  step2a(cost, solution, star, prime, covered_cols, covered_rows);
}

/********************************************************/
void step5(Matrix& cost,
           Solution& solution,
           MatrixXb& star,
           MatrixXb& prime,
           VectorXb& covered_cols,
           VectorXb& covered_rows)
{
  auto const uncovered_rows = find(covered_rows.array() == false);
  auto const uncovered_cols = find(covered_cols.array() == false);
  double const h = [&] {
    if (uncovered_rows.size() > 0 and uncovered_cols.size() > 0)
    {
      return cost(uncovered_rows, uncovered_cols).minCoeff();
    }
    return std::numeric_limits<double>::infinity();
  }();
  cost(find(covered_rows), Eigen::all).array() += h;
  cost(Eigen::all, uncovered_cols).array() -= h;
  step3(cost, solution, star, prime, covered_cols, covered_rows);
}

}  // namespace ttb::hungarian
