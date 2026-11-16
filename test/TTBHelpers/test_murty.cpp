#include "gtest/gtest.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Misc/MurtyAlgorithm.h"

#include <iostream>

#include <vector>
#include <chrono>

constexpr double TOL{ 1e-5 };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

double const inf = std::numeric_limits<double>::infinity();

using MatrixXb = Eigen::Matrix<bool, -1, -1>;
using VectorXb = Eigen::Vector<bool, -1>;

/// find true indices
ttb::Indices find(VectorXb const& bools)
{
  ttb::Indices idx = (bools).select(ttb::Indices::LinSpaced(bools.size(), 0, bools.size() - 1),
                                    std::numeric_limits<ttb::Index>::lowest());
  std::ranges::sort(idx, std::greater{});
  idx.conservativeResize(bools.cast<ttb::Index>().sum());
  return idx.reverse();
}

TEST(Murty, all_solutions)
{
  ttb::Matrix<double> costMatrix{ { 1, 0, 3 }, { 0, 1, 2 }, { 6, 1, 0 } };

  ttb::Matrix<double> cost = costMatrix;
  // make positive
  double const minVal = cost.minCoeff();
  cost.array() -= minVal;
  assert((cost.array() >= 0).all());
  // replace infinities
  auto [rankedAss, costs] = ttb::murty::getAssignments(costMatrix, 20);
  ASSERT_TRUE(rankedAss.cols() == 6);
  ASSERT_TRUE(costs.rows() == 6);
}

TEST(Murty, one_solution_possible)
{
  ttb::Matrix<double> costMatrix{ { 1, inf, inf }, { inf, inf, 2 }, { inf, 1, inf } };
  auto [rankedAss, costs] = ttb::murty::getAssignments(costMatrix, 20);
  ASSERT_TRUE(rankedAss.cols() == 1);
  ASSERT_TRUE(costs.rows() == 1);
  ASSERT_TRUE(costs(0) == 4);
}

TEST(Murty, no_solution_possible)
{
  ttb::Matrix<double> costMatrix{ { 1, inf, inf }, { inf, inf, inf }, { inf, 1, inf } };
  auto [rankedAss, costs] = ttb::murty::getAssignments(costMatrix, 20);
  ASSERT_TRUE(rankedAss.cols() == 0);
  ASSERT_TRUE(costs.rows() == 0);
}

TEST(Murty, more_tracks_as_meas)
{
  ttb::Matrix<double> costMatrix{ { 1, 2 }, { 4, 7 }, { 6, 1 } };
  auto [rankedAss, costs] = ttb::murty::getAssignments(costMatrix, 20);
  ASSERT_TRUE(rankedAss.cols() == 0);
  ASSERT_TRUE(costs.rows() == 0);
}

TEST(Murty, more_tracks_as_meas_zero_filled)
{
  ttb::Matrix<double> costMatrix{ { 1, 2 }, { 4, 7 }, { 6, 1 } };
  costMatrix.transposeInPlace();
  auto [rankedAss, costs] = ttb::murty::getAssignments(costMatrix, 20);
  ASSERT_TRUE(rankedAss.cols() == 6);
  ASSERT_TRUE(costs.rows() == 6);
}