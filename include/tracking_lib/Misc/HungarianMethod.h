#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb::hungarian
{

struct Solution
{
  Indices assignment{};
  double cost{};
};

/// Calculates the best assignment for the given cost matrix and returns the assignment with corresponding cost.
/// Every row is assigned exactly one column, where every column can only be assigned once.
/// This means that columns can be not assigned, whereas each row get a column assigned.
/// The solution consists of the assignment of every row, i.e., row i gets assigned to assignment(i) column, and the sum
/// of all costs given this assignment.
///
/// If the cost matrix has more rows than columns, no solution can be found.
/// This is indicated by an assignment that contains -1 and infinite cost.
/// The implementation can deal with infinity costs, this mean that this row/column assignment is not possible.
/// Algorithms for the Assignment and Transportation Problems, James Munkres, Journal of the Society for Industrial and
/// Applied Mathematics 1957 5:1, 32-38 https://doi.org/10.1137/0105003
Solution solve(Matrix const& m);

}  // namespace ttb::hungarian