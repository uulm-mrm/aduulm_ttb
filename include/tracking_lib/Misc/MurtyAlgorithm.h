#pragma once

#include "tracking_lib/Misc/HungarianMethod.h"

namespace ttb::murty
{

/// Calculates the k-best assignments (with minimal costs) for the given cost matrix and returns assignments with
/// corresponding costs. costMatr: Track (Rows) -> Measurement (Cols) assignment costs \returns Matrix A where each
/// column is an assignment, i.e., Track of row i -> Measurement A(i, j)
[[nodiscard]] std::pair<Eigen::MatrixXi, Vector> getAssignments(Matrix const& costMat, unsigned int k);

}  // namespace ttb::murty
