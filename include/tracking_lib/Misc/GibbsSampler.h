#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb::gibbs
{

/// Solve the assignment problem with Gibbs sampling
[[nodiscard]] std::pair<Eigen::MatrixXi, Vector> sample_assignments(Matrix const& log_cost_matrix,
                                                                    std::size_t numSols,
                                                                    Eigen::VectorXi const& initAssignment,
                                                                    std::size_t maxTries,
                                                                    std::size_t max_consecutive_old_sols);

}  // namespace ttb::gibbs