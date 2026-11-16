#pragma once

#include "tracking_lib/Misc/Assignments.h"

namespace ttb::murty
{

[[nodiscard]] std::vector<Assignment> murty(Cost const& cost, std::size_t num_sols);

[[nodiscard]] std::pair<Eigen::MatrixXi, Vector<double>> getAssignments(Matrix<double> const& costMatrix,
                                                                        unsigned int numSols);

}  // namespace ttb::murty
