#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Misc/Assignments.h"

namespace ttb::gibbs
{

/// Solve the assignment problem with Gibbs sampling
[[nodiscard]] std::pair<MatrixIndex, Vector<double>> sample_assignments(Matrix<double> const& log_cost_matrix,
                                                                        std::size_t numSols,
                                                                        VectorIndex const& initAssignment,
                                                                        std::size_t maxTries,
                                                                        std::size_t max_consecutive_old_sols);
template <class UDT>
[[nodiscard]] std::vector<Assignment> sample(TTBManager<UDT> const& manager,
                                             Cost const& cost,
                                             std::size_t num_sols,
                                             Assignment const& init_assignment,
                                             std::size_t max_tries,
                                             std::size_t max_consecutive_old_sols);

}  // namespace ttb::gibbs