#include "tracking_lib/Misc/GibbsSampler.h"
#include <random>

#include <tracy/tracy/Tracy.hpp>

namespace ttb::gibbs
{

// calculates costs of a given assignment vector, according to cost matrix
double assignment_costs(Matrix const& cost_matrix, Eigen::VectorXi const& assignment)
{
  double cost = 0;
  for (int i = 0; i < cost_matrix.rows(); i++)
  {
    cost += cost_matrix(i, assignment(i));
  }
  return cost;
}

std::pair<Eigen::MatrixXi, Vector> sample_assignments(Matrix const& log_cost_matrix,
                                                      std::size_t numSols,
                                                      Eigen::VectorXi const& initAssignment,
                                                      std::size_t maxTries,
                                                      std::size_t max_consecutive_old_sols)
{
  assert(log_cost_matrix.rows() > 0 and log_cost_matrix.cols() >= log_cost_matrix.rows());
  ZoneScopedN("Gibbs::sample");
  if (numSols == 0)
  {
    return { {}, {} };
  }
  std::mt19937 static thread_local gen(0);  ///< SEED 0
  LOG_DEB("Gibbs sampling");
  if (log_cost_matrix.rows() < 3 and log_cost_matrix.cols() < 10)
  {
    numSols = std::min(
        numSols,
        static_cast<std::size_t>(Eigen::VectorXi::LinSpaced(log_cost_matrix.rows(),
                                                            log_cost_matrix.cols() - log_cost_matrix.rows() + 1,
                                                            log_cost_matrix.cols())
                                     .prod()));
  }
  Matrix const cost_matrix = Eigen::exp(-1 * log_cost_matrix.array());
  Eigen::MatrixXi assignments(log_cost_matrix.rows(), numSols);
  assignments.col(0) = initAssignment;
  Vector costs(numSols);
  costs(0) = assignment_costs(log_cost_matrix, initAssignment);
  std::size_t sol = 1;
  std::size_t tries = 0;
  std::size_t consecutive_old_sols = 0;
  for (; sol < numSols and tries < maxTries and consecutive_old_sols < max_consecutive_old_sols;
       ++tries, ++consecutive_old_sols)
  {
    Eigen::VectorXi const& lastSol = assignments(Eigen::all, sol - 1);
    Eigen::VectorXi newSol(log_cost_matrix.rows());
    for (std::size_t i = 0; i < cost_matrix.rows(); ++i)
    {
      Eigen::RowVectorXd current_costs = cost_matrix.row(i);
      // set already in newSol chosen cols to zero
      current_costs(newSol(Eigen::seq(0, i - 1))).array() = 0.0;
      // set other already chosen cols to zero
      current_costs(lastSol(Eigen::seq(i + 1, log_cost_matrix.rows() - 1))).array() = 0.0;
      std::discrete_distribution<> dist(current_costs.begin(), current_costs.end());
      newSol(i) = dist(gen);
    }
    bool newSolUnique{ [&] {
      for (std::size_t tmp = 0; tmp < sol; ++tmp)
      {
        if (newSol == assignments(Eigen::all, tmp))
        {
          return false;
        }
      }
      return true;
    }() };
    if (newSolUnique)
    {
      consecutive_old_sols = 0;
      costs(sol) = assignment_costs(log_cost_matrix, newSol);
      assignments.col(sol) = std::move(newSol);
      ++sol;
    }
  }
  std::string info_str = "#Assignments requested " + std::to_string(numSols) +
                         "\n#Assignments found: " + std::to_string(sol) + "\n#Tries: " + std::to_string(tries) +
                         "\n#Rows: " + std::to_string(log_cost_matrix.rows()) +
                         "\n#Cols: " + std::to_string(log_cost_matrix.cols());
  ZoneText(info_str.c_str(), info_str.size());
  return { assignments(Eigen::all, Eigen::seq(0, sol - 1)), costs(Eigen::seq(0, sol - 1)) };
}

}  // namespace ttb::gibbs