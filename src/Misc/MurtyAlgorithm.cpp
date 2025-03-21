#include "tracking_lib/Misc/MurtyAlgorithm.h"

#include <tracy/tracy/Tracy.hpp>

namespace ttb::murty
{

constexpr const auto tracy_color = tracy::Color::PaleTurquoise;

std::pair<Eigen::MatrixXi, Vector> performMurtyAlgorithm(Matrix const& costMatrix, unsigned int numSols);

std::pair<Eigen::MatrixXi, Vector> getAssignments(Matrix const& costMat, unsigned int k)
{
  ZoneScopedNC("Murty::getAssignments", tracy_color);
  std::string info_str = "#Solutions requested: " + std::to_string(k);
  ZoneText(info_str.c_str(), info_str.size());
  info_str = "#Tracks: " + std::to_string(costMat.rows());
  ZoneText(info_str.c_str(), info_str.size());
  info_str = "#Measurements: " + std::to_string(costMat.cols() - costMat.rows());
  ZoneText(info_str.c_str(), info_str.size());
  assert(costMat.rows() > 0 and costMat.cols() > 0);
  auto [assignments, costs] = performMurtyAlgorithm(costMat, k);
  info_str = "#Solutions computed: " + std::to_string(costs.size());
  ZoneText(info_str.c_str(), info_str.size());
  return { assignments, costs };
}

/// See paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=599256
std::pair<Eigen::MatrixXi, Vector> performMurtyAlgorithm(Matrix const& costMatrix, unsigned int numSols)
{
  // adjust sizes of output matrices
  Eigen::MatrixXi assignments(costMatrix.rows(), numSols);
  Vector costs(numSols);
  auto [current_assignment, cost] = hungarian::solve(costMatrix);
  if (not std::isfinite(cost) or (current_assignment.array() < 0).any() or
      numSols == 0)  // ready since sol is not anymore finite
  {
    return {};
  }
  struct SubProblem
  {
    Matrix costMat;
    Indices sol;
    double cost;
  };
  std::vector<SubProblem> subProblems{ SubProblem{ .costMat = costMatrix, .sol = current_assignment, .cost = cost } };

  for (std::size_t sol = 0; sol < numSols; ++sol)
  {
    if (subProblems.empty())
    {
      return { assignments(Eigen::all, Eigen::seq(0, sol - 1)), costs(Eigen::seq(0, sol - 1)) };
    }
    auto currentProblemIt = std::min_element(subProblems.begin(),
                                             subProblems.end(),
                                             [](SubProblem const& a, SubProblem const& b) { return a.cost < b.cost; });
    assignments(Eigen::all, sol) = currentProblemIt->sol.cast<int>();
    costs(sol) = currentProblemIt->cost;
    Matrix P = currentProblemIt->costMat;
    subProblems.erase(currentProblemIt);
    for (std::size_t i = 0; i < costMatrix.rows(); ++i)
    {
      Matrix nextCostMatrix = P;
      double tmpCost = nextCostMatrix(i, assignments(i, sol));
      nextCostMatrix(i, assignments(i, sol)) = std::numeric_limits<double>::infinity();
      auto [next_assignment, next_cost] = hungarian::solve(nextCostMatrix);
      if (not(next_assignment.array() < 0).any())
      {
        subProblems.push_back(
            SubProblem{ .costMat = std::move(nextCostMatrix), .sol = std::move(next_assignment), .cost = next_cost });
      }
      P(i, Eigen::all).array() = std::numeric_limits<double>::infinity();
      P(i, assignments(i, sol)) = tmpCost;
    }
  }
  return { assignments, costs };
}

}  // namespace ttb::murty
