#include "tracking_lib/Misc/MurtyAlgorithm.h"
#include "tracking_lib/Misc/HungarianMethod.h"

#include <ranges>
#include <tracy/tracy/Tracy.hpp>

namespace ttb::murty
{

std::vector<Assignment> murty(Cost const& cost, std::size_t num_sols)
{
  assert(cost.size() > 0);
  assert([&] {
    if (cost.front().alive.size() != 1)
    {
      LOG_FATAL("Our murty alog can only solve 2D problems .....");
      return false;
    }
    return true;
  }());
  bool const joint_prediction_and_update =
      std::ranges::any_of(cost, [](StateLikelihood const& state_cost) { return state_cost.surv_likelihood < 1; });
  std::unordered_map<MeasurementId, Index> meas_index;
  Index current_meas_id = 0;
  for (StateLikelihood const& state_cost : cost)
  {
    assert(state_cost.alive.size() == 1);
    for (MeasurementId const& meas_id : state_cost.alive.front().second | std::views::keys)
    {
      if (meas_id != NOT_DETECTED)
      {
        if (auto const meas_it = meas_index.find(meas_id); meas_it == meas_index.end())
        {
          meas_index[meas_id] = current_meas_id;
          ++current_meas_id;
        }
      }
    }
  }
  Index const num_meas = meas_index.size();
  assert(num_meas >= 0);
  Matrix<double> cost_matrix{ [&] {
    if (joint_prediction_and_update)
    {
      return std::numeric_limits<double>::infinity() * Matrix<double>::Ones(cost.size(), num_meas + 2 * cost.size());
    }
    return std::numeric_limits<double>::infinity() * Matrix<double>::Ones(cost.size(), num_meas + cost.size());
  }() };
  for (std::size_t i = 0; i < cost.size(); ++i)
  {
    // meas cost
    for (std::size_t j = 0; j < cost.at(i).alive.front().second.size() - 1; ++j)
    {
      assert(cost.at(i).alive.front().second.at(j).first != NOT_DETECTED);
      cost_matrix(i, meas_index.at(cost.at(i).alive.front().second.at(j).first)) =
          -std::log(cost.at(i).alive.front().second.at(j).second);
    }
    // not detected
    assert(cost.at(i).alive.front().second.back().first == NOT_DETECTED);
    cost_matrix(i, i + num_meas) = -std::log(cost.at(i).alive.front().second.back().second);
    if (joint_prediction_and_update)
    {
      cost_matrix(i, i + num_meas + cost.size()) = -std::log(1 - cost.at(i).surv_likelihood);
    }
  }
  auto [assignments, costs] = getAssignments(cost_matrix, num_sols);
  std::vector<Assignment> assign;
  for (Index solI = 0; solI < assignments.cols(); solI++)
  {
    Assignment assi;
    for (auto const& [n, track] : std::views::enumerate(cost))
    {
      Index const measIndex = assignments(n, solI);
      if (measIndex >= num_meas + cost.size())  // track does not exist
      {
        assi.push_back(StateAssignment{ .id = track.id, .alive = false });
        continue;
      }
      MeasurementId const measId{ [&]() {
        if (measIndex < num_meas)  // track detected
        {
          auto const meas_it =
              std::ranges::find_if(meas_index, [&](auto const& meas_id_id) { return meas_id_id.second == measIndex; });
          assert(meas_it != meas_index.end());
          return meas_it->first;
        }
        return NOT_DETECTED;
      }() };
      assi.push_back(StateAssignment{
          .id = track.id, .alive = true, .meas_assignments = { { cost.front().alive.front().first, measId } } });
    }
    assert(is_valid(assi));
    assign.push_back(std::move(assi));
  }
  return assign;
}

/// See paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=599256
std::pair<Eigen::MatrixXi, Vector<double>> getAssignments(Matrix<double> const& costMatrix, unsigned int numSols)
{
  ZoneScopedN("Assignments::murty");
  assert(costMatrix.rows() > 0 and costMatrix.cols() > 0);
  // adjust sizes of output matrices
  Eigen::MatrixXi assignments(costMatrix.rows(), numSols);
  Vector<double> costs(numSols);
  auto [current_assignment, cost] = hungarian::solve(costMatrix);
  if (not std::isfinite(cost) or (current_assignment.array() < 0).any() or
      numSols == 0)  // ready since sol is not anymore finite
  {
    return {};
  }
  struct SubProblem
  {
    Matrix<double> costMat;
    Indices sol;
    double cost;
  };
  std::vector subProblems{ SubProblem{ .costMat = costMatrix, .sol = current_assignment, .cost = cost } };

  for (std::size_t sol = 0; sol < numSols; ++sol)
  {
    if (subProblems.empty())
    {
      return { assignments(Eigen::all, Eigen::seq(0, sol - 1)), costs(Eigen::seq(0, sol - 1)) };
    }
    auto currentProblemIt =
        std::ranges::min_element(subProblems, [](SubProblem const& a, SubProblem const& b) { return a.cost < b.cost; });
    assignments(Eigen::all, sol) = currentProblemIt->sol.cast<int>();
    costs(sol) = currentProblemIt->cost;
    if (sol + 1 < numSols)
    {
      Matrix<double> P = currentProblemIt->costMat;
      subProblems.erase(currentProblemIt);
      for (std::size_t i = 0; i < costMatrix.rows(); ++i)
      {
        Matrix<double> nextCostMatrix = P;
        double const tmpCost = nextCostMatrix(i, assignments(i, sol));
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
  }
  return { assignments, costs };
}
}  // namespace ttb::murty
