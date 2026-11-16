#include "tracking_lib/Misc/GibbsSampler.h"
#include <random>
#include <ranges>
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/States/State.h"
#include <tracy/tracy/Tracy.hpp>

namespace ttb::gibbs
{

// calculates costs of a given assignment vector, according to cost matrix
double assignment_costs(Matrix<double> const& cost_matrix, VectorIndex const& assignment)
{
  double cost = 0;
  for (int i = 0; i < cost_matrix.rows(); i++)
  {
    cost += cost_matrix(i, assignment(i));
  }
  return cost;
}

std::pair<MatrixIndex, Vector<double>> sample_assignments(Matrix<double> const& log_cost_matrix,
                                                          std::size_t numSols,
                                                          VectorIndex const& initAssignment,
                                                          std::size_t maxTries,
                                                          std::size_t max_consecutive_old_sols)
{
  assert(log_cost_matrix.rows() > 0 and log_cost_matrix.cols() >= log_cost_matrix.rows());
  if (numSols == 0)
  {
    return { {}, {} };
  }
  std::mt19937 thread_local gen(0);  ///< SEED 0
  LOG_DEB("Gibbs sampling");
  if (log_cost_matrix.rows() < 3 and log_cost_matrix.cols() < 10)
  {
    numSols =
        std::min(numSols,
                 static_cast<std::size_t>(VectorIndex::LinSpaced(log_cost_matrix.rows(),
                                                                 log_cost_matrix.cols() - log_cost_matrix.rows() + 1,
                                                                 log_cost_matrix.cols())
                                              .prod()));
  }
  Matrix<double> const cost_matrix = Eigen::exp(-1 * log_cost_matrix.array());
  MatrixIndex assignments(log_cost_matrix.rows(), numSols);
  assignments.col(0) = initAssignment;
  Vector<double> costs(numSols);
  costs(0) = assignment_costs(log_cost_matrix, initAssignment);
  std::size_t sol = 1;
  std::size_t tries = 0;
  std::size_t consecutive_old_sols = 0;
  for (; sol < numSols and tries < maxTries and consecutive_old_sols < max_consecutive_old_sols;
       ++tries, ++consecutive_old_sols)
  {
    VectorIndex const& lastSol = assignments(Eigen::all, sol - 1);
    VectorIndex newSol(log_cost_matrix.rows());
    for (Index i = 0; i < cost_matrix.rows(); ++i)
    {
      RowVector current_costs = cost_matrix.row(i);
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
      assignments.col(sol) = newSol;
      ++sol;
    }
  }
  return { assignments(Eigen::all, Eigen::seq(0, sol - 1)), costs(Eigen::seq(0, sol - 1)) };
}

// the used measurements of the first n-1 states for the first_assignment and the n+1 last states of the
// seconto_string(d assignment
std::unordered_set<MeasurementId> used_measurement(Index n,
                                                   Assignment const& first_assignment,
                                                   Assignment const& second_assignment)
{
  assert([&] {
    if (n > static_cast<Index>(first_assignment.size()))
    {
      LOG_FATAL(std::format("n {} bigger than the first assignment {}", n, first_assignment.size()));
      LOG_FATAL(std::format("first assignment\n{}", to_string(first_assignment)));
      LOG_FATAL(std::format("second assignment\n{}", to_string(second_assignment)));
      return false;
    }
    if (first_assignment.size() > second_assignment.size())
    {
      LOG_FATAL("second assignment is smaller than the first. Must not happen, because the second should represent the "
                "previous, complete assignment");
      LOG_FATAL(std::format("first assignment\n{}", to_string(first_assignment)));
      LOG_FATAL(std::format("second assignment\n{}", to_string(second_assignment)));
      return false;
    }
    for (auto const& [first, second] : std::views::zip(first_assignment, second_assignment))
    {
      if (first.id != second.id)
      {
        LOG_FATAL("different order of state ids...");
        return false;
      }
    }
    return true;
  }());
  std::unordered_set<MeasurementId> used_meas;
  for (Index i = 0; i <= n - 1; ++i)
  {
    for (auto const& [source_id, meas] : first_assignment.at(i).meas_assignments)
    {
      if (meas != NOT_DETECTED)
      {
        used_meas.insert(meas);
      }
    }
  }
  for (Index i = n + 1; i < static_cast<Index>(second_assignment.size()); ++i)
  {
    for (auto const& [source_id, meas] : second_assignment.at(i).meas_assignments)
    {
      if (meas != NOT_DETECTED)
      {
        used_meas.insert(meas);
      }
    }
  }
  return used_meas;
}

// the used measurements of the first n-1 states for the first_assignment and the n+1 last states of the
// seconto_string(d assignment
std::unordered_set<MeasurementId> used_measurement(Assignment const& assignment)
{
  std::unordered_set<MeasurementId> used_meas;
  for (StateAssignment const& state_assignment : assignment)
  {
    for (auto const& [source_id, meas] : state_assignment.meas_assignments)
    {
      if (meas != NOT_DETECTED)
      {
        used_meas.insert(meas);
      }
    }
  }
  return used_meas;
}

std::vector<std::pair<StateId, double>> state_alive(Cost const& cost, Assignment const& assignment)
{
  std::vector<std::pair<StateId, double>> alive_num;
  std::vector<std::pair<StateId, std::vector<std::pair<DataSourceContainerId, double>>>> PSI;

  for (auto const& [n, state_cost] : std::views::enumerate(cost))
  {
    auto const used_meas = used_measurement(n, assignment, assignment);
    alive_num.emplace_back(state_cost.id, 1);
    PSI.emplace_back(state_cost.id, std::vector<std::pair<DataSourceContainerId, double>>{});
    for (auto const& [source_id, meas_assignments] : state_cost.alive)
    {
      PSI.back().second.emplace_back(source_id, 0);
      for (auto const& [meas, prob] : meas_assignments)
      {
        if (meas == NOT_DETECTED or not used_meas.contains(meas))  // can assign this meas
        {
          PSI.back().second.back().second += prob;
        }
      }
    }
  }
  for (std::size_t i = 0; i < alive_num.size(); ++i)
  {
    assert(alive_num.at(i).first == PSI.at(i).first);
    for (auto const& [source, prob] : PSI.at(i).second)
    {
      alive_num.at(i).second *= prob;
    }
  }
  for (std::size_t i = 0; i < alive_num.size(); ++i)
  {
    assert(cost.at(i).id == alive_num.at(i).first);
    alive_num.at(i).second /= 1 - cost.at(i).surv_likelihood + alive_num.at(i).second;
    alive_num.at(i).second = cost.at(i).surv_likelihood;
  }
  return alive_num;
}

/// return the next assignment for the flexible state and source, all assignments below the flexible state are not fixed
/// yet
void next_assignment_rec(Cost const& cost,
                         Assignment const& current,
                         std::size_t current_state,
                         bool current_alive,
                         std::size_t current_source,
                         std::vector<Assignment>& all_assignments)
{
  if (current_state == cost.size())
  {  // processed all states
    assert(is_valid(current));
    all_assignments.push_back(current);
    return;
  }
  if (not current_alive)  // this state is not alive -> go down
  {
    if (cost.at(current_state).surv_likelihood < 1)
    {
      Assignment next = current;
      next.push_back(StateAssignment{ .id = cost.at(current_state).id, .alive = false, .meas_assignments = {} });
      next_assignment_rec(cost, next, current_state + 1, false, 0, all_assignments);
    }
    // make this alive
    Assignment alive_next = current;
    alive_next.push_back(StateAssignment{ .id = cost.at(current_state).id, .alive = true, .meas_assignments = {} });
    next_assignment_rec(cost, alive_next, current_state, true, 0, all_assignments);
    return;
  }
  // this state alive
  if (current_source == cost.at(current_state).alive.size())  // processed all sources -> go to next state
  {
    next_assignment_rec(cost, current, current_state + 1, false, 0, all_assignments);
    return;
  }
  std::unordered_set<MeasurementId> const used = used_measurement(current);
  for (auto const& [meas_id, weight] : cost.at(current_state).alive.at(current_source).second)
  {
    if (not used.contains(meas_id))
    {
      Assignment next_meas = current;
      next_meas.back().meas_assignments.emplace_back(cost.at(current_state).alive.at(current_source).first, meas_id);
      next_assignment_rec(cost, next_meas, current_state, true, current_source + 1, all_assignments);
    }
  }
}

template std::vector<Assignment> sample(TTBManager<double> const& manager,
                                        Cost const& cost,
                                        std::size_t num_sols,
                                        Assignment const& init_assignment,
                                        std::size_t max_tries,
                                        std::size_t max_consecutive_old_sols);
template std::vector<Assignment> sample(TTBManager<AD> const& manager,
                                        Cost const& cost,
                                        std::size_t num_sols,
                                        Assignment const& init_assignment,
                                        std::size_t max_tries,
                                        std::size_t max_consecutive_old_sols);
template <class UDT>
std::vector<Assignment> sample(TTBManager<UDT> const& manager,
                               Cost const& cost,
                               std::size_t num_sols,
                               Assignment const& init_assignment,
                               std::size_t max_tries,
                               std::size_t max_consecutive_old_sols)
{
  ZoneScopedN("Gibbs");
  if (num_sols == 0)
  {
    return {};
  }
  /// maximum number of possible assignments
  /// for this estimation we assume the states and measurements are completly independent
  auto const [poss_assignments, num_meas] = [&] {
    double possibilities = 1;
    std::size_t meas = 0;
    for (StateLikelihood const& state_cost : cost)
    {
      double meas_possibilities = 1;
      for (auto const& [data_source, ass] : state_cost.alive)
      {
        meas += ass.size();
        meas_possibilities *= ass.size();
      }
      if (state_cost.surv_likelihood < 1)
      {
        meas_possibilities += 1;
      }
      possibilities *= meas_possibilities;
    }
    return std::pair{ possibilities, meas };
  }();

  if (poss_assignments < 1e3)
  {
    ZoneScopedN("Gibbs::enumerate");
    std::vector<Assignment> all_assignments;
    next_assignment_rec(cost, {}, 0, false, 0, all_assignments);
    if (all_assignments.size() <= num_sols)
    {
      return all_assignments;
    }
    std::vector<std::pair<std::size_t, double>> costs;
    double sum = 0;
    for (auto const& [ind, assi] : std::views::enumerate(all_assignments))
    {
      double const assi_cost = cost_assignment<double>(cost, assi);
      sum += assi_cost;
      costs.emplace_back(ind, assi_cost);
    }
    std::ranges::sort(costs, std::greater{}, &std::pair<std::size_t, double>::second);
    std::vector<Assignment> best_assignments;
    double sum_best = 0;
    for (auto const& [i, assi_cost] : costs | std::views::take(num_sols))
    {
      sum_best += assi_cost;
      best_assignments.push_back(std::move(all_assignments.at(i)));
    }
    LOG_DEB(std::format("Did not process all possible assocs: {} / {} with weight {} / {} rel {}",
                        num_sols,
                        all_assignments.size(),
                        sum_best,
                        sum,
                        sum_best / sum));
    return best_assignments;
  }
  {
    ZoneScopedN("Gibbs::sample");
    std::mt19937 gen(0);
    assert(is_valid(init_assignment));
    assert([&] {
      if (init_assignment.size() != cost.size())
      {
        LOG_FATAL("different size of init sol and cost");
        LOG_FATAL(to_string(init_assignment));
        LOG_FATAL(to_string(cost));
        return false;
      }
      return true;
    }());
    std::size_t const chain_length =
        10 * num_sols;  // restart the gibbs chain at the initial solution after reaching this length
    std::vector<Assignment> assignments{ init_assignment };
    Assignment current_assignment = init_assignment;
    std::size_t tries = 1;
    std::size_t consecutive_old_sols = 0;
    std::size_t chain_tries = 1;
    for (; tries < max_tries and assignments.size() < std::min(num_sols, static_cast<std::size_t>(poss_assignments)) and
           consecutive_old_sols < max_consecutive_old_sols;
         ++tries, ++consecutive_old_sols, ++chain_tries)
    {
      std::vector<std::pair<StateId, double>> const alive_prob = state_alive(cost, current_assignment);
      Assignment next_assignment;
      for (auto const [n, state_cost] : std::views::enumerate(cost))
      {
        assert(cost.at(n).id == alive_prob.at(n).first);
        if (std::discrete_distribution<> alive_dist({ alive_prob.at(n).second, 1 - alive_prob.at(n).second });
            alive_dist(gen) == 0)  // alive
        {
          StateAssignment state_assignment(
              state_cost.id, true, std::vector<std::pair<DataSourceContainerId, MeasurementId>>{});
          std::unordered_set<MeasurementId> const used_meas = used_measurement(n, next_assignment, current_assignment);

          for (auto const& [source_id, meas_assignments] : state_cost.alive)
          {
            std::vector<double> probs;
            probs.reserve(meas_assignments.size());
            for (auto const& [meas_id, prob] : meas_assignments)
            {
              if (not used_meas.contains(meas_id))
              {
                probs.push_back(prob);
              }
              else
              {
                probs.push_back(0);
              }
            }
            assert([&] {
              if (std::accumulate(probs.begin(), probs.end(), 0.0) == 0.0)
              {
                LOG_FATAL(to_string(cost));
                LOG_FATAL(to_string(next_assignment));
                LOG_FATAL(to_string(current_assignment));
                LOG_FATAL("Current State: " << state_cost.id.value_);
                for (MeasurementId meas_id : used_meas)
                {
                  LOG_FATAL("used: " << meas_id);
                }
                for (auto const& [meas_id, prob] : meas_assignments)
                {
                  if (not used_meas.contains(meas_id))
                  {
                    LOG_FATAL(std::format("chain tries {}, chain lenght {}", chain_tries, chain_length));
                  }
                  else
                  {
                    LOG_FATAL(std::format("State {} can not use meas: {}", state_cost.id, meas_id));
                  }
                }
                return false;
              }
              return true;
            }());
            std::discrete_distribution<> dist(probs.begin(), probs.end());
            std::size_t const meas_index = dist(gen);
            state_assignment.meas_assignments.emplace_back(source_id, meas_assignments.at(meas_index).first);
          }
          next_assignment.push_back(std::move(state_assignment));
        }
        else  // dead
        {
          next_assignment.emplace_back(
              state_cost.id, false, std::vector<std::pair<DataSourceContainerId, MeasurementId>>{});
        }
      }
      assert(is_valid(next_assignment));
      {
        if (std::ranges::find(assignments, next_assignment) == assignments.end())
        {
          LOG_DEB(std::format("found new sol with try {}", tries));
          assignments.push_back(next_assignment);
          consecutive_old_sols = 0;
        }
      }
      if (tries % chain_length == 0)  // restart chain
      {
        current_assignment = init_assignment;
        chain_tries = 1;
      }
      else
      {
        current_assignment = std::move(next_assignment);
      }
    }
    return assignments;
  }
}

}  // namespace ttb::gibbs
