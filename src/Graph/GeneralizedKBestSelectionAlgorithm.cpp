#include "tracking_lib/Graph/GeneralizedKBestSelectionAlgorithm.h"

#include <numeric>
#include <ranges>

namespace ttb::graph::generalized_k_best_selection
{

GKBS::GKBS(const MatrixWList& W, std::size_t k, const std::vector<graph_impl::Edge>& edges)
  : W_(W), V_(W_.empty() ? 0 : W_.front().size()), k_(k), pool_(4096), edges_(edges)
{
  setMaxPossibleCombination();
  assert(!W_.empty() && "W must not be empty.");

  if (V_ == 0)
    LOG_WARN("V must be > 0, for getting results");

  assert(!(V_ == 0 && max_solutions_ != 0) && "V is 0 but possible max number of combinations not");

  weights_log_.resize(W_.size());
  for (std::size_t j = 0; j < W_.size(); ++j)
  {
    weights_log_[j].resize(W_[j].size());
    for (std::size_t s = 0; s < W_[j].size(); ++s)
    {
      weights_log_[j][s].resize(W_[j][s].size());
      for (std::size_t i = 0; i < W_[j][s].size(); ++i)
      {
        weights_log_[j][s][i] = edges_.at(W_[j][s][i])._weight;
      }
    }
  }

  // Reserve approximately enough room for all states.
  // The exact number of simultaneously existing nodes is
  // generally much smaller than k, but this reduces rehashing.
  seenNodes_.reserve(max_solutions_ * 2 + 1024);
  seenNodes_.max_load_factor(0.70f);
}
void GKBS::setMaxPossibleCombination()
{
  std::size_t maxPossibleCombs = 0;
  for (const auto& matrix : W_)
  {
    if (matrix.empty())
    {
      LOG_DEB("array empty -> continue");
      continue;
    }
    assert(!(V_ > 0 && (V_ != matrix.size())) && "subgraph has different number of nodes. Implementation does not "
                                                 "support this!");
    V_ = matrix.size();
    std::size_t max_combs_matrix = 1;
    for (const auto& row : matrix)
    {
      assert(!row.empty() && "Row from at least one submatrix ist empty");
      std::size_t new_val = max_combs_matrix * row.size();
      // prevent too big values!
      if (new_val > k_ || (row.size() > 0 && new_val == 0))
      {
        max_combs_matrix = k_;
        break;
      }
      max_combs_matrix *= row.size();
    }
    if (max_combs_matrix >= k_)
    {
      maxPossibleCombs = k_;
    }
    else
    {
      maxPossibleCombs += max_combs_matrix;
    }
    if (maxPossibleCombs > k_)
    {
      maxPossibleCombs = k_;
      break;
    }
  }
  max_solutions_ = maxPossibleCombs;
}

void GKBS::initSortedIndicesList()
{
  size_t u = 0;
  std::vector<double> maxWeightsPerSubgraph(W_.size(), 0.);
  for (auto const& W : W_)
  {
    if (W.empty())
    {
      maxWeightsPerSubgraph[u] = 0;
      u++;
      continue;
    }
    double res = 0.0;
    for (auto const& WRow : W)
    {
      for (auto const& val : WRow)
        res += edges_.at(val)._weight;
    }
    maxWeightsPerSubgraph[u] = res;
    u++;
  }
  // initialize original index locations
  std::vector<std::size_t> idx(maxWeightsPerSubgraph.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::ranges::stable_sort(idx, [&maxWeightsPerSubgraph](std::size_t a, std::size_t b) -> bool {
    return maxWeightsPerSubgraph[a] > maxWeightsPerSubgraph[b];
  });
  sortedIndices_ = std::move(idx);
  next_subgraph_id_ = sortedIndices_[0];
}

void GKBS::setNextSubgraphId()
{
  counter_++;
  if (counter_ < sortedIndices_.size())
  {
    next_subgraph_id_ = sortedIndices_[counter_];
    root_children_are_all_explored_ = false;
    return;
  }
  root_children_are_all_explored_ = true;
}

IndexVector GKBS::getIndexVector(const Node* node) const
{
  IndexVector v(V_, 0);

  while (node != nullptr && !node->root)
  {
    assert(!(node->parent->root && node->changed_index != NoChangeGiven));
    if (node->changed_index == NoChangeGiven && node->parent->root)
    {
      break;
    }
    ++v[node->changed_index];
    node = node->parent;
  }
  return v;
}

Node* GKBS::nextBestChildGKBS(const Node* parent)
{
  assert(parent != nullptr && "parent is nullptr");
  assert(sortedIndices_.size() > 0 && "sortedIndices is empty");
  if (parent->root)
  {
    // calculation of nextBestChild depends on subgraphId list
    if (root_children_are_all_explored_)
    {
      return nullptr;
    }
    const std::size_t j = next_subgraph_id_;
    setNextSubgraphId();
    if (W_[j].empty())
    {
      return nullptr;
    }
    IndexVector v(V_, 0);
    const double score = getScore(v, j);
    Node* child = pool_.allocate();
    child->changed_index = NoChangeGiven;
    child->j = static_cast<std::uint32_t>(j);
    child->score = score;
    child->id = hashSelection(std::move(v));
    child->parent = parent;
    child->root = false;
    return child;
  }
  // normal kBestSelection case when parent is not a rootNode
  return nextBestChildKBS(parent);
}

Node* GKBS::nextBestChildKBS(const Node* parent)
{
  IndexVector v_candidate = getIndexVector(parent);
  Node* candidate = nullptr;
  double bestScore = -std::numeric_limits<double>::infinity();
  for (std::size_t s = 0; s < V_; ++s)
  {
    const std::size_t nextIndex = v_candidate[s] + 1;
    if (nextIndex >= max_solutions_ || nextIndex >= W_[parent->j][s].size())
    {
      // Enough Solutions visited!
      continue;
    }
    // adapt v to get possible child
    const Index oldValue = v_candidate[s];
    const Index newValue = oldValue + 1;
    v_candidate[s] = newValue;

    const UniqueId hash = updateHash(parent->id, s, oldValue, newValue);
    // is solution already in solutions or potential solutions?
    if (const StateKey key{ parent->j, hash }; seenNodes_.contains(key))
    {
      v_candidate[s] = oldValue;
      continue;
    }
    const double score = getNextScore(parent, s, oldValue, newValue);
    if (score >= bestScore)
    {
      bestScore = score;
      if (candidate == nullptr)
      {
        candidate = pool_.allocate();
      }
      candidate->changed_index = s;
      candidate->j = parent->j;
      candidate->score = score;
      candidate->id = hash;
      candidate->parent = const_cast<Node*>(parent);
      candidate->root = false;
    }
    // Restore original vector.
    v_candidate[s] = oldValue;
  }
  return candidate;
}

double GKBS::getNextScore(const Node* parent,
                          const std::size_t sensor,
                          const std::size_t oldIndex,
                          const std::size_t newIndex) const
{
  assert(parent != nullptr);

  const SubgraphId j = parent->j;
  assert(j < W_.size());
  assert(sensor < W_[j].size());

  const auto& row = weights_log_[j][sensor];
  const double oldWeight = row[oldIndex];
  const double newWeight = row[newIndex];
  return parent->score + newWeight - oldWeight;
}

double GKBS::getScore(const IndexVector& v, const SubgraphId j) const
{
  assert(j < W_.size() && "subgraph id does not exist!");
  if (j >= W_.size())
  {
    LOG_FATAL("Subgraph id value" << j << "is higher than allowed (" << W_.size() << ")");
    return -std::numeric_limits<double>::infinity();
  }

  double ret = 0.0;
  for (auto [index, weights_row] : std::views::zip(v, weights_log_[j]))
  {
    ret += weights_row[index];
  }

  assert([&] {  // NOLINT
    if (std::isnan(ret))
    {
      for (auto [index, weights_row] : std::views::zip(v, weights_log_[j]))
      {
        LOG_FATAL("weight: " << weights_row[index]);
      }
      LOG_FATAL("product weight is nan: " << ret);
      return false;
    }
    return true;
  }());
  return ret;
}

std::vector<std::pair<Selection, double>> GKBS::generateSelectionsList(const std::vector<Node*>& selections) const
{
  std::vector<std::pair<Selection, double>> ret;
  if (selections.empty())
  {
    if (max_solutions_ != 0)
    {
      LOG_WARN("Just for your information. GKBEST delivers an empty result.");
      assert([&]() {  // NOLINT
        for (const auto& matrix : weights_log_)
        {
          std::vector sum_row(matrix.size(), 0.0);
          for (const auto& row : matrix)
          {
            std::size_t counter = 0;
            for (const auto& val : row)
            {
              sum_row[counter] += val;
            }
          }
          bool has_inf_value = false;
          for (const auto& val : sum_row)
          {
            if (std::abs(val) == std::numeric_limits<double>::infinity())
            {
              has_inf_value = true;
            }
          }
          if (!has_inf_value)
          {
            LOG_FATAL("Result of generalizedKBest algo is empty. But there should be an result... K= "
                      << k_ << " maxPossSol: " << max_solutions_);
            for (const auto& matrix_tmp : weights_log_)
            {
              LOG_FATAL("next matrix: ");
              for (const auto& row : matrix_tmp)
              {
                std::stringstream ss;
                for (const auto& val : row)
                {
                  ss << val << " ";
                }
                LOG_FATAL(ss.str());
              }
            }
            return false;
          }
        }
        return true;
      }());
    }
    return ret;
  }
  ret.reserve(selections.size());

  for (const auto& node : selections)
  {
    assert([&] {  // NOLINT
      if (!node)
      {
        LOG_FATAL("node of selection is nullptr!");
        return false;
      }
      return true;
    }());

    assert(!(node->root) && "Selected node is root, how is this possible? Check!");
    Selection s;
    s.reserve(V_);

    IndexVector v = getIndexVector(node);
    assert(v.size() == W_[node->j].size());
    for (auto [index, sensorSelection] : std::views::zip(v, W_[node->j]))
    {
      s.emplace_back(sensorSelection[index]);
    }
    ret.emplace_back(std::move(s), node->score);
  }
  return ret;
}

}  // namespace ttb::graph::generalized_k_best_selection