#pragma once

#include "tracking_lib/Graph/GraphImpl.h"

#include <queue>
#include <unordered_set>

namespace ttb::graph::generalized_k_best_selection
{
using MatrixRow = std::vector<graph_impl::EdgeId>;  // results belonging to Sensor V
using MatrixW = std::vector<MatrixRow>;             // results of all Sensors for one specific u
using MatrixWList = std::vector<MatrixW>;           // results of all u
using Selection = MatrixRow;                        // One possible result (selection) is given by a MatrixRow
using SelectionList = std::vector<Selection>;
using SubgraphId = std::size_t;  // subgraph id
using IndexVector = std::vector<std::size_t>;
using Index = std::size_t;

struct UniqueId
{
  std::uint64_t h1 = 0;
  std::uint64_t h2 = 0;

  bool operator==(const UniqueId& other) const noexcept
  {
    return h1 == other.h1 && h2 == other.h2;
  }
};

static inline std::uint64_t splitmix64(std::uint64_t x) noexcept
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static UniqueId hashSelection(const Selection& v)
{
  std::uint64_t h1 = 0x243f6a8885a308d3ULL;
  std::uint64_t h2 = 0x13198a2e03707344ULL;

  for (std::size_t i = 0; i < v.size(); ++i)
  {
    const std::uint64_t x = static_cast<std::uint64_t>(v[i]);
    h1 ^= splitmix64(x + 0x9e3779b97f4a7c15ULL * (i + 1));
    h2 += splitmix64(x ^ (0xbf58476d1ce4e5b9ULL * (i + 1)));
    h1 = (h1 << 7) | (h1 >> 57);
    h2 = (h2 << 11) | (h2 >> 53);
  }

  return { h1, h2 };
}

static inline UniqueId
updateHash(UniqueId hash, const Index position, const std::size_t oldValue, const std::size_t newValue) noexcept
{
  const std::uint64_t p = static_cast<std::uint64_t>(position) + 1;
  hash.h1 ^= splitmix64(static_cast<std::uint64_t>(oldValue) + 0x9e3779b97f4a7c15ULL * p);
  hash.h1 ^= splitmix64(newValue + 0x9e3779b97f4a7c15ULL * p);
  hash.h2 ^= splitmix64(oldValue + 0xbf58476d1ce4e5b9ULL * p);
  hash.h2 ^= splitmix64(newValue + 0xbf58476d1ce4e5b9ULL * p);
  return hash;
}

constexpr Index NoChangeGiven = std::numeric_limits<Index>::infinity();
struct Node
{
  Index changed_index = NoChangeGiven;  // defines possible solution, by saving the information which index in vector v
                                        // changed compared to parent!
  std::uint32_t j = 0;                  // id of subgraph
  double score = std::numeric_limits<double>::infinity();  // sum weight of solution (because values of MatrixWList are
                                                           // logarithmic!!)
  UniqueId id{};                                           // unique id of this solution...
  const Node* parent = nullptr;
  bool root = false;
};

// Key of created solution node or potential solution node
struct StateKey
{
  SubgraphId j;  // id of subgraph
  UniqueId id{};

  bool operator==(const StateKey& other) const noexcept
  {
    return j == other.j && id == other.id;
  }
};

struct StateKeyHash
{
  std::size_t operator()(const StateKey& key) const noexcept
  {
    std::uint64_t x = key.id.h1;
    x ^= key.id.h2 + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    x ^= static_cast<std::uint64_t>(key.j) + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    return splitmix64(x);
  }
};

// Priority queue comparator
struct NodeCompare
{
  bool operator()(const Node* a, const Node* b) const noexcept
  {
    if (a->score != b->score)
      return a->score < b->score;

    if (a->j != b->j)
      return a->j > b->j;

    if (a->id.h1 != b->id.h1)
      return a->id.h1 > b->id.h1;

    return a->id.h2 > b->id.h2;
  }
};

using PriorityQueue = std::priority_queue<Node*, std::vector<Node*>, NodeCompare>;

class NodePool
{
private:
  std::vector<std::unique_ptr<Node[]>> blocks_;
  std::size_t blockSize_;
  std::size_t position_ = 0;

public:
  explicit NodePool(std::size_t blockSize = 4096) : blockSize_(blockSize)
  {
  }

  Node* allocate()
  {
    if (blocks_.empty() || position_ >= blockSize_)
    {
      blocks_.push_back(std::make_unique<Node[]>(blockSize_));
      position_ = 0;
    }
    return &blocks_.back()[position_++];
  }

  void clear()
  {
    blocks_.clear();
    position_ = 0;
  }
};

class GKBS
{
public:
  GKBS(const MatrixWList& W, std::size_t k, const std::vector<graph_impl::Edge>& edges);

  /// Run GeneralizedKBestAlgorithm published in
  /// C. Hermann, A. Scheible, M. Buchholz and K. Dietmayer, "An Efficient Implementation of the Fast Product
  /// Multi-Sensor Labeled Multi-Bernoulli Filter," 2024 IEEE International Conference on Multisensor Fusion and
  /// Integration for Intelligent Systems (MFI), Pilsen, Czech Republic, 2024, pp. 1-8,
  /// doi: 10.1109/MFI62651.2024.10705774.
  std::vector<std::pair<Selection, double>> runAlgorithm()
  {
    // Reset state from previous execution.
    while (!PotentialSolutions_.empty())
      PotentialSolutions_.pop();
    seenNodes_.clear();
    pool_.clear();
    sortedIndices_.clear();

    std::vector<Node*> solutions;
    solutions.reserve(max_solutions_);
    if (k_ == 0)
    {
      LOG_FATAL("k = 0 -> no solutions are calculated");
      return {};
    }

    if (V_ == 0)
    {
      LOG_FATAL("V = 0 -> no solutions are calculated");
      return {};
    }

    // Initialization
    initSortedIndicesList();
    Node* root = pool_.allocate();
    root->changed_index = NoChangeGiven;
    root->j = 0;
    root->score = 0.0;
    IndexVector v(V_, 0);
    UniqueId dummy_hash;
    root->id = hashSelection(std::move(v));  // this does not matter...
    root->parent = nullptr;
    root->root = true;

    // First candidate
    Node* firstChild = nextBestChildGKBS(root);
    if (firstChild != nullptr)
    {
      insertPending(firstChild);
    }
    // init done, start of main part of algorithm
    for (std::size_t n = 0; n < max_solutions_; ++n)
    {
      assert(!PotentialSolutions_.empty());
      if (PotentialSolutions_.empty())
      {
        LOG_FATAL("PotentialSolutions_ is empty. Can we reach this part?! Check!");  // should not happen anymore
        break;
      }
      Node* best = PotentialSolutions_.top();
      assert(best != nullptr && "best is nullptr");
      PotentialSolutions_.pop();
      if (best->score == -std::numeric_limits<double>::infinity())
      {
        // case can happen if whe have -inf values in weight_ matrix...
        LOG_WARN("Just for your information. GKBEST breaks, because of best score value -inf.");
        assert([&]() {  // NOLINT
          bool has_inf_value = false;
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

            for (const auto& val : sum_row)
            {
              if (std::abs(val) == std::numeric_limits<double>::infinity())
              {
                has_inf_value = true;
              }
            }
            if (has_inf_value)
            {
              break;
            }
          }
          if (!has_inf_value)
          {
            LOG_FATAL("An inf solution should not exist! weights_log is printed now");
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
          return true;
        }());
        break;
      }
      solutions.push_back(best);
      if (n == max_solutions_ - 1)
      {
        // enough solutions added
        break;
      }
      Node* child = nextBestChildGKBS(best);
      if (child != nullptr)
      {
        insertPending(child);
      }
      Node* sibling = nextBestChildGKBS(best->parent);
      if (sibling != nullptr)
      {
        insertPending(sibling);
      }
    }
    return generateSelectionsList(solutions);
  }

private:
  const MatrixWList& W_;                                       /// contains the edge_ids
  std::vector<std::vector<std::vector<double>>> weights_log_;  /// contains the weights of the edge_ids
  std::size_t V_;
  const std::size_t k_;  // solutions to be calculated
  NodePool pool_;        // solution???
  PriorityQueue PotentialSolutions_;
  std::unordered_set<StateKey, StateKeyHash> seenNodes_;
  std::vector<SubgraphId> sortedIndices_;       // sorted indices of subgraphs, size is u!
  std::size_t max_solutions_;                   // maximum number of possible solutions (<=k_!)
  const std::vector<graph_impl::Edge>& edges_;  // Saves the edges of GraphImpl!!
  bool root_children_are_all_explored_ = false;
  size_t counter_ = 0;
  SubgraphId next_subgraph_id_ = 0;

  void initSortedIndicesList();
  void setMaxPossibleCombination();
  void setNextSubgraphId();
  double getScore(const IndexVector& v, const SubgraphId j) const;
  double getNextScore(const Node* parent, std::size_t sensor, std::size_t oldIndex, std::size_t newIndex) const;
  IndexVector getIndexVector(const Node* node) const;
  Node* nextBestChildGKBS(const Node* parent);
  Node* nextBestChildKBS(const Node* parent);
  std::vector<std::pair<Selection, double>> generateSelectionsList(const std::vector<Node*>& selections) const;
  // Insert candidate into P (potential solutions)
  void insertPending(Node* node)
  {
    const StateKey key{ node->j, node->id };
    // insert returns false if this node was already seen.
    const auto [it, inserted] = seenNodes_.insert(key);
    if (!inserted)
    {
      LOG_FATAL("Node we want to insert already exists. This should be checked!");
      return;
    }
    PotentialSolutions_.push(node);
  }
};
}  // namespace ttb::graph::generalized_k_best_selection
