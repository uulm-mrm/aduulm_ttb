#pragma once

#include "tracking_lib/Graph/GraphImpl.h"

#include <queue>
#include <unordered_set>

namespace ttb::graph::generalized_k_best_selection
{
using Selection = std::vector<graph_impl::EdgeId>;
using SelectionArray = std::vector<Selection>;
using SelectionArrayList = std::vector<SelectionArray>;
using SubgraphId = std::size_t;
using SubgraphIdsList = std::vector<SubgraphId>;

namespace generalized_kbest_impl
{
struct Node;
using NodePtr = std::shared_ptr<Node>;

struct Node
{
  using Indices = std::vector<std::size_t>;

  Node(SubgraphId subgraphId_, Indices indices_, double score_, NodePtr parent_, std::size_t hash_id, bool isRootNode_);

  constexpr bool operator<(const Node& other) const noexcept;

  SubgraphId _subgraphId;  // corresponds to the id of the subgraph
  Indices _indices;        // vector with indices defining the selection
  double _score;           // resulting product weight
  NodePtr _parent;
  std::size_t _hash;  // unique hash of every node
  bool _isRootNode;   // information if this is the root/dummy node
};

struct NodePtrCompare
{
  bool operator()(const NodePtr& lhs, const NodePtr& rhs) noexcept;
};

struct Hasher
{
  // calculates Hashes of nodes for one specific subgraph
  explicit Hasher(const SelectionArray& array);
  /**
   * Calculation of hash of specific child based on parent node and sensor index which value changes
   * @param n parent node
   * @param index sensor index
   * @return hash of next child node
   */
  [[nodiscard]] std::size_t getNext(const Node& n, std::size_t index) const;

protected:
  std::vector<std::size_t> _increments;  // based on initial weight matrix for one subgraph id i. Example: weight matrix
                                         // with two rows. First row contains three relevant values. Second row contains
                                         // two relevant values -> increments = [1 3]
};

struct HasherList
{
  // saves calculation of hashes for every subgraph id
  explicit HasherList(const SelectionArrayList& arrayList);
  [[nodiscard]] std::size_t getNext(const SubgraphId subgraphId, const Node& n, std::size_t index) const;

protected:
  std::vector<Hasher> _incrementsList;
};

/*
 * returns the index vector of the in descending order sorted vector v without changing the order in v
 */
std::vector<std::size_t> sort_indexes(const std::vector<double>& v);

struct RootNodeInformations
{
  explicit RootNodeInformations(const SelectionArrayList& weightMatricesList,
                                const std::vector<graph_impl::Edge>& edges);

  std::size_t getNextSubgraphId()
  {
    if (_subgraphIdsList.empty())
    {
      throw std::runtime_error("rootID list is empty!!!");
    }
    selectId(_counter);  // subgraph id
    std::size_t ret = _nextSubgraphId;
    setNextSubgraphId();
    return ret;
  }

  void selectId(SubgraphId u)
  {  // Just for debugging purposes
    if (_selectedIds[u])
    {
      throw std::runtime_error("Already selected RootID is selected again. This should not happen!");
    }
    _selectedIds[u] = true;
  }

  void setNextSubgraphId()
  {
    _counter++;
    if (_counter < _subgraphIdsList.size())
    {
      _nextSubgraphId = _subgraphIdsList[_counter];
      _rootChildsAreAllSet = false;
      return;
    }
    _rootChildsAreAllSet = true;
  }

  bool allRootChildsSet()
  {
    return _rootChildsAreAllSet;
  }

protected:
  SubgraphIdsList _subgraphIdsList;  // contains all subgraph ids sorted by highest product weight of each subgraph id
                                     // weight matrix
  std::vector<bool> _selectedIds;  // saves information if subgraph id is already selected (either in potentialSolutions
                                   // or in solutions). This is just for debugging and not needed for the algorithm
  SubgraphId _nextSubgraphId;      // saves information about next best subgraph id
  size_t _counter;
  bool _rootChildsAreAllSet;
};

struct Graph
{
  /**
   * Graph for algorithm is initialized
   * @param weightMatricesList list of sorted weight matrices. One weight matrix corresponds to one subgraph. Sorting of
   * matrix must be in descending order
   * @param edges contain edge_ids of graph, which is solved
   * @param numSolutions number of solutions which shall be calculated
   */
  Graph(const SelectionArrayList& weightMatricesList,
        const std::vector<graph_impl::Edge>& edges,
        std::size_t numSolutions);
  double weightProduct(const Node::Indices& indices, const SubgraphId& u) const;
  SubgraphId getNextSubgraphId()
  {
    return _rootNodeInfos.getNextSubgraphId();
  };
  bool allRootChildsSet()
  {
    return _rootNodeInfos.allRootChildsSet();
  }
  auto
  generateSelectionsList(const std::vector<NodePtr>& selections) const -> std::vector<std::pair<Selection, double>>;
  auto nextBestChild(const NodePtr& parent) -> NodePtr;

private:
  const SelectionArrayList& _weightMatricesList;  // List of sorted weight matrices. Each weight matrix belongs to one
                                                  // subgraph
  const std::vector<graph_impl::Edge>& _edges;    // Saves the edges of GraphImpl!!
  RootNodeInformations _rootNodeInfos;  // contains information's about sorted subgraph's indices. The sorting is done
                                        // by the highest product weight of each corresponding weight matrix
  HasherList _hasherList;  // list of Hasher for each subgraph id. Each Hasher is able to calculate the hash for the
                           // nodes
  std::unordered_set<std::size_t> _visitedChildren;
  std::vector<std::unordered_set<std::size_t>> _visitedChildrenList;  // Childrens of every subgraph id u are held
                                                                      // separately
  std::size_t _k;                                                     // number of solutions to calculate
};

}  // namespace generalized_kbest_impl

/**
 * Functor that creates all possible combinations, given a SelectionArray.
 */
struct AllCombinations
{
  /** Calculates all solutions
   *
   * @param weightMatricesList list of sorted weight matrices. One weight matrix corresponds to one subgraph. Sorting of
   * matrix must be in descending order
   * @param edges contain edge_ids of graph, which is solved
   * @return vector containing the solutions and the corresponding product weight
   */
  auto operator()(const SelectionArrayList& weightMatricesList,
                  const std::vector<graph_impl::Edge>& edges) const -> std::vector<std::pair<Selection, double>>;

  auto generateCombinationsSubgraph(const SelectionArray& weightMatrix, const std::vector<graph_impl::Edge>& edges)
      const -> std::vector<std::pair<Selection, double>>;
};

/**
 * The generalizedKBestSelection algorithm solves the k shortest path problem for specific graph's. For algorithm
 * details, see: C. Hermann, A. Scheible, M. Buchholz and K. Dietmayer, "An Efficient Implementation of the Fast Product
 * Multi-Sensor Labeled Multi-Bernoulli Filter," 2024 IEEE International Conference on Multisensor Fusion and
 * Integration for Intelligent Systems (MFI), Pilsen, Czech Republic, 2024, pp. 1-8, doi: 10.1109/MFI62651.2024.10705774
 * (https://ieeexplore.ieee.org/document/10705774).
 *
 * For details on the kBestSelection algorithm, which is also contained, see:
 * M. Herrmann, C. Hermann and M. Buchholz, "Distributed Implementation of the Centralized Generalized Labeled
 * Multi-Bernoulli Filter," in IEEE Transactions on Signal Processing, vol. 69, pp. 5159-5174, 2021,
 * doi: 10.1109/TSP.2021.3107632. (https://ieeexplore.ieee.org/document/9524466)
 */
struct GeneralizedKBest
{
  explicit GeneralizedKBest(std::size_t k) : _k(k)
  {
  }

  /** Performs the generalizedKBestSelection/kBestSelection algorithm
   *
   * @param weightMatricesList list of sorted weight matrices. One weight matrix corresponds to one subgraph. Sorting of
   * matrix must be in descending order
   * @param edges contain edge_ids of graph, which is solved
   * @return vector containing the solutions and the corresponding product weight
   */
  auto operator()(const SelectionArrayList& weightMatricesList,
                  const std::vector<graph_impl::Edge>& edges) const -> std::vector<std::pair<Selection, double>>
  {
    LOG_INF("Calculate the " << _k << " best solutions!");
    if (weightMatricesList.empty() || _k == 0)
    {
      if (weightMatricesList.empty())
      {
        LOG_WARN("ArrayList is empty! GeneralizedKBest algorithm is not necessary");
      }
      if (_k == 0)
      {
        LOG_WARN("k is set to 0. No results will be calculated! Choose a value higher than 0!");
      }
      return {};
    }

    std::size_t maxPossibleCombs = 0;
    std::size_t numSensors = static_cast<std::size_t>(0);
    for (const auto& matrix : weightMatricesList)
    {
      if (matrix.empty())
      {
        LOG_DEB("array empty -> continue");
        continue;
      }
      if (numSensors > 0 && (numSensors != matrix.size()))
      {
        LOG_WARN("subgraph has different number of nodes");
        if (matrix.size() > numSensors)
        {
          numSensors = matrix.size();
        }
      }
      else
      {
        numSensors = matrix.size();
        LOG_INF("numSensors is set to " << numSensors);
      }
      std::size_t localComps = 1;
      for (const auto& list : matrix)
      {
        if (list.empty())
        {
          throw std::runtime_error("List from at least one sensor ist empty");
        }
        localComps *= list.size();
      }
      maxPossibleCombs += localComps;
      if (maxPossibleCombs > _k)
      {
        maxPossibleCombs = _k;
        break;
      }
    }

    using namespace generalized_kbest_impl;
    LOG_DEB("numSensors: " << numSensors);
    LOG_DEB("maxPossibleCombs: " << maxPossibleCombs);
    if (numSensors == 0 && maxPossibleCombs != 0)
    {
      throw std::runtime_error("relevant number of sensors is 0 but possible max number of combinations not");
    }
    if (numSensors == 0)
    {
      // there is nothing to fuse!
      LOG_DEB("return empty selection list!");
      std::vector<std::pair<Selection, double>> ret(weightMatricesList.size());
      return ret;
    }
    generalized_kbest_impl::Graph graph(weightMatricesList, edges, maxPossibleCombs);
    std::vector<NodePtr> solutions;
    std::priority_queue<NodePtr, std::vector<NodePtr>, NodePtrCompare> potentialSolutions;
    solutions.reserve(std::min(_k + 1, maxPossibleCombs + 1));  // reserve k+1 solutions instead of k, since first
                                                                // solution is dummy node!
    // Weight of first solution is irrelevant since it is a dummy node!
    solutions.emplace_back(std::make_shared<Node>(0, Node::Indices(numSensors), 0, nullptr, 0, true));
    auto child = graph.nextBestChild(solutions.front());
    if (nullptr != child)
    {
      potentialSolutions.emplace(std::move(child));
    }
    // init done
    for (std::size_t step = 0; step < _k && !potentialSolutions.empty(); ++step)
    {
      // const cast is necessary because std::priority_queue is stoopid with two 'o'
      auto best = std::move(const_cast<NodePtr&>(potentialSolutions.top()));
      potentialSolutions.pop();
      child = graph.nextBestChild(best);
      if (nullptr != child)
      {
        potentialSolutions.emplace(std::move(child));
      }
      auto sibling = graph.nextBestChild(best->_parent);
      if (nullptr != sibling)
      {
        potentialSolutions.emplace(std::move(sibling));
      }
      solutions.emplace_back(std::move(best));
    }

    return graph.generateSelectionsList(solutions);
  }

private:
  std::size_t _k;
};
}  // namespace ttb::graph::generalized_k_best_selection
