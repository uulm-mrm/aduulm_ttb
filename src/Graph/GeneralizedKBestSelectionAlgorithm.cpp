#include "tracking_lib/Graph/GeneralizedKBestSelectionAlgorithm.h"

#include <numeric>
#include <ranges>

namespace ttb::graph::generalized_k_best_selection
{
auto AllCombinations::generateCombinationsSubgraph(const SelectionArray& weightMatrix,
                                                   const std::vector<graph_impl::Edge>& edges) const
    -> std::vector<std::pair<Selection, double>>
{
  if (weightMatrix.empty())
  {
    return {};
  }

  const std::size_t numSensors = weightMatrix.size();
  std::vector<std::size_t> indices(numSensors);
  auto sizeView = weightMatrix | std::views::transform([](const Selection& list) { return list.size(); });
  auto numCombinations = std::accumulate(sizeView.begin(),
                                         sizeView.end(),
                                         std::size_t{ 1 },
                                         std::multiplies<>());  // calculates product of std::vector
                                                                // (number of elements in first row of weightMatrix
                                                                // *...* number of elements in last row of weightMatrix)

  std::vector<std::pair<Selection, double>> ret;
  ret.reserve(numCombinations);
  while (numCombinations-- > 0)
  {
    std::pair<Selection, double> res;
    Selection s;
    s.reserve(numSensors);
    std::size_t increment = 1;
    double weight = 1.0;
    for (auto [idx, selection] : std::views::zip(indices, weightMatrix))
    {
      weight *= edges.at(selection[idx])._weight;
      s.push_back(selection[idx]);
      idx += increment;
      if (idx >= selection.size())
      {
        idx = 0;
      }
      else
      {
        increment = 0;
      }
    }
    ret.emplace_back(std::move(s), weight);
  }
  return ret;
}

auto AllCombinations::operator()(const SelectionArrayList& weightMatricesList,
                                 const std::vector<graph_impl::Edge>& edges) const
    -> std::vector<std::pair<Selection, double>>
{
  if (weightMatricesList.empty())
  {
    return {};
  }

  std::vector<std::pair<Selection, double>> ret;
  for (const auto& weightMatrix : weightMatricesList)
  {
    if (weightMatrix.empty())
    {
      continue;
    }
    std::vector<std::pair<Selection, double>> selections = generateCombinationsSubgraph(weightMatrix, edges);
    ret.insert(ret.end(), std::make_move_iterator(selections.begin()), std::make_move_iterator(selections.end()));
  }
  return ret;
}

namespace generalized_kbest_impl
{
Node::Node(SubgraphId subgraphId_,
           Indices indices_,
           double score_,
           NodePtr parent_,
           std::size_t hash_id,
           bool isRootNode_)
  : _subgraphId(subgraphId_)
  , _indices(std::move(indices_))
  , _score(score_)
  , _parent(std::move(parent_))
  , _hash(hash_id)
  , _isRootNode(isRootNode_)
{
}

constexpr bool Node::operator<(const Node& other) const noexcept
{
  return _score < other._score;
}

bool NodePtrCompare::operator()(const NodePtr& lhs, const NodePtr& rhs) noexcept  // inverse version to std::greater
{
  return *lhs < *rhs;
}

Hasher::Hasher(const SelectionArray& array)
{
  auto sizeView = array | std::views::transform([acc = 1UL](const auto& sel) mutable noexcept {
                    auto ret = acc;
                    acc *= sel.size();
                    return ret;
                  });

  _increments = std::vector<std::size_t>(sizeView.begin(), sizeView.end());
}

std::size_t Hasher::getNext(const Node& n, std::size_t index) const
{
  assert(index < _increments.size() && index < n._indices.size());
  return n._hash + _increments[index];
}

HasherList::HasherList(const SelectionArrayList& arrayList)
{
  _incrementsList.clear();
  for (const auto& array : arrayList)
  {
    _incrementsList.push_back(Hasher(array));
  }
}

std::size_t HasherList::getNext(const SubgraphId subgraphId, const Node& n, std::size_t index) const
{
  return _incrementsList[subgraphId].getNext(n, index);
}

std::vector<std::size_t> sort_indexes(const std::vector<double>& v)
{
  // initialize original index locations
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v using std::stable_sort instead of std::sort to avoid unnecessary index
  // re-orderings when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(), [&v](std::size_t i1, std::size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

RootNodeInformations::RootNodeInformations(const SelectionArrayList& weightMatricesList,
                                           const std::vector<graph_impl::Edge>& edges)
{
  using Indices = std::vector<std::size_t>;

  // save weightProduct of every best solution (always first column of weightMatrix!) of each weightMatrix
  Indices indices(weightMatricesList.begin()->size(), static_cast<std::size_t>(0.));

  size_t u = 0;
  auto maxWeightPerSubgraph =
      weightMatricesList | std::views::transform([indices, u, edges](const auto& list) mutable noexcept {
        if (list.empty())
        {
          u++;
          return 0.0;
        }
        // calculate product weight of column of matrix
        double res = 1.0;
        for (auto [index, sensorSelection] : std::views::zip(indices, list))
        {
          res *= edges.at(sensorSelection[index])._weight;
        }
        u++;
        return res;
      });

  std::vector<double> weightVec = std::vector<double>(maxWeightPerSubgraph.begin(), maxWeightPerSubgraph.end());

  _subgraphIdsList = sort_indexes(weightVec);
  std::vector<bool> tmp(_subgraphIdsList.size(), false);
  _selectedIds = tmp;
  _counter = static_cast<size_t>(0);
  _rootChildsAreAllSet = false;
  _nextSubgraphId = _subgraphIdsList[0];
}

Graph::Graph(const SelectionArrayList& weightMatricesList,
             const std::vector<graph_impl::Edge>& edges,
             std::size_t numSolutions)
  : _weightMatricesList(weightMatricesList)
  , _edges(edges)
  , _rootNodeInfos(weightMatricesList, edges)
  , _hasherList(weightMatricesList)
  , _k(numSolutions)
{
  std::vector<std::unordered_set<std::size_t>> visitedChildrenList_(weightMatricesList.size());
  _visitedChildrenList = visitedChildrenList_;  // initialize emtpy visitedChildrenList with #subgraph's elements
}

double Graph::weightProduct(const Node::Indices& indices, const SubgraphId& u) const
{
  if (u >= _weightMatricesList.size())
  {
    std::stringstream msg;
    msg << "Subgraph id value" << u << "is higher than allowed (" << _weightMatricesList.size() << ")";
    throw std::runtime_error(msg.str());
  }
  double ret = 1.0;
  for (auto [index, sensorSelection] : std::views::zip(indices, _weightMatricesList[u]))
  {
    ret *= _edges.at(sensorSelection[index])._weight;
  }
  return ret;
}

auto Graph::generateSelectionsList(const std::vector<NodePtr>& selections) const
    -> std::vector<std::pair<Selection, double>>
{
  LOG_DEB("generateSelectionsList - number of subgraphs: " << _weightMatricesList.size());
  std::vector<std::pair<Selection, double>> ret;
  if (selections.empty() || selections.size() == 1)
  {
    if (selections.empty())
    {
      LOG_ERR("Result of generalizedKBest algo is empty");
    }
    if (selections.size() == 1)
    {
      LOG_ERR("Result of generalizedKBest algo only contains dummy node!");
    }
    throw std::runtime_error("Result of generalizedKBest algo is empty");
    return ret;
  }

  for (const auto& node : selections)
  {
    if (node->_isRootNode)
    {
      // root node is dummy node with no relevant information for output!
      continue;
    }
    Selection s;
    s.reserve(node->_indices.size());
    assert(node->_indices.size() == _weightMatricesList[node->_subgraphId].size());
    for (auto [index, sensorSelection] : std::views::zip(node->_indices, _weightMatricesList[node->_subgraphId]))
    {
      s.emplace_back(sensorSelection[index]);
    }
    ret.emplace_back(std::move(s), node->_score);
  }
  return ret;
}

auto Graph::nextBestChild(const NodePtr& parent) -> NodePtr
{
  NodePtr ret = nullptr;

  if (parent->_isRootNode)
  {
    // calculation of nextBestChild depends on subgraphId list
    if (allRootChildsSet())
    {
      // all siblings are already explored
      return ret;
    }
    std::vector<std::size_t> childIndices(parent->_indices.size(), 0);  // first solution is guaranteed to be the
                                                                        // highest of the specific subgraph
    SubgraphId subgraphId = getNextSubgraphId();
    if (_weightMatricesList[subgraphId].empty())
    {
      return ret;
    }
    double score = weightProduct(childIndices, subgraphId);
    ret = std::make_shared<Node>(subgraphId, parent->_indices, score, parent, 0, false);
    return ret;
  }

  // normal kBestSelection case when parent is not a rootNode
  for (std::size_t sensorIndex = 0; sensorIndex < parent->_indices.size(); ++sensorIndex)
  {
    const std::size_t nextIndex = parent->_indices[sensorIndex] + 1;
    if (nextIndex >= _k || nextIndex >= _weightMatricesList[parent->_subgraphId][sensorIndex].size())
    {
      // Enough Solutions visited!
      continue;
    }

    const std::size_t newHash = _hasherList.getNext(
        parent->_subgraphId, *parent, sensorIndex);  // ToDo(hermann): Check if nodes with different level can have the
                                                     // same hash!!! (If yes --> Bug!)
    // Child was already visited
    if (_visitedChildrenList[parent->_subgraphId].contains(newHash))
    {
      continue;
    }

    std::vector<std::size_t> childIndices = parent->_indices;
    SubgraphId childSubgraphId = parent->_subgraphId;
    ++childIndices[sensorIndex];
    double score = weightProduct(childIndices, childSubgraphId);
    if (nullptr == ret || score > ret->_score)
    {
      ret = std::make_shared<Node>(std::move(childSubgraphId), std::move(childIndices), score, parent, newHash, false);
    }
  }

  if (nullptr != ret)
  {
    // add best child
    _visitedChildrenList[parent->_subgraphId].emplace(ret->_hash);
  }

  return ret;
}
}  // namespace generalized_kbest_impl
}  // namespace ttb::graph::generalized_k_best_selection
