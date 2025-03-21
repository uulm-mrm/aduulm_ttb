#include "tracking_lib/Graph/GraphImpl.h"
#include "tracking_lib/Graph/GeneralizedKBestSelectionAlgorithm.h"
#include <algorithm>
#include <tracy/tracy/Tracy.hpp>

namespace ttb::graph_impl
{
constexpr auto tracy_color = tracy::Color::Beige;

std::string Node::toString() const
{
  return "Node with ID: " + std::to_string(_id) + '\n';
}

std::string Edge::toString() const
{
  return "Edge with internal Id: " + std::to_string(_id) + " Node " + std::to_string(_begin) + " -> " +
         std::to_string(_end) + " with  " + std::to_string(_weight) + '\n';
}

std::vector<std::vector<NodeId>> DiGraph::components() const
{
  ZoneScopedNC("GraphImpl::components", tracy_color);
  resetFlags();
  std::vector<std::vector<NodeId>> comps;
  while (true)
  {
    bool finished = true;
    for (auto const& node : _nodes)
    {
      if (not node.flag_1)
      {
        comps.emplace_back(bfs(node._id, {}, false));
        finished = false;
        break;
      }
    }
    if (finished)
    {
      return comps;
    }
  }
}

std::vector<std::pair<std::vector<EdgeId>, double>>
DiGraph::kShortestPath(SELECTION_STRATEGY strategy, std::size_t k, NodeId startNodeID, NodeId endNodeID) const
{
  ZoneScopedNC("GraphImpl::kShortestPath", tracy_color);
  LOG_DEB("start kShortestPath");
  LOG_DEB("startNode: " << startNodeID);
  LOG_DEB("endNodeId: " << endNodeID);
  if (startNodeID == endNodeID)
  {
    LOG_ERR("start node and end node is the same! Can not perform kShortestPath!");
    throw std::runtime_error("start node and end node is the same! Can not perform kShortestPath!");
  }
  std::vector<EdgeId> edgesFromStart = edges_from(startNodeID);
  graph::generalized_k_best_selection::SelectionArrayList arrayList;
  // generate the sorted matrices needed for the generalizedKBestAlgorithm!
  // either all edgesFromStart have the same end node or all have different end nodes and none the same
  if (edgesHaveSameEndNode(startNodeID))
  {
    // could also be implemented only with the kBestSelectionAlgorithm of https://ieeexplore.ieee.org/document/9524466
    LOG_DEB("kBestSelection algorithm");
    graph::generalized_k_best_selection::SelectionArray array;
    NodeId endNode = startNodeID;
    while (endNode != endNodeID)
    {
      assert(edgesHaveSameEndNode(endNode));
      LOG_DEB("While - endNode: " << endNode);
      std::vector<EdgeId> edgesFrom = edges_from(endNode);
      assert([&] {  // NOLINT
        if (edgesFrom.empty())
        {
          LOG_FATAL("Empty edges from node: " << endNode);
          LOG_FATAL("Graph: " << toString());
          LOG_FATAL("Start Node: " << _nodes.at(startNodeID).toString());
          LOG_FATAL("End Node: " << _nodes.at(endNodeID).toString());
          return false;
        }
        return true;
      }());
      endNode = _edges.at(edgesFrom.front())._end;
      // sort edges bei weight in descending order
      std::ranges::sort(edgesFrom, [&edges = _edges](const EdgeId& a, const EdgeId& b) {
        return edges.at(a)._weight > edges.at(b)._weight;
      });
      array.push_back(std::move(edgesFrom));
    }
    arrayList.push_back(std::move(array));
  }
  else
  {
    // only possible to solve with the GeneralizedKBestSelectionAlgorithm
    // (https://ieeexplore.ieee.org/document/10705774)
    LOG_DEB("generalizedKBestSelection algorithm");

    for (const auto& edge : edgesFromStart)
    {
      graph::generalized_k_best_selection::SelectionArray array;
      std::vector<EdgeId> dummyEdge;
      dummyEdge.push_back(edge);
      array.push_back(dummyEdge);
      NodeId endNode = _edges.at(edge)._end;
      while (endNode != endNodeID)
      {
        std::vector<EdgeId> edgesFrom = edges_from(endNode);
        if (edgesFrom.empty())
        {
          LOG_FATAL("This graph structure is not allowed for kShortestPath!");
          throw std::runtime_error("This graph structure is not allowed for kShortestPath!");
        }
        endNode = _edges.at(edgesFrom.at(0))._end;
        // sort edges bei weight in descending order
        std::sort(edgesFrom.begin(), edgesFrom.end(), [&edges = _edges](const EdgeId& a, const EdgeId& b) {
          return edges.at(a)._weight > edges.at(b)._weight;
        });
        array.push_back(std::move(edgesFrom));
      }
      arrayList.push_back(std::move(array));
    }
  }

  // GeneralizedKBestSelection algorithm
  std::vector<std::pair<std::vector<EdgeId>, double>> selectionsList;
  switch (strategy)
  {
    case SELECTION_STRATEGY::ALL_COMBINATIONS:
    {
      auto apportionStrategy = graph::generalized_k_best_selection::AllCombinations{};
      selectionsList = apportionStrategy(arrayList, _edges);
      break;
    }
    case SELECTION_STRATEGY::GENERALIZED_K_BEST:
    {
      auto apportionStrategy = graph::generalized_k_best_selection::GeneralizedKBest{ k };
      selectionsList = apportionStrategy(arrayList, _edges);
      break;
    }
    default:
      throw std::runtime_error("Selection Strategy is not known!!! Change it in config file");
  }
  std::ranges::sort(selectionsList,
                    [](std::pair<std::vector<EdgeId>, double> const& first,
                       std::pair<std::vector<EdgeId>, double> const& second) { return first.second < second.second; });
  return selectionsList;
}

std::vector<EdgeId> DiGraph::edges_to(NodeId node) const
{
  std::vector<EdgeId> out;
  for (EdgeId edge_id : _graph.at(node))
  {
    if (_edges.at(edge_id)._end == node)
    {
      out.push_back(edge_id);
    }
  }
  return out;
}

std::vector<EdgeId> DiGraph::edges_from(NodeId node) const
{
  std::vector<EdgeId> out;
  for (EdgeId edge_id : _graph.at(node))
  {
    if (_edges.at(edge_id)._begin == node)
    {
      out.push_back(edge_id);
    }
  }
  return out;
}

bool DiGraph::edgesHaveSameEndNode(NodeId startNode) const
{
  auto const& edges = edges_from(startNode);
  if (edges.size() <= 1)
  {
    return true;
  }
  return std::ranges::all_of(edges,
                             [&](EdgeId edge_id) { return _edges.at(edge_id)._end == _edges.at(edges.front())._end; });
}

void DiGraph::resetFlags() const
{
  for (auto& node : _nodes)
  {
    node.flag_1 = false;
    node.flag_2 = false;
  }
  for (auto& edge : _edges)
  {
    edge.flag_1 = false;
    edge.flag_2 = false;
  }
}

std::vector<NodeId> DiGraph::bfs(NodeId start, std::optional<NodeId> target, bool do_reset) const
{
  ZoneScopedNC("GraphImpl::bfs", tracy_color);
  if (do_reset)
  {
    resetFlags();
  }
  std::vector<NodeId> out;
  std::vector<NodeId> open{ start };
  while (not open.empty())
  {
    NodeId current = open.back();
    if (_nodes.at(current).flag_1)
    {
      // already seen
      open.pop_back();
      continue;
    }
    out.push_back(current);
    _nodes.at(current).flag_1 = true;
    open.pop_back();
    if (target.has_value())
    {
      if (current == target)
      {
        return out;
      }
    }
    auto new_edges = edges_from(current);
    for (EdgeId edge_id : new_edges)
    {
      NodeId new_node = _edges.at(edge_id)._end;
      if (not _nodes.at(new_node).flag_1)  // not visited already
      {
        open.push_back(_edges.at(edge_id)._end);
      }
    }
  }
  return out;
}

std::string DiGraph::toString() const
{
  std::string out =
      "Graph with " + std::to_string(_nodes.size()) + " Nodes and " + std::to_string(_edges.size()) + "Edges";
  for (auto node : _nodes)
  {
    out += node.toString() + "\n";
  }
  for (auto edge : _edges)
  {
    out += edge.toString() + "\n";
  }
  return out;
}

DiGraph::DiGraph(std::size_t nodes_num, std::vector<Edge> const& edges)
  : _nodes(nodes_num), _edges(edges.size()), _graph(nodes_num)
{
  for (NodeId node_id = 0; node_id < nodes_num; ++node_id)
  {
    _nodes.at(node_id) = Node{ ._id = NodeId{ node_id } };
  }
  for (Edge const& edge : edges)
  {
    assert([&] {  // NOLINT
      if (edge._begin >= nodes_num or edge._end >= nodes_num)
      {
        LOG_FATAL("node id is higher than number of nodes!");
        return false;
      }
      return true;
    }());
    _edges.at(edge._id) = edge;
    _graph.at(edge._begin).push_back(edge._id);
    _graph.at(edge._end).push_back(edge._id);
  }
}

Graph::Graph(std::size_t num_nodes, std::vector<Edge> edges)
  : DiGraph([&] {
    std::vector<Edge> uni_edges;
    uni_edges.reserve((2 * edges.size()));
    for (auto const& edge : edges)
    {
      uni_edges.emplace_back(EdgeId{ 2 * edge._id }, edge._begin, edge._end, edge._weight);
      uni_edges.emplace_back(EdgeId{ 2 * edge._id + 1 }, edge._end, edge._begin, edge._weight);
    }
    return DiGraph(num_nodes, std::move(uni_edges));
  }())
{
}

}  // namespace ttb::graph_impl