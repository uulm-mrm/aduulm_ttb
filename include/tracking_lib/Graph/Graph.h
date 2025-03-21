#pragma once

#include "tracking_lib/Graph/GraphImpl.h"
#include <ranges>

namespace ttb::graph
{

template <class NodeId>
struct Node
{
  NodeId id;
};

template <class EdgeId, class NodeId, class Weight>
struct Edge
{
  EdgeId id;
  NodeId begin;
  NodeId end;
  Weight weight;
};

template <class NodeId, class EdgeId, class Weight>
class DiGraph
{
public:
  DiGraph() = default;
  DiGraph(std::vector<NodeId> ext_nodes, std::vector<Edge<EdgeId, NodeId, Weight>> ext_edges)
  {
    for (auto const& [intern_node_id, node] : std::views::enumerate(ext_nodes))
    {
      _ext2node.emplace(node, intern_node_id);
      _node2ext.emplace(intern_node_id, node);
    }
    std::vector<graph_impl::Edge> edges;
    edges.reserve(ext_edges.size());
    for (auto const& [intern_edge_id, edge] : std::views::enumerate(ext_edges))
    {
      _ext2edge.emplace(edge.id, intern_edge_id);
      _edge2ext.emplace(intern_edge_id, edge.id);
      edges.emplace_back(intern_edge_id, _ext2node.at(edge.begin), _ext2node.at(edge.end), edge.weight);
    }
    _diGraph = graph_impl::DiGraph(ext_nodes.size(), std::move(edges));
  }

  [[nodiscard]] std::string toString() const
  {
    std::string out = std::format("Graph with {} nodes and {} edges\n", _diGraph._nodes.size(), _diGraph._edges.size());
    for (auto const& [ext_node, node] : _ext2node)
    {
      out += std::format("Node: {}\n", ext_node);
    }
    for (auto const& [ext_edge, edge] : _ext2edge)
    {
      out += std::format("Edge {}: {} -> {}: {}\n",
                         ext_edge,
                         _node2ext.at(_diGraph._edges.at(edge)._begin),
                         _node2ext.at(_diGraph._edges.at(edge)._end),
                         _diGraph._edges.at(edge)._weight);
    }
    return out;
  }

  [[nodiscard]] std::vector<NodeId> bfs(NodeId start,
                                        std::optional<NodeId> target = std::nullopt,
                                        bool do_reset = true) const
  {
    auto const intern_target = [&]() -> std::optional<graph_impl::NodeId> {
      if (target.has_value())
      {
        return _ext2node.at(target.value());
      }
      return {};
    }();
    std::vector<graph_impl::NodeId> const intern_ids = _diGraph.bfs(_ext2node.at(start), intern_target, do_reset);
    std::vector<NodeId> out;
    for (graph_impl::NodeId inter_id : intern_ids)
    {
      out.push_back(_node2ext.at(inter_id));
    }
    return out;
  }

  /// find all connected components of the Graph using bfs
  [[nodiscard]] std::vector<std::vector<NodeId>> components() const
  {
    std::vector<std::vector<graph_impl::NodeId>> const intern_comps = _diGraph.components();
    std::vector<std::vector<NodeId>> comps;
    for (std::vector<graph_impl::NodeId> const& intern_comp : intern_comps)
    {
      std::vector<NodeId> comp;
      for (graph_impl::NodeId inter : intern_comp)
      {
        comp.push_back(_node2ext.at(inter));
      }
      comps.emplace_back(std::move(comp));
    }
    return comps;
  }

  /// find the k-shortest paths between two nodes in a graph with very special structure
  [[nodiscard]] std::vector<std::pair<std::vector<EdgeId>, Weight>> k_shortest_paths(NodeId start,
                                                                                     NodeId end,
                                                                                     std::size_t k) const
  {
    std::vector<std::pair<std::vector<graph_impl::EdgeId>, double>> intern_best =
        _diGraph.kShortestPath(SELECTION_STRATEGY::GENERALIZED_K_BEST, k, _ext2node.at(start), _ext2node.at(end));
    std::vector<std::pair<std::vector<EdgeId>, Weight>> out_paths;
    for (const auto& [intern_path, cost] : intern_best)
    {
      std::vector<EdgeId> out_path(intern_path.size());
      for (auto const& [idx, intern] : std::views::enumerate(intern_path))
      {
        out_path.at(idx) = _edge2ext.at(intern);
      }
      out_paths.emplace_back(std::move(out_path), cost);
    }
    return out_paths;
  }

  [[nodiscard]] bool edges_have_same_end_nodes(NodeId node) const
  {
    return _diGraph.edgesHaveSameEndNode(_ext2node.at(node));
  }

private:
  graph_impl::DiGraph _diGraph;
  std::map<NodeId, graph_impl::NodeId> _ext2node;
  std::map<graph_impl::NodeId, NodeId> _node2ext;
  std::map<EdgeId, graph_impl::EdgeId> _ext2edge;
  std::map<graph_impl::EdgeId, EdgeId> _edge2ext;
};

/// This represents the special case of a unidirectional IndexedGraph
/// Attention: Only adds reverted edges
template <class ENodeId, class EEdgeId, class EWeight>
class Graph : public DiGraph<ENodeId, EEdgeId, EWeight>
{
public:
  Graph(std::vector<ENodeId> ext_nodes, std::vector<Edge<EEdgeId, ENodeId, EWeight>> ext_edges)
    : DiGraph<ENodeId, EEdgeId, EWeight>([&] {
      std::vector<Edge<EEdgeId, ENodeId, EWeight>> uni_edges;
      uni_edges.reserve(2 * ext_edges.size());
      for (Edge<EEdgeId, ENodeId, EWeight> const& edge : ext_edges)
      {
        uni_edges.emplace_back(2 * edge.id, edge.begin, edge.end, edge.weight);
        uni_edges.emplace_back(2 * edge.id + 1, edge.end, edge.begin, edge.weight);
      }
      return DiGraph(std::move(ext_nodes), std::move(uni_edges));
    }())
  {
  }
};

}  // namespace ttb::graph