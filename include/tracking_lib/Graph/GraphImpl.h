#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

#include <type_safe/strong_typedef.hpp>

namespace ttb::graph_impl
{

using NodeId = std::size_t;
using EdgeId = std::size_t;
using Weight = double;

struct Node
{
  NodeId _id;
  /// Flags to easy the implementation of certain algorithms
  mutable bool flag_1 = false;
  mutable bool flag_2 = false;
  [[nodiscard]] std::string toString() const;
};

/// ####################################################################
struct Edge
{
  EdgeId _id;
  NodeId _begin;
  NodeId _end;
  Weight _weight;
  /// Flags to easy the implementation of certain algorithms
  mutable bool flag_1 = false;
  mutable bool flag_2 = false;
  [[nodiscard]] std::string toString() const;
};

/// #################################################################################

/// This represents a (weighted) directional graph
/// It assumes (consecutive) indices starting from zero
class DiGraph
{
public:
  DiGraph() = default;
  DiGraph(std::size_t nodes_num, std::vector<Edge> const& edges);
  /// toString
  [[nodiscard]] std::string toString() const;
  /// perform a breath first search based on the start node start
  /// If no target node is specified all reachable nodes will be returned
  /// do_reset: reset the flags before the search
  [[nodiscard]] std::vector<NodeId> bfs(NodeId start,
                                        std::optional<NodeId> target = std::nullopt,
                                        bool do_reset = true) const;
  /// find all connected components of the Graph using a bfs
  [[nodiscard]] std::vector<std::vector<NodeId>> components() const;

  /// Find the k shortest paths between two nodes.
  /// Attention: The weight of a path is the PRODUCT of the edge weights!
  /// @param strategy strategy type
  /// @param k max number of solutions which shall be calculated
  /// @param startNodeID start node of graph
  /// @param endNodeID end node of graph
  /// @return vector of paths with product weight
  [[nodiscard]] std::vector<std::pair<std::vector<EdgeId>, double>>
  kShortestPath(SELECTION_STRATEGY strategy, std::size_t k, NodeId startNodeID, NodeId endNodeID) const;
  /// reset the Node and Edge Flags to false
  void resetFlags() const;
  /// return the outgoing edges from a node
  [[nodiscard]] std::vector<EdgeId> edges_from(NodeId node) const;
  /// return the incoming edges to a node
  [[nodiscard]] std::vector<EdgeId> edges_to(NodeId node) const;
  /// check if all edges starting at startNode ends at the SAME node
  [[nodiscard]] bool edgesHaveSameEndNode(NodeId startNode) const;
  /// Node_Id -> Nodes
  std::vector<Node> _nodes;
  /// Edge_Id -> Edges
  std::vector<Edge> _edges;
  /// Node_Id -> vector of adjacent Edges
  std::vector<std::vector<EdgeId>> _graph;
};

/// This represents the special case of a unidirectional Graph
/// This duplicates all specified edges and builds a directional graph
class Graph : public DiGraph
{
public:
  Graph(std::size_t num_nodes, std::vector<Edge> edges);
};

}  // namespace ttb::graph_impl