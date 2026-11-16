#include "gtest/gtest.h"
#include "tracking_lib/Graph/Graph.h"

#include "tracking_lib/Misc/logger_setup.h"

namespace ttb
{

TEST(IndexedDiGraph, IndexedDiGraph)
{
  graph::DiGraph<std::string, std::size_t, double> g(
      { "node_1", "node_2", "node_3", "node_6" },
      { { 0, "node_1", "node_2", 3 }, { 2, "node_2", "node_1", 1.1 }, { 9, "node_3", "node_6", 2.2 } });
  auto comps = g.components();
  EXPECT_EQ(g.bfs("node_1").size(), 2);
  EXPECT_EQ(g.bfs("node_3").size(), 2);
  EXPECT_EQ(g.bfs("node_6").size(), 1);
}

TEST(IndexedDiGraph, edgesHaveDifferentEndNodes)
{
  graph::DiGraph<int, int, double> g({ 0, 1, 2, 3, 4, 5 },
                                     { { 0, 0, 1, 1 },
                                       { 1, 0, 3, 1 },
                                       { 7, 1, 2, 0.4 },
                                       { 9, 1, 2, 0.35 },
                                       { 10, 1, 2, 0.25 },
                                       { 12, 3, 4, 0.2 },
                                       { 5, 3, 4, 0.8 },
                                       { 2, 2, 5, 1 },
                                       { 3, 4, 5, 1 } });
  auto comps = g.components();
  EXPECT_EQ(comps.size(), 1);
  EXPECT_EQ(g.bfs(0).size(), 6);

  ASSERT_EQ(g.edges_have_same_end_nodes(0), false);
}

TEST(IndexedDiGraph, edgesHaveSameEndNode)
{
  graph::DiGraph<int, int, double> g({ 1, 2, 3 },
                                     { { 7, 1, 2, 0.4 }, { 9, 1, 2, 0.35 }, { 10, 1, 2, 0.25 }, { 1, 2, 3, 1 } });
  auto comps = g.components();
  ASSERT_EQ(g.edges_have_same_end_nodes(1), true);
}

std::string to_string(auto const& k_longest_paths)
{
  std::string out = "\nSolution of k_shortest_path\n";
  for (auto const& [path, cost] : k_longest_paths)
  {
    out += "Path: ";
    for (auto const& edge : path)
    {
      out += std::format("{} ", edge);
    }
    out += std::format(" cost: {}\n", cost);
  }
  return out;
}

TEST(kLongestPaths, GeneralizedKBest)
{
  double val1 = std::log(1);
  double val2 = std::log(2);
  graph::DiGraph<int, int, double> g({ 0, 1, 2 },
                                     { { 0, 0, 1, val1 }, { 1, 0, 1, val2 }, { 2, 1, 2, val1 }, { 3, 1, 2, val2 } });
  auto const res = g.k_longest_paths(0, 2, 5);
  EXPECT_EQ(res.size(), 4);
  EXPECT_EQ(std::exp(res.front().second), 1);
  EXPECT_EQ(std::exp(res.at(1).second), 2);
  EXPECT_EQ(std::exp(res.at(2).second), 2);
  EXPECT_EQ(std::exp(res.at(3).second), 4);

  auto const res2 = g.k_longest_paths(0, 2, 3);

  EXPECT_EQ(res2.size(), 3);
  EXPECT_EQ(std::exp(res2.front().second), 2);
  EXPECT_EQ(std::exp(res2.at(1).second), 2);
  EXPECT_EQ(std::exp(res2.at(2).second), 4);
}

TEST(kLongestPaths, GeneralizedKBest2)
{
  double val1 = std::log(1);
  double val2 = std::log(2);
  double val3 = std::log(3);
  double val4 = std::log(4);
  graph::DiGraph<int, std::string, double> g({ 0, 1, 2, 3, 4, 5 },
                                             { { "a", 0, 1, 0 },
                                               { "b", 0, 4, 0 },
                                               { "c", 2, 3, 0 },
                                               { "d", 5, 3, 0 },
                                               { "e", 1, 2, 0 },
                                               { "f", 1, 2, val2 },
                                               { "g", 4, 5, val3 },
                                               { "h", 4, 5, val4 } });
  auto const res = g.k_longest_paths(0, 3, 6);
  EXPECT_EQ(res.size(), 4);
  EXPECT_EQ(res.front().first.at(0), "a");
  EXPECT_EQ(res.front().first.at(1), "e");
  EXPECT_EQ(res.front().first.at(2), "c");
  EXPECT_EQ(std::exp(res.front().second), 1);
  EXPECT_EQ(std::exp(res.at(1).second), 2);
  EXPECT_NEAR(std::exp(res.at(2).second), 3, 1e-10);
  EXPECT_EQ(std::exp(res.at(3).second), 4);
}

TEST(kLongestPaths, GeneralizedKBest_FPM)
{
  // prior comp weights
  std::vector<int> factors;
  factors.push_back(1);
  factors.push_back(1);

  std::vector<std::vector<std::vector<int>>> sensor_weights;
  sensor_weights.resize(factors.size());
  sensor_weights[0] = { { 1, 2 }, { 1, 3 } };  // results for prior id 0
  sensor_weights[1] = { { 4, 3 }, { 1 } };     // results for prior id 1

  int numSensors = 2;
  int numPriorComps = 2;
  std::vector<int> nodes;
  std::vector<graph::Edge<int, int, double>> edges;
  int priorCompCounter = 1;
  int edgeId = 0;
  // save start and end node
  int startNode = 0;
  nodes.push_back(startNode);
  int highestPossibleVal = numPriorComps * (numSensors + 1) + 1;
  int endNode = highestPossibleVal;
  // Add nodes
  for (int priorComp = 0; priorComp < numPriorComps; priorComp++)
  {
    int sensorCounter = 1;
    // Add Dummy Edge containing weight of 1/(V-1)*priorCompWeight
    int first_node_id = (priorCompCounter - 1) * (numSensors + 1) + 1;
    edges.emplace_back(edgeId, startNode, first_node_id, -std::log(factors[priorComp]));
    edgeId++;
    for (std::size_t sensor = 0; sensor < numSensors; sensor++)
    {
      size_t node1 = (priorCompCounter - 1) * (numSensors + 1) + sensorCounter;
      sensorCounter++;
      size_t node2 = (priorCompCounter - 1) * (numSensors + 1) + sensorCounter;
      if (node2 == priorCompCounter * (numSensors + 1))
      {
        // last node of this prior component id
        node2 = endNode;
      }
      nodes.push_back(node1);
      nodes.push_back(node2);
      for (const auto& val : sensor_weights[priorComp][sensor])
      {
        edges.emplace_back(edgeId, node1, node2, std::log(val));
        edgeId++;
      }
    }
    priorCompCounter++;
  }
  // remove duplicated nodes
  sort(nodes.begin(), nodes.end());
  nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());

  graph::DiGraph<int, int, double> g(nodes, edges);
  std::vector<std::pair<std::vector<int>, double>> selections = g.k_longest_paths(startNode, endNode, 10);

  //   real solution GeneralizedKBest
  std::vector<std::pair<std::vector<int>, int>> true_solution;
  true_solution = {
    { std::pair<std::vector<int>, int>({ 0, 1, 3 }, 1) }, { std::pair<std::vector<int>, int>({ 0, 2, 3 }, 2) },
    { std::pair<std::vector<int>, int>({ 0, 1, 4 }, 3) }, { std::pair<std::vector<int>, int>({ 5, 7, 8 }, 3) },
    { std::pair<std::vector<int>, int>({ 5, 6, 8 }, 4) }, { std::pair<std::vector<int>, int>({ 0, 2, 4 }, 6) },
  };

  // compare
  ASSERT_EQ(true_solution.size(), selections.size());
  for (std::size_t solution = 0; solution < true_solution.size(); solution++)
  {
    ASSERT_NEAR(true_solution[solution].second, std::exp(selections[solution].second), 1e-10);
    ASSERT_EQ(true_solution[solution].first.size(), selections[solution].first.size());
    for (std::size_t elem = 0; elem < true_solution[solution].first.size(); elem++)
    {
      ASSERT_EQ(true_solution[solution].first[elem], selections[solution].first[elem]);
    }
  }
}

TEST(K_SHORTEST_PATH, add_tracks)
{
  using Edge = std::size_t;
  using Node = std::size_t;  // nodes have no meaning
  for (auto hyps : { 5, 500 })
  {
    for (auto tracks : { 1, 3, 5, 10 })
    {
      Node start_node{ 0 };
      Node end_node{ 1 };
      std::vector<Node> nodes{ start_node, end_node };  // start = 0
      nodes.reserve(2 + hyps * tracks);
      std::vector<graph::Edge<Edge, Node, double>> edges;
      edges.reserve(hyps * (2 + tracks * 2));
      std::size_t next_node = 2;
      Edge edge = 0;
      for (std::size_t hyp = 0; hyp < hyps; ++hyp)
      {
        double hyp_weight = ttb::Vector<double>::Random(1)(0);
        hyp_weight = std::abs(hyp_weight);
        edges.emplace_back(edge++, start_node, next_node, std::log(hyp_weight));
        for (std::size_t track = 0; track < tracks; ++track)
        {
          double prob = ttb::Vector<double>::Random(1)(0);
          prob = std::abs(prob);
          edges.emplace_back(edge++, next_node, next_node + 1, std::log(prob));
          edges.emplace_back(edge++, next_node, next_node + 1, std::log(1 - prob));
          nodes.push_back(next_node);
          ++next_node;
        }
        edges.emplace_back(edge++, next_node, end_node, 0);
        nodes.push_back(next_node);
        ++next_node;
      }
      graph::DiGraph graph(std::move(nodes), std::move(edges));
      auto best_paths = graph.k_longest_paths(start_node, end_node, 3000);
    }
  }
}

TEST(K_SHORTEST_PATH, predict_tracks)
{
  using Edge = std::size_t;
  using Node = std::size_t;  // nodes have no meaning
  for (auto hyps : { 5, 500 })
  {
    for (auto largest_num_tracks : { 0, 1, 3, 5 })
    {
      Node start_node{ 0 };
      Node end_node{ 1 };
      std::vector<Node> nodes{ start_node, end_node };  // start = 0

      nodes.reserve(2 + hyps * largest_num_tracks);
      std::vector<graph::Edge<Edge, Node, double>> edges;
      edges.reserve(hyps * (2 + largest_num_tracks * 2));
      std::size_t next_node = 2;
      Edge edge_ctr = 0;
      for (std::size_t hyp = 0; hyp < hyps; ++hyp)
      {
        std::size_t hyp_tracks = static_cast<std::size_t>(ttb::Vector<double>::Random(1)(0) * largest_num_tracks);
        double hyp_weight = std::abs(ttb::Vector<double>::Random(1)(0));
        if (largest_num_tracks > 0)
        {
          edges.emplace_back(edge_ctr++, start_node, next_node, std::log(hyp_weight));
          for (std::size_t i = 0; i < largest_num_tracks; ++i)
          {
            if (i < hyp_tracks)
            {
              double track_weight = std::abs(ttb::Vector<double>::Random(1)(0));
              edges.emplace_back(edge_ctr++, next_node, next_node + 1, std::log(track_weight));
              edges.emplace_back(edge_ctr++, next_node, next_node + 1, std::log(1 - track_weight));
              nodes.push_back(next_node);
              ++next_node;
            }
            else  // add dummy edge so all sub-graphs have the same size, this is needed for the k_shortest_path
                  // algorithm
            {
              edges.emplace_back(edge_ctr++, next_node, next_node + 1, 0);
              nodes.push_back(next_node);
              ++next_node;
            }
          }
          edges.emplace_back(edge_ctr++, next_node, end_node, 0);
          nodes.push_back(next_node);
          ++next_node;
        }
        else
        {
          edges.emplace_back(edge_ctr++, start_node, end_node, std::log(hyp_weight));
        }
      }
      graph::DiGraph graph(std::move(nodes), std::move(edges));
      auto best_paths = graph.k_longest_paths(start_node, end_node, 10000);
    }
  }
}

}  // namespace ttb
