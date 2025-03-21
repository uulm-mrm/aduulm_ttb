#include "gtest/gtest.h"
#include "tracking_lib/Graph/GraphImpl.h"

#include "tracking_lib/Misc/logger_setup.h"

namespace ttb
{

TEST(DiGraphImpl, print_bfs)
{
  graph_impl::DiGraph g(
      3,
      { graph_impl::Edge{ graph_impl::EdgeId{ 0 }, graph_impl::NodeId{ 0 }, graph_impl::NodeId{ 2 }, 1.1 },
        graph_impl::Edge{ graph_impl::EdgeId{ 1 }, graph_impl::NodeId{ 0 }, graph_impl::NodeId{ 1 }, 2.2 },
        graph_impl::Edge{ graph_impl::EdgeId{ 2 }, graph_impl::NodeId{ 1 }, graph_impl::NodeId{ 2 }, 9.9 } });
  std::cout << g.toString() << std::endl;
  auto component = g.bfs(graph_impl::NodeId{ 0 });
  std::sort(component.begin(), component.end());
  for (auto node_id : component)
  {
    std::cout << node_id << std::endl;
  }
  EXPECT_EQ(component.at(0), 0);
  EXPECT_EQ(component.at(1), 1);
  EXPECT_EQ(component.at(2), 2);
  component = g.bfs(graph_impl::NodeId{ 1 });
  std::sort(component.begin(), component.end());
  EXPECT_EQ(component.at(0), 1);
  EXPECT_EQ(component.at(1), 2);
  component = g.bfs(graph_impl::NodeId{ 2 });
  std::ranges::sort(component);
  EXPECT_EQ(component.at(0), 2);
  EXPECT_EQ(component.size(), 1);
  EXPECT_EQ(g.edges_from(graph_impl::NodeId{ 1 }).at(0), 2);
  EXPECT_EQ(g.edges_to(graph_impl::NodeId{ 1 }).at(0), 1);
  EXPECT_EQ(g.edges_from(graph_impl::NodeId{ 0 }).size(), 2);
  EXPECT_EQ(g.edges_from(graph_impl::NodeId{ 2 }).size(), 0);
}

TEST(DiGraphImpl, components)
{
  graph_impl::DiGraph g(
      4,
      { graph_impl::Edge{ graph_impl::EdgeId{ 0 }, graph_impl::NodeId{ 0 }, graph_impl::NodeId{ 1 }, 1.1 },
        graph_impl::Edge{ graph_impl::EdgeId{ 1 }, graph_impl::NodeId{ 1 }, graph_impl::NodeId{ 0 }, 1.1 },
        graph_impl::Edge{ graph_impl::EdgeId{ 2 }, graph_impl::NodeId{ 2 }, graph_impl::NodeId{ 3 }, 2.2 },
        graph_impl::Edge{ graph_impl::EdgeId{ 3 }, graph_impl::NodeId{ 3 }, graph_impl::NodeId{ 2 }, 2.2 } });
  std::cout << g.toString() << std::endl;
  auto first_comp = g.bfs(graph_impl::NodeId{ 0 });
  for (auto comp : first_comp)
  {
    std::cout << comp << " ";
  }
  std::cout << std::endl << std::endl;
  auto comps = g.components();
  EXPECT_EQ(comps.size(), 2);
  for (auto const& comp : comps)
  {
    std::cout << "Comp: ";
    for (graph_impl::NodeId i : comp)
    {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }
}

TEST(GraphImpl, Graph)
{
  graph_impl::Graph g(
      4,
      { graph_impl::Edge{ graph_impl::EdgeId{ 0 }, graph_impl::NodeId{ 0 }, graph_impl::NodeId{ 1 }, 1.1 },
        graph_impl::Edge{ graph_impl::EdgeId{ 1 }, graph_impl::NodeId{ 3 }, graph_impl::NodeId{ 2 }, 2.2 } });
  std::cout << g.toString() << std::endl;
  auto comps = g.components();
  for (auto const& comp : comps)
  {
    std::cout << "Comp: ";
    for (graph_impl::NodeId i : comp)
    {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }
  EXPECT_EQ(comps.size(), 2);
}

}  // namespace ttb