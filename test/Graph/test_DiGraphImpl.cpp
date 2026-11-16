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
  auto component = g.bfs(graph_impl::NodeId{ 0 });
  std::sort(component.begin(), component.end());

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
  auto first_comp = g.bfs(graph_impl::NodeId{ 0 });
  auto comps = g.components();
  EXPECT_EQ(comps.size(), 2);
}

}  // namespace ttb