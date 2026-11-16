#include "tracking_lib/Map/Map.h"

#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/TTBManager/Visualization.h"

#include <boost/geometry/algorithms/within.hpp>
#include <boost/geometry/algorithms/correct.hpp>

namespace ttb
{

template<class UDT>
constexpr auto check_overlap = [](State<UDT> const& track, Polygon::MultiPolygon poly) {
  auto const [model, dist] = track.getEstimate();
  auto const [xs, ys] = to_box(dist.mean(), track._manager->state_comps(model));
  std::vector<Point> points;

  for (auto [x, y] : std::views::zip(xs, ys))
  {
    points.emplace_back(x, y);
  }
  Polygon::SinglePolygon state_area;
  boost::geometry::assign_points(state_area, points);
  boost::geometry::correct(state_area);
  return boost::geometry::intersects(state_area, poly);
};

bool Building::contains(Vector2 const& xy_pos) const
{
  return boost::geometry::within(Point(xy_pos.x(), xy_pos.y()), footprint.multiPolygon);
}

template<class UDT>
bool Building::overlaps(State<UDT> const& state) const
{
  return check_overlap<UDT>(state, footprint.multiPolygon);
}

bool Way::contains(Vector2 const& xy_pos) const
{
  return boost::geometry::within(Point(xy_pos.x(), xy_pos.y()), outline.multiPolygon);
}

template<class UDT>
bool Way::overlaps(State<UDT> const& state) const
{
  return check_overlap<UDT>(state, outline.multiPolygon);
}

}  // namespace ttb