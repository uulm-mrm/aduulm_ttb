#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Misc/Region.h"

namespace ttb
{

struct Building
{
  Polygon footprint;
  /// check whether the xy-point is on this way
  [[nodiscard]] bool contains(Vector2 const& xy_pos) const;
  /// check whether the Vector overlaps with the way
  template<class UDT>
  [[nodiscard]] bool overlaps(State<UDT> const& state) const;
};

struct Way
{
  Interval centerline;
  Polygon outline;
  int num_lanes_forward{};
  int num_lanes_backward{};
  double total_width{};
  std::optional<double> speed_limit{};   ///< m/s
  std::vector<CLASS> usable_by{};
  /// check whether the xy-point is on this way
  [[nodiscard]] bool contains(Vector2 const& xy_pos) const;
  /// check whether the Vector overlaps with the way
  template<class UDT>
  [[nodiscard]] bool overlaps(State<UDT> const& state) const;
};

struct ParkingArea
{
  Polygon area;
};

struct TrafficSignal
{
};

struct Map
{
  std::vector<Building> buildings{};
  std::vector<Way> ways{};
  std::vector<ParkingArea> parking_areas{};
  std::vector<TrafficSignal> traffic_signals{};
};

}  // namespace ttb