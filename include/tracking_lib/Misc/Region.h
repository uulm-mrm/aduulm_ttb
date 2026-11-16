#pragma once

#include "tracking_lib/TTBTypes/Components.h"

#include <type_traits>
#include <boost/geometry/algorithms/area_result.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/multi_linestring.hpp>
#include <boost/geometry/geometries/register/point.hpp>

namespace ttb
{
struct Point
{
  Vector3 _point;
  Point() : Point(second_dimension_constant, second_dimension_constant, second_dimension_constant)
  {
  }
  explicit Point(double x) : Point(x, second_dimension_constant, second_dimension_constant)
  {
  }
  Point(double x, double y) : Point(x, y, second_dimension_constant)
  {
  }
  Point(double x, double y, double z) : _point(x, y, z)
  {
  }

private:
  static constexpr std::integral_constant<int, 0> second_dimension_constant{};
};
}  // namespace ttb

BOOST_GEOMETRY_REGISTER_POINT_2D(ttb::Point, double, boost::geometry::cs::cartesian, _point.x(), _point.y())

namespace ttb
{
/// 1D closed set
struct Interval
{
  Components _comps;
  using LineString = boost::geometry::model::linestring<Point>;
  using MultiLineString = boost::geometry::model::multi_linestring<LineString>;
  MultiLineString _multi_line{};
  /// does the polygon contains the to the a,b components projected point?
  /// If the Vector does not contain the COMPONENT d1, return true, e.g., Interval Z, Point XY -> true
  template <class UDT>
  [[nodiscard]] bool contains(Vector<UDT> const& point, Components const& comps) const;
  /// return the area of this field of view
  [[nodiscard]] double area() const;
  /// merge this with other field of view
  [[nodiscard]] Interval merge(Interval const& other) const;
  /// intersect this with other field of view
  [[nodiscard]] Interval intersect(Interval const& other) const;
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
};

/// 2D closed set
struct Polygon
{
  Components _comps;
  using SinglePolygon = boost::geometry::model::polygon<Point>;
  using MultiPolygon = boost::geometry::model::multi_polygon<SinglePolygon>;
  MultiPolygon multiPolygon{};
  /// does the polygon contains the to the a,b components projected point?
  /// If the Vector does not contain the COMPONENT d1, d2, return true, e.g., Polygon XY, Point X -> true
  template <class UDT>
  [[nodiscard]] bool contains(Vector<UDT> const& point, Components const& comps) const;
  /// return the area of this field of view
  [[nodiscard]] double area() const;
  /// merge this with other field of view
  [[nodiscard]] Polygon merge(Polygon const& other) const;
  /// intersect this with other field of view
  [[nodiscard]] Polygon intersect(Polygon const& other) const;
  [[nodiscard]] Polygon difference(Polygon const& other) const;
  [[nodiscard]] std::string toString(std::string prefix = "") const;
};

/// represents a region in a space
/// currently supported are 2D Polygons, 1D intervals, and cartesian products of that,
///     e.g., a polygon for the XY-POS and the other components with min/max values
class Region
{
public:
  /// check whether the given point is inside this field of view
  template <class UDT>
  [[nodiscard]] bool contains(Vector<UDT> const& point, Components const& comps) const;
  /// return the area of this field of view for the given Components
  [[nodiscard]] double area(Components const&) const;
  /// merge this with other field of view, common components must be defined with the same type (Interval or Polygon)
  [[nodiscard]] Region merge(Region const& other) const;
  /// intersect this with other field of view
  [[nodiscard]] Region intersect(Region const& other) const;
  /// string representation
  [[nodiscard]] std::string toString(std::string prefix = "") const;
  /// check whether all areas > 0 and every COMPONENT only once
  [[nodiscard]] bool isValid() const;
  /// transform this fov
  void transform(SE3Trafo const& trafo, SE3Trafo const& default_tracking_to_default_sensor_coords);
  /// center point per component
  [[nodiscard]] std::map<COMPONENT, double> center(Components const& comps) const;

  std::map<Components, Interval> _intervals{};
  std::map<Components, Polygon> _polygons{};
};

}  // namespace ttb