#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/multi_linestring.hpp>

namespace ttb
{

/// 1D closed set
struct Interval1D
{
  COMPONENT d1;
  struct SimpleInterval
  {
    double min;
    double max;
  };
  std::vector<SimpleInterval> _intervals;
  /// does the polygon contains the to the a,b components projected point?
  /// If the Vector does not contain the COMPONENT d1, return true, e.g., Interval Z, Point XY -> true
  [[nodiscard]] bool contains(Vector const& point, Components const& comps) const;
  /// return the area of this field of view
  [[nodiscard]] double area() const;
  /// merge this with other field of view
  [[nodiscard]] Interval1D merge(Interval1D const& other) const;
  /// intersect this with other field of view
  [[nodiscard]] Interval1D intersect(Interval1D const& other) const;
  [[nodiscard]] std::string toString(std::string prefix = "") const;
};

/// 2D closed set
struct Polygon2D
{
  COMPONENT d1;
  COMPONENT d2;
  using Point = boost::geometry::model::d2::point_xy<double>;
  using Polygon = boost::geometry::model::polygon<Point>;
  using MultiPolygon = boost::geometry::model::multi_polygon<Polygon>;
  MultiPolygon multiPolygon = {};
  /// does the polygon contains the to the a,b components projected point?
  /// If the Vector does not contain the COMPONENT d1, d2, return true, e.g., Polygon XY, Point X -> true
  [[nodiscard]] bool contains(Vector const& point, Components const& comps) const;
  /// return the area of this field of view
  [[nodiscard]] double area() const;
  /// merge this with other field of view
  [[nodiscard]] Polygon2D merge(Polygon2D const& other) const;
  /// intersect this with other field of view
  [[nodiscard]] Polygon2D intersect(Polygon2D const& other) const;
  [[nodiscard]] Polygon2D difference(Polygon2D const& other) const;
  [[nodiscard]] std::string toString(std::string prefix = "") const;
};

/// represents the field of view of an sensor
/// currently supported are 2D Polygons, 1D intervals, and cartesian products of that,
///     e.g., a polygon for the XY-POS and the other components with min/max values
class FieldOfView
{
public:
  /// check whether the given point is inside this field of view
  [[nodiscard]] bool contains(Vector const& point, Components const& comps) const;
  /// return the area of this field of view for the given Components
  [[nodiscard]] double area(Components const&) const;
  /// merge this with other field of view, common components must be defined with the same type (Interval or Polygon)
  [[nodiscard]] FieldOfView merge(FieldOfView const& other) const;
  /// intersect this with other field of view
  [[nodiscard]] FieldOfView intersect(FieldOfView const& other) const;
  /// string representation
  [[nodiscard]] std::string toString(std::string prefix = "") const;
  /// check whether all areas > 0 and every COMPONENT only once
  [[nodiscard]] bool isValid() const;
  /// transform this fov
  void transform(Vector3 const& trans, Matrix33 const& rot);
  std::map<COMPONENT, Interval1D> _intervals{};
  std::map<std::vector<COMPONENT>, Polygon2D> _polygons{};
};

/// Dynamic Sensor Information with respect to the tracking frame
class SensorInformation
{
public:
  ///  The transformation from tracking coordinates to sensor coordinates
  SE3Trafo _to_sensor_cs{ SE3Trafo::Identity() };
  /// Field of view of the sensor in tracking coords
  std::optional<FieldOfView> _sensor_fov{};
  /// string representation
  [[nodiscard]] std::string toString(std::string prefix = "") const;
};

}  // namespace ttb