#include "tracking_lib/Misc/Region.h"

// silence boost
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <boost/geometry/algorithms/union.hpp>
#include <boost/geometry/algorithms/is_valid.hpp>
#include <boost/geometry/algorithms/difference.hpp>
#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/algorithms/simplify.hpp>

namespace ttb
{

template bool Interval::contains(Vector<double> const& point, Components const& comps) const;
template bool Interval::contains(Vector<AD> const& point, Components const& comps) const;
template <class UDT>
bool Interval::contains(Vector<UDT> const& point, Components const& comps) const
{
  std::optional<Indices> const inds = comps.indexOf(_comps);
  if (not inds.has_value())
  {
    return true;
  }
  assert(inds.value().rows() == 1);
  return boost::geometry::within(Point(to_val(point(inds.value()(0)))), _multi_line);
}

double Interval::area() const
{
  return boost::geometry::length(_multi_line);
}

Interval Interval::merge(Interval const& other) const
{
  assert([&] {  // NOLINT
    if (other._comps != _comps)
    {
      LOG_FATAL("Merge of Intervals with different components: " + to_string(_comps) +
                " and: " + to_string(other._comps));
      std::terminate();
      return false;
    }
    return true;
  }());
  MultiLineString merge;
  boost::geometry::union_(_multi_line, other._multi_line, merge);
  MultiLineString simplified;
  boost::geometry::simplify(merge, simplified, 0.1);
  return { ._comps = _comps, ._multi_line = std::move(simplified) };
}

Interval Interval::intersect(Interval const& other) const
{
  assert([&] {  // NOLINT
    if (other._comps != _comps)
    {
      LOG_FATAL("Intersect of Intervals with different components: " + to_string(_comps) +
                " and: " + to_string(other._comps));
      std::terminate();
      return false;
    }
    return true;
  }());
  MultiLineString intersect;
  boost::geometry::intersection(_multi_line, other._multi_line, intersect);
  return { ._comps = _comps, ._multi_line = std::move(intersect) };
}

std::string Interval::toString(std::string const& prefix) const
{
  std::string out = prefix + "Intervals\n";
  for (LineString const& simple : _multi_line)
  {
    out += prefix +
           "|\t[";  // + std::to_string(simple.at(0).get<0>()) + ", " + std::to_string(simple.at(1).get<0>()) + "]\n";

    for (Point const& point : simple)
    {
      out += std::to_string(point._point.x()) + ", ";
    }
    out += "]\n";
  }
  return out;
}

template bool Polygon::contains(Vector<double> const& point, Components const& comps) const;
template bool Polygon::contains(Vector<AD> const& point, Components const& comps) const;
template <class UDT>
bool Polygon::contains(Vector<UDT> const& point, Components const& comps) const
{
  std::optional<Indices> const inds = comps.indexOf(_comps);
  if (not inds.has_value())
  {
    return true;
  }
  assert(inds.value().rows() == 2);
  Point const p{ to_val(point(inds.value()(0))), to_val(point(inds.value()(1))) };
  return boost::geometry::within(p, multiPolygon);
}

double Polygon::area() const
{
  return boost::geometry::area(multiPolygon);
}

Polygon Polygon::merge(Polygon const& other) const
{
  assert([&] {  // NOLINT
    if (other._comps != _comps)
    {
      LOG_FATAL("Merge of Polygons with different components: " + to_string(_comps) +
                " and: " + to_string(other._comps));
      std::terminate();
      return false;
    }
    return true;
  }());
  MultiPolygon merge;
  boost::geometry::union_(multiPolygon, other.multiPolygon, merge);
  MultiPolygon simplified;
  boost::geometry::simplify(merge, simplified, 0.1);
  return { ._comps = _comps, .multiPolygon = std::move(simplified) };
}

Polygon Polygon::intersect(Polygon const& other) const
{
  assert([&] {  // NOLINT
    if (other._comps != _comps)
    {
      LOG_FATAL("Intersect of Polygons with different components: " + to_string(_comps) +
                " and: " + to_string(other._comps));
      std::terminate();
      return false;
    }
    return true;
  }());
  MultiPolygon intersect;
  boost::geometry::intersection(multiPolygon, other.multiPolygon, intersect);
  return { ._comps = _comps, .multiPolygon = std::move(intersect) };
}

Polygon Polygon::difference(Polygon const& other) const
{
  assert([&] {  // NOLINT
    if (other._comps != _comps)
    {
      LOG_FATAL("Difference of Polygons with different components: " + to_string(_comps) +
                " and: " + to_string(other._comps));
      std::terminate();
      return false;
    }
    return true;
  }());
  MultiPolygon intersect;
  boost::geometry::difference(multiPolygon, other.multiPolygon, intersect);
  return { ._comps = _comps, .multiPolygon = std::move(intersect) };
}

std::string Polygon::toString(std::string prefix) const
{
  std::string out{ prefix + "Polygons\n" };
  for (auto const& polygon : multiPolygon)
  {
    out += prefix + "|\tPolygon Outer\n";
    for (auto const& point : polygon.outer())
    {
      out +=
          prefix + "|\t|\t" + "(" + std::to_string(point._point.x()) + ", " + std::to_string(point._point.y()) + ")\n";
    }
    if (not polygon.inners().empty())
    {
      out += prefix + "|\tPolygon Inners\n";
      for (auto const& inner : polygon.inners())
      {
        out += prefix + "|\t|\tInner\n";
        for (auto const& point : inner)
        {
          out += prefix + "|\t|\t|\t" + "(" + std::to_string(point._point.x()) + ", " +
                 std::to_string(point._point.y()) + "\n";
        }
      }
    }
  }
  return out;
}

template bool Region::contains(Vector<double> const& point, Components const& comps) const;
template bool Region::contains(Vector<AD> const& point, Components const& comps) const;
template <class UDT>
bool Region::contains(Vector<UDT> const& point, Components const& comps) const
{
  for (auto const& [_, inter] : _intervals)
  {
    if (not inter.contains(point, comps))
    {
      return false;
    }
  }
  for (auto const& [_, poly] : _polygons)
  {
    if (not poly.contains(point, comps))
    {
      return false;
    }
  }
  return true;
}

double Region::area(Components const& comps) const
{
  double prod = 1;
  for (auto const& [c, poly] : _polygons)
  {
    if (comps.indexOf(c).has_value())
    {
      prod *= poly.area();
    }
  }
  for (auto const& [c, inter] : _intervals)
  {
    if (comps.indexOf(c).has_value())
    {
      prod *= inter.area();
    }
  }
  return prod;
}

Region Region::merge(Region const& other) const
{
  Region merged;
  std::vector<std::vector<COMPONENT>> other_poly_processed;
  for (auto const& [c, poly] : _polygons)
  {
    if (other._polygons.contains(c))
    {
      merged._polygons.emplace(c, poly.merge(other._polygons.at(c)));
      other_poly_processed.push_back(c._comps);
    }
    else
    {
      merged._polygons.emplace(c, poly);
    }
  }
  for (auto const& [other_c, other_poly] : other._polygons)
  {
    if (std::ranges::find(other_poly_processed, other_c._comps) == other_poly_processed.end())
    {
      merged._polygons.emplace(other_c, other_poly);
    }
  }
  std::vector<COMPONENT> other_inter_processed;
  for (auto const& [c, inter] : _intervals)
  {
    if (other._intervals.contains(c))
    {
      merged._intervals.emplace(c, inter.merge(other._intervals.at(c)));
      other_inter_processed.push_back(c._comps.at(0));
    }
    else
    {
      merged._intervals.emplace(c, inter);
    }
  }
  for (auto const& [other_c, other_inter] : other._intervals)
  {
    if (std::ranges::find(other_inter_processed, other_c._comps.at(0)) == other_inter_processed.end())
    {
      merged._intervals.emplace(other_c, other_inter);
    }
  }
  std::vector<COMPONENT> all_comps;
  for (auto const& [c, _] : merged._polygons)
  {
    if (std::ranges::find(all_comps, c._comps.at(0)) != all_comps.end())
    {
      LOG_FATAL("Component: " + to_string(c._comps.at(0)) + " multiple time in merge");
      throw std::runtime_error("Component: " + to_string(c._comps.at(0)) + " multiple time in merge");
    }
    all_comps.push_back(c._comps.at(0));
    if (std::ranges::find(all_comps, c._comps.at(1)) != all_comps.end())
    {
      LOG_FATAL("Component: " + to_string(c._comps.at(0)) + " multiple time in merge");
      throw std::runtime_error("Component: " + to_string(c._comps.at(1)) + " multiple time in merge");
    }
    all_comps.push_back(c._comps.at(1));
  }
  for (auto const& [c, _] : _intervals)
  {
    if (std::ranges::find(all_comps, c._comps.at(0)) != all_comps.end())
    {
      LOG_FATAL("Component: " + to_string(c._comps.at(0)) + " multiple time in merge");
      throw std::runtime_error("Component: " + to_string(c._comps.at(0)) + " multiple time in merge");
    }
    all_comps.push_back(c._comps.at(0));
  }
  return merged;
}

Region Region::intersect(Region const& other) const
{
  Region merged;
  for (auto const& [c, poly] : _polygons)
  {
    if (other._polygons.contains(c))
    {
      merged._polygons.emplace(c, poly.intersect(other._polygons.at(c)));
    }
  }
  for (auto const& [c, inter] : _intervals)
  {
    if (other._intervals.contains(c))
    {
      merged._intervals.emplace(c, inter.intersect(other._intervals.at(c)));
    }
  }
  return merged;
}

std::string Region::toString(std::string prefix) const
{
  std::string out = prefix + "FieldOfView\n";
  for (auto const& [c, poly] : _polygons)
  {
    out += prefix + "|\tComponents: " + to_string(c._comps.at(0)) + ", " + to_string(c._comps.at(1)) + "\n";
    out += prefix + poly.toString("|\t");
  }
  for (auto const& [c, inter] : _intervals)
  {
    out += prefix + "|\tComponent: " + to_string(c._comps.at(0)) + "\n";
    out += prefix + inter.toString("|\t");
  }
  return out;
}

bool Region::isValid() const
{
  std::vector<COMPONENT> seen;
  for (auto const& [c, inter] : _intervals)
  {
    if (std::ranges::find(seen, c._comps.at(0)) != seen.end())
    {
      LOG_FATAL("COMPONENT: " + to_string(c._comps.at(0)) + " already seen");
      return false;
    }
    seen.push_back(c._comps.at(0));
    if (inter.area() <= 0)
    {
      LOG_FATAL("Area of Interval of Component: " + to_string(c._comps.at(0)) + " <= 0");
      return false;
    }
  }

  for (auto const& [cs, poly] : _polygons)
  {
    for (COMPONENT c : cs._comps)
    {
      if (std::ranges::find(seen, c) != seen.end())
      {
        LOG_FATAL("COMPONENT: " + to_string(c) + " already seen");
        return false;
      }
      seen.push_back(c);
    }
    std::string reason;
    if (not boost::geometry::is_valid(poly.multiPolygon, reason))
    {
      LOG_FATAL("Polygon " + poly.toString() + " invalid according to boost");
      LOG_FATAL("Reason: " + reason);
      return false;
    }
    if (poly.area() <= 0)
    {
      LOG_FATAL("Area of Polygon of first Component: " + to_string(cs._comps.front()) + " <= 0");
      return false;
    }
  }
  return true;
}

void Region::transform(SE3Trafo const& trafo, SE3Trafo const& default_tracking_to_default_sensor_coords)
{
  for (auto& [cs, multiPoly] : _polygons)
  {
    for (auto& poly : multiPoly.multiPolygon)
    {
      for (auto& point : poly.outer())
      {
        point._point = trafo * default_tracking_to_default_sensor_coords * point._point.homogeneous();
      }
      for (auto& inner : poly.inners())
      {
        for (auto& point : inner)
        {
          point._point = trafo * default_tracking_to_default_sensor_coords * point._point.homogeneous();
        }
      }
    }
  }
}

std::map<COMPONENT, double> Region::center(Components const& comps) const
{
  auto center = boost::geometry::return_centroid<Point>(_polygons.at(comps).multiPolygon);
  std::map<COMPONENT, double> centerOfComp;
  for (auto& comp : comps._comps)
  {
    centerOfComp[comp] = 0.0;
  }
  if (centerOfComp.contains(COMPONENT::POS_X))
  {
    centerOfComp[COMPONENT::POS_X] = center._point.x();
  }
  else
  {
    LOG_FATAL("No POS_X in Region::_roi");
  }
  if (centerOfComp.contains(COMPONENT::POS_Y))
  {
    centerOfComp[COMPONENT::POS_Y] = center._point.y();
  }
  else
  {
    LOG_FATAL("No POS_Y in Region::_roi");
  }
  return centerOfComp;
}

}  // namespace ttb
#pragma GCC diagnostic pop