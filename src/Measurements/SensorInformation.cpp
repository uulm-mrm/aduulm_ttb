#include "tracking_lib/Measurements/SensorInformation.h"

#include <boost/geometry/strategies/strategies.hpp>
#include <boost/geometry/algorithms/union.hpp>
#include <boost/geometry/algorithms/is_valid.hpp>
#include <boost/geometry/algorithms/difference.hpp>
#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/within.hpp>
#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/algorithms/simplify.hpp>
#include <regex>

namespace ttb
{

bool Interval1D::contains(Vector const& point, Components const& comps) const
{
  auto ind = comps.indexOf(d1);
  if (not ind.has_value())
  {
    return true;
  }
  return std::any_of(_intervals.begin(), _intervals.end(), [&](SimpleInterval const& inter) {
    return inter.min <= point(ind.value()) and point(ind.value()) <= inter.max;
  });
}

double Interval1D::area() const
{
  return std::accumulate(_intervals.begin(), _intervals.end(), 0.0, [&](double old, SimpleInterval const& inter) {
    return old + inter.max - inter.min;
  });
}

Interval1D Interval1D::merge(Interval1D const& other) const
{
  if (d1 != other.d1)
  {
    LOG_FATAL("Merge of Intervals with different components: " + to_string(d1) + " and: " + to_string(other.d1));
    throw std::runtime_error("Merge of Intervals with different components: " + to_string(d1) +
                             " and: " + to_string(other.d1));
  }
  std::vector<SimpleInterval> all;
  all.insert(all.end(), _intervals.begin(), _intervals.end());
  all.insert(all.end(), other._intervals.begin(), other._intervals.end());
  std::sort(all.begin(), all.end(), [](auto const& a, auto const& b) { return a.min < b.min; });
  // iteratively merge 2 intervals until all merged
  std::vector<SimpleInterval> merged;
  while (true)
  {
    bool done_merge = false;
    merged = {};
    for (std::size_t i = 0; i + 1 < all.size(); ++i)
    {
      if (all.at(i).max >= all.at(i + 1).min)  // overlapping interval
      {
        merged.push_back(
            { .min = std::min(all.at(i).min, all.at(i + 1).min), .max = std::max(all.at(i).max, all.at(i + 1).max) });
        done_merge = true;
        for (std::size_t j = 0; j < all.size(); ++j)
        {
          if (j != i and j != i + 1)
          {
            merged.push_back(all.at(j));
          }
        }
        all = std::move(merged);
        std::sort(all.begin(), all.end(), [](auto const& a, auto const& b) { return a.min < b.min; });
        break;
      }
    }
    if (not done_merge)
    {
      return { .d1 = d1, ._intervals = std::move(all) };
    }
  }
}

Interval1D Interval1D::intersect(Interval1D const& other) const
{
  if (d1 != other.d1)
  {
    LOG_FATAL("Intersect of Intervals with different components: " + to_string(d1) + " and: " + to_string(other.d1));
    throw std::runtime_error("Intersect of Intervals with different components: " + to_string(d1) +
                             " and: " + to_string(other.d1));
  }
  std::vector<SimpleInterval> intersections;
  for (SimpleInterval const& my : _intervals)
  {
    for (SimpleInterval const& ot : _intervals)
    {
      double min = std::max(my.min, ot.min);
      double max = std::min(my.max, ot.max);
      if (min <= max)
      {
        intersections.push_back({ .min = min, .max = max });
      }
    }
  }
  return { .d1 = d1, ._intervals = std::move(intersections) };
}

std::string Interval1D::toString(std::string prefix) const
{
  std::string out = prefix + "Intervals\n";
  for (SimpleInterval const& simple : _intervals)
  {
    out += prefix + "|\t[" + std::to_string(simple.min) + ", " + std::to_string(simple.max) + "]\n";
  }
  return out;
}

bool Polygon2D::contains(Vector const& point, Components const& comps) const
{
  auto ind = comps.indexOf({ d1, d2 });
  if (not ind.has_value())
  {
    return true;
  }
  Point const p{ point(ind.value()(0)), point(ind.value()(1)) };
  return boost::geometry::within(p, multiPolygon);
}

double Polygon2D::area() const
{
  return boost::geometry::area(multiPolygon);
}

Polygon2D Polygon2D::merge(Polygon2D const& other) const
{
  if (d1 != other.d1 or d2 != other.d2)
  {
    LOG_FATAL("Merge of Annuli with different components: " + to_string(d1) + " and: " + to_string(other.d1) +
              to_string(d2) + " and: " + to_string(other.d2));
    throw std::runtime_error("Merge of Annuli with different components: " + to_string(d1) +
                             " and: " + to_string(other.d1) + to_string(d2) + " and: " + to_string(other.d2));
  }
  MultiPolygon merge;
  boost::geometry::union_(multiPolygon, other.multiPolygon, merge);
  MultiPolygon simplified;
  boost::geometry::simplify(merge, simplified, 0.1);
  return { .d1 = d1, .d2 = d2, .multiPolygon = std::move(merge) };
}

Polygon2D Polygon2D::intersect(Polygon2D const& other) const
{
  if (d1 != other.d1 or d2 != other.d2)
  {
    LOG_FATAL("Merge of Annuli with different components: " + to_string(d1) + " and: " + to_string(other.d1) +
              to_string(d2) + " and: " + to_string(other.d2));
    throw std::runtime_error("Merge of Annuli with different components: " + to_string(d1) +
                             " and: " + to_string(other.d1) + to_string(d2) + " and: " + to_string(other.d2));
  }
  MultiPolygon intersect;
  boost::geometry::intersection(multiPolygon, other.multiPolygon, intersect);
  return { .d1 = d1, .d2 = d2, .multiPolygon = std::move(intersect) };
}

Polygon2D Polygon2D::difference(Polygon2D const& other) const
{
  if (d1 != other.d1 or d2 != other.d2)
  {
    LOG_FATAL("Merge of Annuli with different components: " + to_string(d1) + " and: " + to_string(other.d1) +
              to_string(d2) + " and: " + to_string(other.d2));
    throw std::runtime_error("Merge of Annuli with different components: " + to_string(d1) +
                             " and: " + to_string(other.d1) + to_string(d2) + " and: " + to_string(other.d2));
  }
  MultiPolygon intersect;
  boost::geometry::difference(multiPolygon, other.multiPolygon, intersect);
  return { .d1 = d1, .d2 = d2, .multiPolygon = std::move(intersect) };
}

std::string Polygon2D::toString(std::string prefix) const
{
  std::string out{ prefix + "Polygons\n" };
  for (auto const& polygon : multiPolygon)
  {
    out += prefix + "|\tPolygon Outer\n";
    for (auto const& point : polygon.outer())
    {
      out += prefix + "|\t|\t" + "(" + std::to_string(point.x()) + ", " + std::to_string(point.y()) + ")\n";
    }
    if (not polygon.inners().empty())
    {
      out += prefix + "|\tPolygon Inners\n";
      for (auto const& inner : polygon.inners())
      {
        out += prefix + "|\t|\tInner\n";
        for (auto const& point : inner)
        {
          out += prefix + "|\t|\t|\t" + "(" + std::to_string(point.x()) + ", " + std::to_string(point.y()) + "\n";
        }
      }
    }
  }
  return out;
}

bool FieldOfView::contains(Vector const& point, Components const& comps) const
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

double FieldOfView::area(Components const& comps) const
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

FieldOfView FieldOfView::merge(FieldOfView const& other) const
{
  FieldOfView merged;
  std::vector<std::vector<COMPONENT>> other_poly_processed;
  for (auto const& [c, poly] : _polygons)
  {
    if (other._polygons.contains(c))
    {
      merged._polygons.emplace(c, poly.merge(other._polygons.at(c)));
      other_poly_processed.push_back(c);
    }
    else
    {
      merged._polygons.emplace(c, poly);
    }
  }
  for (auto const& [other_c, other_poly] : other._polygons)
  {
    if (std::find(other_poly_processed.begin(), other_poly_processed.end(), other_c) == other_poly_processed.end())
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
      other_inter_processed.push_back(c);
    }
    else
    {
      merged._intervals.emplace(c, inter);
    }
  }
  for (auto const& [other_c, other_inter] : other._intervals)
  {
    if (std::find(other_inter_processed.begin(), other_inter_processed.end(), other_c) == other_inter_processed.end())
    {
      merged._intervals.emplace(other_c, other_inter);
    }
  }
  std::vector<COMPONENT> all_comps;
  for (auto const& [c, _] : merged._polygons)
  {
    if (std::find(all_comps.begin(), all_comps.end(), c.at(0)) != all_comps.end())
    {
      LOG_FATAL("Component: " + to_string(c.at(0)) + " multiple time in merge");
      throw std::runtime_error("Component: " + to_string(c.at(0)) + " multiple time in merge");
    }
    all_comps.push_back(c.at(0));
    if (std::find(all_comps.begin(), all_comps.end(), c.at(1)) != all_comps.end())
    {
      LOG_FATAL("Component: " + to_string(c.at(0)) + " multiple time in merge");
      throw std::runtime_error("Component: " + to_string(c.at(1)) + " multiple time in merge");
    }
    all_comps.push_back(c.at(1));
  }
  for (auto const& [c, _] : _intervals)
  {
    if (std::find(all_comps.begin(), all_comps.end(), c) != all_comps.end())
    {
      LOG_FATAL("Component: " + to_string(c) + " multiple time in merge");
      throw std::runtime_error("Component: " + to_string(c) + " multiple time in merge");
    }
    all_comps.push_back(c);
  }
  return merged;
}

FieldOfView FieldOfView::intersect(FieldOfView const& other) const
{
  FieldOfView merged;
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

std::string FieldOfView::toString(std::string prefix) const
{
  std::string out = prefix + "FieldOfView\n";
  for (auto const& [c, poly] : _polygons)
  {
    out += prefix + "|\tComponents: " + to_string(c.at(0)) + ", " + to_string(c.at(1)) + "\n";
    out += prefix + poly.toString("|\t");
  }
  for (auto const& [c, inter] : _intervals)
  {
    out += prefix + "|\tComponent: " + to_string(c) + "\n";
    out += prefix + inter.toString("|\t");
  }
  return out;
}

bool FieldOfView::isValid() const
{
  std::vector<COMPONENT> seen;
  for (auto const& [c, inter] : _intervals)
  {
    if (std::find(seen.begin(), seen.end(), c) != seen.end())
    {
      LOG_FATAL("COMPONENT: " + to_string(c) + " already seen");
      return false;
    }
    seen.push_back(c);
    if (inter.area() <= 0)
    {
      LOG_FATAL("Area of Interval of Component: " + to_string(c) + " <= 0");
      return false;
    }
  }

  for (auto const& [cs, poly] : _polygons)
  {
    for (COMPONENT c : cs)
    {
      if (std::find(seen.begin(), seen.end(), c) != seen.end())
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
      LOG_FATAL("Area of Polygon of first Component: " + to_string(cs.front()) + " <= 0");
      return false;
    }
  }
  return true;
}

void FieldOfView::transform(Vector3 const& trans, Matrix33 const& rot)
{
  for (auto& [cs, multiPoly] : _polygons)
  {
    if ((multiPoly.d1 == COMPONENT::POS_X and multiPoly.d2 == COMPONENT::POS_Y) or
        (multiPoly.d1 == COMPONENT::POS_Z and multiPoly.d2 == COMPONENT::POS_X))
    {
      for (auto& poly : multiPoly.multiPolygon)
      {
        for (auto& point : poly.outer())
        {
          Vector point_3d = [&] {
            if (multiPoly.d1 == COMPONENT::POS_X and multiPoly.d2 == COMPONENT::POS_Y)
            {
              return Vector{ { point.x(), point.y(), 0 } };
            }
            return Vector{ { point.y(), 0, point.x() } };
          }();
          Vector const trafo = rot * point_3d + trans;
          point = Polygon2D::Point(trafo(0), trafo(1));
        }
        for (auto& inner : poly.inners())
        {
          for (auto& point : inner)
          {
            Vector point_3d = [&] {
              if (multiPoly.d1 == COMPONENT::POS_X and multiPoly.d2 == COMPONENT::POS_Y)
              {
                return Vector{ { point.x(), point.y(), 0 } };
              }
              return Vector{ { point.y(), 0, point.x() } };
            }();
            Vector const trafo = rot * point_3d + trans;
            point = Polygon2D::Point(trafo(0), trafo(1));
          }
        }
      }
    }
  }
}

std::string SensorInformation::toString(std::string prefix) const
{
  std::string out = prefix + "Sensor Information\n";
  std::ostringstream pos_ss;
  pos_ss << prefix + "|\tPosition: " << _to_sensor_cs.translation();
  auto pos_str = pos_ss.str();
  pos_str = std::regex_replace(pos_str, std::regex("\n"), "\n" + prefix + "|\t          ");
  out += pos_str + "\n";
  std::ostringstream rot_ss;
  rot_ss << prefix + "|\tRotation: " << _to_sensor_cs.rotation();
  auto rot_str = rot_ss.str();
  rot_str = std::regex_replace(rot_str, std::regex("\n"), "\n" + prefix + "|\t          ");
  out += rot_str + "\n";
  if (_sensor_fov.has_value())
  {
    out += _sensor_fov.value().toString(prefix + "|\t");
  }
  return out;
}
}  // namespace ttb