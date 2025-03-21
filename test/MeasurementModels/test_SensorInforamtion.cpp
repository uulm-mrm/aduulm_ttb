#include "gtest/gtest.h"
#include "tracking_lib/Measurements//SensorInformation.h"
#include <boost/geometry/algorithms/correct.hpp>

using namespace std::literals;

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(SensorInformation, Interval)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Debug);
  ttb::Interval1D inter{ .d1 = ttb::COMPONENT::POS_Z, ._intervals = { { .min = 0, .max = 1 } } };

  ttb::Interval1D inter2{ .d1 = ttb::COMPONENT::VEL_X, ._intervals = { { .min = -2, .max = 10 } } };

  EXPECT_NEAR(inter.area(), 1, 1e-5);
  EXPECT_NEAR(inter2.area(), 12, 1e-5);

  ttb::FieldOfView fov{ ._intervals = { { ttb::COMPONENT::POS_Z, inter }, { ttb::COMPONENT::VEL_X, inter2 } } };
  EXPECT_NEAR(fov.area(ttb::Components({ ttb::COMPONENT::POS_Z, ttb::COMPONENT::VEL_X })), 12, 1e-5);
  EXPECT_NEAR(fov.area(ttb::Components({ ttb::COMPONENT::POS_Z })), 1, 1e-5);
  LOG_FATAL("area passed");
  EXPECT_TRUE(fov.contains(ttb::Vector{ { 0.5 } }, ttb::Components({ ttb::COMPONENT::POS_Z })));
  EXPECT_TRUE(not fov.contains(ttb::Vector{ { 1.5 } }, ttb::Components({ ttb::COMPONENT::POS_Z })));
  LOG_FATAL("contains 1D passed");

  EXPECT_TRUE(
      fov.contains(ttb::Vector{ { 0.5, 0 } }, ttb::Components({ ttb::COMPONENT::POS_Z, ttb::COMPONENT::VEL_X })));
  EXPECT_TRUE(
      not fov.contains(ttb::Vector{ { 0.5, -3 } }, ttb::Components({ ttb::COMPONENT::POS_Z, ttb::COMPONENT::VEL_X })));
  LOG_FATAL("contains 2d passed");

  EXPECT_TRUE(fov.contains(ttb::Vector{ { 0.5, 0, 0 } },
                           ttb::Components({ ttb::COMPONENT::POS_Z, ttb::COMPONENT::VEL_X, ttb::COMPONENT::VEL_Y })));
  LOG_FATAL("contains passed");

  ttb::Interval1D inter3{ .d1 = ttb::COMPONENT::POS_Z, ._intervals = { { .min = 1, .max = 4 } } };

  auto merge = inter.merge(inter3);
  LOG_FATAL(merge.toString("MERGED Inter"));
  EXPECT_TRUE(merge._intervals.size() == 1);
  EXPECT_TRUE(merge._intervals.front().min == 0 and merge._intervals.front().max == 4);

  ttb::Interval1D inter4{ .d1 = ttb::COMPONENT::POS_Z, ._intervals = { { .min = 6, .max = 9 } } };
  merge = merge.merge(inter4);
  LOG_FATAL(merge.toString("MERGED Inter"));
  EXPECT_TRUE(merge._intervals.size() == 2);
  EXPECT_TRUE(merge.contains(ttb::Vector{ { 7 } }, ttb::Components({ ttb::COMPONENT::POS_Z })));

  ttb::Interval1D inter5{ .d1 = ttb::COMPONENT::POS_Z, ._intervals = { { .min = -5, .max = 10 } } };
  merge = merge.merge(inter5);
  LOG_FATAL(merge.toString("MERGED Inter"));
  EXPECT_TRUE(merge._intervals.size() == 1);
  EXPECT_TRUE(merge._intervals.front().min == -5 and merge._intervals.front().max == 10);

  ttb::Polygon2D::Point p1{ 0, 0 };
  ttb::Polygon2D::Point p2{ 1, 0 };
  ttb::Polygon2D::Point p3{ 1, 1 };
  ttb::Polygon2D::Point p4{ 0, 1 };
  ttb::Polygon2D::Polygon poly;
  poly.outer().push_back(p1);
  poly.outer().push_back(p2);
  poly.outer().push_back(p3);
  poly.outer().push_back(p4);

  ttb::Polygon2D::MultiPolygon multiPolygon{ poly };
  boost::geometry::correct(multiPolygon);

  ttb::Polygon2D pilyfov{ .d1 = ttb::COMPONENT::POS_X, .d2 = ttb::COMPONENT::POS_Y, .multiPolygon = multiPolygon };
  LOG_FATAL("poly1" + pilyfov.toString());

  fov._polygons[{ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }] = pilyfov;

  EXPECT_TRUE(
      fov.contains(ttb::Vector{ { 0.5, 0.5 } }, ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y })));

  ttb::Polygon2D::Point p5{ 0, 0 };
  ttb::Polygon2D::Point p6{ 2, 0 };
  ttb::Polygon2D::Point p7{ 2, 1 };
  ttb::Polygon2D::Point p8{ 0, 1 };
  ttb::Polygon2D::Polygon poly2;
  poly2.outer().push_back(p5);
  poly2.outer().push_back(p6);
  poly2.outer().push_back(p7);
  poly2.outer().push_back(p8);
  ttb::Polygon2D::MultiPolygon multiPolygon2{ poly2 };
  boost::geometry::correct(multiPolygon2);
  ttb::Polygon2D pilyfov2{ .d1 = ttb::COMPONENT::POS_X, .d2 = ttb::COMPONENT::POS_Y, .multiPolygon = multiPolygon2 };
  LOG_FATAL("pilyfov2" + pilyfov2.toString());

  auto polymerge = pilyfov.merge(pilyfov2);
  LOG_FATAL(polymerge.toString("merged Polygon"));

  ttb::Polygon2D::Point p9{ 1, 0 };
  ttb::Polygon2D::Point p10{ 1, 1 };
  ttb::Polygon2D::Point p11{ 0.5, 1 };
  ttb::Polygon2D::Point p12{ 0.5, 0 };
  ttb::Polygon2D::Polygon poly3;
  poly3.outer().push_back(p9);
  poly3.outer().push_back(p10);
  poly3.outer().push_back(p11);
  poly3.outer().push_back(p12);
  ttb::Polygon2D::MultiPolygon multiPolygon3{ poly3 };
  boost::geometry::correct(multiPolygon3);
  ttb::Polygon2D pilyfov3{ .d1 = ttb::COMPONENT::POS_X, .d2 = ttb::COMPONENT::POS_Y, .multiPolygon = multiPolygon3 };
  LOG_FATAL("pilyfov3" + pilyfov3.toString());

  ttb::FieldOfView fov2;
  fov2._polygons[{ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }] = pilyfov3;

  LOG_FATAL(fov.toString());

  LOG_FATAL(fov2.toString());

  LOG_FATAL("Intersect");
  LOG_FATAL(fov.intersect(fov2).toString());

  LOG_FATAL("Merge");
  LOG_FATAL(fov.merge(fov2).toString());

  EXPECT_TRUE(fov.isValid());
}
