#include "gtest/gtest.h"
#include "tracking_lib/Misc/Region.h"
#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/length.hpp>

using namespace std::literals;

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(SensorInformation, Interval)
{
  _setLogLevel(aduulm_logger::LoggerLevel::Fatal);

  ttb::Interval inter{ ._comps = ttb::Components({ ttb::COMPONENT::POS_Z }),
                       ._multi_line = { { ttb::Point(0), ttb::Point(1) } } };
  ttb::Interval inter2{ ._comps = ttb::Components({ ttb::COMPONENT::VEL_X }),
                        ._multi_line = { { ttb::Point(10), ttb::Point(-2) } } };

  EXPECT_NEAR(inter.area(), 1, 1e-5);
  EXPECT_NEAR(inter2.area(), 12, 1e-5);

  ttb::Region fov{ ._intervals = { { ttb::Components({ ttb::COMPONENT::POS_Z }), inter },
                                   { ttb::Components({ ttb::COMPONENT::VEL_X }), inter2 } } };
  EXPECT_NEAR(fov.area(ttb::Components({ ttb::COMPONENT::POS_Z, ttb::COMPONENT::VEL_X })), 12, 1e-5);
  EXPECT_NEAR(fov.area(ttb::Components({ ttb::COMPONENT::POS_Z })), 1, 1e-5);
  EXPECT_TRUE(fov.contains(ttb::Vector<double>{ { 0.5 } }, ttb::Components({ ttb::COMPONENT::POS_Z })));
  EXPECT_TRUE(not fov.contains(ttb::Vector<double>{ { 1.5 } }, ttb::Components({ ttb::COMPONENT::POS_Z })));
  // LOG_FATAL("contains 1D passed");
  //
  EXPECT_TRUE(fov.contains(ttb::Vector<double>{ { 0.5, 0 } },
                           ttb::Components({ ttb::COMPONENT::POS_Z, ttb::COMPONENT::VEL_X })));
  EXPECT_TRUE(not fov.contains(ttb::Vector<double>{ { 0.5, -3 } },
                               ttb::Components({ ttb::COMPONENT::POS_Z, ttb::COMPONENT::VEL_X })));

  EXPECT_TRUE(fov.contains(ttb::Vector<double>{ { 0.5, 0, 0 } },
                           ttb::Components({ ttb::COMPONENT::POS_Z, ttb::COMPONENT::VEL_X, ttb::COMPONENT::VEL_Y })));

  ttb::Interval inter3{ ._comps = ttb::Components({ ttb::COMPONENT::POS_Z }),
                        ._multi_line = { { ttb::Point(1), ttb::Point(4) } } };

  auto merge = inter.merge(inter3);

  ttb::Interval inter4{ ._comps = ttb::Components({ ttb::COMPONENT::POS_Z }),
                        ._multi_line = { { ttb::Point(6), ttb::Point(9) } } };
  merge = merge.merge(inter4);
  EXPECT_TRUE(merge.contains(ttb::Vector<double>{ { 7 } }, ttb::Components({ ttb::COMPONENT::POS_Z })));

  ttb::Interval inter5{ ._comps = ttb::Components({ ttb::COMPONENT::POS_Z }),
                        ._multi_line = { { ttb::Point(-5), ttb::Point(10) } } };
  merge = merge.merge(inter5);
  EXPECT_NEAR(boost::geometry::length(merge._multi_line), 15, 1e-5);
  ttb::Point p1{ 0, 0 };
  ttb::Point p2{ 1, 0 };
  ttb::Point p3{ 1, 1 };
  ttb::Point p4{ 0, 1 };
  ttb::Polygon::SinglePolygon poly;
  poly.outer().push_back(p1);
  poly.outer().push_back(p2);
  poly.outer().push_back(p3);
  poly.outer().push_back(p4);

  ttb::Polygon::MultiPolygon multiPolygon{ poly };
  boost::geometry::correct(multiPolygon);

  ttb::Polygon pilyfov{ ._comps = ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }),
                        .multiPolygon = multiPolygon };

  fov._polygons.emplace(ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }), pilyfov);

  EXPECT_TRUE(fov.contains(ttb::Vector<double>{ { 0.5, 0.5 } },
                           ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y })));

  ttb::Point p5{ 0, 0 };
  ttb::Point p6{ 2, 0 };
  ttb::Point p7{ 2, 1 };
  ttb::Point p8{ 0, 1 };
  ttb::Polygon::SinglePolygon poly2;
  poly2.outer().push_back(p5);
  poly2.outer().push_back(p6);
  poly2.outer().push_back(p7);
  poly2.outer().push_back(p8);
  ttb::Polygon::MultiPolygon multiPolygon2{ poly2 };
  boost::geometry::correct(multiPolygon2);
  ttb::Polygon pilyfov2{ ._comps = ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }),
                         .multiPolygon = multiPolygon2 };

  auto polymerge = pilyfov.merge(pilyfov2);

  ttb::Point p9{ 1, 0 };
  ttb::Point p10{ 1, 1 };
  ttb::Point p11{ 0.5, 1 };
  ttb::Point p12{ 0.5, 0 };
  ttb::Polygon::SinglePolygon poly3;
  poly3.outer().push_back(p9);
  poly3.outer().push_back(p10);
  poly3.outer().push_back(p11);
  poly3.outer().push_back(p12);
  ttb::Polygon::MultiPolygon multiPolygon3{ poly3 };
  boost::geometry::correct(multiPolygon3);
  ttb::Polygon pilyfov3{ ._comps = ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }),
                         .multiPolygon = multiPolygon3 };
  ttb::Region fov2;
  fov2._polygons.emplace(ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }), pilyfov3);

  EXPECT_TRUE(fov.isValid());
}
