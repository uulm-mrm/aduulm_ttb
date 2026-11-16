#include "gtest/gtest.h"

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/Transformations/Transformation.h"

TEST(HULL, transform)
{
  ttb::Components stateComps({ ttb::COMPONENT::POS_X,
                               ttb::COMPONENT::POS_Y,
                               ttb::COMPONENT::ROT_Z,
                               ttb::COMPONENT::LENGTH,
                               ttb::COMPONENT::WIDTH });
  ttb::Vector<double> mean = ttb::Vector<double>::Zero(stateComps._comps.size());
  mean(stateComps.indexOf(ttb::COMPONENT::POS_X).value()) = 1000;
  mean(stateComps.indexOf(ttb::COMPONENT::POS_Y).value()) = 2000;
  mean(stateComps.indexOf(ttb::COMPONENT::ROT_Z).value()) = 0;
  mean(stateComps.indexOf(ttb::COMPONENT::LENGTH).value()) = 99;
  mean(stateComps.indexOf(ttb::COMPONENT::WIDTH).value()) = 5;
  ttb::Matrix<double> cov = ttb::Matrix<double>::Random(stateComps._comps.size(), stateComps._comps.size());
  cov *= cov.transpose();
  if ((cov.eigenvalues().real().array() < 0).any())
  {
    assert(false);
  }
  ttb::Components meas_comps({ ttb::COMPONENT::HULL_POS_X_MIN,
                               ttb::COMPONENT::HULL_POS_X_MAX,
                               ttb::COMPONENT::HULL_POS_Y_MIN,
                               ttb::COMPONENT::HULL_POS_Y_MAX });
  ttb::SE3Trafo sensor_pose = ttb::SE3Trafo::Identity();
  sensor_pose.translation() = ttb::Vector<double>{ { 0, 0, 0 } };
  auto transformed = ttb::transformation::transform(mean, stateComps, meas_comps, sensor_pose);
  assert(transformed.has_value());

  double rot_z = 1;
  double const c = std::cos(rot_z);
  double const s = std::sin(rot_z);
  sensor_pose.matrix() = ttb::Matrix<double>{ { c, -s, 0, 7 }, { s, c, 0, -5 }, { 0, 0, 1, 1 }, { 0, 0, 0, 1 } };
  transformed = ttb::transformation::transform(mean, stateComps, meas_comps, sensor_pose);

  assert(transformed.has_value());
  for (double rz = 0; rz < 7; rz += 0.1)
  {
    double const cn = std::cos(rot_z);
    double const sn = std::sin(rot_z);
    sensor_pose.matrix() = ttb::Matrix<double>{ { cn, -sn, 0, 7 }, { sn, cn, 0, -5 }, { 0, 0, 1, 1 }, { 0, 0, 0, 1 } };
    auto transformedn = ttb::transformation::transform(mean, stateComps, meas_comps, sensor_pose);
    ASSERT_TRUE((transformedn.value() - transformed.value()).norm() < 1);
  }
}
