#include "gtest/gtest.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Transformations/Transformation.h"

using namespace std::literals;
constexpr double TOL{ 1e-5 };

DEFINE_LOGGER_LIBRARY_INTERFACE_HEADER

TEST(Transformations, transformation)
{
  ttb::Components from({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y });
  ttb::Components to({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y, ttb::COMPONENT::POS_Z });
  ttb::Vector state{ { 0, 0 } };
  ttb::Matrix cov = ttb::Matrix::Identity(2, 2);
  auto trans = ttb::transformation::transform(state, cov, from, to);
  ASSERT_TRUE(not trans.has_value());
}

TEST(Transformations, rp_transformation)
{
  ttb::Components from({ ttb::COMPONENT::POS_X,
                         ttb::COMPONENT::POS_Y,
                         ttb::COMPONENT::ROT_Z,
                         ttb::COMPONENT::LENGTH,
                         ttb::COMPONENT::WIDTH });
  for (std::size_t i = 0; i < 50; ++i)
  {
    ttb::Vector state = ttb::Vector::Random(5);
    state(2) *= std::numbers::pi;
    state(3) *= state(3);
    state(4) *= state(4);

    auto trans_bl =
        ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::BACK_LEFT);
    auto trans_br = ttb::transformation::transform(
        trans_bl, from, ttb::REFERENCE_POINT::BACK_LEFT, ttb::REFERENCE_POINT::BACK_RIGHT);
    auto trans_fl = ttb::transformation::transform(
        trans_br, from, ttb::REFERENCE_POINT::BACK_RIGHT, ttb::REFERENCE_POINT::FRONT_LEFT);
    auto trans_fr = ttb::transformation::transform(
        trans_fl, from, ttb::REFERENCE_POINT::FRONT_LEFT, ttb::REFERENCE_POINT::FRONT_RIGHT);
    auto trans_c =
        ttb::transformation::transform(trans_fr, from, ttb::REFERENCE_POINT::FRONT_RIGHT, ttb::REFERENCE_POINT::CENTER);
    ASSERT_TRUE((trans_c - state).norm() < 1e-7);
  }
}

TEST(Transformations, rp_transformation_edges)
{
  ttb::Components from({ ttb::COMPONENT::POS_X,
                         ttb::COMPONENT::POS_Y,
                         ttb::COMPONENT::ROT_Z,
                         ttb::COMPONENT::LENGTH,
                         ttb::COMPONENT::WIDTH });
  for (std::size_t i = 0; i < 50; ++i)
  {
    ttb::Vector state = ttb::Vector::Random(5);
    state(2) *= std::numbers::pi;
    state(3) *= state(3);
    state(4) *= state(4);

    auto trans_b =
        ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::BACK);
    auto trans_l =
        ttb::transformation::transform(trans_b, from, ttb::REFERENCE_POINT::BACK, ttb::REFERENCE_POINT::LEFT);
    auto trans_r =
        ttb::transformation::transform(trans_l, from, ttb::REFERENCE_POINT::LEFT, ttb::REFERENCE_POINT::RIGHT);
    trans_b = ttb::transformation::transform(trans_r, from, ttb::REFERENCE_POINT::RIGHT, ttb::REFERENCE_POINT::BACK);
    auto trans_c =
        ttb::transformation::transform(trans_b, from, ttb::REFERENCE_POINT::BACK, ttb::REFERENCE_POINT::CENTER);
    ASSERT_TRUE((trans_c - state).norm() < 1e-7);
  }
}

TEST(Transformations, rp_transformation_edges_values)
{
  ttb::Components from({ ttb::COMPONENT::POS_X,
                         ttb::COMPONENT::POS_Y,
                         ttb::COMPONENT::ROT_Z,
                         ttb::COMPONENT::LENGTH,
                         ttb::COMPONENT::WIDTH });

  ttb::Vector state{ { 0, 0, 0, 1, 2 } };

  ASSERT_TRUE((ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::BACK) -
               ttb::Vector{ { -0.5, 0, 0, 1, 2 } })
                  .norm() < 1e-7);
  ASSERT_TRUE((ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::FRONT) -
               ttb::Vector{ { 0.5, 0, 0, 1, 2 } })
                  .norm() < 1e-7);
  ASSERT_TRUE((ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::LEFT) -
               ttb::Vector{ { 0, 1, 0, 1, 2 } })
                  .norm() < 1e-7);
  ASSERT_TRUE((ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::RIGHT) -
               ttb::Vector{ { 0, -1, 0, 1, 2 } })
                  .norm() < 1e-7);
  ASSERT_TRUE(
      (ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::BACK_RIGHT) -
       ttb::Vector{ { -0.5, -1, 0, 1, 2 } })
          .norm() < 1e-7);
  ASSERT_TRUE(
      (ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::FRONT_LEFT) -
       ttb::Vector{ { 0.5, 1, 0, 1, 2 } })
          .norm() < 1e-7);
  ASSERT_TRUE(
      (ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::FRONT_RIGHT) -
       ttb::Vector{ { 0.5, -1, 0, 1, 2 } })
          .norm() < 1e-7);
  ASSERT_TRUE(
      (ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::BACK_LEFT) -
       ttb::Vector{ { -0.5, 1, 0, 1, 2 } })
          .norm() < 1e-7);

  state = ttb::Vector{ { 0, 0, std::numbers::pi, 1, 2 } };
  ASSERT_TRUE(
      (ttb::transformation::transform(state, from, ttb::REFERENCE_POINT::CENTER, ttb::REFERENCE_POINT::FRONT_LEFT) -
       ttb::Vector{ { -0.5, -1, std::numbers::pi, 1, 2 } })
          .norm() < 1e-7);
}