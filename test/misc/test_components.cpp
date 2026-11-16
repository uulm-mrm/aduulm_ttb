#include "tracking_lib/TTBTypes/Components.h"

#include "gtest/gtest.h"

TEST(components, index)
{
  ttb::Components comps({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y });
  ASSERT_TRUE(comps.indexOf(ttb::COMPONENT::POS_X).has_value());
  {
    ttb::Indices inds_2 = {{0, 1}};
  }
  auto inds = comps.indexOf({ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y});

  ASSERT_TRUE(inds.has_value());

  auto ind_2 = comps.indexOf(ttb::COMPONENT::VEL_X);

  ASSERT_TRUE(not ind_2.has_value());

  auto ind_3 = comps.indexOf({ttb::COMPONENT::VEL_X, ttb::COMPONENT::VEL_Y});

  ASSERT_TRUE(not ind_3.has_value());


}