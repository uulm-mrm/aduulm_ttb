#include "gtest/gtest.h"

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/Misc/LockedData.h"

#include <tracking_lib/Misc/Numeric.h>

TEST(fold, test)
{
  ttb::Matrix<double> first_m{ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
  ttb::Matrix<double> second_m{ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };

  ttb::Matrix<double> fold_1 = ttb::numeric::fold(first_m, second_m);

}
