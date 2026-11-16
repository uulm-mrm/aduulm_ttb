#include "gtest/gtest.h"
#include "tracking_lib/Misc/GibbsSampler.h"

/*
 checks if it does not crash and prints results
*/
TEST(Gibbs, testcase1)
{
  // load a test matrix
  double Inf = std::numeric_limits<double>::infinity();
  ttb::Matrix<double> costMatrix{ { 0.030459, Inf, Inf, Inf, 7.4186, Inf, Inf, Inf, -0.87002, Inf, Inf },
                          { Inf, 0.030459, Inf, Inf, Inf, 7.4186, Inf, Inf, Inf, -0.43676, Inf },
                          { Inf, Inf, 0.030459, Inf, Inf, Inf, 7.4186, Inf, Inf, Inf, 0.34407 },
                          { Inf, Inf, Inf, 0.030459, Inf, Inf, Inf, 7.4186, Inf, Inf, Inf } };

  // create initial values
  ttb::VectorIndex initial{ { 4, 5, 6, 7 } };
  auto [assigments, costs] = ttb::gibbs::sample_assignments(costMatrix, 15, initial, 30, 30);

  // test succeeds if sampls could be generated
  ASSERT_TRUE(true);
}

TEST(Gibbs, testcase2)
{
  ttb::Matrix<double> costMatrix{ { 1, 2 }, { 3, 6 } };

  // create initial values
  ttb::VectorIndex initial{ { 0, 1 } };

  auto [assigments, costs] = ttb::gibbs::sample_assignments(costMatrix, 15, initial, 15, 30);


  ASSERT_TRUE(assigments.cols() == 1);
}

TEST(Gibbs, testcase3)
{
  // load a test matrix
  double Inf = std::numeric_limits<double>::infinity();
  ttb::Matrix<double> costMatrix{ { 1, 2, 3, Inf }, { 3, 6, Inf, 3 } };

  // create initial values
  ttb::VectorIndex initial{ { 2, 3 } };

  auto [assigments, costs] = ttb::gibbs::sample_assignments(costMatrix, 15, initial, 100, 30);

  // test succeeds if samples could be generated
  ASSERT_TRUE(true);
}

TEST(Gibbs, testcase4)
{
  // load a test matrix
  double Inf = std::numeric_limits<double>::infinity();
  ttb::Matrix<double> costMatrix{ { 1, 2, 3, Inf, 5, Inf }, { 3, 6, Inf, 3, Inf, 5 } };

  // create initial values
  ttb::VectorIndex initial{ { 2, 3 } };

  auto [assigments, costs] = ttb::gibbs::sample_assignments(costMatrix, 15, initial, 30, 30);


  ASSERT_TRUE(true);
}

TEST(Gibbs, testcase5)
{
  // load a test matrix
  ttb::Matrix<double> costMatrix{ { 1, 2 } };

  // create initial values
  ttb::VectorIndex initial{ { 1 } };

  auto [assigments, costs] = ttb::gibbs::sample_assignments(costMatrix, 15, initial, 30, 30);

  ASSERT_TRUE(assigments.cols() == 2);
}

TEST(Gibbs, testcase6)
{
  // load a test matrix
  double Inf = std::numeric_limits<double>::infinity();
  ttb::Matrix<double> costMatrix{ { 1, 2, Inf, Inf, Inf } };

  // create initial values
  ttb::VectorIndex initial{ { 1 } };

  auto [assigments, costs] = ttb::gibbs::sample_assignments(costMatrix, 15, initial, 30, 30);

  ASSERT_TRUE(assigments.cols() == 2);
}