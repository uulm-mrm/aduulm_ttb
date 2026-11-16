#include "gtest/gtest.h"

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/Transformations/Transformation.h"
#include <tracy/tracy/Tracy.hpp>

TEST(unscentedUpdate, update)
{
  std::vector<int> tmps(100);
  hpx::for_each(hpx::execution::par, tmps.begin(), tmps.end(), [](int) {
    ZoneScopedN("Test");
    ttb::Components stateComps(
        { ttb::COMPONENT::POS_X, ttb::COMPONENT::VEL_X, ttb::COMPONENT::POS_Y, ttb::COMPONENT::VEL_Y });
    ttb::Vector<double> mean = ttb::Vector<double>::Zero(stateComps._comps.size());
    mean(stateComps.indexOf(ttb::COMPONENT::VEL_X).value()) = 1;
    mean(stateComps.indexOf(ttb::COMPONENT::VEL_Y).value()) = 1;
    ttb::Matrix<double> cov = ttb::Matrix<double>::Random(stateComps._comps.size(), stateComps._comps.size());
    cov *= cov.transpose();
    if ((cov.eigenvalues().real().array() < 0).any())
    {
      assert(false);
    }
    ttb::Components measComps({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y, ttb::COMPONENT::ROT_Z });
    ttb::Vector<double> measMean = ttb::Vector<double>::Zero(measComps._comps.size());
    measMean(measComps.indexOf(ttb::COMPONENT::POS_X).value()) = -0.1;
    measMean(measComps.indexOf(ttb::COMPONENT::POS_Y).value()) = 0.2;
    measMean(measComps.indexOf(ttb::COMPONENT::ROT_Z).value()) = 0;
    ttb::Matrix<double> measCov = ttb::Matrix<double>::Identity(measComps._comps.size(), measComps._comps.size());
    measCov(measComps.indexOf(ttb::COMPONENT::ROT_Z).value(), measComps.indexOf(ttb::COMPONENT::ROT_Z).value()) = 0.001;
    if ((measCov.eigenvalues().real().array() < 0).any())
    {
      assert(false);
    }
    auto transformedState = ttb::transformation::transform(mean, cov, stateComps, measComps);
    if ((transformedState.value().cov.eigenvalues().real().array() < 0).any())
    {
      assert(false);
    }
    ttb::Vector<double> res = measMean - transformedState->mean;
    ttb::Matrix<double> S = measCov + transformedState->cov;
    ttb::Matrix<double> iS = S.inverse();

    ttb::Matrix<double> K = transformedState.value().cross_cov.value() * iS;
    ttb::Vector<double> mean_upd = mean + K * res;
    ttb::Matrix<double> cov_upd = cov - K * S * K.transpose();
    if ((cov_upd.eigenvalues().real().array() < 0).any())
    {
      assert(false);
    }
  });
}
