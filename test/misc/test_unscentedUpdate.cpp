#include "gtest/gtest.h"

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/Transformations/Transformation.h"

TEST(unscentedUpdate, update)
{
  for (std::size_t tttt = 0; tttt < 100; ++tttt)
  {
    ttb::Components stateComps(
        { ttb::COMPONENT::POS_X, ttb::COMPONENT::VEL_X, ttb::COMPONENT::POS_Y, ttb::COMPONENT::VEL_Y });
    LOG_FATAL("State Comps: " << stateComps.toString());
    ttb::Vector mean = ttb::Vector::Zero(stateComps._comps.size());
    mean(stateComps.indexOf(ttb::COMPONENT::VEL_X).value()) = 1;
    mean(stateComps.indexOf(ttb::COMPONENT::VEL_Y).value()) = 1;
    ttb::Matrix cov = ttb::Matrix::Random(stateComps._comps.size(), stateComps._comps.size());
    cov *= cov.transpose();
    if ((cov.eigenvalues().real().array() < 0).any())
    {
      LOG_FATAL("not ok");
      assert(false);
    }
    LOG_FATAL("state mean: " << mean);
    LOG_FATAL("state cov: " << cov);
    ttb::Components measComps({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y, ttb::COMPONENT::ROT_Z });
    LOG_FATAL("Meas Comps: " << measComps.toString());
    ttb::Vector measMean = ttb::Vector::Zero(measComps._comps.size());
    measMean(measComps.indexOf(ttb::COMPONENT::POS_X).value()) = -0.1;
    measMean(measComps.indexOf(ttb::COMPONENT::POS_Y).value()) = 0.2;
    measMean(measComps.indexOf(ttb::COMPONENT::ROT_Z).value()) = 0;
    ttb::Matrix measCov = ttb::Matrix::Identity(measComps._comps.size(), measComps._comps.size());
    measCov(measComps.indexOf(ttb::COMPONENT::ROT_Z).value(), measComps.indexOf(ttb::COMPONENT::ROT_Z).value()) = 0.001;
    if ((measCov.eigenvalues().real().array() < 0).any())
    {
      LOG_FATAL("not ok");
      assert(false);
    }
    LOG_FATAL("meas mean: " << measMean);
    LOG_FATAL("meas cov: " << measCov);
    auto transformedState = ttb::transformation::transform(mean, cov, stateComps, measComps);
    if (not transformedState.has_value())
    {
      LOG_FATAL("not possible");
    }
    LOG_FATAL("predicted meas mean: " << transformedState.value().mean);
    LOG_FATAL("predicted meas cov: " << transformedState.value().cov);
    if ((transformedState.value().cov.eigenvalues().real().array() < 0).any())
    {
      LOG_FATAL("not ok");
      assert(false);
    }
    LOG_FATAL("Update -----------------");
    ttb::Vector res = measMean - transformedState->mean;
    ttb::Matrix S = measCov + transformedState->cov;
    ttb::Matrix iS = S.inverse();

    ttb::Matrix K = transformedState.value().cross_cov.value() * iS;
    LOG_FATAL("cross cov: " << transformedState.value().cross_cov.value());
    LOG_FATAL("K " << K);
    ttb::Vector mean_upd = mean + K * res;
    ttb::Matrix cov_upd = cov - K * S * K.transpose();

    LOG_FATAL("mean_upd: " << mean_upd);
    LOG_FATAL("cov_upd: " << cov_upd);

    LOG_FATAL("cov_upd evalues: " << cov_upd.eigenvalues());
    if ((cov_upd.eigenvalues().real().array() < 0).any())
    {
      LOG_FATAL("not ok");
      assert(false);
    }
  }
}
