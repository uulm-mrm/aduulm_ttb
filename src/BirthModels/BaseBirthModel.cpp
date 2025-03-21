#include "tracking_lib/BirthModels/BaseBirthModel.h"
#include "tracking_lib/Misc/SeparatingAxisTest.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/StateModels/BaseStateModel.h"

namespace ttb
{

double BaseBirthModel::getBirthProbability(State const& newBornState, std::vector<State> const& existingTracks) const
{
  if (manager()->params().birth_model->allow_overlapping)
  {
    LOG_DEB("Allow overlapping birth");
    return manager()->params().birth_model->default_birth_existence_prob;
  }
  std::vector const requiredComponents_fallback = { COMPONENT::POS_X, COMPONENT::POS_Y };
  auto [new_model, new_dist] = newBornState.bestState();
  for (State const& track : existingTracks)  // existing tracks
  {
    auto [existing_model, existing_dist] = track.bestState();
    if (manager()->getStateModel(new_model).state_comps().indexOf(sat::requiredComponents).has_value() and
        manager()->getStateModel(existing_model).state_comps().indexOf(sat::requiredComponents).has_value())
    {
      if (sat::is_overlapping(new_dist.mean(),
                              manager()->getStateModel(new_model).state_comps(),
                              existing_dist.mean(),
                              manager()->getStateModel(existing_model).state_comps()))
      {
        LOG_DEB("Overlaps with existing Track");
        return 0;
      }
    }
    else  // don't perform SAT as not all required states are provided
    {
      auto new_inds = manager()->getStateModel(new_model).state_comps().indexOf({ COMPONENT::POS_X, COMPONENT::POS_X });
      auto existing_inds =
          manager()->getStateModel(existing_model).state_comps().indexOf({ COMPONENT::POS_X, COMPONENT::POS_X });
      if (not new_inds.has_value() or not existing_inds.has_value())
      {
        return 0;
      }
      Matrix S = new_dist.covariance()(new_inds.value(), new_inds.value()) +
                 existing_dist.covariance()(existing_inds.value(), existing_inds.value());
      Vector res = new_dist.mean()(new_inds.value()) - existing_dist.mean()(existing_inds.value());
      double mhd2 = res.transpose() * S.llt().solve(res);
      if (mhd2 < manager()->params().birth_model->min_mhd_4_overlapping_wo_extent)
      {
        LOG_DEB("MHD2 < _minMHD4OverlappingWoExtent");
        return 0;
      }
    }
  }
  LOG_DEB("Does not overlap with existing Track");
  return manager()->params().birth_model->default_birth_existence_prob;
}

}  // namespace ttb
