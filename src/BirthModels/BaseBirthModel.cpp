#include "tracking_lib/BirthModels/BaseBirthModel.h"
#include "tracking_lib/Misc/SeparatingAxisTest.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/StateModels/BaseStateModel.h"

namespace ttb
{

template double getBirthProbability<double>(TTBManager<double> const& manager,
                                            State<double> const& newBornState,
                                            std::vector<State<double>> const& existingTracks);

template AD getBirthProbability<AD>(TTBManager<AD> const& manager,
                                            State<AD> const& newBornState,
                                            std::vector<State<AD>> const& existingTracks);
template <class UDT>
UDT getBirthProbability(TTBManager<UDT> const& manager,
                           State<UDT> const& newBornState,
                           std::vector<State<UDT>> const& existingTracks)
{
  if (manager.params()->birth_model->allow_overlapping)
  {
    LOG_DEB("Allow overlapping birth");
    return manager.params()->birth_model->default_birth_existence_prob;
  }
  std::vector const requiredComponents_fallback = { COMPONENT::POS_X, COMPONENT::POS_Y };
  auto [new_model, new_dist] = newBornState.getEstimate();
  for (State<UDT> const& track : existingTracks)  // existing tracks
  {
    auto [existing_model, existing_dist] = track.getEstimate();
    if (manager.state_comps(new_model).indexOf(sat::requiredComponents).has_value() and
        manager.state_comps(existing_model).indexOf(sat::requiredComponents).has_value())
    {
      if (sat::is_overlapping(new_dist.mean(),
                              manager.state_comps(new_model),
                              existing_dist.mean(),
                              manager.state_comps(existing_model)))
      {
        LOG_DEB("Overlaps with existing Track");
        return 0;
      }
    }
    else  // don't perform SAT as not all required states are provided
    {
      auto new_inds = manager.state_comps(new_model).indexOf({ COMPONENT::POS_X, COMPONENT::POS_X });
      auto existing_inds = manager.state_comps(existing_model).indexOf({ COMPONENT::POS_X, COMPONENT::POS_X });
      if (not new_inds.has_value() or not existing_inds.has_value())
      {
        return 0;
      }
      Matrix<UDT> S = new_dist.covariance()(new_inds.value(), new_inds.value()) +
                 existing_dist.covariance()(existing_inds.value(), existing_inds.value());
      Vector<UDT> res = new_dist.mean()(new_inds.value()) - existing_dist.mean()(existing_inds.value());
      UDT const mhd2 = res.transpose() * S.llt().solve(res);
      if (mhd2 < manager.params()->birth_model->min_mhd_4_overlapping_wo_extent)
      {
        LOG_DEB("MHD2 < _minMHD4OverlappingWoExtent");
        return 0;
      }
    }
  }
  LOG_DEB("Does not overlap with existing Track");
  return manager.params()->birth_model->default_birth_existence_prob;
}


}  // namespace ttb
