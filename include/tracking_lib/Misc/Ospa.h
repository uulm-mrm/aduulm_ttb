#pragma once

#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// D. Schuhmacher, B. -T. Vo and B. -N. Vo, "A Consistent Metric for Performance Evaluation of Multi-Object Filters,"
/// in IEEE Transactions on Signal Processing, vol. 56, no. 8, pp. 3447-3457, Aug. 2008, doi: 10.1109/TSP.2008.920469.
template <typename UDT>
[[nodiscard]] OspaResult eval_ospa(TTBManager<UDT> const& manager,
                                   std::vector<State<UDT>> const& tracks,
                                   std::vector<State<UDT>> const& ground_truths,
                                   double c = 10,
                                   double p = 2,
                                   std::vector<COMPONENT> const& components = { COMPONENT::POS_X, COMPONENT::POS_Y });

/// A. S. Rahmathullah, Á. F. García-Fernández and L. Svensson, "Generalized optimal sub-pattern assignment metric,"
/// 2017 20th International Conference on Information Fusion (Fusion), Xi'an, China, 2017, pp. 1-8,
/// doi: 10.23919/ICIF.2017.8009645.
template <typename UDT>
[[nodiscard]] GospaResult eval_gospa(TTBManager<UDT> const& manager,
                                     std::vector<State<UDT>> const& tracks,
                                     std::vector<State<UDT>> const& ground_truths,
                                     double c = 10,
                                     double p = 2,
                                     std::vector<COMPONENT> const& components = { COMPONENT::POS_X, COMPONENT::POS_Y });

/// M. Beard, B. T. Vo and B. -N. Vo, "A Solution for Large-Scale Multi-Object Tracking," in IEEE Transactions on Signal
/// Processing, vol. 68, pp. 2754-2769, 2020, doi: 10.1109/TSP.2020.2986136.
template <typename UDT>
[[nodiscard]] Ospa2Result eval_ospa2(TTBManager<UDT> const& manager,
                                     std::map<Label, SlidingWindow<std::optional<State<UDT>>>> const& gt_trajectories,
                                     std::map<Label, SlidingWindow<std::optional<State<UDT>>>> const& track_trajectories,
                                     double c = 10,
                                     double p = 2,
                                     std::vector<COMPONENT> const& components = { COMPONENT::POS_X, COMPONENT::POS_Y });

}  // namespace ttb