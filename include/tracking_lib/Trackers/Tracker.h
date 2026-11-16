#pragma once
#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
#include "tracking_lib/MultiObjectStateDistributions/GLMBDistribution.h"
#include "tracking_lib/MultiObjectStateDistributions/PMBMDistribution.h"

#include "tracking_lib/Trackers/ICTracker.h"
#include "tracking_lib/Trackers/MSTracker.h"
#include "tracking_lib/Trackers/GNN_Tracker.h"
#include "tracking_lib/Trackers/NO_Tracker.h"
#include "tracking_lib/Trackers/NN_Tracker.h"
#include <tracking_lib/Trackers/Id_Tracker.h>
#include "tracking_lib/Trackers/Nothing_Tracker.h"

#include "tracking_lib/TTTFilters/NoTTTFilter.h"
#include "tracking_lib/TTTFilters/TransTTTFilter.h"
#include <variant>

namespace ttb
{

template<class UDT>
class Tracker
{
public:
  explicit Tracker(TTBManager<UDT>* manager);
  void cycle(std::vector<DataSourceContainer<UDT>> measContainers);
  [[nodiscard]] std::vector<BernoulliState<UDT>> get_estimate() const;
  [[nodiscard]] std::vector<StateId> all_tracks() const;
  void reset();
  [[nodiscard]] Time time() const;
  TTBManager<UDT>* _manager;
  using AllFilters = std::variant<ICTracker<UDT, LMBDistribution<UDT>, TTBManager<UDT>>,
                                  MSTracker<UDT, LMBDistribution<UDT>, TTBManager<UDT>>,
                                  ICTracker<UDT, GLMBDistribution<UDT>, TTBManager<UDT>>,
                                  MSTracker<UDT, GLMBDistribution<UDT>, TTBManager<UDT>>,
                                  ICTracker<UDT, PMBMDistribution<UDT>, TTBManager<UDT>>,
                                  GNN_Tracker<UDT>,
                                  NO_Tracker<UDT>,
                                  NN_Tracker<UDT>,
                                  Id_Tracker<UDT>,
                                  NoTTTFilter<UDT>,
                                  TransTTTFilter<UDT>,
                                  Nothing_Tracker<UDT>>;
  AllFilters _tracker;
};

static_assert(CPTTracker<ICTracker<double, LMBDistribution<double>, TTBManager<double>>, double>);
static_assert(CPTTracker<MSTracker<double, LMBDistribution<double>, TTBManager<double>>, double>);
static_assert(CPTTracker<ICTracker<double, GLMBDistribution<double>, TTBManager<double>>, double>);
static_assert(CPTTracker<MSTracker<double, GLMBDistribution<double>, TTBManager<double>>, double>);
static_assert(CPTTracker<ICTracker<double, PMBMDistribution<double>, TTBManager<double>>, double>);
static_assert(CPTTracker<GNN_Tracker<double>, double>);
static_assert(CPTTracker<Id_Tracker<double>, double>);
static_assert(CPTTracker<NN_Tracker<double>, double>);
static_assert(CPTTracker<NO_Tracker<double>, double>);
static_assert(CPTTracker<NoTTTFilter<double>, double>);
static_assert(CPTTracker<TransTTTFilter<double>, double>);

}  // namespace ttb