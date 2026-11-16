#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/MultiObjectStateDistributions/PoissonPointProcess.h"
#include "tracking_lib/MultiObjectStateDistributions/MBMDistribution.h"

namespace ttb
{

template<class UDT>
class PMBMDistribution final
{
public:
  explicit PMBMDistribution(TTBManager<UDT>* manager);
  PMBMDistribution(TTBManager<UDT>* manager, PoissonPointProcess<UDT> ppp, MBMDistribution<UDT> mbm);

  /// Predict this distribution deltaT time into the future.
  void predict(Duration deltaT, PredictionInformation const& prediction_info);
  /// adding new born but yet undetected tracks to PPP
  void add_tracks(std::vector<BernoulliState<UDT>> states);
  /// Update this PMBM distribution with the given MeasurementContainer.
  /// The update method is selected depending on the parameters.
  void update(DataSourceContainer<UDT> const& measurementContainer);
  /// return all tracks of the filter
  [[nodiscard]] std::vector<StateId> all_tracks() const;
  /// the probability/weight a measurement contributed to the update of any track in the last update
  std::map<MeasurementId, UDT> meas_assignment_prob(DataSourceContainer<UDT> const& data_container) const;

  std::vector<BernoulliState<UDT>> get_estimate() const;
  /// check if all tracks are valid
  [[nodiscard]] bool is_valid() const;

  TTBManager<UDT>* _manager;
  /// all tracks
  std::vector<State<UDT>> _tracks;
  /// new detected tracks
  std::vector<State<UDT>> _newTracks;
  /// undetected targets
  PoissonPointProcess<UDT> _ppp;
  /// detected targets
  MBMDistribution<UDT> _mbm;
};

static_assert(CPTCommonMODist<PMBMDistribution<double>, double>);
static_assert(CPTICUpdate<PMBMDistribution<double>, double>);

}  // namespace ttb
