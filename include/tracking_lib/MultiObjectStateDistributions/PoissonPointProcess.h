#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/DataContainer/DataSourceContainer.h"

namespace ttb
{

template<class UDT>
class PoissonPointProcess
{
public:
  PoissonPointProcess(TTBManager<UDT>* manager, std::vector<State<UDT>> initial_intensity);
  explicit PoissonPointProcess(TTBManager<UDT>* ttb_manager);

  void predict(Duration deltaT, PredictionInformation const& pred);

  void birth(std::vector<BernoulliState<UDT>> states);

  /// updates ppp with undetected targets
  void update(DataSourceContainer<UDT> const& measurementContainer);

  std::pair<std::vector<UDT>, UDT> normalizeLogWeights(std::vector<UDT> const& log_weights);

  /// prune all tracks with probability smaller than threshold
  void prune(double threshold);

  void removeEmptyStates();

  TTBManager<UDT>* _manager;

  std::vector<State<UDT>> _intensity;  // PPP intensity is mixture of Gaussian
  std::vector<State<UDT>> _detected_for_the_first_time;
};

}  // namespace ttb
