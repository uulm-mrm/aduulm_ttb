#include "tracking_lib/MultiObjectStateDistributions/PMBMDistribution.h"

#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/MultiObjectStateDistributions/PoissonPointProcess.h"
#include "tracking_lib/MultiObjectStateDistributions/MBMDistribution.h"

#include <utility>

namespace ttb
{
constexpr const auto tracy_color = tracy::Color::LightSalmon;


template class PMBMDistribution<double>;
template class PMBMDistribution<AD>;

template <class UDT>
PMBMDistribution<UDT>::PMBMDistribution(TTBManager<UDT>* manager)
  : _manager{ manager }, _ppp{ _manager }, _mbm{ manager }
{
}

template <class UDT>
PMBMDistribution<UDT>::PMBMDistribution(TTBManager<UDT>* manager, PoissonPointProcess<UDT> ppp, MBMDistribution<UDT> mbm)
  : _manager{ manager }, _ppp{ std::move(ppp) }, _mbm{ std::move(mbm) }
{
}

template <class UDT>
void PMBMDistribution<UDT>::predict(Duration deltaT, PredictionInformation const& prediction_info)
{
  LOG_DEB("Predict PMBM Distribution with " + std::to_string(_ppp._intensity.size()) + " PPP Components and "
          << std::to_string(_mbm._bernoulli_basis.size()) + " MBM Components");
  assert(std::ranges::all_of(_ppp._intensity, std::identity{}, &State<UDT>::isValid));
  assert(std::ranges::all_of(_mbm._bernoulli_basis | std::views::values, std::identity{}, &State<UDT>::isValid));
  _mbm.predict(deltaT, prediction_info);
  _ppp.predict(deltaT, prediction_info);
}

template <class UDT>
void PMBMDistribution<UDT>::add_tracks(std::vector<BernoulliState<UDT>> states)
{
  LOG_DEB("Birth of new but undetected targets");

  _ppp.birth(std::move(states));
}

template <class UDT>
void PMBMDistribution<UDT>::update(DataSourceContainer<UDT> const& measurementContainer)
{
  LOG_DEB("Update PMBM Distribution");
  _ppp.update(measurementContainer);

  _mbm._new_bernoullis.clear();
  for (State<UDT>& new_track : _ppp._detected_for_the_first_time)
  {
    StateId const id = new_track._id;
    _mbm._new_bernoullis.emplace(id, std::move(new_track));
  }

  _mbm.update(measurementContainer);

  if (_manager->params()->pmbm_distribution.mbm.enable_projection)
  {
    _mbm.projectToPMB();
  }

  if (_manager->params()->pmbm_distribution.mbm.enable_recycling)
  {
    // Recycling and updating global hypotheses
    std::vector<State<UDT>> recycled_states = _mbm.recycleBernoulliStates();
    std::vector<StateId> recycled_ids;

    for (State<UDT> const& state : recycled_states)
    {
      recycled_ids.push_back(state._id);
    }

    LOG_DEB("# of recycled states: " << recycled_states.size());
    _ppp._intensity.insert(_ppp._intensity.end(),
                           std::make_move_iterator(recycled_states.begin()),
                           std::make_move_iterator(recycled_states.end()));

    _mbm.cleanupBernoulliBasis();
    LOG_DEB("# of cleaned up Bernoulli Ids after recycling: " << _mbm._bernoulli_basis.size());

    _mbm.updateGlobalHypotheses(recycled_ids);
  }

  LOG_DEB("# of final global hypotheses: " << _mbm._global_hypotheses.size());

  _ppp.prune(_manager->params()->pmbm_distribution.ppp.pruning_threshold);

  LOG_DEB("# of Bernoullis after update: " << _mbm._bernoulli_basis.size());
  TracyPlotConfig("# PPP components", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("# PPP components", static_cast<int64_t>(_ppp._intensity.size()));
  TracyPlotConfig("# Bernoullis", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("# Bernoullis", static_cast<int64_t>(_mbm._bernoulli_basis.size()));
}

template <class UDT>
std::vector<BernoulliState<UDT>> PMBMDistribution<UDT>::get_estimate() const
{
  LOG_DEB("getEstimate PMBM Distribution");
  return _mbm.getEstimate();
}

template <class UDT>
bool PMBMDistribution<UDT>::is_valid() const
{
  return std::ranges::all_of(_tracks, [&](State<UDT> const& track) {
    if (not track.isValid())
    {
      LOG_FATAL("Track " + track.toString() + "not valid");
      return false;
    }
    return true;
  });
}

template <class UDT>
std::vector<StateId> PMBMDistribution<UDT>::all_tracks() const
{
  std::vector<StateId> state_ids;
  for (State<UDT> const& state : _tracks)
  {
    state_ids.push_back(state._id);
    _manager->database().states.insert(state);
  }
  return state_ids;
}

template <class UDT>
std::map<MeasurementId, UDT> PMBMDistribution<UDT>::meas_assignment_prob(DataSourceContainer<UDT> const&) const
{
  return {};
}

}  // namespace ttb
