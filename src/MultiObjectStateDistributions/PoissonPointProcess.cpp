#include "tracking_lib/MultiObjectStateDistributions/PoissonPointProcess.h"

#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/MultiObjectStateDistributions/Utils.h"

#include <utility>

namespace ttb
{

template class PoissonPointProcess<double>;
template class PoissonPointProcess<AD>;

template <class UDT>
PoissonPointProcess<UDT>::PoissonPointProcess(TTBManager<UDT>* manager, std::vector<State<UDT>> initial_intensity)
  : _manager{ manager }, _intensity{ std::move(initial_intensity) }
{
}
template <class UDT>
PoissonPointProcess<UDT>::PoissonPointProcess(TTBManager<UDT>* manager) : PoissonPointProcess(manager, {})
{
}

template <class UDT>
void PoissonPointProcess<UDT>::predict(Duration deltaT, PredictionInformation const& pred)
{
  LOG_DEB("PPPDistribution::predict");
  LOG_DEB("Number of ppp intensity states " << _intensity.size());
  LOG_DEB("Before predict");

  std::unordered_map<StateId, State<UDT>> predicted_states =
      utils::predict<UDT>(_intensity, deltaT, pred._egoMotion, {}, [&](State<UDT>& state) {
        UDT const survival_probability =
            _manager->getPersistenceModel().getPersistenceProbability(state, deltaT, pred._roi);
        state._log_weight += log(survival_probability);
      });
  LOG_DEB("After predict");
  _intensity.clear();
  for (State<UDT>& state : predicted_states | std::views::values)
  {
    _intensity.push_back(std::move(state));
  }
}

template <class UDT>
void PoissonPointProcess<UDT>::birth(std::vector<BernoulliState<UDT>> birth_tracks)
{
  LOG_DEB("PPPDistribution::birth");
  LOG_DEB("Insert " << birth_tracks.size() << " Birth Tracks");
  for (BernoulliState<UDT>& new_born : birth_tracks)
  {
    double birth_weight = _manager->params()->pmbm_distribution.ppp.birth_weight;
    new_born.state._log_weight = std::log(birth_weight);
    _manager->database().state_existences.emplace(new_born.state._id, new_born.existence_probability);
    _intensity.push_back(std::move(new_born.state));
  }
  LOG_DEB("Number of intensity components: " << _intensity.size());
}

template <class UDT>
void PoissonPointProcess<UDT>::update(DataSourceContainer<UDT> const& measurementContainer)
{
  LOG_DEB("PoissonPointProcess::update");
  std::vector<State<UDT>> updated_potential_tracks;
  std::vector<State<UDT>> updated_undetected_tracks;

  // Group all updates per measurement
  std::unordered_map<StateId, std::vector<State<UDT>>> meas_to_states;

  std::unordered_map<StateId, UDT> clutter_to_state;
  // Clutter is a measurement-model property, not a by-product of PPP innovations.
  auto const& measurement_model = _manager->getMeasModel(measurementContainer._source_id);
  for (State<UDT> const& meas : measurementContainer._measurements)
  {
    UDT const kappa = measurement_model.calculate_clutter_intensity(meas, measurementContainer._sensorInfo);
    clutter_to_state.emplace(meas._id, kappa);
  }

  {
    // no gating for ppp comps
    Params params = *_manager->params();
    params.meas_model_params(measurementContainer._source_id).gating_prob = 1;
    _manager->update_params(params);
    std::vector<std::tuple<StateDetection<UDT>,
                           std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>>,
                           std::vector<MeasClutter<UDT>>>>
        updates = utils::innovate(_intensity, measurementContainer, std::identity{});
    params.meas_model_params(measurementContainer._source_id).gating_prob =
        _manager->original_params().meas_model_params(measurementContainer._source_id).gating_prob;
    _manager->update_params(std::move(params));
    for (auto& [state_detection, state_meas_updates, meas_infos] : updates)
    {
      for (auto& [old_state_id, meas_id, updated_state, meas_log_likelihood] : state_meas_updates)
      {
        if (meas_id != NOT_DETECTED)
        {
          auto const clutter_info = std::ranges::find_if(
              meas_infos, [&](MeasClutter<UDT> const& meas_info) { return meas_info.meas_id == meas_id; });
          assert(clutter_info != meas_infos.end());
          clutter_to_state.emplace(meas_id, clutter_info->clutter_intensity);
        }
        if (meas_id == NOT_DETECTED)
        {
          updated_state._log_weight += std::log(1 - state_detection.detection_probability);
          _manager->database().states.insert(updated_state);
          updated_undetected_tracks.emplace_back(std::move(updated_state));
        }
        else
        {
          updated_state._log_weight += std::log(state_detection.detection_probability) + meas_log_likelihood;
          _manager->database().states.insert(updated_state);
          meas_to_states[meas_id].push_back(std::move(updated_state));
        }
      }
    }
  }
  LOG_DEB("# of NOT GATED meas_ids after innovation: " << meas_to_states.size());
  {
    for (State<UDT> const& measurement : measurementContainer._measurements)
    {
      if (meas_to_states.contains(measurement._id))
      {
        // Do Moment Matching
        std::vector<State<UDT>> const& states = meas_to_states.at(measurement._id);
        assert(not states.empty());
        LOG_DEB("# of states to be merged: " << states.size());
        std::vector<UDT> log_weights;
        for (State<UDT> const& state : states)
        {
          log_weights.push_back(state._log_weight);
        }

        const auto& [normalizedLogWeights, log_weight_sum] = normalizeLogWeights(log_weights);
        LOG_DEB("log_weight_sum: " << log_weight_sum);
        {
          assert([&]() {  // NOLINT
            double control_sum = 0.0;
            for (const auto& norm : normalizedLogWeights)
            {
              control_sum += std::exp(to_val(norm));
            }
            return std::abs(control_sum - 1.0) < 1e-6;
          }() && "Normalized log weights do not sum to 1.0");
        }
        std::vector<std::pair<UDT, StateId>> for_merging;
        for (auto [idx, state] : std::views::enumerate(states))
        {
          // Normalized weights avoid underflow for very small PPP intensities.
          for_merging.emplace_back(std::exp(normalizedLogWeights.at(idx)), state._id);
        }

        State result = merge(*_manager, for_merging);
        assert(result._meta_data._meas_update_weight.at(measurementContainer._source_id).size() == 1);
        UDT const log_likelihood = std::log(clutter_to_state.at(measurement._id) + std::exp(log_weight_sum));
        assert(!std::isnan(log_likelihood));
        UDT const result_existenceProbability = std::exp(log_weight_sum - log_likelihood);
        result._log_weight = log_likelihood;
        result._label = State<UDT>::_labelGenerator.getID();
        result._type = STATE_TYPE::BIRTH;
        LOG_DEB("Merged result state:\n" << result.toString());
        if (not result.isEmpty())
        {
          assert(result.isValid());
          _manager->database().state_existences.emplace(result._id, result_existenceProbability);
          updated_potential_tracks.emplace_back(std::move(result));
        }
        else
        {
          throw std::runtime_error("PPP moment matching returned an empty state for a positive PPP detection weight");
        }
      }
      else
      {
        // No PPP component explains this measurement: only the clutter term remains.
        UDT const kappa = clutter_to_state.at(measurement._id);
        State dummyState(_manager, {});
        dummyState._log_weight = kappa > 0.0 ? std::log(kappa) : UDT{ -std::numeric_limits<double>::infinity() };
        dummyState._meta_data._meas_update_weight = { { measurementContainer._source_id,
                                                        { { measurement._id, UDT{ 1.0 } } } } };
        _manager->database().state_existences.emplace(dummyState._id, 0.0);
        LOG_DEB("Pure-clutter / impossible new-object branch: kappa=" << kappa
                                                                      << ", log_weight=" << dummyState._log_weight);
        updated_potential_tracks.emplace_back(std::move(dummyState));
      }
    }
  }
  assert(updated_potential_tracks.size() == measurementContainer._measurements.size());
  _intensity = std::move(updated_undetected_tracks);
  _detected_for_the_first_time = std::move(updated_potential_tracks);
  removeEmptyStates();
}

template <class UDT>
std::pair<std::vector<UDT>, UDT> PoissonPointProcess<UDT>::normalizeLogWeights(const std::vector<UDT>& log_weights)
{
  std::size_t n = log_weights.size();
  std::vector<UDT> normalized_log_weights(n);
  UDT log_sum_w = 0.0;

  assert(n);
  if (n == 1)
  {
    log_sum_w = log_weights[0];
    normalized_log_weights[0] = 0.0;  // log(1)
    return { normalized_log_weights, log_sum_w };
  }

  // Log-Sum-Exp: stable
  auto max_iter = std::ranges::max_element(log_weights);
  UDT max_log_w = *max_iter;

  UDT sum_exp = 0.0;
  for (UDT lw : log_weights)
  {
    sum_exp += std::exp(lw - max_log_w);
  }

  log_sum_w = max_log_w + std::log(sum_exp);

  for (std::size_t i = 0; i < n; ++i)
  {
    normalized_log_weights[i] = log_weights[i] - log_sum_w;
  }

  return { normalized_log_weights, log_sum_w };
}

template <class UDT>
void PoissonPointProcess<UDT>::prune(double threshold)
{
  LOG_DEB("PoissonPointProcess::prune");

  for (auto it = _intensity.begin(); it != _intensity.end();)
  {
    if (it->_log_weight < std::log(threshold) || it->isEmpty())
    {
      it = _intensity.erase(it);  // erase returns next iterator
    }
    else
    {
      ++it;
    }
  }
}

template <class UDT>
void PoissonPointProcess<UDT>::removeEmptyStates()
{
  LOG_DEB("PoissonPointProcess::removeEmptyStates");

  for (auto it = _intensity.begin(); it != _intensity.end();)
  {
    if (it->isEmpty())
    {
      LOG_DEB("State with ID: " << it->_id << " is empty and was removed");
      it = _intensity.erase(it);  // erase returns next iterator
    }
    else
    {
      ++it;
    }
  }
}

}  // namespace ttb
