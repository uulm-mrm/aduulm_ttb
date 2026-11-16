#include "tracking_lib/BirthModels/DynamicBirthModel.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/StateModels/BaseStateModel.h"

namespace ttb
{

template class DynamicBirthModel<double>;
template class DynamicBirthModel<AD>;

template <class UDT>
DynamicBirthModel<UDT>::DynamicBirthModel(TTBManager<UDT>* manager) : _manager{ manager }
{
}

template <class UDT>
BIRTH_MODEL_TYPE DynamicBirthModel<UDT>::type() const
{
  return BIRTH_MODEL_TYPE::DYNAMIC;
}

template <class UDT>
TTBManager<UDT>* DynamicBirthModel<UDT>::manager() const
{
  return _manager;
}

template <class UDT>
BirthTracks<UDT> DynamicBirthModel<UDT>::getBirthTracks(DataSourceContainer<UDT> const& measContainer,
                                                        std::map<MeasurementId, UDT> const& assignProb,
                                                        std::vector<State<UDT>> const& existingTracks)
{
  LOG_DEB("get dynmaic birth tracks");
  assert(measContainer._measurements.size() == assignProb.size());
  auto const& meas_model_id = measContainer._source_id;
  if (not manager()->params()->meas_model_params(measContainer._source_id).can_init)
  {
    LOG_DEB("Meas Model: " + meas_model_id.value_ + " can not init");
    if (manager()->params()->show_gui)
    {
      _manager->vizu()._meas_model_data.wlock().data[measContainer._source_id]._birth_tracks.emplace_back(
          measContainer._time, std::vector<BernoulliState<UDT>>{});
    }
    return {};
  }
  if (_oldMeasurements.contains(meas_model_id))
  {
    for (auto& meas : _oldMeasurements.at(meas_model_id))
    {
      meas.compensateEgoMotion(measContainer._time - meas._time, measContainer._predictionInfo._egoMotion);
    }
  }
  BirthTracks<UDT> birth_tracks;
  std::vector<State<UDT>> newBirthCandidates;
  UDT adaptive_birth_denumerator = 0;
  for (const auto& [measID, associationProb] : assignProb)
  {
    if (associationProb < manager()->params()->birth_model->dynamic_model.birth_threshold)  // set up new state
    {
      auto measIt =
          std::ranges::find_if(measContainer._measurements, [&](State<UDT> const& m) { return m._id == measID; });
      if (measIt == measContainer._measurements.end())
      {
        continue;
      }
      adaptive_birth_denumerator += 1 - associationProb;
      State<UDT> const& meas = *measIt;
      std::optional<State<UDT>> newState =
          _manager->getMeasModel(meas_model_id)
              .create_state(meas, manager()->params()->birth_model->dynamic_model.force_one_step_birth);

      if (newState.has_value())
      {
        LOG_DEB("Can directly init");
        if (UDT const exProbSpatial = getBirthProbability<UDT>(*_manager, newState.value(), existingTracks);
            exProbSpatial > 0)
        {
          newState.value()._misc["adaptive_birth_numerator"] =
              UDT{ (1 - associationProb) * manager()->params()->birth_model->dynamic_model.mean_num_birth };
          if (manager()->params()->birth_model->dynamic_model.use_default_values)
          {
            // overwrite birth covariance of tracks
            std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> birth_dist;
            CLASS measured_class = measIt->_classification.getEstimate();
            for (auto& [model_id, model_dist] : newState.value()._state_dist)
            {
              for (auto const& [weight, gauss] : model_dist.second.to_gaussian_components())
              {
                Matrix<UDT> cov = gauss.covariance();
                Vector<UDT> mean = gauss.mean();
                for (auto [ind, comp] : std::views::enumerate(_manager->state_comps(model_id)._comps))
                {
                  auto const default_values =
                      default_val(manager()->params()->birth_model->dynamic_model.default_values, comp, measured_class);
                  if (default_values.mean.has_value())
                  {
                    LOG_FATAL("Overwrite mean of " + to_string(comp) + " with default value of " +
                              std::to_string(default_values.mean.value()));
                    mean(ind) = default_values.mean.value();
                  }
                  if (default_values.var.has_value())
                  {
                    LOG_DEB("Overwrite var of " + to_string(comp) + " with default value of " +
                            std::to_string(default_values.var.value()));
                    cov(ind, ind) = default_values.var.value();
                  }
                }
                birth_dist[model_id].emplace_back(model_dist.first * weight,
                                                  GaussianDistribution(std::move(mean), std::move(cov)));
              }
            }
            newState.value()._state_dist = newState.value().build_consistent_state_dist(std::move(birth_dist));
          }
          birth_tracks.birth_tracks.emplace_back(std::move(newState.value()), exProbSpatial);
          birth_tracks.birth_measurements.push_back(meas);
        }
      }
    }
  }
  _oldMeasurements[meas_model_id] = std::move(newBirthCandidates);
  assert([&]() {  // NOLINT
    if (birth_tracks.birth_tracks.size() != birth_tracks.birth_measurements.size())
    {
      LOG_FATAL("measurements inconsistent with birth");
      return false;
    }
    for (auto const& [track, meas] : std::views::zip(birth_tracks.birth_tracks, birth_tracks.birth_measurements))
    {
      if (not track.state.isValid())
      {
        LOG_FATAL("invalid birth track: " << track.state.toString());
        return false;
      }
      LOG_DEB("valid birth track:" << track.state.toString());
    }
    return true;
  }());
  for (BernoulliState<UDT>& birth_state : birth_tracks.birth_tracks)
  {
    birth_state.existence_probability = std::min(birth_state.existence_probability,
                                                 std::any_cast<UDT>(birth_state.state._misc["adaptive_birth_"
                                                                                            "numerator"]) /
                                                     adaptive_birth_denumerator);
    if (auto const it =
            std::ranges::find(manager()->params()->birth_model->dynamic_model.meas_model_specific_existence_probs,
                              measContainer._source_id.value_,
                              &BirthModelExistenceParams::source_id);
        it != manager()->params()->birth_model->dynamic_model.meas_model_specific_existence_probs.end())
    {
      birth_state.existence_probability = it->existence_prob;
    }
    birth_state.state._type = STATE_TYPE::BIRTH;
    birth_state.state._time = measContainer._time;
  }
  if (manager()->params()->show_gui)
  {
    _manager->vizu()._meas_model_data.wlock().data[measContainer._source_id]._birth_tracks.emplace_back(
        measContainer._time, birth_tracks.birth_tracks);
  }

  return birth_tracks;
}

template <class UDT>
void DynamicBirthModel<UDT>::reset()
{
  LOG_DEB("Reset Dynamic Birth Model");
  _oldMeasurements.clear();
}

}  // namespace ttb
