#include "tracking_lib/BirthModels/StaticBirthModel.h"
#include "tracking_lib/BirthModels/BaseBirthModel.h"

#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/StateModels/BaseStateModel.h"

namespace ttb
{

template class StaticBirthModel<double>;
template class StaticBirthModel<AD>;

template <class UDT>
StaticBirthModel<UDT>::StaticBirthModel(TTBManager<UDT>* manager) : _manager{ manager }
{
}

template <class UDT>
BIRTH_MODEL_TYPE StaticBirthModel<UDT>::type() noexcept
{
  return BIRTH_MODEL_TYPE::STATIC;
}

template <class UDT>
TTBManager<UDT>* StaticBirthModel<UDT>::manager() const
{
  return _manager;
}

template <class UDT>
BirthTracks<UDT> StaticBirthModel<UDT>::getBirthTracks(DataSourceContainer<UDT> const& measContainer,
                                                       std::map<MeasurementId, UDT> const& /*unused*/,
                                                       std::vector<State<UDT>> const& existingTracks)
{
  if (not manager()->params()->meas_model_params(measContainer._source_id).can_init)
  {
    LOG_DEB("Meas Model: " + measContainer._source_id.value_ + " can not init");
    if (manager()->params()->show_gui)
    {
      _manager->vizu()._meas_model_data.wlock().data[measContainer._source_id]._birth_tracks.emplace_back(
          measContainer._time, std::vector<BernoulliState<UDT>>{});
    }
    return {};
  }
  BirthTracks<UDT> birthTracks;
  for (StaticBirthLocation const& location : manager()->params()->birth_model->static_model.locations)
  {
    std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> birth_dist;
    for (auto [model_ctr, modelId] : std::views::enumerate(manager()->params()->state.multi_model.use_state_models))
    {
      Components const& state_comps = _manager->state_comps(modelId);
      Vector<UDT> mean(state_comps._comps.size());
      Matrix<UDT> cov = Matrix<UDT>::Zero(state_comps._comps.size(), state_comps._comps.size());
      for (COMPONENT comp : state_comps._comps)
      {
        auto ind = state_comps.indexOf(comp).value();
        if (auto const it = location.mean.find(to_string(comp)); it == location.mean.end())
        {
          LOG_FATAL("Location mean does not contain comp: " + to_string(comp) +
                    " which is needed by state model: " + std::to_string(modelId.value_));
          std::terminate();
        }
        else
        {
          mean(ind) = it->second;
        }
        if (auto const it = location.var.find(to_string(comp)); it == location.var.end())
        {
          LOG_FATAL("Location var does not contain comp: " + to_string(comp) +
                    " which is needed by state model: " + std::to_string(modelId.value_));
          std::terminate();
        }
        else
        {
          cov(ind, ind) = it->second;
        }
      }
      birth_dist[modelId].emplace_back(manager()->params()->state.multi_model.birth_weights.at(model_ctr),
                                       GaussianDistribution(mean, cov));
    }
    State newState(_manager, std::move(birth_dist));
    UDT const newState_existenceProbability = getBirthProbability<UDT>(*_manager, newState, existingTracks);
    newState._score = log(newState_existenceProbability / (1 - newState_existenceProbability));
    newState._type = STATE_TYPE::BIRTH;
    newState._time = measContainer._time;
    if (newState_existenceProbability > 0)
    {
      birthTracks.birth_tracks.emplace_back(std::move(newState), newState_existenceProbability);
      birthTracks.birth_measurements.emplace_back(std::nullopt);
    }
  }
  if (manager()->params()->show_gui)
  {
    _manager->vizu()._meas_model_data.wlock().data[measContainer._source_id]._birth_tracks.emplace_back(
        measContainer._time, birthTracks.birth_tracks);
  }
  return birthTracks;
}

template <class UDT>
void StaticBirthModel<UDT>::reset()
{
  LOG_DEB("Reset Static Birth Model");
}

}  // namespace ttb