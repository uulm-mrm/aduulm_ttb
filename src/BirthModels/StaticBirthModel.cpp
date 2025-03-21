#include "tracking_lib/BirthModels/StaticBirthModel.h"

#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"

#include <tracy/tracy/Tracy.hpp>

namespace ttb
{
StaticBirthModel::StaticBirthModel(TTBManager* manager) : _manager{ manager }
{
  LOG_DEB("Init static birth model");
  loadBirthLocations();
}

BIRTH_MODEL_TYPE StaticBirthModel::type() const noexcept
{
  return BIRTH_MODEL_TYPE::STATIC;
}

TTBManager* StaticBirthModel::manager() const
{
  return _manager;
}

std::vector<State> StaticBirthModel::getBirthTracks(MeasurementContainer const& measContainer,
                                                    std::map<MeasurementId, Probability> const& /*unused*/,
                                                    std::vector<State> const& existingTracks)
{
  ZoneScopedN("StaticBirthModel::birthTracks");
  if (not _manager->meas_model_params(measContainer._id).can_init)
  {
    LOG_DEB("Meas Model: " + measContainer._id.value_ + " can not init");
    return {};
  }
  std::vector<State> birthTracks;
  for (std::size_t i = 0; i < _birthLocations.size(); ++i)
  {
    State newState = _manager->createState();
    std::size_t ctr = 0;
    for (StateModelId modelId : _manager->params().state.multi_model.use_state_models)
    {
      BaseStateModel const& sm = *_manager->getStateModelMap().at(modelId);
      Vector mean(sm.state_comps()._comps.size());
      Matrix cov = Matrix::Zero(sm.state_comps()._comps.size(), sm.state_comps()._comps.size());
      for (COMPONENT comp : sm.state_comps()._comps)
      {
        auto ind = sm.state_comps().indexOf(comp).value();
        mean(ind) = _birthLocations.at(i).at(comp);
        cov(ind, ind) = _birthVariances.at(i).at(comp);
      }
      newState._state_dist.at(modelId)->merge(std::make_unique<GaussianDistribution>(
          mean, cov, _manager->params().state.multi_model.birth_weights.at(ctr), REFERENCE_POINT::CENTER));
      ++ctr;
    }
    newState._existenceProbability = getBirthProbability(newState, existingTracks);
    newState._score = log(newState._existenceProbability / (1 - newState._existenceProbability));
    if (newState._existenceProbability > 0)
    {
      birthTracks.push_back(std::move(newState));
    }
  }
  std::string info_str = "#BirthTracks: " + std::to_string(birthTracks.size());
  ZoneText(info_str.c_str(), info_str.size());
  return birthTracks;
}

void StaticBirthModel::loadBirthLocations()
{
  State const state = _manager->createState();
  Components allComps = _manager->getStateModel(state._state_dist.begin()->first).state_comps();
  for (auto const& [modelId, _] : state._state_dist)
  {
    allComps = allComps.merge(_manager->getStateModel(modelId).state_comps());
  }
  int bIdx = 0;
  for (StaticBirthLocation const& location : _manager->params().birth_model->static_model.locations)
  {
    std::map<COMPONENT, double> bstate;
    std::map<COMPONENT, double> bvar;
    for (COMPONENT comp : allComps._comps)
    {
      if (not location.mean.contains(to_string(comp)))
      {
        LOG_FATAL("Location " + std::to_string(bIdx) + " MEAN does not contain needed comp: " + to_string(comp));
      }
      bstate.emplace(comp, location.mean.at(to_string(comp)));
      if (not location.var.contains(to_string(comp)))
      {
        LOG_FATAL("Location " + std::to_string(bIdx) + " VAR does not contain needed comp: " + to_string(comp));
      }
      bvar.emplace(comp, location.var.at(to_string(comp)));
    }
    _birthLocations.emplace_back(std::move(bstate));
    _birthVariances.emplace_back(std::move(bvar));
    bIdx++;
  }

  if (_birthLocations.empty())
  {
    LOG_FATAL("No birth locations specified");
  }
  if (_birthVariances.empty())
  {
    LOG_ERR("No birth covariance specified");
    throw std::runtime_error("No birth covariance specified");
  }
  LOG_INF("birth means:");
  for (auto const& loc : _birthLocations)
  {
    LOG_DEB("Mean x: " << loc.at(COMPONENT::POS_X) << " \tMean y: " << loc.at(COMPONENT::POS_Y));
  }
  for (auto const& var : _birthVariances)
  {
    LOG_DEB("VAR x: " << var.at(COMPONENT::POS_X) << " \tVAR y: " << var.at(COMPONENT::POS_Y));
  }
  LOG_INF("Allocated birth locations: " << _birthLocations.size());
  LOG_INF("Allocated birth variances: " << _birthVariances.size());
}

void StaticBirthModel::reset() noexcept
{
  LOG_DEB("Reset Static Birth Model");
}

}  // namespace ttb