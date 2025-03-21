#include "tracking_lib/BirthModels/DynamicBirthModel.h"
// #####################################################################################################################
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
// #####################################################################################################################
#include <tracy/tracy/Tracy.hpp>

namespace ttb
{
DynamicBirthModel::DynamicBirthModel(TTBManager* manager) : _manager{ manager }
{
}

BIRTH_MODEL_TYPE DynamicBirthModel::type() const
{
  return BIRTH_MODEL_TYPE::DYNAMIC;
}

TTBManager* DynamicBirthModel::manager() const
{
  return _manager;
}

std::vector<State> DynamicBirthModel::getBirthTracks(MeasurementContainer const& measContainer,
                                                     std::map<MeasurementId, Probability> const& assignProb,
                                                     std::vector<State> const& existingTracks)
{
  ZoneScopedN("DynamicBirthModel::birthTracks");
  assert(measContainer._data.size() == assignProb.size());
  auto const& meas_model_id = measContainer._id;
  auto const& meas_model = *_manager->getMeasModelMap().at(meas_model_id);
  if (not _manager->meas_model_params(measContainer._id).can_init)
  {
    LOG_DEB("Meas Model: " + meas_model_id.value_ + " can not init");
    return {};
  }
  LOG_DEB("initBaseBirthTracks");
  LOG_DEB("Ego compensate existing birth candidates");
  if (_oldMeasurements.contains(meas_model_id))
  {
    for (auto& meas : _oldMeasurements.at(meas_model_id))
    {
      for (BaseDistribution* dist : meas._dist->dists())
      {
        assert(dist->type() == DISTRIBUTION_TYPE::GAUSSIAN);
        EgoCompensated egoComp = compensateEgoMotion(dist->mean(),
                                                     dist->covariance(),
                                                     meas._meas_comps,
                                                     measContainer._time - meas._time,
                                                     measContainer._egoMotion);
        dist->set(std::move(egoComp.state), std::move(egoComp.cov));
      }
    }
  }
  std::vector<State> birth_tracks;
  std::vector<Measurement> newBirthCandidates;
  double adaptive_birth_denumerator = 0;
  for (const auto& [measID, associationProb] : assignProb)
  {
    if (associationProb < _manager->params().birth_model->dynamic_model.birth_threshold)  // set up new state
    {
      auto measIt = std::ranges::find_if(measContainer._data, [&](Measurement const& m) { return m._id == measID; });
      if (measIt == measContainer._data.end())
      {
        continue;
      }
      adaptive_birth_denumerator += 1 - associationProb;
      Measurement const& meas = *measIt;
      std::optional<State> newState =
          meas_model.createState(meas, _manager->params().birth_model->dynamic_model.force_one_step_birth);
      if (newState.has_value())
      {
        LOG_DEB("Can directly init");
        if (double const exProbSpatial = getBirthProbability(newState.value(), existingTracks); exProbSpatial > 0)
        {
          newState.value()._existenceProbability = exProbSpatial;
          newState.value()._misc["adaptive_birth_numerator"] =
              (1 - associationProb) * _manager->params().birth_model->dynamic_model.mean_num_birth;
          if (_manager->params().birth_model->dynamic_model.use_default_values)
          {
            // overwrite birth covariance of tracks
            CLASS measured_class = measIt->_classification.getEstimate();
            for (auto& [model_id, model_dist] : newState.value()._state_dist)
            {
              Matrix cov = model_dist->covariance();
              Vector mean = model_dist->mean();
              for (auto [ind, comp] : std::views::enumerate(_manager->getStateModel(model_id).state_comps()._comps))
              {
                auto const default_values =
                    default_val(_manager->params().birth_model->dynamic_model.default_values, comp, measured_class);
                if (default_values.mean.has_value())
                {
                  LOG_DEB("Overwrite mean of " + to_string(comp) + " with default value of " +
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
              for (auto& dist : model_dist->dists())
              {
                dist->set(mean);
                dist->set(cov);
              }
            }
          }
          birth_tracks.push_back(std::move(newState.value()));
        }
      }
      else
      {
        LOG_DEB("Do a 2 step init");
        if (_oldMeasurements.contains(meas_model_id))
        {
          LOG_DEB("#Old Birth Candidates for Meas Model " << meas_model_id << ": "
                                                          << _oldMeasurements.at(meas_model_id).size());
          LOG_DEB("Already have some Candidates");
          LOG_DEB("Try to create new State Dist");
          auto initializedDist =
              meas_model.initStage2(_oldMeasurements.at(meas_model_id), *measIt, measContainer._sensorInfo);
          // ensure that we were really able to initialize a track!
          if (initializedDist.has_value())
          {
            LOG_DEB("Initializing a new track, rz entry: " << associationProb);
            auto birth_model_prob = getBirthProbability(initializedDist.value(), existingTracks);
            initializedDist.value()._existenceProbability = birth_model_prob;
            initializedDist.value()._misc["adaptive_birth_numerator"] =
                (1 - associationProb) * _manager->params().birth_model->dynamic_model.mean_num_birth;
            initializedDist.value()._score = log(birth_model_prob / (1 - birth_model_prob));
            LOG_DEB("existence probability of new born track: " << birth_model_prob);
            LOG_DEB("birthModelProb: " << birth_model_prob);
            LOG_DEB("meanNumBirth: " << _manager->params().birth_model->dynamic_model.mean_num_birth);
            LOG_DEB("assoc Prob: " << associationProb);
            LOG_DEB("initializedDist->_state_dist.size(): " << initializedDist->_state_dist.size());
            birth_tracks.push_back(std::move(initializedDist.value()));
          }
          else  // the birth candidates have not fit to the measurement, add the measurement as birth candidate to
                // the next time step
          {
            LOG_DEB("Measurement does nit fit to the birth candidates");
            // if a measurement was not able to initialize a track, it may still be a birth candidate for the next
            // time step.
            newBirthCandidates.push_back(*measIt);
          }
          LOG_DEB("#Next step Birth Candidates Model " << meas_model_id << ": " << newBirthCandidates.size());
        }
        else
        {
          newBirthCandidates.push_back(*measIt);
        }
      }
    }
  }
  _oldMeasurements[meas_model_id] = std::move(newBirthCandidates);
  LOG_DEB("#BirthTracks: " << birth_tracks.size());
  assert([&]() {  // NOLINT
    for (auto const& track : birth_tracks)
    {
      if (not track.isValid())
      {
        LOG_FATAL("invalid birth track: " << track.toString());
        return false;
      }
      LOG_DEB("valid birth track:" << track.toString());
    }
    return true;
  }());
  std::string birth_tracks_str = "#Birth Tracks: " + std::to_string(birth_tracks.size());
  ZoneText(birth_tracks_str.c_str(), birth_tracks_str.size());
  for (State& birth_state : birth_tracks)
  {
    birth_state._existenceProbability =
        std::min(birth_state._existenceProbability,
                 std::any_cast<double>(birth_state._misc["adaptive_birth_numerator"]) / adaptive_birth_denumerator);
  }
  return birth_tracks;
}

void DynamicBirthModel::reset() noexcept
{
  LOG_DEB("Reset Dynamic Birth Model");
  _oldMeasurements.clear();
}

}  // namespace ttb
