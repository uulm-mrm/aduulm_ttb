#include "tracking_lib/TTTFilters/TTTUncorrelatedTracks.h"
#include "tracking_lib/States/Innovation.h"
#include <boost/math/distributions/chi_squared.hpp>
#include "tracking_lib/Trackers/BaseTracker.h"
#include "tracking_lib/Trackers/LMB_FPM_Tracker.h"
#include <tracy/tracy/Tracy.hpp>

namespace ttb::uncorrelated_t2t_fusion
{

TTTUncorrelatedTracks::TTTUncorrelatedTracks(TTBManager* manager,
                                             TTTUncorrelatedTracksParams params,
                                             std::vector<tttfusion::TracksSensor>&& lmbs)
  : _manager(manager), _t2t_params{ std::move(params) }, _tracksVec(std::move(lmbs)), _numTrackers(_tracksVec.size())
{
}

Innovation TTTUncorrelatedTracks::calculateInnovationT2T(MeasurementContainer const& measContainer,
                                                         State const& dist,
                                                         bool gatingActivated) const
{
  LOG_DEB("Calculate Innovation for Gaussian Measurement Model for " << measContainer._data.size()
                                                                     << " Pseudo Measurements");

  ZoneScopedN("TTTUncorrelatedTracks::calculateInnovationT2T");
  ZoneText("StateDistribution", 17);
  ZoneValue(dist._id.value_);
  ZoneText("#Measurements", 13);
  ZoneValue(measContainer._data.size());
  LOG_DEB("Calculate Innovation for Gaussian Measurement Model for " << measContainer._data.size() << " Measurements");
  LOG_DEB("State Distribution:\n" << dist.toString());
  if (not dist.isValid())
  {
    LOG_FATAL("invalid dist" << dist.toString());
    throw std::runtime_error("invalid dist");
  }
  assert([&] {
    if (not dist.isValid())
    {
      LOG_FATAL(dist.toString());
      return false;
    }
    return true;
  }());

  if (dist._state_dist.size() > 1)
  {
    LOG_ERR("FPM Birth track fusion currently not tested for multi model state distributions!");
  }
  const auto& measModelMap = _manager->getMeasModelMap();
  const auto& stateModelMap = _manager->getStateModelMap();

  LOG_DEB("MEasContainer: " << measContainer.toString());
  GaussianMeasurementModel const& measModel = dynamic_cast<GaussianMeasurementModel&>(
      *measModelMap.at(measContainer._id));  // Todo(hermann): How can I check if this cast was successful?!
  GatingDistanceCache gatingDistCache;

  double const detection_prob{ [&]() -> double {
    double pD{ 0 };
    for (const auto& [model_id, model_dist] : dist._state_dist)
    {
      BaseStateModel const& stateModel = *stateModelMap.at(model_id);
      Components measComps = transformation::transformableComps(
                                 stateModel.state_comps(),
                                 transformation::TransformOptions{
                                     _manager->state_model_params(model_id).assume_orient_in_velocity_direction })
                                 .all.intersection(measModel._model_comps);

      // iterate over mixture of gaussian distribution
      for (const auto& dist_comp : model_dist->dists())
      {
        //        profilerData._numMixtureComps++;
        if (dist_comp->type() != DISTRIBUTION_TYPE::GAUSSIAN)
        {
          LOG_ERR("Compute Innovation for state which is a mixture of NON-GAUSSIAN components - sth. may be wrong "
                  << "you have been warned");
        }
        assert(dist_comp->refPoint() == REFERENCE_POINT::CENTER);
        Vector const x = dist_comp->mean();
        //        PseudoPredictedMeasurement predMeas;
        Vector const& z_pred{ [&] {
          std::optional<Vector> predictedMeas = transformation::transform(
              x, stateModel.state_comps(), measComps, measContainer._sensorInfo._to_sensor_cs);
          if (predictedMeas.has_value())
          {
            return predictedMeas.value();
          }
          LOG_FATAL("state: " << dist.toString());
          LOG_FATAL("Measurement Space: " << measComps.toString());
          throw std::runtime_error("could not transform the state to the measurement space");
        }() };
        //        predMeas.z_pred = std::move(z_pred);
        // Todo(hermann): How to handle the case of a two step initialization at the sensors?!
        double pD_component =
            measModel.getDetectionProbability(z_pred, measComps, measContainer._sensorInfo._sensor_fov);
        //        predMeas.detection_prob = pD_component;
        pD += dist_comp->sumWeights() * pD_component;  // if fov is empty, detection_prob given in config is returned
        //      predMeasCache.emplace(std::make_tuple(model_id,dist_comp->id()),std::move(predMeas));
        // contains on element? Check! If yes then this calculation is not necessary
        // calculate gating Distance if not already happened
        if (not gatingDistCache.contains(model_id))
        {
          // the gating dist is not in the cache
          double distance;
          try
          {
            distance =
                boost::math::quantile(boost::math::chi_squared_distribution<>(stateModel.state_comps()._comps.size()),
                                      _t2t_params.gating_prob);
          }
          catch (std::overflow_error const& e)
          {
            LOG_ERR(" gating distance of uncorrelated t2t fusion is set to infinity!");
            distance = std::numeric_limits<double>::infinity();
          }
          gatingDistCache.emplace(model_id, distance);
        }
      }
    }
    return pD;
  }() };
  assert(detection_prob <= 1. && "pD > 1");
  assert(detection_prob >= 0. && "pD < 0");
  Innovation innovation{ ._detectionProbability = detection_prob };

  // LOG_DEB("Setted Up PredMeasMap");
  // std::map<std::vector<COMPONENT>, PredMeasMap> featurePredMeasMap;       // measured_features to PredMeasMap
  //  MeasContainer not empty
  //  Iterate over measurements

  LOG_DEB("Iterate over all Measurements");

  //  double weightSum = 0.0;
  for (Measurement const& meas : measContainer._data)
  {
    LOG_DEB("Calculate Innovation for Pseudo Measurement:\n" + meas.toString());
    innovation._updates.emplace(meas._id, Innovation::Update{ .updated_dist = _manager->createState() });
    innovation._updates.at(meas._id).updated_dist._label = dist._label;
    innovation._updates.at(meas._id).updated_dist._time = meas._time;
    innovation._updates.at(meas._id).updated_dist._meta_data._numPredictions = dist._meta_data._numPredictions;
    innovation._updates.at(meas._id).updated_dist._meta_data._numUpdates = dist._meta_data._numUpdates + 1;
    innovation._updates.at(meas._id).updated_dist._meta_data._timeOfLastAssociation = meas._time;
    innovation._updates.at(meas._id).updated_dist._meta_data._durationSinceLastAssociation = Duration::zero();
    innovation._updates.at(meas._id).updated_dist._meta_data._lastAssociatedMeasurement = meas._id;
    innovation._updates.at(meas._id).updated_dist._misc = dist._misc;
    innovation._updates.at(meas._id).updated_dist._score = dist._score;

    double measWeightSum = 0.;
    // Todo(hermann): Do I need to consider the measurement components for the calculation of clutter? But then: How to
    // save the informations about the real measurements in the track Todo(hermann): How is the clutterIntensity
    // calculated in a two step initialization of the sensors? For 1 step it is the clutter intensity of the sensor...
    double const kappa = measModel.getClutterIntensity(meas, measContainer._sensorInfo);

    innovation._updates.at(meas._id).updated_dist._classification =
        classification::StateClassification(dist._classification);

    innovation._updates.at(meas._id).clutter_intensity = kappa;

    std::vector<std::unique_ptr<BaseDistribution>> updatedComponents;
    double mixtureWeightSum = 0.;

    assert(meas._dist->refPoint() == REFERENCE_POINT::CENTER);  // Ensure RP_CENTER

    // iterate over dist
    for (const auto& [model_id, model_mixture_dist] : dist._state_dist)
    {
      const auto& stateModel = stateModelMap.at(model_id);
      if (stateModel->state_comps()._comps != meas._meas_comps._comps)
      {
        LOG_WARN("components of tracks are not equal! -> Measurement is ignored!!!");
        continue;
      }
      bool mixtureIsGated = true;
      // iterate over all components
      //      LOG_DEB("Iterate over all Mixture Components");
      for (const auto& component : model_mixture_dist->dists())
      {
        assert(component->refPoint() == REFERENCE_POINT::CENTER);  // Ensure RP_CENTER
                                                                   //          profilerData._numMixtureComps++;
                                                                   // get predicted measurement
        //            if (not predMeasCache.contains({ model_id, component->id() }))  // the prediction is not in the
        //            cache
        //            {
        //              LOG_FATAL("predicted Measurement is not in the cache!... Should not happen -> Bug!");
        //              throw std::runtime_error("predicted Measurement is not in the cache!... Should not happen ->
        //              Bug!");
        //            }
        //            PseudoPredictedMeasurement const& pred_meas = predMeasCache.at({ model_id, component->id() });
        if (not gatingDistCache.contains(model_id))  // the prediction is not in the cache
        {
          LOG_FATAL("gating distance is not in the cache!... Should not happen -> Bug!");
          throw std::runtime_error("gating distance is not in the cache!... Should not happen -> Bug!");
        }
        double const gatingDistance = gatingDistCache.at(model_id);
        TTTUpdateInfo info{ [&] {
          return calcUpdateInfoT2T(
              meas, component, gatingDistance, stateModel->state_comps().indexOf(COMPONENT::ROT_Z));
        }() };
        if (info.isGated && gatingActivated)
        {
          LOG_DEB("Component " << component->toString() << " is GATED!");
          continue;
        }
        LOG_DEB("Component " << component->toString() << " is associated");
        LOG_DEB("with Measurement " << meas.toString());
        mixtureIsGated = false;
        double const likelihood = getMeasurementLikelihood(info.distance,
                                                           _t2t_params.gating_prob,
                                                           info.T);  // /
                                                                     // observRefPoints.size();
                                                                     // // every refPoint is
                                                                     // equally likely
        LOG_DEB("Distance: " << info.distance);
        LOG_DEB("Git Likelihood: " + std::to_string(likelihood));
        // avgCompLikelihood += likelihood;
        const Vector x1 = component->mean();
        const Matrix P1 = component->covariance();
        const Vector x2 = meas._dist->mean();
        const Matrix P2 = meas._dist->covariance();
        // Track to track fusion equations for uncorrelated tracks (Bar-Shalom: A Handbook of Algorithms(yellow book),
        // 2011, pages 580-583
        Vector upd_x = P2 * info.iT * x1 + P1 * info.iT * x2;
        Matrix upd_P = P1 * info.iT * P2;
        LOG_DEB("upd_x: " << upd_x);
        LOG_DEB("upd_P: " << upd_P);

        // normalize orientation
        auto orient_ind = stateModel->state_comps().indexOf(COMPONENT::ROT_Z);
        if (orient_ind.has_value())
        {
          angles::normalizeAngle(upd_x(orient_ind.value()));
        }

        double newWeight = component->sumWeights() * likelihood;

        if (not std::isfinite(newWeight))
        {
          LOG_FATAL("obtained infinite weight for update ");
          LOG_FATAL("Component: " << component->toString());
          LOG_FATAL("Distance: " << info.distance);
          LOG_FATAL("likelihood: " << likelihood);
          LOG_FATAL("residuum: " << info.residuum);
          LOG_FATAL("T: " << info.T);
          LOG_FATAL("iT: " << info.iT);
          LOG_FATAL("upd_x: " << upd_x);
          LOG_FATAL("upd_P: " << upd_P);
          LOG_FATAL("Discarded comp infinite weight");
          throw std::runtime_error("infinite component weight");
        }
        mixtureWeightSum += newWeight;
        updatedComponents.emplace_back(std::make_unique<GaussianDistribution>(
            std::move(upd_x), std::move(upd_P), newWeight, component->refPoint()));
        updatedComponents.back()->setPriorId(component->id());

        //            updatedComponents_refPoints.emplace_back(
        //                std::make_unique<GaussianDistribution>(std::move(x), std::move(P), newWeight,
        //                component->refPoint()));
        //            LOG_DEB("component weight " << component->sumWeights());
        //            LOG_DEB("pD += " << component->sumWeights() * pred_meas.detection_prob);
        //            updatedComponents_refPoints.back()->_misc = "likelihood: " + std::to_string(likelihood);
        //            updatedComponents_refPoints.back()->setPriorId(component->id()); //keep Component id of predicted
        //            component (needed for fpm and pm lmb fusion)
      }
      if (not mixtureIsGated)
      {
        measWeightSum += mixtureWeightSum;
        innovation._updates.at(meas._id).updated_dist._state_dist.at(model_id)->merge(std::move(updatedComponents));
      }
    }
    if (measWeightSum > 0)
    {
      innovation._updates.at(meas._id).updated_dist._classification.update(meas._classification);
      double logLikelihood = std::log(measWeightSum) - std::log(kappa);
      if (not std::isfinite(logLikelihood))
      {
        LOG_FATAL("obtained infinite neglogcost=" << logLikelihood << "\nWeightSum=" << log(measWeightSum)
                                                  << "\nKappa:" << kappa << " for update");
        assert(false);
      }
      LOG_DEB("updatedDistWeights: " << innovation._updates.at(meas._id).updated_dist.sumWeights());
      double const weight = innovation._updates.at(meas._id).updated_dist.sumWeights();
      innovation._updates.at(meas._id).log_likelihood = std::log(weight);
      innovation._updates.at(meas._id).updated_dist._score = innovation._updates.at(meas._id).updated_dist._score +
                                                             std::log(detection_prob) + std::log(weight) -
                                                             std::log(kappa);
      innovation._updates.at(meas._id).updated_dist.multiplyWeights(1 / weight);  // normalize dist
      assert([&] {
        if (not innovation._updates.at(meas._id).updated_dist.isValid())
        {
          LOG_FATAL("Update of state: " + dist.toString() + " with Measurement " + meas.toString() + " is INVALID");
          LOG_FATAL("Update: " + innovation._updates.at(meas._id).updated_dist.toString());
          return false;
        }
        return true;
      }());
      LOG_DEB("updated Dists weight: " << weight);
    }
  }  // Iterate over all Measurements
  /// add original distribution to model a misdetection
  State misdetection = dist;
  misdetection._meta_data._lastAssociatedMeasurement = NOT_DETECTED;
  misdetection._id = State::_idGenerator.getID();
  misdetection._time = measContainer._time;
  misdetection._score = misdetection._score + std::log(1 - detection_prob);
  for (auto& [model, inno] : misdetection._innovation)  // we do not want to copy the innovations of the state
  {
    inno._updates.clear();
  }
  innovation._updates.emplace(NOT_DETECTED,
                              Innovation::Update{
                                  .updated_dist = std::move(misdetection),
                              });

  /// remove Updates where all components have been gated
  std::map<MeasurementId, Innovation::Update> filteredUpdates;
  for (auto& [key, val] : innovation._updates)
  {
    if (not val.updated_dist.isEmpty())  // TODO: Adapt for multi model case
    {
      filteredUpdates.emplace(key, std::move(val));
    }
  }
  innovation._updates = std::move(filteredUpdates);

  LOG_DEB("Innovation done");
  return innovation;
}

LMBDistribution TTTUncorrelatedTracks::calcInnovationsTracks(LMBDistribution&& lmbdist,
                                                             MeasurementContainer const& Z,
                                                             bool gatingActivated,
                                                             bool deactivate_parallelization) const
{
  LOG_DEB("Innovation LMB Distribution with " + std::to_string(lmbdist._tracks.size()) + " Tracks and " +
          std::to_string(Z._data.size()) + " Pseudo Measurements (Birth Tracks of other Sensors)");
  if (_manager->params().thread_pool_size > 0 && !deactivate_parallelization)
  {
    _manager->thread_pool().detach_loop(std::size_t{ 0 }, lmbdist._tracks.size(), [&](std::size_t i) {
      lmbdist._tracks.at(i)._innovation.at(Z._id) = calculateInnovationT2T(Z, lmbdist._tracks.at(i), gatingActivated);
    });
    _manager->thread_pool().wait();
  }
  else
  {
    std::for_each(std::execution::seq, lmbdist._tracks.begin(), lmbdist._tracks.end(), [&](State& track) {
      track._innovation.at(Z._id) = calculateInnovationT2T(Z, track, gatingActivated);
    });
  }
  return lmbdist;
  //  LOG_DEB("After calcInnovation:\n" + toString());
}

tttfusion::TracksSensor TTTUncorrelatedTracks::addPseudoBirthTracks(MeasurementContainer const& measurementContainer,
                                                                    std::map<MeasurementId, double> const& rzMap,
                                                                    std::size_t num_empty_updates,
                                                                    bool gatingActivated,
                                                                    bool deactivate_parallelization) const
{
  std::map<StateId, State> birthTracks;
  tttfusion::TracksSensor bt;
  for (const auto& [measID, associationProb] : rzMap)
  {
    if (associationProb < _t2t_params.pseudo_birth_threshold)  // set up new state
    {
      auto measIt = std::find_if(measurementContainer._data.begin(),
                                 measurementContainer._data.end(),
                                 [&](Measurement const& m) { return m._id == measID; });
      if (measIt == measurementContainer._data.end())
      {
        LOG_WARN("Measurment id " << measID << " is not in measurement container!");
        continue;
      }
      State track = _manager->createState();
      std::vector<std::unique_ptr<BaseDistribution>> components;
      components.emplace_back(std::make_unique<GaussianDistribution>(std::move(measIt->_dist->mean()),
                                                                     std::move(measIt->_dist->covariance())));
      track._state_dist.at(_measId2stateModelId.at(measID))->merge(std::move(components));
      LMBDistribution tmpLMB(_manager);
      track._time = measIt->_time;
      track._meta_data._timeOfLastAssociation = measIt->_time;
      track._existenceProbability = _t2t_params.pseudo_birth_prob;
      tmpLMB._tracks.emplace_back(std::move(track));
      tmpLMB = doEmptyLMBUpdate(
          std::move(tmpLMB), num_empty_updates, measurementContainer._id, gatingActivated, deactivate_parallelization);
      for (const auto& trackLMB : tmpLMB._tracks)
      {
        birthTracks.emplace(trackLMB._id, std::move(trackLMB));
      }
    }
  }
  return { ._id = measurementContainer._id,
           ._trackMapSensor = std::move(birthTracks),
           ._pD = 0,
           ._sensorInfo = std::move(measurementContainer._sensorInfo) };
}

LMBDistribution TTTUncorrelatedTracks::doEmptyLMBUpdate(LMBDistribution&& dist,
                                                        std::size_t num_empty_updates,
                                                        MeasModelId id,
                                                        bool gatingActivated,
                                                        bool deactivate_parallelization) const
{
  // calculate empty updates -> each track must have the same number of total updates!
  for (std::size_t u = 0; u < num_empty_updates; u++)
  {
    MeasurementContainer emptyContainer;
    emptyContainer._id = id;
    dist = calcInnovationsTracks(std::move(dist), emptyContainer, gatingActivated, deactivate_parallelization);
    dist.update(emptyContainer);
  }
  return std::move(dist);
}

void TTTUncorrelatedTracks::overwriteCovariances(LMBDistribution& lmbDist) const
{
  // overwrite birth covariance!
  if (_t2t_params.overwrite_var)
  {
    if (_manager->params().thread_pool_size > 0)
    {
      _manager->thread_pool().detach_loop(std::size_t{ 0 }, lmbDist._tracks.size(), [&](std::size_t i) {
        for (auto& [stateModelId, stateDist] : lmbDist._tracks.at(i)._state_dist)
        {
          BaseStateModel const& sm = *_manager->getStateModelMap().at(stateModelId);

          for (auto* dist : stateDist->dists())
          {
            Matrix cov = dist->covariance();
            for (const auto& [str, val] : _t2t_params.var)
            {
              COMPONENT comp = ttb::to_COMPONENT(str).value();
              if (std::find(sm.state_comps()._comps.begin(), sm.state_comps()._comps.end(), comp) !=
                  sm.state_comps()._comps.end())
              {
                auto ind = sm.state_comps().indexOf(comp).value();
                cov(ind, ind) = val;
              }
            }
            dist->set(cov);
          }
        }
      });
      _manager->thread_pool().wait();
    }
    else
    {
      for (auto& track : lmbDist._tracks)
      {
        for (auto& [stateModelId, stateDist] : track._state_dist)
        {
          BaseStateModel const& sm = *_manager->getStateModelMap().at(stateModelId);

          for (auto* dist : stateDist->dists())
          {
            Matrix cov = dist->covariance();
            for (const auto& [str, val] : _t2t_params.var)
            {
              COMPONENT comp = ttb::to_COMPONENT(str).value();
              if (std::find(sm.state_comps()._comps.begin(), sm.state_comps()._comps.end(), comp) !=
                  sm.state_comps()._comps.end())
              {
                auto ind = sm.state_comps().indexOf(comp).value();
                cov(ind, ind) = val;
              }
            }
            dist->set(cov);
          }
        }
      }
    }
  }
}

LMBDistribution TTTUncorrelatedTracks::fpm_version(LMBDistribution lmbDist,
                                                   std::vector<MeasurementContainer>&& measContainerList)
{
  LOG_DEB("start cycle");
  Time time = std::chrono::high_resolution_clock::now();
  std::size_t numUpdatesNeeded = _numTrackers - 1;
  const auto& stateModelMap = _manager->getStateModelMap();
  if (measContainerList.empty())
  {
    LOG_ERR("It is possible to arrive here?!?");
    return lmbDist;
  }

  std::vector<MeasModelId> measModelIds;
  for (const auto& measurementContainer : measContainerList)
  {
    measModelIds.push_back(measurementContainer._id);
  }

  // calculate single sensor updates for each sensor independently
  LMBDistribution fused_birth_dist(_manager);
  for (std::size_t u = 0; u < numUpdatesNeeded; u++)
  {
    auto [fusedDist, birthLMBs] = fpmUpdateAndFusion(std::move(lmbDist), std::move(measContainerList));
    LOG_DEB("fusedDist: " << fusedDist.toString());
    fused_birth_dist.merge(std::move(fusedDist));
    // create updated measurementContainer
    if (u == numUpdatesNeeded - 1)
    {
      // it is not necessary to fill the measurement containers since the fusion is done!
      std::size_t countNonEmptyDists = 0;
      for (const auto& birthLMB : birthLMBs)
      {
        if (!birthLMB._trackMapSensor.empty())
        {
          countNonEmptyDists++;
          for (const auto& track : birthLMB._trackMapSensor)
          {
            fused_birth_dist._tracks.emplace_back(std::move(track.second));
          }
        }
      }
      // algorithm is finished!
      if (countNonEmptyDists > 1)
      {
        LOG_FATAL("There should only be one birthLMB at last iteration of for loop!");
        throw std::runtime_error("More than one sensor delivers birthTracks....");
      }
      break;
    }
    measContainerList.clear();
    lmbDist._tracks.clear();
    bool hasBirthTracks = false;
    std::size_t birthCounter = 0;
    for (const auto& birthLMB : birthLMBs)
    {
      LOG_DEB("#BirthTracks " << birthLMB._trackMapSensor.size() << " for meas model id " << birthLMB._id);
      if (!birthLMB._trackMapSensor.empty())
      {
        hasBirthTracks = true;
        for (const auto& track : birthLMB._trackMapSensor)
        {
          lmbDist._tracks.emplace_back(std::move(track.second));
        }
        break;
      }
      birthCounter++;
    }
    if (!hasBirthTracks)
    {
      // nothing to do anymore!
      break;
    }

    LOG_DEB("print prior density! " << lmbDist.toString());
    birthLMBs.erase(birthLMBs.begin() + birthCounter);
    // create measurement containers
    LOG_DEB("Create Measurement container");
    std::vector<MeasModelId> measModelIds_copy = measModelIds;
    for (auto& birth_tracks : birthLMBs)
    {
      LOG_DEB("next birth tracks as measurements: #birthTracks: " << birth_tracks._trackMapSensor.size()
                                                                  << " birth_tracks id: " << birth_tracks._id);
      MeasurementContainer measContainer;
      measContainer._id = birth_tracks._id;
      measContainer._sensorInfo = std::move(birth_tracks._sensorInfo);
      auto it = std::find(measModelIds_copy.begin(), measModelIds_copy.end(), birth_tracks._id);
      if (it == measModelIds.end())
      {
        LOG_FATAL("BUG ALERT!!!");
        throw std::runtime_error("BUG ALERT!!!");
      }
      LOG_DEB("delete meas model id: " << it->value_ << " from following vector: ");
      measModelIds_copy.erase(it);
      // add tracks as measurements
      for (auto& [track_id, track] : birth_tracks._trackMapSensor)
      {
        if (track._state_dist.size() > 1)
        {
          LOG_WARN("measurements with multi-model informations are not supported! Only informations of first state "
                   "model "
                   "are used! You are warned!");
          //        throw std::runtime_error("measurements with multi-model informations are not supported!");
        }
        StateModelId smId = stateModelMap.at(track._state_dist.begin()->first)->id();
        for (auto& [model_id, mixture] : track._state_dist)
        {
          const std::vector<BaseDistribution*>& mixtureComponents = mixture->dists();

          if (mixtureComponents.size() > 1)
          {
            LOG_WARN("Track contains more mixture distribution and not gaussian distribution. means are covs are "
                     "combined!");
          }
          for (const auto& prior_component : mixtureComponents)
          {
            if (prior_component->refPoint() != REFERENCE_POINT::CENTER)
            {
              LOG_FATAL("dist of track does not have ref point center!!");
              throw std::runtime_error(" dist of track does not have ref point center!!");
            }
            assert(prior_component->refPoint() == REFERENCE_POINT::CENTER && " dist of track does not have ref point "
                                                                             "center!!");
          }
          break;
        }
        LOG_DEB("add measurement with");
        Vector mean = (*track._state_dist.begin()->second).mean();
        Matrix cov = (*track._state_dist.begin()->second).covariance();
        LOG_DEB("mean: " << mean);
        LOG_DEB("cov: " << cov);
        auto meas_dist =
            std::make_unique<GaussianDistribution>(std::move((*track._state_dist.begin()->second).mean()),
                                                   std::move((*track._state_dist.begin()->second).covariance()),
                                                   1,
                                                   REFERENCE_POINT::CENTER);
        Measurement meas{ std::move(meas_dist), time, stateModelMap.at(smId)->state_comps() };
        _measId2stateModelId.emplace(meas._id, smId);
        measContainer._data.push_back(std::move(meas));
      }
      measContainerList.push_back(std::move(measContainer));
    }
    // add empty measurement containers for the update
    for (const auto& id : measModelIds_copy)
    {
      MeasurementContainer emptyContainer;
      emptyContainer._id = id;
      measContainerList.push_back(std::move(emptyContainer));
    }
  }
  LOG_DEB("final fused birth lmb dist: " << fused_birth_dist.toString());
  // overwrite birth covariance!
  overwriteCovariances(fused_birth_dist);
  return fused_birth_dist;
}

std::pair<LMBDistribution, std::vector<tttfusion::TracksSensor>>
TTTUncorrelatedTracks::fpmUpdateAndFusion(LMBDistribution&& lmbdist,
                                          std::vector<MeasurementContainer>&& measContainerList,
                                          bool gatingActivated,
                                          bool deactivate_parallelization) const
{
  std::map<StateModelId, double> distID2mergeDist;  // save values of yaml file for later
  for (auto const& [state_model_id, state_model] : _manager->getStateModelMap())
  {
    distID2mergeDist.emplace(state_model_id,
                             _manager->state_model_params(state_model_id).distribution.post_process.merging_distance);
  }

  LOG_DEB("start cycle with " << measContainerList.size() << " pseudo measurements containers!");
  for (const auto& measurementContainer : measContainerList)
  {
    LOG_DEB("calc inno for meas id " << measurementContainer._id);
    lmbdist =
        calcInnovationsTracks(std::move(lmbdist), measurementContainer, gatingActivated, deactivate_parallelization);
    LOG_DEB("after calc inno: " << lmbdist.toString());
  }
  LOG_DEB("after innovation: " << lmbdist.toString());

  // deactivate merging of state distributions for single sensor updates
  for (auto const& [state_model_id, state_model] : _manager->getStateModelMap())
  {
    _manager->state_model_next_params(state_model_id).distribution.post_process.merging_distance = 0;
  }
  _manager->update_params();

  // calculate single sensor updates for each sensor independently
  std::vector<LMBDistribution> singleSensorLMBUpdates;
  std::vector<tttfusion::TracksSensor> birthLMBs;
  std::mutex updated_single_sensors_lmbs_mutex;

  if (_manager->params().thread_pool_size > 0 && !deactivate_parallelization)
  {
    _manager->next_params().lmb_distribution.calculate_single_sensor_group_updates_parallel = false;
    _manager->update_params();
    _manager->thread_pool().detach_loop(std::size_t{ 0 }, measContainerList.size(), [&](std::size_t i) {
      LOG_DEB("Start single sensor updates!");
      LOG_DEB("Processing Msg. of sensor " + measContainerList.at(i)._id.value_
              << " with " << measContainerList.at(i)._data.size() << " Detections");

      LMBDistribution prior_copy = lmbdist;
      prior_copy.update(measContainerList.at(i));
      if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
      {
        tttfusion::TracksSensor birth = addPseudoBirthTracks(
            measContainerList.at(i), prior_copy.meas_assignment_prob(), 0, gatingActivated, deactivate_parallelization);
        {
          std::unique_lock lock(updated_single_sensors_lmbs_mutex);
          birthLMBs.push_back(std::move(birth));
        }
      }
      //    LOG_DEB("Print LMB Density after postProcessUpdate: " << updatedLMBDist.toString());
      assert(prior_copy.isValid() && "end of measurement update");
      LOG_DEB("Nr. Tracks: " << prior_copy._tracks.size());
      {
        std::unique_lock lock(updated_single_sensors_lmbs_mutex);
        singleSensorLMBUpdates.push_back(std::move(prior_copy));
      }
    });
    _manager->thread_pool().wait();
    _manager->next_params().lmb_distribution.calculate_single_sensor_group_updates_parallel = true;
    _manager->update_params();
    // Do postprocessing, where the processing per track is parallelized!
    for (auto& tmp : singleSensorLMBUpdates)
    {
      if (_manager->params().thread_pool_size > 0 && !deactivate_parallelization)
      {
        LOG_ERR("calc state post processing parallel");
        _manager->thread_pool().detach_loop(
            std::size_t{ 0 }, tmp._tracks.size(), [&](std::size_t i) { tmp._tracks.at(i).postProcess(); });
        _manager->thread_pool().wait();
      }
      else
      {
        LOG_ERR("Do not calc state post processing parallel");
        for (State& state : tmp._tracks)
        {
          state.postProcess();
        }
      }
    }
  }
  else
  {
    //    LOG_ERR("No parallelization of single sensor updates in fpm_lmb stage");
    for (auto const& measurementContainer : measContainerList)
    {
      LOG_DEB("Start single sensor updates!");
      LMBDistribution prior_copy = lmbdist;
      prior_copy.update(measurementContainer);

      LOG_DEB("Add Pseudo tracks");
      tttfusion::TracksSensor birth = addPseudoBirthTracks(measurementContainer, prior_copy.meas_assignment_prob(), 0);
      birthLMBs.push_back(std::move(birth));
      assert(prior_copy.isValid() && "end of measurement update");
      LOG_DEB("Nr. Tracks: " << prior_copy._tracks.size());
      singleSensorLMBUpdates.push_back(std::move(prior_copy));
    }
  }
  // activate merging of state distributions for fusion
  for (auto const& [state_model_id, state_model] : _manager->getStateModelMap())
  {
    _manager->state_model_next_params(state_model_id).distribution.post_process.merging_distance =
        distID2mergeDist.at(state_model_id);
  }
  bool calculate_fpm_fusion_tracks_parallel_memory =
      _manager->params().filter.lmb_fpm.calculate_fpm_fusion_tracks_parallel;
  if (deactivate_parallelization)
  {
    _manager->next_params().filter.lmb_fpm.calculate_fpm_fusion_tracks_parallel = false;
  }
  _manager->update_params();
  // FPM-LMB fusion
  lmbdist.fpm_fusion(std::move(singleSensorLMBUpdates), true);
  if (deactivate_parallelization)
  {
    _manager->next_params().filter.lmb_fpm.calculate_fpm_fusion_tracks_parallel =
        calculate_fpm_fusion_tracks_parallel_memory;
  }
  _manager->update_params();
  return { std::move(lmbdist), std::move(birthLMBs) };
}

LMBDistribution TTTUncorrelatedTracks::ic_version(LMBDistribution lmbDist,
                                                  std::vector<MeasurementContainer>&& measContainerList,
                                                  bool gatingActivated,
                                                  bool deactivate_parallelization)
{
  LOG_DEB("start cycle");
  std::size_t numUpdatesDone = 0;
  for (const auto& measurementContainer : measContainerList)
  {
    LOG_ERR("calc inno for sensor: " << measurementContainer._id);
    lmbDist =
        calcInnovationsTracks(std::move(lmbDist), measurementContainer, gatingActivated, deactivate_parallelization);
    LOG_DEB("after calc inno: " << lmbDist.toString());

    lmbDist.update(measurementContainer);
    // Add measurements which are not used as tracks to the density
    lmbDist.postProcessUpdate();
    LOG_DEB("Add Pseudo tracks");
    numUpdatesDone++;
    tttfusion::TracksSensor birthTracks = addPseudoBirthTracks(measurementContainer,
                                                               lmbDist.meas_assignment_prob(),
                                                               numUpdatesDone,
                                                               gatingActivated,
                                                               deactivate_parallelization);
    for (const auto& track : birthTracks._trackMapSensor)
    {
      lmbDist._tracks.emplace_back(std::move(track.second));
    }
  }
  LOG_DEB("Succesfull finished fusion!");
  // overwrite birth covariance!
  overwriteCovariances(lmbDist);
  return lmbDist;
}

LMBDistribution TTTUncorrelatedTracks::fusion_without_t2tAssociation_before()
{
  Time time = std::chrono::high_resolution_clock::now();
  const auto& stateModelMap = _manager->getStateModelMap();
  LMBDistribution lmbDist(_manager);
  // initialize fused birth lmb density with birth tracks of first sensor!
  // Todo(hermann): Check if the result of the fusion depends on which sensor is used as prior density
  LOG_DEB("start fusion without t2tAssociation before for " << _tracksVec.size() << " birth densities");
  std::size_t idOfPriorTracks = 0;
  bool birthTracksGiven = false;
  for (const auto& tracks : _tracksVec)
  {
    if (!tracks._trackMapSensor.empty())
    {
      birthTracksGiven = true;
      for (const auto& [_, track] : tracks._trackMapSensor)
      {
        lmbDist._tracks.emplace_back(std::move(track));
      }
      break;
    }
    idOfPriorTracks++;
  }
  if (!birthTracksGiven)
  {
    return lmbDist;
  }

  LOG_DEB("print prior density! " << lmbDist.toString());
  _tracksVec.erase(_tracksVec.begin() + idOfPriorTracks);
  // create measurement containers
  LOG_DEB("Create Measurement container");
  std::vector<MeasurementContainer> measContainerList;
  for (auto& birth_tracks : _tracksVec)  // todo(hermann): parallelize?!
  {
    LOG_DEB("next birth tracks as measurements - number of birth tracks: " << birth_tracks._trackMapSensor.size());
    MeasurementContainer measContainer;
    measContainer._id = birth_tracks._id;
    measContainer._sensorInfo = std::move(birth_tracks._sensorInfo);
    // add tracks as measurements
    for (auto& [track_id, track] : birth_tracks._trackMapSensor)
    {
      LOG_DEB("Track with label: " << track._label << " and id: " << track._id);
      if (track._state_dist.size() > 1)
      {
        LOG_WARN("measurements with multi-model informations are not supported! Only informations of first state model "
                 "are used! You are warned!");
        //        throw std::runtime_error("measurements with multi-model informations are not supported!");
      }
      StateModelId smId = stateModelMap.at(track._state_dist.begin()->first)->id();
      for (auto& [model_id, mixture] : track._state_dist)
      {
        const std::vector<BaseDistribution*>& mixtureComponents = mixture->dists();

        if (mixtureComponents.size() > 1)
        {
          LOG_WARN("Track contains more mixture distribution and not gaussian distribution. means are covs are "
                   "combined!");
        }
        for (const auto& prior_component : mixtureComponents)
        {
          if (prior_component->refPoint() != REFERENCE_POINT::CENTER)
          {
            LOG_FATAL("dist of track does not have ref point center!!");
            throw std::runtime_error(" dist of track does not have ref point center!!");
          }
          assert(prior_component->refPoint() == REFERENCE_POINT::CENTER && " dist of track does not have ref point "
                                                                           "center!!");
        }
        break;
      }
      LOG_DEB("add measurement with");
      Vector mean = (*track._state_dist.begin()->second).mean();
      Matrix cov = (*track._state_dist.begin()->second).covariance();
      LOG_DEB("mean: " << mean);
      LOG_DEB("cov: " << cov);
      auto meas_dist =
          std::make_unique<GaussianDistribution>(std::move((*track._state_dist.begin()->second).mean()),
                                                 std::move((*track._state_dist.begin()->second).covariance()),
                                                 1,
                                                 REFERENCE_POINT::CENTER);
      Measurement meas{ std::move(meas_dist), time, stateModelMap.at(smId)->state_comps() };
      _measId2stateModelId.emplace(meas._id, smId);
      measContainer._data.push_back(std::move(meas));
    }
    measContainerList.push_back(std::move(measContainer));
  }

  // calculate innovations to birth tracks of other sensors (other sensors are treated as measurements)
  switch (_t2t_params.strategy)
  {
    case TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::FPM_LMB_VERSION_WITHOUT_ASSOCIATION_BEFORE:
      return fpm_version(std::move(lmbDist), std::move(measContainerList));
      break;
    case TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::IC_LMB_VERSION_WITHOUT_ASSOCIATION_BEFORE:
      return ic_version(std::move(lmbDist), std::move(measContainerList));
      break;
    default:
      LOG_FATAL("TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY is not known! Can't perform fusion!");
  }
  LOG_ERR("Should never reach this point. There is probably a bug in your config!");
  LMBDistribution fusedLMBDist(_manager);
  return fusedLMBDist;
}

LMBDistribution TTTUncorrelatedTracks::fusion_with_t2tAssociation_before()
{
  Time time = std::chrono::high_resolution_clock::now();
  std::size_t num_samples = 20;
  const auto& stateModelMap = _manager->getStateModelMap();
  StateModelId sm_id{ 0 };
  for (const auto& tracks : _tracksVec)
  {
    if (!tracks._trackMapSensor.empty())
    {
      sm_id = tracks._trackMapSensor.begin()->second._state_dist.begin()->first;
      break;
    }
  }
  tttfusion::Associations associations = t2ta_stochastic_optimization(std::move(_tracksVec), num_samples, sm_id);
  //  LOG_ERR("The following states are associated:");
  //  for(const auto& clusters : associations._clusters)
  //  {
  //    std::stringstream clStr;
  //    std::stringstream sensorsStr;
  //    for(const auto& state : clusters._cluster)
  //    {
  //      clStr << state << " ";
  //      sensorsStr << associations._trackID2sensor.at(state) << " ";
  //    }
  //    LOG_ERR("Cluster contains the following states: " );
  //    LOG_ERR(clStr.str() << "\n");
  //    LOG_ERR(sensorsStr.str() << "\n");
  //  }

  LMBDistribution fusedLMBDist(_manager);
  std::vector<MeasModelId> measModelIDs;
  for (const auto& sensor : _tracksVec)
  {
    measModelIDs.push_back(sensor._id);
  }

  bool use_grouping_memory = _manager->params().lmb_distribution.use_grouping;
  _manager->next_params().lmb_distribution.use_grouping =
      false;  // grouping is not needed since grouping is done in this method before
  _manager->update_params();
  std::mutex fused_tracks_mutex;
  if (_manager->params().thread_pool_size > 0)
  {
    LOG_ERR("Parallelization of groups! Cluster size: " << associations._clusters.size());
    _manager->next_params().lmb_distribution.calculate_single_sensor_group_updates_parallel = false;
    _manager->update_params();
    _manager->thread_pool().detach_loop(std::size_t{ 0 }, associations._clusters.size(), [&](std::size_t i) {
      LOG_WARN("######################CLUSTER " << i << "#############################");
      std::vector<MeasModelId> unusedMeasModels = measModelIDs;
      // first element is a priori track
      LMBDistribution lmbDist(_manager);
      std::vector<MeasurementContainer> measurementContainerList;
      if (!associations._clusters.at(i)._cluster.empty())
      {
        auto it = std::find(
            unusedMeasModels.begin(),
            unusedMeasModels.end(),
            _tracksVec.at(associations._trackID2sensor.at(associations._clusters.at(i)._cluster.front()))._id);
        if (it != unusedMeasModels.end())
        {
          unusedMeasModels.erase(it);
        }
        else
        {
          throw std::runtime_error("Measurement model id is more than one time in cluster. This is not allowed -> "
                                   "BUG!");
        }
        lmbDist._tracks.emplace_back(
            std::move(_tracksVec.at(associations._trackID2sensor.at(associations._clusters.at(i)._cluster.front()))
                          ._trackMapSensor.at(associations._clusters.at(i)._cluster.front())));
      }
      else
      {
        LOG_ERR("clusters are empty! Can this be possible?!");
      }
      // add the other cluster members as measurements
      std::vector<MeasurementContainer> measContainerList;
      for (auto memberIt = associations._clusters.at(i)._cluster.begin() + 1;
           memberIt < associations._clusters.at(i)._cluster.end();
           memberIt++)
      {
        MeasurementContainer measContainer;
        measContainer._id = _tracksVec.at(associations._trackID2sensor.at(*memberIt))._id;
        auto it = std::find(unusedMeasModels.begin(), unusedMeasModels.end(), measContainer._id);
        if (it != unusedMeasModels.end())
        {
          unusedMeasModels.erase(it);
        }
        else
        {
          throw std::runtime_error("Measurement model id is more than one time in cluster. This is not allowed -> "
                                   "BUG!");
        }
        measContainer._sensorInfo =
            _tracksVec.at(associations._trackID2sensor.at(*memberIt))._sensorInfo;  // copy is needed...
        auto meas_dist = std::make_unique<GaussianDistribution>(
            std::move(
                (std::move(_tracksVec.at(associations._trackID2sensor.at(*memberIt))._trackMapSensor.at(*memberIt))
                     ._state_dist.begin()
                     ->second)
                    ->mean()),
            std::move(
                (std::move(_tracksVec.at(associations._trackID2sensor.at(*memberIt))._trackMapSensor.at(*memberIt))
                     ._state_dist.begin()
                     ->second)
                    ->covariance()),
            1,
            REFERENCE_POINT::CENTER);
        Measurement meas{ std::move(meas_dist), time, stateModelMap.at(sm_id)->state_comps() };
        _measId2stateModelId.emplace(meas._id, sm_id);
        measContainer._data.push_back(std::move(meas));
        measContainerList.push_back(std::move(measContainer));
      }
      // add empty measurement containers to ensure #measContainers==numSensors-1!

      for (const auto& mmID : unusedMeasModels)
      {
        MeasurementContainer emptyContainer;
        emptyContainer._id = mmID;
        measContainerList.push_back(std::move(emptyContainer));
      }

      switch (_t2t_params.strategy)
      {
        case TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::FPM_LMB_VERSION_WITH_ASSOCIATION_BEFORE:
        {
          auto [dist, birthLMBs] = fpmUpdateAndFusion(std::move(lmbDist), std::move(measContainerList), false, true);
          // only for checks -> remove
          std::size_t counter = 0;
          for (const auto& birthLMB : birthLMBs)
          {
            if (!birthLMB._trackMapSensor.empty())
            {
              LOG_ERR("There are birth tracks for sensor " << counter);
              for (const auto& t : birthLMB._trackMapSensor)
              {
                LOG_ERR("birth tracks have stateID " << t.first);
              }
            }
            counter++;
          }
          if (dist._tracks.size() > 1)
          {
            LOG_ERR("There is more than one track in the density!!!");
            throw std::runtime_error("There is more than one track in the density!!!");
          }
          {
            std::unique_lock lock(fused_tracks_mutex);
            for (auto const& track : dist._tracks)
            {
              fusedLMBDist._tracks.emplace_back(std::move(track));
            }
          }
          break;
        }
        case TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::IC_LMB_VERSION_WITH_ASSOCIATION_BEFORE:
        {
          auto dist = ic_version(std::move(lmbDist), std::move(measContainerList), false, true);
          if (dist._tracks.size() > 1)
          {
            LOG_ERR("There is more than one track in the density!!!");
            throw std::runtime_error("There is more than one track in the density!!!");
          }
          std::unique_lock lock(fused_tracks_mutex);
          for (auto const& track : dist._tracks)
          {
            fusedLMBDist._tracks.emplace_back(std::move(track));
          }
          break;
        }
        default:
          LOG_FATAL("TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY is not known! Can't perform fusion!");
      }
      LOG_ERR("End cluster stuff");
    });
    _manager->thread_pool().wait();
    _manager->next_params().lmb_distribution.calculate_single_sensor_group_updates_parallel = true;
    _manager->update_params();
  }
  else
  {
    for (auto const& cluster : associations._clusters)
    {
      std::vector<MeasModelId> unusedMeasModels = measModelIDs;
      // first element is a priori track
      LMBDistribution lmbDist(_manager);
      std::vector<MeasurementContainer> measurementContainerList;
      if (!cluster._cluster.empty())
      {
        auto it = std::find(unusedMeasModels.begin(),
                            unusedMeasModels.end(),
                            _tracksVec.at(associations._trackID2sensor.at(cluster._cluster.front()))._id);
        if (it != unusedMeasModels.end())
        {
          unusedMeasModels.erase(it);
        }
        else
        {
          throw std::runtime_error("Measurement model id is more than one time in cluster. This is not allowed -> "
                                   "BUG!");
        }
        lmbDist._tracks.emplace_back(std::move(_tracksVec.at(associations._trackID2sensor.at(cluster._cluster.front()))
                                                   ._trackMapSensor.at(cluster._cluster.front())));
      }
      else
      {
        LOG_ERR("clusters are empty! Can this be possible?!");
      }
      // add the other cluster members as measurements
      std::vector<MeasurementContainer> measContainerList;
      for (auto memberIt = cluster._cluster.begin() + 1; memberIt < cluster._cluster.end(); memberIt++)
      {
        MeasurementContainer measContainer;
        measContainer._id = _tracksVec.at(associations._trackID2sensor.at(*memberIt))._id;
        auto it = std::find(unusedMeasModels.begin(), unusedMeasModels.end(), measContainer._id);
        if (it != unusedMeasModels.end())
        {
          unusedMeasModels.erase(it);
        }
        else
        {
          throw std::runtime_error("Measurement model id is more than one time in cluster. This is not allowed -> "
                                   "BUG!");
        }
        measContainer._sensorInfo =
            _tracksVec.at(associations._trackID2sensor.at(*memberIt))._sensorInfo;  // copy is needed...
        auto meas_dist = std::make_unique<GaussianDistribution>(
            std::move(
                (std::move(_tracksVec.at(associations._trackID2sensor.at(*memberIt))._trackMapSensor.at(*memberIt))
                     ._state_dist.begin()
                     ->second)
                    ->mean()),
            std::move(
                (std::move(_tracksVec.at(associations._trackID2sensor.at(*memberIt))._trackMapSensor.at(*memberIt))
                     ._state_dist.begin()
                     ->second)
                    ->covariance()),
            1,
            REFERENCE_POINT::CENTER);
        Measurement meas{ std::move(meas_dist), time, stateModelMap.at(sm_id)->state_comps() };
        _measId2stateModelId.emplace(meas._id, sm_id);
        measContainer._data.push_back(std::move(meas));
        measContainerList.push_back(std::move(measContainer));
      }
      // add empty measurement containers to ensure #measContainers==numSensors-1!

      for (const auto& mmID : unusedMeasModels)
      {
        MeasurementContainer emptyContainer;
        emptyContainer._id = mmID;
        measContainerList.push_back(std::move(emptyContainer));
      }

      switch (_t2t_params.strategy)
      {
        case TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::FPM_LMB_VERSION_WITH_ASSOCIATION_BEFORE:
        {
          auto [dist, birthLMBs] = fpmUpdateAndFusion(std::move(lmbDist), std::move(measContainerList), false);
          // only for checks -> remove
          std::size_t counter = 0;
          for (const auto& birthLMB : birthLMBs)
          {
            if (!birthLMB._trackMapSensor.empty())
            {
              LOG_ERR("There are birth tracks for sensor " << counter);
              for (const auto& t : birthLMB._trackMapSensor)
              {
                LOG_ERR("birth tracks have stateID " << t.first);
              }
            }
            counter++;
          }
          if (dist._tracks.size() > 1)
          {
            LOG_ERR("There is more than one track in the density!!!");
            throw std::runtime_error("There is more than one track in the density!!!");
          }
          for (auto const& track : dist._tracks)
          {
            fusedLMBDist._tracks.emplace_back(std::move(track));
          }
          break;
        }
        case TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::IC_LMB_VERSION_WITH_ASSOCIATION_BEFORE:
        {
          auto dist = ic_version(std::move(lmbDist), std::move(measContainerList), false);
          if (dist._tracks.size() > 1)
          {
            LOG_ERR("There is more than one track in the density!!!");
            throw std::runtime_error("There is more than one track in the density!!!");
          }
          for (auto const& track : dist._tracks)
          {
            fusedLMBDist._tracks.emplace_back(std::move(track));
          }
          break;
        }
        default:
          LOG_FATAL("TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY is not known! Can't perform fusion!");
      }
    }
  }
  _manager->next_params().lmb_distribution.use_grouping = use_grouping_memory;
  _manager->update_params();
  overwriteCovariances(fusedLMBDist);
  return fusedLMBDist;
}

LMBDistribution TTTUncorrelatedTracks::fuseTracksOfDifferentSensors()
{
  switch (_t2t_params.strategy)
  {
    case TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::FPM_LMB_VERSION_WITHOUT_ASSOCIATION_BEFORE:
      return fusion_without_t2tAssociation_before();
      break;
    case TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::IC_LMB_VERSION_WITHOUT_ASSOCIATION_BEFORE:
      return fusion_without_t2tAssociation_before();
      break;
    case TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::FPM_LMB_VERSION_WITH_ASSOCIATION_BEFORE:
      return fusion_with_t2tAssociation_before();
      break;
    case TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::IC_LMB_VERSION_WITH_ASSOCIATION_BEFORE:
      return fusion_with_t2tAssociation_before();
      break;
    default:
      LOG_FATAL("TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY is not known! Can't perform fusion!");
  }
  LOG_ERR("Should never reach this point. There is probably a bug in your config!");
  LMBDistribution fusedLMBDist(_manager);
  return fusedLMBDist;
}

}  // namespace ttb::uncorrelated_t2t_fusion
