#pragma once

#include "tracking_lib/DataContainer/DataSourceContainer.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include <aduulm_logger/aduulm_logger.hpp>
#include <tracy/tracy/Tracy.hpp>
#include <tracy/tracy/TracyC.h>

namespace ttb::utils
{

struct NoTrans
{
  void operator()(auto&) const noexcept
  {
  }
};

/// predict all states inside the state_container
/// proj: should map from a single element of the container to a State&
/// trans: additional steps after the prediction
template <class UDT, class Ts, class Trans = NoTrans, class Proj = std::identity>
  requires requires(Ts const& data, Proj proj, Trans trans) {
    { std::invoke(proj, *data.begin()) } -> std::convertible_to<State<UDT>>;
  }
[[nodiscard]] std::unordered_map<StateId, State<UDT>> predict(Ts const& state_container,
                                                              Duration dt,
                                                              EgoMotionDistribution const& ego_motion_distribution,
                                                              Proj proj = {},
                                                              Trans trans = {})
{
  std::unordered_map<StateId, State<UDT>> predicted_states;
  for (auto const& T : state_container)
  {
    State copy = std::invoke(proj, T);
    copy.predict(dt, ego_motion_distribution);
    if (not copy.isEmpty())
    {
      std::invoke(trans, copy);
    }
    predicted_states.emplace(std::invoke(proj, T)._id, std::move(copy));
  }
  return predicted_states;
}

/// innovate all states inside the state_container
/// proj: should map from a single element of the container to a State&
template <class UDT, class Ts, class Proj = std::identity>
  requires requires(Ts const& data, Proj proj) {
    { std::invoke(proj, *data.begin()) } -> std::convertible_to<State<UDT>>;
  }
[[nodiscard]] std::vector<std::tuple<StateDetection<UDT>,
                                     std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>>,
                                     std::vector<MeasClutter<UDT>>>>
innovate(Ts const& state_container, DataSourceContainer<UDT> const& measurement_container, Proj proj = {})
{
  std::vector<std::tuple<StateDetection<UDT>,
                         std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>>,
                         std::vector<MeasClutter<UDT>>>>
      updates;
  for (auto const& T : state_container)
  {
    updates.push_back(std::invoke(proj, T).innovate(measurement_container));
  }
  return updates;
}

/// predict the states with the given ids, return vector of old_state_id -> predicted_state_id
template <class UDT, class Trans = NoTrans>
[[nodiscard]] std::unordered_map<StateId, StateId> predict_db(TTBManager<UDT>* manager,
                                                              std::vector<StateId> const& ids,
                                                              Duration dt,
                                                              EgoMotionDistribution const& ego_motion_distribution,
                                                              Trans trans = {})
{
  TracyFiberEnter("Utils::predict");
  TracyCZone(ctx, 1);
  // ZoneScopedN("Utils::predict");
  std::vector<std::pair<StateId, std::optional<State<UDT>>>> predicted;
  predicted.reserve(ids.size());
  for (StateId id : ids)
  {
    predicted.emplace_back(id, std::nullopt);
  }

  auto task = [&](std::pair<StateId, std::optional<State<UDT>>>& to_predict) {
    State<UDT> state = manager->get_state(to_predict.first);
    state.predict(dt, ego_motion_distribution);
    if (not state.isEmpty())
    {
      std::invoke(trans, state);
      to_predict.second = std::move(state);
    }
  };
  if (manager->params()->thread_pool_size > 0)
  {
    hpx::for_each(hpx::execution::par_unseq, predicted.begin(), predicted.end(), task);
  }
  else
  {
    hpx::for_each(hpx::execution::seq, predicted.begin(), predicted.end(), task);
  }

  std::unordered_map<StateId, StateId> predicted_ids;
  {
    ZoneScopedN("Utils::predict_db::save");
    for (auto& [old_id, predicted_state] : predicted)
    {
      if (predicted_state.has_value())
      {
        predicted_ids.emplace(old_id, predicted_state.value()._id);
        manager->database().states.insert(std::move(predicted_state.value()));
      }
      else
      {
        predicted_ids.emplace(old_id, EMPTY_STATE);
      }
    }
  }
  assert([&] {  // NOLINT
    for (StateId id : ids)
    {
      if (not predicted_ids.contains(id))
      {
        return false;
      }
    }
    return true;
  }());
  TracyFiberEnter("Utils::predict");
  TracyCZoneEnd(ctx);
  return predicted_ids;
}

/// innovate all states inside the state_container
template <class UDT>
void innovate_db(TTBManager<UDT>* manager,
                 std::vector<StateId> const& state_container,
                 DataSourceContainer<UDT> const& measurement_container)
{
  TracyFiberEnter("Utils::innovate");
  TracyCZone(ctx, 1);
  // hpx::tracy::region("Utils::innovate_db", 0, 0);
  std::vector<std::tuple<StateDetection<UDT>,
                         std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>>,
                         std::vector<MeasClutter<UDT>>>>
      results;
  results.reserve(state_container.size());
  for (auto const& [ind, state_id] : std::views::enumerate(state_container))
  {
    results.emplace_back(StateDetection<UDT>{ .state_id = state_id },
                         std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>>{},
                         std::vector<MeasClutter<UDT>>{});
  }
  auto const task = [&](auto& state_update_result) {
    State<UDT> const& state = manager->get_state(std::get<0>(state_update_result).state_id);
    state_update_result = state.innovate(measurement_container);
  };
  LOG_DEB("Compute state update");
  hpx::for_each(hpx::execution::par_unseq, results.begin(), results.end(), task);
  LOG_DEB("computation done");
  {
    ZoneScopedN("Utils::innovate_db::save");
    for (auto& [state_innovation, state_meas_innovations, meas_innovations] : results)
    {
      manager->database().state_detection_probabilities.insert(
          { .state_id = state_innovation.state_id,
            .data_source_container_id = measurement_container._id,
            .detection_probability = state_innovation.detection_probability });
      std::vector<StateId> relevant_meas;
      for (auto& [state_id, meas_id, updated_state, meas_log_likelihood] : state_meas_innovations)
      {
        manager->database().state_updates.insert({ .prior_state_id = state_id,
                                                   .data_container_id = measurement_container._id,
                                                   .meas_id = meas_id,
                                                   .updated_state_id = updated_state._id });
        manager->database().state_meas_log_likelihood.insert(
            { .state_id = state_id, .meas_id = meas_id, .meas_log_likelihood = meas_log_likelihood });
        manager->database().states.insert(std::move(updated_state));
        relevant_meas.push_back(meas_id);
      }
      manager->database().relevant_meas.insert({ .prior_state_id = state_innovation.state_id,
                                                 .data_container_id = measurement_container._id,
                                                 .relevant_meas_ids = std::move(relevant_meas) });
      for (MeasClutter<UDT> const& inno : meas_innovations)
      {
        manager->database().meas_clutter_intensity.insert(
            { .meas_id = inno.meas_id, .clutter_intensity = inno.clutter_intensity });
      }
    }
  }
  TracyFiberEnter("Utils::innovate");
  TracyCZoneEnd(ctx);
}

}  // namespace ttb::utils
