#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/States/State.h"

#include "tracking_lib/BirthModels/DynamicBirthModel.h"
#include "tracking_lib/BirthModels/StaticBirthModel.h"
#include "tracking_lib/BirthModels/ROIBirthModel.h"

namespace ttb
{

static_assert(CPTBirthModel<StaticBirthModel<double>, double>);
static_assert(CPTBirthModel<DynamicBirthModel<double>, double>);
static_assert(CPTBirthModel<ROIBirthModel<double>, double>);
static_assert(CPTBirthModel<StaticBirthModel<AD>, AD>);
static_assert(CPTBirthModel<DynamicBirthModel<AD>, AD>);
static_assert(CPTBirthModel<ROIBirthModel<AD>, AD>);

/// The interface of the BirthModels.
/// A birth model is responsible for setting up new tracks.
template <class UDT>
class BirthModel
{
public:
  using AllModels = std::variant<DynamicBirthModel<UDT>, StaticBirthModel<UDT>, ROIBirthModel<UDT>>;

  /// the type of the birth model.
  [[nodiscard]] BIRTH_MODEL_TYPE type() const
  {
    return std::visit([&](auto const& model) { return model.type(); }, _model);
  }
  /// get access to the manager.

  [[nodiscard]] TTBManager<UDT>* manager() const
  {
    return std::visit([&](auto const& model) { return model.manager(); }, _model);
  }

  /// return the new birth tracks
  [[nodiscard]] BirthTracks<UDT> getBirthTracks(DataSourceContainer<UDT> const& measContainer,
                                                std::map<MeasurementId, UDT> const& assignProb,
                                                std::vector<State<UDT>> const& existingTracks)
  {
    return std::visit([&](auto& model) { return model.getBirthTracks(measContainer, assignProb, existingTracks); },
                      _model);
  }

  /// reset the birth model.
  void reset()
  {
    std::visit([&](auto& model) { model.reset(); }, _model);
  }
  AllModels _model;
};

/// Compute the existence prob of the given state.
/// This depends on the existing tracks only when overlapping birth tracks are not allowed.
template <class UDT>
[[nodiscard]] UDT getBirthProbability(TTBManager<UDT> const&,
                                      State<UDT> const& newBornState,
                                      std::vector<State<UDT>> const& existingTracks);

}  // namespace ttb
