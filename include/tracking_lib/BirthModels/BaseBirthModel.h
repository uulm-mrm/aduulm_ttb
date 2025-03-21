#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/States/State.h"

namespace ttb
{
class TTBManager;
class State;
class BaseMeasurementModel;

/// The interface of the BirthModels.
/// A birth model is responsible for setting up new tracks.
class BaseBirthModel
{
public:
  virtual ~BaseBirthModel() = default;
  /// the type of the birth model.
  [[nodiscard]] virtual BIRTH_MODEL_TYPE type() const = 0;
  /// get access to the manager.
  [[nodiscard]] virtual TTBManager* manager() const = 0;
  /// return the new birth tracks.
  [[nodiscard]] virtual std::vector<State> getBirthTracks(MeasurementContainer const& measContainer,
                                                          std::map<MeasurementId, Probability> const& assignProb,
                                                          std::vector<State> const& existingTracks) = 0;
  /// Estimate the birth prob of the given state.
  /// This depends on the existing tracks only when overlapping birth tracks are not allowed.
  [[nodiscard]] virtual double getBirthProbability(State const& newBornState,
                                                   std::vector<State> const& existingTracks) const;
  /// reset the birth model.
  virtual void reset() = 0;
};

}  // namespace ttb
