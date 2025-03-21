#pragma once

#include "tracking_lib//TTBTypes/TTBTypes.h"
// ######################################################################################################################

namespace ttb
{

class TTBManager;
class State;

/// This represents a (class dependent) Markov Transition Matrix for a Multi Model Filter
class MarkovTransition final
{
public:
  explicit MarkovTransition(TTBManager* manager);

  /// return the Markov transition matrix
  [[nodiscard]] Matrix transitionMatrix(State const& state) const;

  TTBManager* _manager;
};

}  // namespace ttb