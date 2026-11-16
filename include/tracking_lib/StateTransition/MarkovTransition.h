#pragma once

#include "tracking_lib//TTBTypes/TTBTypes.h"

namespace ttb
{

/// This represents a (class dependent) Markov Transition Matrix for a Multi Model Filter
template <class UDT>
class MarkovTransition final
{
public:
  explicit MarkovTransition(TTBManager<UDT> const& manager);

  /// return the Markov transition matrix
  [[nodiscard]] Matrix<double> transitionMatrix(State<UDT> const& state) const;

  TTBManager<UDT> const& _manager;
};

}  // namespace ttb