#include "tracking_lib/StateTransition/MarkovTransition.h"
// ######################################################################################################################
#include "tracking_lib/States/State.h"
#include "tracking_lib/TTBManager/TTBManager.h"

namespace ttb
{

template class MarkovTransition<double>;
template class MarkovTransition<AD>;

template <class UDT>
MarkovTransition<UDT>::MarkovTransition(TTBManager<UDT> const& manager) : _manager{ manager }
{
  if (_manager.params()->state.multi_model.enable_markov_transition)
  {
    for (MarkovTransitionParams const& transitionParams : _manager.params()->state.multi_model.markov_transition)
    {
      if (not transitionParams.transition_matrix.rowwise().sum().isApprox(
              Vector<double>::Ones(_manager.params()->state.multi_model.use_state_models.size())))
      {
        LOG_FATAL("Transition Matrix is either not valid (columns for every row must sum to 1) or #rows does not fit "
                  "the "
                  "#use_state_models");
        LOG_FATAL("Transition Matrix: " << transitionParams.transition_matrix);
        throw std::runtime_error("Transition Matrix not valid");
      }
    }
  }
}

template <class UDT>
Matrix<double> MarkovTransition<UDT>::transitionMatrix(State<UDT> const& state) const
{
  if (not _manager.params()->state.multi_model.enable_markov_transition)
  {
    return Matrix<double>::Identity(state._state_dist.size(), state._state_dist.size());
  }
  CLASS const type{ state._classification.getEstimate() };
  for (MarkovTransitionParams const& transitionParams : _manager.params()->state.multi_model.markov_transition)
  {
    if (std::ranges::find(transitionParams.type, type) != transitionParams.type.end())
    {
      return transitionParams.transition_matrix;
    }
  }
  LOG_FATAL("No Transition Matrix given for State: " + state.toString());
  throw std::runtime_error("No Transition Matrix given for State: " + state.toString());
}

}  // namespace ttb