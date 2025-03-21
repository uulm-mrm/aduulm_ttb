#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/States/StateContainer.h"

namespace ttb
{

class BaseTTTFilter
{
public:
  virtual ~BaseTTTFilter() = default;
  /// compute on cycle for a given time
  virtual void cycle(Time time, std::vector<StateContainer> trackContainers) = 0;
  /// current estimation
  [[nodiscard]] virtual std::vector<State> getEstimate() const = 0;

  [[nodiscard]] virtual TTBManager* manager() const = 0;

  [[nodiscard]] virtual Time time() const = 0;

  [[nodiscard]] virtual TTT_FILTER_TYPE type() const = 0;

  /// Reset the filter
  virtual void reset() = 0;
};

}  // namespace ttb