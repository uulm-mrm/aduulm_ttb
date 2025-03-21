#pragma once

#include "tracking_lib/TTTFilters/BaseTTTFilter.h"

#include "tracking_lib/TTBTypes/Params.h"

namespace ttb
{

class TTBManager;

class NoTTTFilter : public BaseTTTFilter
{
public:
  explicit NoTTTFilter(TTBManager* manager);

  void cycle(Time time, std::vector<StateContainer> trackContainers) override;

  [[nodiscard]] std::vector<State> getEstimate() const override;

  [[nodiscard]] TTBManager* manager() const override;

  [[nodiscard]] Time time() const override;

  [[nodiscard]] TTT_FILTER_TYPE type() const override;

  void reset() override;

  TTBManager* _manager;
  std::vector<State> _tracks;
};

}  // namespace ttb