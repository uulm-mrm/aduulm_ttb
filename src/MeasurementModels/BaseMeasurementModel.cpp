#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"

namespace ttb
{
BaseMeasurementModel::DefaultVal default_val(std::vector<DefaultValuesParams> const& params,
                                             COMPONENT comp,
                                             CLASS clazz)
{
  const auto it = std::ranges::find_if(params, [&](DefaultValuesParams const& class_vars) {
    return std::ranges::find(class_vars.type, clazz) != class_vars.type.end();
  });
  if (it == params.end())
  {
    LOG_FATAL("Class " + to_string(clazz) + " not found in default_values");
    return {};
  }
  return default_val(*it, comp);
}

BaseMeasurementModel::DefaultVal default_val(DefaultValuesParams const& params, COMPONENT comp)
{
  auto mean = [&] -> std::optional<double> {
    auto const mean_it = params.mean.find(to_string(comp));
    if (mean_it == params.mean.end())
    {
      return std::nullopt;
    }
    return mean_it->second;
  }();
  auto var = [&] -> std::optional<double> {
    auto const var_it = params.var.find(to_string(comp));
    if (var_it == params.var.end())
    {
      return std::nullopt;
    }
    return var_it->second;
  }();
  return BaseMeasurementModel::DefaultVal{ .mean = mean, .var = var };
}

}  // namespace ttb