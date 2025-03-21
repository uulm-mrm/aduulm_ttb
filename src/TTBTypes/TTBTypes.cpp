#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

double to_seconds(Duration dur)
{
  return std::chrono::duration<double, std::ratio<1, 1>>(dur).count();
}

double to_milliseconds(Duration dur)
{
  return std::chrono::duration<double, std::milli>(dur).count();
}

int64_t to_nanoseconds(Duration dur)
{
  return std::chrono::duration<int64_t, std::nano>(dur).count();
}

std::string to_string(Duration dur)
{
  return "Duration: " + std::to_string(to_seconds(dur)) + "s";
}

std::string to_string(Time time)
{
  return "Time: " + std::to_string(to_nanoseconds(time.time_since_epoch()) / 1000000000ULL) + "s " +
         std::to_string(to_nanoseconds(time.time_since_epoch()) % 1000000000ULL) + "ns";
}

namespace impl
{
template <class TYPE>
std::optional<TYPE> to_TYPE(std::string param, std::map<TYPE, std::string> data)
{
  for (auto const& [state, str] : data)
  {
    if (str == param)
    {
      return state;
    }
  }
  LOG_ERR("Unknown Type " + param);
  return {};
}
template <class TYPE>
std::string to_string(TYPE type, std::map<TYPE, std::string> data)
{
  if (data.contains(type))
  {
    return data.at(type);
  }
  LOG_ERR("Unknown Type: ");
  for (auto const& [_, str] : data)
  {
    LOG_ERR("Impl map str: " << str);
  }
  return {};
}
}  // namespace impl

std::string to_string(SELECTION_STRATEGY type)
{
  return impl::to_string<SELECTION_STRATEGY>(type, impl::SELECTION_STRATEGY_2_STRING);
}
std::optional<SELECTION_STRATEGY> to_SELECTION_STRATEGY(std::string type)
{
  return impl::to_TYPE<SELECTION_STRATEGY>(std::move(type), impl::SELECTION_STRATEGY_2_STRING);
}

std::string to_string(STAGE type)
{
  return impl::to_string(type, impl::STAGE_2_STRING);
}
std::optional<STAGE> to_STAGE_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::STAGE_2_STRING);
}

std::string to_string(CLASSIFICATION_TYPE type)
{
  return impl::to_string(type, impl::CLASSIFICATION_TYPE_2_STRING);
}
std::optional<CLASSIFICATION_TYPE> to_CLASSIFICATION_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::CLASSIFICATION_TYPE_2_STRING);
}

std::string to_string(CLASS type)
{
  return impl::to_string(type, impl::CLASS_2_STRING);
}
std::optional<CLASS> to_CLASS(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::CLASS_2_STRING);
}

std::optional<CLASS> unify(CLASS type)
{
  if (impl::CLASS_2_CLASS_unify.contains(type))
  {
    return impl::CLASS_2_CLASS_unify.at(type);
  }
  return {};
}

std::string to_string(SENSOR_CALIBRATION_TYPE type)
{
  return impl::to_string(type, impl::SENSOR_CALIBRATION_TYPE_2_STRING);
}
std::optional<SENSOR_CALIBRATION_TYPE> to_SENSOR_CALIBRATION_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::SENSOR_CALIBRATION_TYPE_2_STRING);
}

std::string to_string(STATE_DISTRIBUTION_EXTRACTION_TYPE type)
{
  return impl::to_string(type, impl::STATE_DISTRIBUTION_EXTRACTION_TYPE_2_STRING);
}
std::optional<STATE_DISTRIBUTION_EXTRACTION_TYPE> to_STATE_DISTRIBUTION_EXTRACTION_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::STATE_DISTRIBUTION_EXTRACTION_TYPE_2_STRING);
}

std::string to_string(TRANSITION_TYPE type)
{
  return impl::to_string(type, impl::TRANSITION_TYPE_2_STRING);
}
std::optional<TRANSITION_TYPE> to_TRANSITION_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::TRANSITION_TYPE_2_STRING);
}

std::string to_string(BUILD_MODE type)
{
  return impl::to_string(type, impl::BUILD_MODE_2_STRING);
}
std::optional<BUILD_MODE> to_BUILD_MODE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::BUILD_MODE_2_STRING);
}

std::string to_string(TTT_FILTER_TYPE type)
{
  return impl::to_string(type, impl::TTT_FILTER_TYPE_2_STRING);
}
std::optional<TTT_FILTER_TYPE> to_TTT_FILTER_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::TTT_FILTER_TYPE_2_STRING);
}

std::optional<TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY> to_TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY_2_STRING);
}

std::string to_string(TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY type)
{
  return impl::to_string(type, impl::TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY_2_STRING);
}

std::string to_string(LMB_UPDATE_METHOD type)
{
  return impl::to_string(type, impl::LMB_UPDATE_METHOD_2_STRING);
}
std::optional<LMB_UPDATE_METHOD> to_LMB_UPDATE_METHOD(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::LMB_UPDATE_METHOD_2_STRING);
}

std::string to_string(FILTER_TYPE type)
{
  return impl::to_string(type, impl::FILTER_TYPE_2_STRING);
}
std::optional<FILTER_TYPE> to_FILTER_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::FILTER_TYPE_2_STRING);
}

std::string to_string(EXTENT type)
{
  return impl::to_string(type, impl::EXTENT_2_STRING);
}
std::optional<EXTENT> to_EXTENT(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::EXTENT_2_STRING);
}

std::string to_string(COMPONENT type)
{
  return impl::to_string(type, impl::COMPONENT_2_STRING);
}
std::optional<COMPONENT> to_COMPONENT(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::COMPONENT_2_STRING);
}
std::optional<COMPONENT> derivate(COMPONENT comp)
{
  if (impl::COMPONENT_DT.contains(comp))
  {
    return impl::COMPONENT_DT.at(comp);
  }
  return {};
}

std::string to_string(MO_DISTRIBUTION_TYPE type)
{
  return impl::to_string(type, impl::MO_DISTRIBUTION_TYPE_2_STRING);
}
std::optional<MO_DISTRIBUTION_TYPE> to_MO_DISTRIBUTION_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::MO_DISTRIBUTION_TYPE_2_STRING);
}

std::string to_string(LMB_2_GLMB_CONVERISON_TYPE type)
{
  return impl::to_string(type, impl::LMB_2_GLMB_CONVERISON_TYPE_2_STRING);
}
std::optional<LMB_2_GLMB_CONVERISON_TYPE> to_LMB_2_GLMB_CONVERISON_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::LMB_2_GLMB_CONVERISON_TYPE_2_STRING);
}

std::string to_string(MO_DISTRIBUTION_EXTRACTION_TYPE type)
{
  return impl::to_string(type, impl::MO_DISTRIBUTION_EXTRACTION_TYPE_2_STRING);
}
std::optional<MO_DISTRIBUTION_EXTRACTION_TYPE> to_MO_DISTRIBUTION_EXTRACTION_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::MO_DISTRIBUTION_EXTRACTION_TYPE_2_STRING);
}

REFERENCE_POINT inverseRP(REFERENCE_POINT rp)
{
  switch (rp)
  {
    case REFERENCE_POINT::CENTER:
      return REFERENCE_POINT::CENTER;
    case REFERENCE_POINT::FRONT:
      return REFERENCE_POINT::BACK;
    case REFERENCE_POINT::BACK:
      return REFERENCE_POINT::FRONT;
    case REFERENCE_POINT::LEFT:
      return REFERENCE_POINT::RIGHT;
    case REFERENCE_POINT::RIGHT:
      return REFERENCE_POINT::LEFT;
    case REFERENCE_POINT::TOP:
      return REFERENCE_POINT::BOTTOM;
    case REFERENCE_POINT::BOTTOM:
      return REFERENCE_POINT::TOP;
    case REFERENCE_POINT::FRONT_LEFT:
      return REFERENCE_POINT::BACK_RIGHT;
    case REFERENCE_POINT::FRONT_RIGHT:
      return REFERENCE_POINT::BACK_LEFT;
    case REFERENCE_POINT::BACK_LEFT:
      return REFERENCE_POINT::FRONT_RIGHT;
    case REFERENCE_POINT::BACK_RIGHT:
      return REFERENCE_POINT::FRONT_LEFT;
  }
  DEBUG_ASSERT_MARK_UNREACHABLE;
}
std::string to_string(REFERENCE_POINT ref)
{
  return impl::to_string(ref, impl::REFERENCE_POINT_2_STRING);
}
std::optional<REFERENCE_POINT> to_REFERENCE_POINT(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::REFERENCE_POINT_2_STRING);
}

std::string to_string(BIRTH_MODEL_TYPE type)
{
  return impl::to_string(type, impl::BIRTH_MODEL_TYPE_2_STRING);
}
std::optional<BIRTH_MODEL_TYPE> to_BIRTH_MODEL(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::BIRTH_MODEL_TYPE_2_STRING);
}

std::string to_string(TRANSITION_MODEL_TYPE type)
{
  return impl::to_string(type, impl::TRANSITION_MODEL_TYPE_2_STRING);
}
std::optional<TRANSITION_MODEL_TYPE> to_TRANSITION_MODEL(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::TRANSITION_MODEL_TYPE_2_STRING);
}

std::string to_string(OCCLUSION_MODEL_TYPE type)
{
  return impl::to_string(type, impl::OCCLUSION_MODEL_TYPE_2_STRING);
}
std::optional<OCCLUSION_MODEL_TYPE> to_OCCLUSION_MODEL(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::OCCLUSION_MODEL_TYPE_2_STRING);
}

std::string to_string(PERSISTENCE_MODEL_TYPE type)
{
  return impl::to_string(type, impl::PERSISTENCE_MODEL_TYPE_2_STRING);
}
std::optional<PERSISTENCE_MODEL_TYPE> to_PERSISTENCE_MODEL(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::PERSISTENCE_MODEL_TYPE_2_STRING);
}

std::string to_string(DISTRIBUTION_TYPE type)
{
  return impl::to_string(type, impl::DISTRIBUTION_TYPE_2_STRING);
}
std::optional<DISTRIBUTION_TYPE> to_DISTRIBUTION_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::DISTRIBUTION_TYPE_2_STRING);
}

std::string to_string(DISTRIBUTION_EXTRACTION type)
{
  return impl::to_string(type, impl::DISTRIBUTION_EXTRACTION_2_STRING);
}
std::optional<DISTRIBUTION_EXTRACTION> to_DISTRIBUTION_EXTRACTION(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::DISTRIBUTION_EXTRACTION_2_STRING);
}

std::string to_string(STATE_MODEL_TYPE type)
{
  return impl::to_string(type, impl::STATE_MODEL_TYPE_2_STRING);
}
std::optional<STATE_MODEL_TYPE> to_STATE_MODEL_TYPE(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::STATE_MODEL_TYPE_2_STRING);
}

std::string to_string(MEASUREMENT_MODEL_TYPE model)
{
  return impl::to_string(model, impl::MEASUREMENT_MODEL_TYPE_2_STRING);
}
std::optional<MEASUREMENT_MODEL_TYPE> to_MEASUREMENT_MODEL(std::string name)
{
  return impl::to_TYPE(std::move(name), impl::MEASUREMENT_MODEL_TYPE_2_STRING);
}

std::string to_string(GLMB_ASSIGNMENT_METHOD type)
{
  return impl::to_string(type, impl::RANKED_ASSIGNMENT_ALGORITHM_TYPE_2_STRING);
}
std::optional<GLMB_ASSIGNMENT_METHOD> to_GLMB_UPDATE_METHOD(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::RANKED_ASSIGNMENT_ALGORITHM_TYPE_2_STRING);
}

std::string to_string(MULTI_SENSOR_UPDATE_METHOD type)
{
  return impl::to_string(type, impl::MULTI_SENSOR_UPDATE_METHOD_TYPE_2_STRING);
}
std::optional<MULTI_SENSOR_UPDATE_METHOD> to_MULTI_SENSOR_UPDATE_METHOD(std::string type)
{
  return impl::to_TYPE(std::move(type), impl::MULTI_SENSOR_UPDATE_METHOD_TYPE_2_STRING);
}
}  // namespace ttb
