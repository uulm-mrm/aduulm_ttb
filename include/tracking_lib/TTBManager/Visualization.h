#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Distributions/GammaDistribution.h"
#include "tracking_lib/Distributions/DirichletDistribution.h"

#include <memory>
#include <vector>
#include <map>
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/States/StateContainer.h"

namespace ttb
{
class TTBManager;

struct VizuMeasModelData
{
  std::vector<Time> _discarded_data;
  std::vector<std::pair<Time, Duration>> _delay;
  std::vector<std::pair<Time, Duration>> _estimated_delay;
  std::vector<std::pair<Time, Duration>> _duration_in_buffer;
  std::vector<std::pair<Time, GammaDistribution>> _clutter_estimation;
  std::vector<std::pair<Time, double>> _clutter_bayes_factor;
  std::vector<std::pair<Time, DirichletDistribution>> _detection_estimation;
  std::vector<std::pair<Time, double>> _detection_bayes_factor;
  std::map<COMPONENT, std::vector<std::tuple<Time, double, double>>> _nis_dof;
  std::vector<std::pair<Time, std::variant<MeasurementContainer, StateContainer>>> _measurements;
  std::vector<std::pair<Time, double>> _num_assoc_measurements;
  std::vector<std::pair<Time, Duration>> _computation_time;
  std::vector<Time> _used_in_cycle;
  std::map<MeasModelId, std::vector<std::pair<Time, std::tuple<Vector2, Vector2, double>>>>
      _distance_to_reference;  // distance, normed distance, V_ABS difference
};

struct VizuCycleData
{
  std::vector<Time> _trigger_time;
  std::map<Label, std::vector<State>> _tracks;
  std::vector<std::tuple<Time, std::size_t>> _num_tracks;
  std::vector<std::tuple<Time, std::size_t>> _num_measurements;
  std::vector<std::tuple<Time, Duration>> computation_time;
  std::vector<std::tuple<Time, Duration>> _filter_delay;
  std::vector<std::tuple<Time, Duration>> _buffer_delay;
  std::vector<std::tuple<Time, std::size_t>> _buffer_size;
  std::vector<std::tuple<Time, std::size_t>> _num_sources;
};

auto constexpr buffer_size = 100;

/// Visualization for the TTBManager
class Vizu
{
public:
  explicit Vizu(TTBManager* manager);
  int viz_loop();

  std::mutex add_data_mutex;
  VizuCycleData _cycle_data{};
  std::map<MeasModelId, VizuMeasModelData> _meas_model_data{};

  void show();
  void show_overview();
  void show_tracks();
  void show_assessment();
  void show_evaluation();
  void show_params();
  void show_debug();

  void reset();

  TTBManager* _manager;
};

}  // namespace ttb