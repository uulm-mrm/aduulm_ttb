#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Distributions/GammaDistribution.h"
#include "tracking_lib/Distributions/DirichletDistribution.h"
#include "tracking_lib/DataContainer/DataSourceContainer.h"
#include "tracking_lib/SelfAssessment/SelfAssessment.h"
#include "tracking_lib/Misc/LockedData.h"

namespace ttb
{

template <class UDT>
std::pair<std::vector<double>, std::vector<double>> to_box(Vector<UDT> const& state, Components const& comps);

struct Assessment
{
  Opinion _clutter_estimation;
  Opinion _clutter_bf;
  Opinion _clutter;
  Opinion _detection_estimation;
  Opinion _detection_bf;
  Opinion _detection;
  std::map<COMPONENT, Opinion> _nis;
};

template <class UDT>
struct VizuMeasModelData
{
  std::vector<Time> _discarded_data;
  std::vector<std::pair<Time, Duration>> _delay;
  std::vector<std::pair<Time, Duration>> _estimated_delay;
  std::vector<std::pair<Time, Duration>> _duration_in_buffer;

  std::vector<std::pair<Time, GammaDistribution>> _clutter_estimation;
  std::vector<std::tuple<Time, double, double, double>> _clutter_estimation_bayes_approx;
  std::vector<std::pair<Time, GammaDistribution>> _approx_clutter_estimation;
  std::vector<std::pair<Time, double>> _clutter_rate;
  std::vector<std::pair<Time, double>> _clutter_bayes_factor;
  std::vector<std::pair<Time, double>> _clutter_cum_bayes_factor;

  std::vector<std::pair<Time, DirichletDistribution>> _detection_estimation;
  std::vector<std::tuple<Time, double, double, double>> _detection_estimation_bayes_approx;
  std::vector<std::pair<Time, DirichletDistribution>> _approx_detection_estimation;

  std::vector<std::pair<Time, double>> _detection_probability;
  std::vector<std::pair<Time, double>> _detection_bayes_factor;
  std::vector<std::pair<Time, double>> _detection_cum_bayes_factor;

  std::vector<std::pair<Time, Assessment>> _assessment;

  std::map<COMPONENT, std::vector<std::tuple<Time, double, double>>> _nis_dof;
  std::vector<std::pair<Time, DataSourceContainer<UDT>>> _measurements;
  std::vector<std::pair<Time, double>> _num_assoc_measurements;
  std::vector<std::pair<Time, Duration>> _computation_time;
  std::vector<Time> _used_in_cycle;
  std::map<SourceId, std::vector<std::pair<Time, std::tuple<Vector2, Vector2, double>>>>
      _distance_to_reference;  // distance, normed distance, V_ABS difference
  std::vector<std::pair<Time, Vector<double>>> _cardinality_distribution;
  std::vector<std::pair<Time, Vector<double>>> _num_hypotheses_per_cardinality;
  std::vector<std::pair<Time, std::size_t>> _num_hypotheses;
  std::vector<std::pair<Time, std::vector<State<UDT>>>> _states_in_sensor_cs;  /// the states after the update with this
                                                                               /// meas model
  std::vector<std::pair<Time, Region>> _fov_in_sensor_cs;
  std::map<Label, std::vector<State<UDT>>> _tracks;  // tracks right after update with this sensor
  std::vector<std::pair<Time, std::vector<BernoulliState<UDT>>>> _birth_tracks;
  std::vector<std::pair<Time, std::vector<std::tuple<std::vector<State<UDT>>, std::size_t, Hypotheses<UDT>>>>>
      _track_groups;
};

template <class UDT>
struct VizuCycleData
{
  std::vector<Time> _trigger_time;
  std::map<Label, std::vector<State<UDT>>> _tracks;
  std::map<Label, std::vector<State<UDT>>> _all_tracks;
  std::map<Label, std::vector<State<UDT>>> _gt_tracks;
  std::vector<std::tuple<Time, std::size_t>> _num_tracks;
  std::vector<std::tuple<Time, std::size_t>> _num_measurements;
  std::vector<std::tuple<Time, Duration>> computation_time;
  std::vector<std::tuple<Time, Duration>> _filter_delay_first;
  std::vector<std::tuple<Time, Duration>> _filter_delay_last;
  std::vector<std::tuple<Time, Duration>> _buffer_delay;
  std::vector<std::tuple<Time, std::size_t>> _buffer_size;
  std::vector<std::tuple<Time, std::size_t>> _num_sources;
  std::vector<std::tuple<Time, GospaResult>> gospa;
  std::vector<std::tuple<Time, OspaResult>> ospa;
  std::vector<std::tuple<Time, Ospa2Result>> ospa2;
  std::map<std::string, std::vector<std::pair<Time, Opinion>>> implementation_op;
  std::vector<std::tuple<Time, double, double, double>> persistence_prob_estimation;
  std::vector<std::tuple<Time, std::vector<State<UDT>>>> _overlapping_tracks;
  std::vector<std::tuple<Time, std::vector<State<UDT>>>> _vanished_tracks;
};

auto constexpr buffer_size = 100;

/// Visualization for the TTBManager
template <class UDT>
class Vizu
{
public:
  explicit Vizu(TTBManager<UDT>* manager);
  int viz_loop();

  Synchronized<VizuCycleData<UDT>> _cycle_data{};
  Synchronized<std::map<SourceId, VizuMeasModelData<UDT>>> _meas_model_data{};

  void show();
  void show_overview();
  void show_tracks();
  void show_assessment_overview();
  void show_assessment();
  void show_birth_info();
  void show_debug();
  void show_tracks_in_sensor_cs();
  void show_grouping();

  void plot_fov(Time now, bool show_names);

  void reset();

  TTBManager<UDT>* _manager;
};

}  // namespace ttb
