#pragma once

#include "tracking_lib/DataContainer/SlidingWindow.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Distributions/GammaDistribution.h"
#include "tracking_lib/Distributions/DirichletDistribution.h"
#include "tracking_lib/MultiObjectStateDistributions/GLMBDistribution.h"
#include "tracking_lib/DataContainer/DataSourceContainer.h"
#include "tracking_lib/Misc/LockedData.h"
namespace ttb
{
enum class SelfAssessmentType
{
  MEASUREMENT_INFLUENCE,  ///< double in (0, /infty): How much has a measurement on the state, this is determined by the
  ///< kullback leibner divergence between the prior and posterior
  MEASUREMENT_OPINION,        ///< Opinion 2: Opinion about the actual Measurement
  MEASUREMENT_MODEL_OPINION,  ///< Opinion 2: Opinion about the Measurement Model
  TRACK_MAP_OPINION,          ///< Opinion 2: How good fit the Track (-Trajectory) to the map
  TRACK_CLASS_OPINION,        ///< Opinion 2: How good fit the state of a track to its estimated class
};
// State, Meas, Source -> kl_divergence
using MeasurementInfluences = std::unordered_map<std::tuple<StateId, StateId, SourceId>, double>;
using MeasurementOpinions = std::unordered_map<StateId, Opinion>;
using MeasurementModelOpinions = std::unordered_map<std::pair<SourceId, Time>, Opinion>;
using TrackMapOpinions = std::unordered_map<StateId, Opinion>;
using TrackClassOpinions = std::unordered_map<StateId, Opinion>;
}  // namespace ttb

namespace ttb
{

struct ClutterEstimate_ProfilerData
{
  std::map<SourceId, std::tuple<double, double, double>> _estimation;  ///< alpha, beta, bayes factor
  Time _time;
};
std::string to_string(ClutterEstimate_ProfilerData const& data);

struct DetectionEstimate_ProfilerData
{
  std::map<SourceId, std::tuple<double, double, double>> _estimation;  /// alpha(0), alpha(1), bayes factor
  Time _time;
};
std::string to_string(DetectionEstimate_ProfilerData const& data);

struct PersistenceEstimate_ProfilerData
{
  double mean;
  double lower;
  double upper;
  Time _time;
};
std::string to_string(PersistenceEstimate_ProfilerData const& data);

struct BayesFactorProfilerData
{
  std::string meas_id;
  Time time;
  double bf;
};
std::string to_string(BayesFactorProfilerData const& data);

struct MeasLikelihoods
{
  double point{};
  double soft_clutter{};
  double soft_detection{};
};

struct BayesFactor
{
  double _clutter_bf{};
  double _detection_bf{};
};

struct GridData
{
  Vector<double> _steps;
  Vector<double> _prior;
  Vector<double> _posterior;
  [[nodiscard]] static double mean(Vector<double> const& steps, Vector<double> const& posterior);
  [[nodiscard]] double mean() const;
  [[nodiscard]] double quantile(double alpha) const;
};

template <class UDT>
class PersistenceModelAssessment
{
public:
  explicit PersistenceModelAssessment(TTBManager<UDT>* manager);
  void update(std::pair<std::vector<Matrix<UDT>>, Duration> const& surviving_info);
  [[nodiscard]] double mean() const;
  [[nodiscard]] double quantile(double alpha) const;
  void reset();
  TTBManager<UDT>* _manager;
  GridData _data;
};

template <class UDT>
class ClutterAssessmentMethodEstimation
{
public:
  explicit ClutterAssessmentMethodEstimation(TTBManager<UDT>* manager);
  /// update only the clutter rate estimation
  void update_clutter_estimate(DataSourceContainer<UDT> const& measurement_container,
                               Vector<UDT> const& clutter_distribution);
  void reset();
  static Opinion create_clutter_opinion(double estimation_lower_quantile,
                                        double estimation_upper_quantile,
                                        double parameter_used);

  TTBManager<UDT>* _manager;
  /// estimation of the clutter rate
  double mean(SourceId id) const;
  double quantile(SourceId id, double alpha) const;
  std::map<SourceId, GridData> _grid_approx_clutter_estimate{};
};

template <class UDT>
class ClutterAssessmentMethodBayesFactor
{
public:
  explicit ClutterAssessmentMethodBayesFactor(TTBManager<UDT>* manager) : _manager{ manager }
  {
  }
  /// update only the clutter rate estimation
  void update_clutter_bf(DataSourceContainer<UDT> const& measurement_container,
                         std::map<UDT, UDT> const& num_detected_and_weights,
                         UDT reference_measurement_likelihood);
  static Opinion to_opinion(double bayes_factor);
  void reset();

  TTBManager<UDT>* _manager;
  std::map<SourceId, boost::circular_buffer<Vector<UDT>>> _old_prior_weight_clutter{};
  std::map<SourceId, boost::circular_buffer<double>> _old_meas_likelihood{};
  std::map<SourceId, double> _bayes_factor{};
  std::map<SourceId, double> _cum_bayes_factor{};
};

template <class UDT>
class ClutterModelAssessment
{
public:
  ClutterModelAssessment(TTBManager<UDT>* manager)
    : _manager{ manager }, _estimation_method{ manager }, _bf_method{ manager }
  {
  }
  void reset();
  TTBManager<UDT>* _manager;
  ClutterAssessmentMethodEstimation<UDT> _estimation_method;
  ClutterAssessmentMethodBayesFactor<UDT> _bf_method;
};

template <class UDT>
class DetectionModelAssessmentEstimationMethod
{
public:
  DetectionModelAssessmentEstimationMethod(TTBManager<UDT>* manager);
  void update_detection_estimate(DataSourceContainer<UDT> const& measurement_container,
                                 std::vector<Matrix<UDT>> const& detection_distributions);
  static Opinion create_detection_opinion(DirichletDistribution const& parameter_estimation, double parameter_used);

  double mean(SourceId id) const;
  double quantile(SourceId id, double alpha) const;

  TTBManager<UDT>* _manager;
  /// estimation of the detection probability
  std::map<SourceId, GridData> _grid_based_detection_estimate{};
  SlidingWindow<std::vector<Matrix<UDT>>> short_term_detection_distributions;
  SlidingWindow<std::vector<Matrix<UDT>>> long_term_detection_distributions;
};

template <class UDT>
class DetectionModelAssessmentBayesFactor
{
public:
  DetectionModelAssessmentBayesFactor(TTBManager<UDT>* manager) : _manager{ manager }
  {
  }
  void update_detection_bf(DataSourceContainer<UDT> const& measurement_container,
                           std::map<std::pair<UDT, UDT>, UDT> const& num_detected_and_weights,
                           UDT reference_measurement_likelihood);
  static Opinion to_opinion(double bayes_factor);

  TTBManager<UDT>* _manager;
  /// old statistics: prior_weights and corresponding number of measurements
  std::map<SourceId, double> _bayes_factor{};
  std::map<SourceId, double> _cum_bayes_factor{};
};

template <class UDT>
class DetectionModelAssessment
{
public:
  DetectionModelAssessment(TTBManager<UDT>* manager)
    : _manager{ manager }, _estimation_method{ manager }, _bf_method{ manager }
  {
  }

  TTBManager<UDT>* _manager;
  DetectionModelAssessmentEstimationMethod<UDT> _estimation_method;
  DetectionModelAssessmentBayesFactor<UDT> _bf_method;
};

template <class UDT>
class StoachsticModelAssessment
{
public:
  explicit StoachsticModelAssessment(TTBManager<UDT>* manager)
    : _manager{ manager }
    , _clutter_model{ manager }
    , _detection_model{ manager }
    , _persistence_model_assessment{ manager }
  {
  }
  void reset();
  /// update the current clutter and detection estimation
  void update_assessment_opinions(DataSourceContainer<UDT> const& measurement_container,
                                  std::vector<State<UDT>> const& estimated_tracks);

  TTBManager<UDT>* _manager;

  ClutterModelAssessment<UDT> _clutter_model;
  DetectionModelAssessment<UDT> _detection_model;
  PersistenceModelAssessment<UDT> _persistence_model_assessment;

  /// opinion about the clutter submodel of the measurement model
  std::map<SourceId, Opinion> _clutter_opinion;

  /// opinion about the detection submodel of the measurement model
  std::map<SourceId, Opinion> _detection_opinion;
  /// old opinions
  std::map<SourceId, boost::circular_buffer<Opinion>> _old_clutter_opinions;
  std::map<SourceId, boost::circular_buffer<Opinion>> _old_detection_opinions;
  std::map<SourceId, std::map<COMPONENT, boost::circular_buffer<std::pair<double, std::size_t>>>> _old_nis_values;

  std::map<SourceId, std::map<COMPONENT, boost::circular_buffer<Opinion>>> _old_nis_opinions;
  std::map<SourceId, std::map<COMPONENT, Opinion>> _nis_filtered_opinions;
};

/// return the overlapping tracks
/// Tracks are overlapping, if they are during the prediction horizon closer than the min_distance
template <class UDT>
[[nodiscard]] std::vector<std::tuple<StateId, StateId, Probability<double>>>
overlapping_tracks(TTBManager<UDT> const& manager,
                   std::vector<StateId> const& current_tracks,
                   Duration prediction_horizon,
                   double min_distance);

template <class UDT>
class UnmodeledAspectsAssessment
{
public:
  explicit UnmodeledAspectsAssessment(TTBManager<UDT>* manager) : _manager{ manager }
  {
  }
  void evaluate_overlapping(std::vector<State<UDT>> const& current_tracks);
  TTBManager<UDT>* _manager;
};

/// evaluate the trajectory
[[nodiscard]] Opinion create_track_trajectory_opinion(std::vector<StateId> trajectory, std::optional<Map> const& map);

/// is the class consistent with some kinematic properties, e.g., extent
template <class UDT>
[[nodiscard]] Opinion create_track_class_opinion(State<UDT> const& track);

template <class UDT>
[[nodiscard]] Opinion create_measurement_opinion(State<UDT> const& measurement);

/// single time opinion about the track and map
template <class UDT>
[[nodiscard]] Opinion create_track_map_opinion(State<UDT> const& track);

template <class UDT>
class TrackingAssessment
{
public:
  explicit TrackingAssessment(TTBManager<UDT>* manager);

  // std::map<Label, std::map<SourceId, std::map<Time, double>>> _kl_divergences;
  TTBManager<UDT>* _manager;
  std::map<StateId, Opinion> _track_assessment;
  StoachsticModelAssessment<UDT> _model_assessment;
  UnmodeledAspectsAssessment<UDT> _unmodeled_aspects;
  MeasurementInfluences _meas_influences;
  MeasurementOpinions _meas_opinions;
  TrackMapOpinions _track_map_opinions;
  TrackClassOpinions _track_class_opinions;
  Synchronized<std::map<std::string, SlidingWindow<Opinion>>> implementation_opinions;
  void compute_implementation_opinion();
  void reset();

  /// compute the track assessment based on the
  void compute_track_assessment(std::vector<StateId> const& track_ids, Duration window_length);
};

}  // namespace ttb
