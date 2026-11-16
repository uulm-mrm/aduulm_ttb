#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/eigen/dense.h>

#include <figcone/configreader.h>
#include <tracking_lib/TTBTypes/TTBTypes.h>
#include <tracking_lib/Misc/logger_setup.h>

#include <boost/geometry/algorithms/assign.hpp>
#include <boost/geometry/algorithms/correct.hpp>

#include <tracking_lib/Bindings/TypeCasters.h>
#include <tracking_lib/TTBManager/TTBManager.h>
#include <tracking_lib/Distributions/GaussianDistribution.h>
#include <tracking_lib/MeasurementModels/BaseMeasurementModel.h>
#include <tracking_lib/PersistenceModels/BasePersistenceModel.h>
#include <tracking_lib/StateModels/BaseStateModel.h>
#include <tracking_lib/States/EgoMotionDistribution.h>
#include <tracking_lib/States/State.h>
#include <tracking_lib/Trackers/Tracker.h>
#include <tracking_lib/Measurements/SensorInformation.h>
#include <tracking_lib/TTBTypes/Components.h>
#include "tracking_lib/Misc/HPXWrapper.h"

namespace nb = nanobind;

using EgoDist = ttb::EgoMotionDistribution;

using SensorInfo = ttb::SensorInformation;
using Components = ttb::Components;
using Component = ttb::COMPONENT;
using Time = ttb::Time;
using ReferencePoint = ttb::REFERENCE_POINT;
using Classification = ttb::classification::MeasClassification;

using ClassLabel = ttb::CLASS;
using SE3Trafo = ttb::SE3Trafo;

void start(int num_threads = -1)
{
  std::vector<char*> argv;
  int argc = static_cast<int>(argv.size());
  hpx::init_params params;
  if (num_threads > 0)
  {
    params.cfg = { std::format("hpx.os_threads={}", num_threads), "hpx.stacks.small_size=0x40000" };
  }
  params.rp_callback = [](hpx::resource::partitioner& rp, hpx::program_options::variables_map const&) {
    // Create a dedicated single-threaded pool for viz
    rp.create_thread_pool("viz_pool", hpx::resource::scheduling_policy::local_priority_fifo);

    // Assign exactly one PU to it (taken from the last core)
    auto& numa_domains = rp.numa_domains();
    auto& last_core = numa_domains.back().cores().back();
    auto& pu = last_core.pus().back();

    rp.add_resource(pu, "viz_pool");  // moves PU out of default pool
  };
  // Non-blocking: returns immediately, HPX runs in background threads
  if (!hpx::start(nullptr, argc, argv.data(), params))
  {
    throw std::runtime_error("Failed to start HPX runtime");
  }
}

void stop()
{
  // finalize() must be called FROM WITHIN the HPX runtime
  hpx::post([] { hpx::finalize(); });
  // Blocks until HPX has fully shut down
  hpx::stop();
}

NB_MODULE(_tracking_lib_python_api, m)
{
  /// HPX Runtime management
  /// All interactions with library must happen within a running HPX runtime environment
  m.def("start_hpx", [](int threads) { start(threads); }, nb::arg("num_threads") = -1);
  m.def("stop_hpx", []() { stop(); });

  /// The TTBManager is the top-level interface to the library
  /// The "default" version uses a plain double for computations, this is not able to compute gradients
  /// Use it for normal filter inference
  nb::class_<ttb::TTBManager<double>>(m, "TTBManager")
      .def("__init__",
           [](ttb::TTBManager<double>* self, std::filesystem::path p) {
             hpx::run_as_hpx_thread([&]() { new (self) ttb::TTBManager<double>(std::move(p)); });
           })
      .def("reset", [](ttb::TTBManager<double>* self) { hpx::run_as_hpx_thread([&] { return self->reset(); }); })
      .def("cycle",
           [](ttb::TTBManager<double>* self, Time time) { hpx::run_as_hpx_thread([&] { return self->cycle(time); }); })
      .def("cycle",
           [](ttb::TTBManager<double>* self, Time time, std::vector<ttb::DataSourceContainer<double>> data, bool flag) {
             hpx::run_as_hpx_thread([&] { return self->cycle(time, data, flag); });
           })
      .def("getEstimate",
           [](ttb::TTBManager<double>* self) { return hpx::run_as_hpx_thread([&] { return self->getEstimate(); }); })
      .def("addData",
           [](ttb::TTBManager<double>* self, ttb::DataSourceContainer<double> data, Time time) {
             hpx::run_as_hpx_thread([&] { return self->addData(data, time); });
           })
      .def("showGui", [](ttb::TTBManager<double>* self) { hpx::run_as_hpx_thread([&] { return self->show_gui(); }); });

  /// This version is able to compute gradients for selected filter operations.
  /// See Alexander Scheible and Michael Buchholz, "Differentiable Labeled Multi-Bernoulli Filtering for Unsupervised
  /// Filter Model Learning in Multi-Object Tracking" in SDF 26 for details
  nb::class_<ttb::TTBManager<ttb::AD>>(m, "TTBManagerAD")
      .def("__init__",
           [](ttb::TTBManager<ttb::AD>* self, std::filesystem::path p) {
             hpx::run_as_hpx_thread([&]() { new (self) ttb::TTBManager<ttb::AD>(std::move(p)); });
           })
      .def("reset", [](ttb::TTBManager<ttb::AD>* self) { hpx::run_as_hpx_thread([&] { return self->reset(); }); })
      .def("cycle",
           [](ttb::TTBManager<ttb::AD>* self, Time time) { hpx::run_as_hpx_thread([&] { return self->cycle(time); }); })
      .def("cycle",
           [](ttb::TTBManager<ttb::AD>* self,
              Time time,
              std::vector<ttb::DataSourceContainer<ttb::AD>> data,
              bool flag) { hpx::run_as_hpx_thread([&] { return self->cycle(time, data, flag); }); })
      .def("getEstimate",
           [](ttb::TTBManager<ttb::AD>* self) { return hpx::run_as_hpx_thread([&] { return self->getEstimate(); }); })
      .def("addData",
           [](ttb::TTBManager<ttb::AD>* self, ttb::DataSourceContainer<ttb::AD> data, Time time) {
             hpx::run_as_hpx_thread([&] { return self->addData(data, time); });
           })
      .def("showGui", [](ttb::TTBManager<ttb::AD>* self) { hpx::run_as_hpx_thread([&] { return self->show_gui(); }); });

  /// Set the log level, i.e., debug, info, warning, error, fatal, of the whole library.
  /// Beware: Logs are printed on stdout and may slow down application
  /// Debug level may be very noisy
  m.def("set_log_level", [](std::string const& log_level) {
    ttb::_setLogLevel(log_level);
    ttb::_setStreamName("");
    ttb::_setPrefix("");
  });

  nb::class_<ttb::PredictionInformation>(m, "PredictionInformation")
      .def(nb::init())
      .def_rw("_ego_motion", &ttb::PredictionInformation::_egoMotion)
      .def_rw("_roi", &ttb::PredictionInformation::_roi);

  nb::class_<ttb::GroundTruthScenarioInfo>(m, "GroundTruthInfo")
      .def(nb::init())
      .def_rw("state_model_info", &ttb::GroundTruthScenarioInfo::state_model_info)
      .def_rw("meas_model_info", &ttb::GroundTruthScenarioInfo::meas_model_info)
      .def_rw("birth_model_info", &ttb::GroundTruthScenarioInfo::birth_model_info);

  nb::class_<ttb::GroundTruthStateModelInfo>(m, "StateModelInfo")
      .def(nb::init())
      .def_rw("noise", &ttb::GroundTruthStateModelInfo::noise_dev)
      .def_rw("persistence_prob", &ttb::GroundTruthStateModelInfo::persistence_prob);

  nb::class_<ttb::GroundTruthBirthModelInfo>(m, "BirthModelInfo")
      .def(nb::init())
      .def_rw("poss_birth_state", &ttb::GroundTruthBirthModelInfo::poss_birth_state)
      .def_rw("prob", &ttb::GroundTruthBirthModelInfo::birth_probability);

  m.def(
      "add_gt_birth_pos",
      [](ttb::GroundTruthBirthModelInfo& birth_info, std::map<ttb::COMPONENT, std::pair<double, double>> const& state) {
        birth_info.poss_birth_state.push_back(state);
      });

  nb::class_<ttb::GroundTruthMeasModelInfo>(m, "MeasModelInfo")
      .def(nb::init())
      .def_rw("detection_prob", &ttb::GroundTruthMeasModelInfo::detection_prob)
      .def_rw("clutter_rate", &ttb::GroundTruthMeasModelInfo::clutter_rate)
      .def_rw("meas_noise", &ttb::GroundTruthMeasModelInfo::meas_xy_dev);

  /// A collection of measurements from one data source, e.g., one sensor, at one specific time
  /// There are two versions because of the data type overload
  nb::class_<ttb::DataSourceContainer<double>>(m, "MeasurementContainer")
      .def(nb::init())
      .def("__repr__", &ttb::DataSourceContainer<double>::toString)
      .def_rw("_id", &ttb::DataSourceContainer<double>::_source_id)
      .def_rw("_measurements", &ttb::DataSourceContainer<double>::_measurements)
      .def_rw("_tracks", &ttb::DataSourceContainer<double>::_tracks)
      .def_rw("_prediction_info", &ttb::DataSourceContainer<double>::_predictionInfo)
      .def_rw("_timestamp", &ttb::DataSourceContainer<double>::_time)
      .def_rw("_sensor_info", &ttb::DataSourceContainer<double>::_sensorInfo)
      .def_rw("_ground_truth_info", &ttb::DataSourceContainer<double>::_ground_truth_info);

  nb::class_<ttb::DataSourceContainer<ttb::AD>>(m, "MeasurementContainerAD")
      .def(nb::init())
      .def("__repr__", &ttb::DataSourceContainer<ttb::AD>::toString)
      .def_rw("_id", &ttb::DataSourceContainer<ttb::AD>::_source_id)
      .def_rw("_measurements", &ttb::DataSourceContainer<ttb::AD>::_measurements)
      .def_rw("_tracks", &ttb::DataSourceContainer<ttb::AD>::_tracks)
      .def_rw("_prediction_info", &ttb::DataSourceContainer<ttb::AD>::_predictionInfo)
      .def_rw("_timestamp", &ttb::DataSourceContainer<ttb::AD>::_time)
      .def_rw("_sensor_info", &ttb::DataSourceContainer<ttb::AD>::_sensorInfo)
      .def_rw("_ground_truth_info", &ttb::DataSourceContainer<ttb::AD>::_ground_truth_info);

  m.def("create_measurement",
        [](ttb::TTBManager<double>* self, ttb::GaussianDistribution<double> gauss, Time time, Components comps) {
          return hpx::run_as_hpx_thread([&] { return create_meas_state(self, gauss, time, comps); });
        });

  m.def("create_measurement_ad",
        [](ttb::TTBManager<ttb::AD>* self, ttb::GaussianDistribution<ttb::AD> gauss, Time time, Components comps) {
          return hpx::run_as_hpx_thread([&] { return create_meas_state<ttb::AD>(self, gauss, time, comps); });
        });

  nb::class_<ttb::GaussianDistribution<double>>(m, "GaussianDistribution")
      .def(nb::init<ttb::Vector<double>, ttb::Matrix<double>>())
      .def("__repr__", &ttb::GaussianDistribution<double>::to_string)
      .def("set",
           nb::overload_cast<ttb::Vector<double>>(&ttb::GaussianDistribution<double>::pb_legacy_set),
           nb::arg("mean").noconvert())
      .def("set",
           nb::overload_cast<ttb::Matrix<double>>(&ttb::GaussianDistribution<double>::pb_legacy_set),
           nb::arg("cov").noconvert())
      .def("set",
           nb::overload_cast<ReferencePoint>(&ttb::GaussianDistribution<double>::pb_legacy_set),
           nb::arg("reference").noconvert());

  nb::class_<ttb::GaussianDistribution<ttb::AD>>(m, "GaussianDistributionAD")
      .def(
          "__init__",
          [](ttb::GaussianDistribution<ttb::AD>* self,
             ttb::Vector<double> const& mean,
             ttb::Matrix<double> const& cov) {
            new (self) ttb::GaussianDistribution<ttb::AD>(mean.cast<ttb::AD>().eval(), cov.cast<ttb::AD>().eval());
          },
          nb::arg("mean"),
          nb::arg("cov"))
      .def("__repr__", &ttb::GaussianDistribution<ttb::AD>::to_string)
      .def(
          "set",
          [](ttb::GaussianDistribution<ttb::AD>& self, ttb::Vector<double> const& mean) {
            self.pb_legacy_set(mean.cast<ttb::AD>().eval());
          },
          nb::arg("mean").noconvert())
      .def(
          "set",
          [](ttb::GaussianDistribution<ttb::AD>& self, ttb::Matrix<double> const& cov) {
            self.pb_legacy_set(cov.cast<ttb::AD>().eval());
          },
          nb::arg("cov").noconvert())
      .def("set",
           nb::overload_cast<ReferencePoint>(&ttb::GaussianDistribution<ttb::AD>::pb_legacy_set),
           nb::arg("reference").noconvert())
      .def("mean",
           [](ttb::GaussianDistribution<ttb::AD> const& self) -> ttb::Vector<double> {
             return ttb::to_val(self.mean());
           })
      .def("covariance", [](ttb::GaussianDistribution<ttb::AD> const& self) -> ttb::Matrix<double> {
        return ttb::to_val(self.covariance());
      });

  nb::class_<EgoDist>(m, "EgoMotionDistribution")
      .def(nb::init<ttb::GaussianDistribution<double>, Components>())
      .def("__repr__", &EgoDist::toString)
      .def("mean_of", &EgoDist::meanOf)
      //      .def_ro("_dist", &EgoDist::_dist, nb::rv_policy::reference)
      .def_ro("_comps", &EgoDist::_comps)
      .def_static("zero", &EgoDist::zero);

  nb::class_<ttb::State<double>>(m, "State")
      .def("__repr__", &ttb::State<double>::toString)
      .def("get_estimate",
           [](ttb::State<double> const& state) {
             auto [model_id, state_dist] = state.getEstimate();
             return std::tuple{ state._manager->getStateModel(model_id).state_comps(),
                                state_dist.mean(),
                                state_dist.covariance() };
           })
      .def("predict", &ttb::State<double>::predict)
      .def("isValid", &ttb::State<double>::isValid)
      .def_rw("_id", &ttb::State<double>::_id)
      .def_rw("_label", &ttb::State<double>::_label)
      .def_rw("_time", &ttb::State<double>::_time)
      .def_rw("_meta_data", &ttb::State<double>::_meta_data)
      .def_rw("_classification", &ttb::State<double>::_classification)
      .def_rw("_meas_classification", &ttb::State<double>::_meas_classification)
      .def_rw("_misc", &ttb::State<double>::_misc);

  nb::class_<ttb::State<ttb::AD>>(m, "StateAD")
      .def("__repr__", &ttb::State<ttb::AD>::toString)
      .def("get_estimate",
           [](ttb::State<ttb::AD> const& state) {
             auto [model_id, state_dist] = state.getEstimate();
             return std::tuple{ state._manager->getStateModel(model_id).state_comps(),
                                ttb::to_val(state_dist.mean()),
                                ttb::to_val(state_dist.covariance()) };
           })
      .def("isValid", &ttb::State<ttb::AD>::isValid)
      .def_rw("_id", &ttb::State<ttb::AD>::_id)
      .def_rw("_label", &ttb::State<ttb::AD>::_label)
      .def_rw("_time", &ttb::State<ttb::AD>::_time)
      .def_rw("_meta_data", &ttb::State<ttb::AD>::_meta_data)
      .def_rw("_classification", &ttb::State<ttb::AD>::_classification)
      .def_rw("_meas_classification", &ttb::State<ttb::AD>::_meas_classification)
      .def_rw("_misc", &ttb::State<ttb::AD>::_misc);

  nb::class_<ttb::BernoulliState<double>>(m, "BernoulliState")
      .def(nb::init<ttb::State<double> const&, double>())
      .def_rw("state", &ttb::BernoulliState<double>::state)
      .def_rw("existence_probability", &ttb::BernoulliState<double>::existence_probability);

  nb::class_<ttb::BernoulliState<ttb::AD>>(m, "BernoulliStateAD")
      .def(
          "__init__",
          [](ttb::BernoulliState<ttb::AD>* self, ttb::State<ttb::AD> const& s, double p) {
            new (self) ttb::BernoulliState<ttb::AD>(s, ttb::AD{ p });
          },
          nb::arg("state"),
          nb::arg("existence_probability"))
      .def_rw("state", &ttb::BernoulliState<ttb::AD>::state)
      .def_prop_rw(
          "existence_probability",
          [](ttb::BernoulliState<ttb::AD> const& self) -> double { return ttb::to_val(self.existence_probability); },
          [](ttb::BernoulliState<ttb::AD>& self, double p) { self.existence_probability = ttb::AD{ p }; });

  nb::class_<Classification>(m, "Classification")
      .def("__repr__", &Classification::toString)
      .def("get_prob", &Classification::getProb)
      .def("get_size", &Classification::getSize)
      .def("get_estimate", &Classification::getEstimate)
      .def_rw("probs", &Classification::m_probs);

  nb::class_<ttb::classification::StateClassification<double>>(m, "StateClassification")
      .def("__repr__", &ttb::classification::StateClassification<double>::toString)
      .def("get_prob", &ttb::classification::StateClassification<double>::getProb)
      .def("get_size", &ttb::classification::StateClassification<double>::getSize)
      .def("get_estimate", &ttb::classification::StateClassification<double>::getEstimate);

  nb::class_<SensorInfo>(m, "SensorInformation")
      .def(nb::init())
      .def("__repr__", &SensorInfo::toString)
      .def_rw("_sensor_pose", &SensorInfo::_to_sensor_cs, nb::rv_policy::reference_internal)
      .def_rw("_sensor_fov", &SensorInfo::_sensor_fov)
      .def("set_sensor_pose",
           [](SensorInfo& sensor_info, nb::ndarray<double, nb::shape<-1, 4>, nb::device::cpu> alignment) {
             if (alignment.shape(0) != 4 || alignment.shape(1) != 4)
             {
               std::cout << "Wrong shape of alignment! 4x4 array is needed" << std::endl;
               return;
             }
             Eigen::Matrix4d matrix(4, 4);
             for (size_t i = 0; i < alignment.shape(0); i++)
             {
               for (size_t j = 0; j < alignment.shape(1); j++)
               {
                 matrix(i, j) = alignment(i, j);
               }
             }
             SE3Trafo trafo(matrix);
             sensor_info._to_sensor_cs = trafo.inverse();
           });

  nb::class_<ttb::Region>(m, "Region")
      .def(nb::init())
      .def("__repr__", &ttb::Region::toString)
      .def("set_covered_area", [](ttb::Region& fov, nb::ndarray<double, nb::shape<-1, 2>, nb::device::cpu> polygon) {
        if (polygon.shape(0) < 3)
        {
          std::cout << "Wrong shape of polygon! More than two given points of polygon needed!" << std::endl;
          return;
        }
        std::vector<ttb::Point> covered_area_pt;
        for (size_t i = 0; i < polygon.shape(0); i++)
        {
          covered_area_pt.emplace_back(polygon(i, 0), polygon(i, 1));
        }
        ttb::Polygon::SinglePolygon covered_area_SC;
        boost::geometry::assign_points(covered_area_SC, covered_area_pt);
        boost::geometry::correct(covered_area_SC);
        ttb::Polygon::MultiPolygon multiPolygon{ covered_area_SC };
        ttb::Polygon poly{ ._comps = Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }),
                           .multiPolygon = std::move(multiPolygon) };
        fov._polygons.emplace(ttb::Components({ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }), std::move(poly));
      });

  nb::class_<Components>(m, "Components")
      .def(nb::init<std::vector<Component>>())
      .def("__repr__", &Components::toString)
      .def("index_of", nb::overload_cast<ttb::COMPONENT>(&ttb::Components::indexOf, nb::const_))
      .def("index_of", nb::overload_cast<Components const&>(&ttb::Components::indexOf, nb::const_))
      .def("intersection", &Components::intersection)
      .def("contains", &Components::contains)
      .def("diff", &Components::diff)
      .def("merge", &Components::merge)
      .def_ro("_comps", &Components::_comps);

  nb::enum_<Component>(m, "Component")
      .value("POS_X", Component::POS_X)
      .value("POS_Y", Component::POS_Y)
      .value("POS_Z", Component::POS_Z)

      .value("VEL_X", Component::VEL_X)
      .value("VEL_Y", Component::VEL_Y)
      .value("VEL_Z", Component::VEL_Z)
      .value("VEL_ABS", Component::VEL_ABS)

      .value("ACC_X", Component::ACC_X)
      .value("ACC_Y", Component::ACC_Y)
      .value("ACC_Z", Component::ACC_Z)
      .value("ACC_ABS", Component::ACC_ABS)

      .value("JERK_X", Component::JERK_X)
      .value("JERK_Y", Component::JERK_Y)
      .value("JERK_Z", Component::JERK_Z)
      .value("JERK_ABS", Component::JERK_ABS)

      .value("ROT_Z", Component::ROT_Z)
      .value("VEL_ROT_Z", Component::VEL_ROT_Z)
      .value("ACC_ROT_Z", Component::ACC_ROT_Z)

      .value("ELEVATION", Component::ELEVATION)
      .value("AZIMUTH", Component::AZIMUTH)

      .value("LENGTH", Component::LENGTH)
      .value("WIDTH", Component::WIDTH)
      .value("HEIGHT", Component::HEIGHT)

      .value("LENGTH_CHANGE", Component::LENGTH_CHANGE)
      .value("WIDTH_CHANGE", Component::WIDTH_CHANGE)
      .value("HEIGHT_CHANGE", Component::HEIGHT_CHANGE)
      .export_values();

  nb::enum_<ReferencePoint>(m, "ReferencePoint")
      .value("CENTER", ReferencePoint::CENTER)
      .value("FRONT", ReferencePoint::FRONT)
      .value("BACK", ReferencePoint::BACK)

      .value("LEFT", ReferencePoint::LEFT)
      .value("RIGHT", ReferencePoint::RIGHT)
      .value("TOP", ReferencePoint::TOP)
      .value("BOTTOM", ReferencePoint::BOTTOM)

      .value("FRONT_LEFT", ReferencePoint::FRONT_LEFT)
      .value("FRONT_RIGHT", ReferencePoint::FRONT_RIGHT)
      .value("BACK_LEFT", ReferencePoint::BACK_LEFT)
      .value("BACK_RIGHT", ReferencePoint::BACK_RIGHT)
      .export_values();

  nb::enum_<ClassLabel>(m, "ClassLabel")
      .value("UNKNOWN", ClassLabel::UNKNOWN)
      .value("PEDESTRIAN", ClassLabel::PEDESTRIAN)
      .value("BICYCLE", ClassLabel::BICYCLE)
      .value("MOTORBIKE", ClassLabel::MOTORBIKE)
      .value("CAR", ClassLabel::CAR)
      .value("VAN", ClassLabel::VAN)
      .value("TRUCK", ClassLabel::TRUCK)
      .value("BUS", ClassLabel::BUS)
      .value("TRAIN", ClassLabel::TRAIN)
      .value("CAR_UNION", ClassLabel::CAR_UNION)
      .value("TRUCK_UNION", ClassLabel::TRUCK_UNION)
      .value("BIKE_UNION", ClassLabel::BIKE_UNION)
      .value("ROAD_OBSTACLE", ClassLabel::ROAD_OBSTACLE)
      .value("ANIMAL", ClassLabel::ANIMAL)
      .value("TRAFFIC_LIGHT_GREEN", ClassLabel::TRAFFIC_LIGHT_GREEN)
      .value("TRAFFIC_LIGHT_YELLOW", ClassLabel::TRAFFIC_LIGHT_YELLOW)
      .value("TRAFFIC_LIGHT_RED", ClassLabel::TRAFFIC_LIGHT_RED)
      .value("TRAFFIC_LIGHT_RED_YELLOW", ClassLabel::TRAFFIC_LIGHT_RED_YELLOW)
      .value("TRAFFIC_LIGHT_NONE", ClassLabel::TRAFFIC_LIGHT_NONE)
      .value("TRAFFIC_SIGN_NONE", ClassLabel::TRAFFIC_SIGN_NONE)
      .value("NOT_CLASSIFIED", ClassLabel::NOT_CLASSIFIED)
      .export_values();
}
