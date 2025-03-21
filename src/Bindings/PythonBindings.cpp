#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/eigen/dense.h>

#include <figcone/configreader.h>
#include "tracking_lib/TTBTypes/TTBTypes.h"

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
#include <tracking_lib/Trackers/BaseTracker.h>
#include <tracking_lib/Measurements/SensorInformation.h>
#include <tracking_lib/TTBTypes/Components.h>
#include <tracking_lib/Misc/logger_setup.h>

namespace nb = nanobind;

using TTBManager = ttb::TTBManager;
using TTBParams = ttb::Params;
using MeasContainer = ttb::MeasurementContainer;
using Meas = ttb::Measurement;
using GaussianDist = ttb::GaussianDistribution;
using EgoDist = ttb::EgoMotionDistribution;
using State = ttb::State;
using SensorInfo = ttb::SensorInformation;
using Point = ttb::Polygon2D::Point;
using Polygon = ttb::Polygon2D::Polygon;
using Polygon2D = ttb::Polygon2D;
using FieldOfView = ttb::FieldOfView;
using Components = ttb::Components;
using Component = ttb::COMPONENT;
using Time = ttb::Time;
using ReferencePoint = ttb::REFERENCE_POINT;
using Classification = ttb::classification::MeasClassification;
using StateClassification = ttb::classification::StateClassification;
using ClassLabel = ttb::CLASS;
using SE3Trafo = ttb::SE3Trafo;

NB_MODULE(_tracking_lib_python_api, m)
{
  nb::class_<TTBManager>(m, "TTBManager")
      .def(nb::init<std::filesystem::path>())
      .def("reset", &TTBManager::reset)
      .def("cycle", nb::overload_cast<Time>(&TTBManager::cycle))
      .def("cycle", nb::overload_cast<Time, std::vector<MeasContainer>, bool>(&TTBManager::cycle))
      .def("cycle", nb::overload_cast<Time, std::vector<ttb::StateContainer>, bool>(&TTBManager::cycle))
      .def("getEstimate", nb::overload_cast<>(&TTBManager::getEstimate, nb::const_))
      .def("addData", nb::overload_cast<MeasContainer, Time>(&TTBManager::addData));

  m.def("set_log_level", [](std::string const& log_level) {
    ttb::_setLogLevel(log_level);
    ttb::_setStreamName("");
    ttb::_setPrefix("");
  });

  nb::class_<MeasContainer>(m, "MeasurementContainer")
      .def(nb::init())
      .def("__repr__", &MeasContainer::toString)
      .def_rw("_id", &MeasContainer::_id)
      .def_rw("_measurements", &MeasContainer::_data)
      .def_rw("_ego_motion", &MeasContainer::_egoMotion)
      .def_rw("_timestamp", &MeasContainer::_time)
      .def_rw("_sensor_info", &MeasContainer::_sensorInfo);

  nb::class_<Meas>(m, "Measurement")
      .def(nb::init<GaussianDist, Time, Components>())
      .def("__repr__", &Meas::toString)
      .def_rw("_id", &Meas::_id)
      .def_rw("_objectId", &Meas::_objectId)
      .def_rw("_time", &Meas::_time)
      .def_rw("_meas_comps", &Meas::_meas_comps)
      .def_rw("_ref_point_measured", &Meas::_ref_point_measured)
      .def_rw("_classification", &Meas::_classification);

  nb::class_<GaussianDist>(m, "GaussianDistribution")
      .def(nb::init())
      .def("__repr__", &GaussianDist::toString)
      .def("set", nb::overload_cast<ttb::Vector>(&GaussianDist::set), nb::arg("mean").noconvert())
      .def("set", nb::overload_cast<ttb::Matrix>(&GaussianDist::set), nb::arg("cov").noconvert())
      .def("set", nb::overload_cast<double>(&GaussianDist::set), nb::arg("weight").noconvert())
      .def("set", nb::overload_cast<ReferencePoint>(&GaussianDist::set), nb::arg("reference").noconvert());

  nb::class_<EgoDist>(m, "EgoMotionDistribution")
      .def(nb::init<GaussianDist, Components>())
      .def("__repr__", &EgoDist::toString)
      .def("mean_of", &EgoDist::meanOf)
      //      .def_ro("_dist", &EgoDist::_dist, nb::rv_policy::reference)
      .def_ro("_comps", &EgoDist::_comps)
      .def_static("zero", &EgoDist::zero);

  nb::class_<State>(m, "State")
      .def("__repr__", &State::toString)
      .def("get_estimate",
           [](const State& state) {
             auto [model_id, state_dist] = state.getEstimate();
             return std::tuple{ state._manager->getStateModel(model_id).state_comps(),
                                state_dist->mean(),
                                state_dist->covariance() };
           })
      .def("predict", &State::predict)
      .def("isValid", &State::isValid)
      .def_rw("_id", &State::_id)
      .def_rw("_label", &State::_label)
      .def_rw("_time", &State::_time)
      .def_rw("_existenceProbability", &State::_existenceProbability)
      .def_rw("_meta_data", &State::_meta_data)
      //.def_rw("_lastAssociatedMeasurement", &State::_meta_data._lastAssociatedMeasurement)
      //.def_rw("_timeOfLastAssociation", &State::_meta_data._timeOfLastAssociation)
      //.def_rw("_durationSinceLastAssociation", &State::_meta_data._durationSinceLastAssociation)
      .def_rw("_classification", &State::_classification)
      .def_rw("_misc", &State::_misc);

  nb::class_<Classification>(m, "Classification")
      .def("__repr__", &Classification::toString)
      .def("get_prob", &Classification::getProb)
      .def("get_size", &Classification::getSize)
      .def("get_estimate", &Classification::getEstimate)
      .def_rw("probs", &Classification::m_probs);

  nb::class_<StateClassification>(m, "StateClassification")
      .def("__repr__", &StateClassification::toString)
      .def("get_prob", &StateClassification::getProb)
      .def("get_size", &StateClassification::getSize)
      .def("get_estimate", &StateClassification::getEstimate);

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

  nb::class_<FieldOfView>(m, "FieldOfView")
      .def(nb::init())
      .def("__repr__", &FieldOfView::toString)
      .def("set_covered_area", [](FieldOfView& fov, nb::ndarray<double, nb::shape<-1, 2>, nb::device::cpu> polygon) {
        if (polygon.shape(0) < 3)
        {
          std::cout << "Wrong shape of polygon! More than two given points of polygon needed!" << std::endl;
          return;
        }
        std::vector<ttb::Polygon2D::Point> covered_area_pt;
        for (size_t i = 0; i < polygon.shape(0); i++)
        {
          covered_area_pt.emplace_back(polygon(i, 0), polygon(i, 1));
        }
        ttb::Polygon2D::Polygon covered_area_SC;
        boost::geometry::assign_points(covered_area_SC, covered_area_pt);
        boost::geometry::correct(covered_area_SC);
        ttb::Polygon2D::MultiPolygon multiPolygon{ covered_area_SC };
        ttb::Polygon2D poly{ .d1 = ttb::COMPONENT::POS_X,
                             .d2 = ttb::COMPONENT::POS_Y,
                             .multiPolygon = std::move(multiPolygon) };
        fov._polygons[{ ttb::COMPONENT::POS_X, ttb::COMPONENT::POS_Y }] = std::move(poly);
      });

  nb::class_<Components>(m, "Components")
      .def(nb::init<std::vector<Component>>())
      .def("__repr__", &Components::toString)
      .def("index_of", nb::overload_cast<ttb::COMPONENT>(&ttb::Components::indexOf, nb::const_))
      .def("index_of", nb::overload_cast<std::vector<ttb::COMPONENT> const&>(&ttb::Components::indexOf, nb::const_))
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
