#!/usr/bin/env python
# %%
"""
=========================================================================
Stone Soup Bridge: Using Tracking Toolbox (TTB) in Stone Soup Environment
=========================================================================
"""
import datetime
import timeit
from datetime import timedelta

import matplotlib
import numpy as np

import pathlib
import os

import stonesoup
from matplotlib.dates import num2date, SecondLocator, MicrosecondLocator

from stonesoup.models.clutter import ClutterModel
from stonesoup.models.measurement import MeasurementModel
# from stonesoup.models.measurement.linear import  LinearGaussian
from stonesoup.models.transition.linear import (ConstantVelocity,
                                                CombinedLinearGaussianTransitionModel, ConstantAcceleration, RandomWalk,
                                                OrnsteinUhlenbeck, Singer, KnownTurnRate)
from stonesoup.models.transition.base import CombinedGaussianTransitionModel
from stonesoup.models.transition.nonlinear import ConstantTurn

from stonesoup.types.track import Track
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.sensor.sensor import SimpleSensor
from stonesoup.base import Property

from tracking_lib import _tracking_lib_python_api as _api
from tracking_lib import TTB
from tracking_lib._tracking_lib_python_api import Component
from tracking_lib.utils import get_estimate_from_state

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

"""
==============================================
TTB SENSOR CLASS FROM STONE SOUP SIMPLE SENSOR
==============================================
"""
class Sensor_TTB(SimpleSensor):
    """A simple passive sensor that generates measurements of targets, using a
        :class:`~.MeasurementModel` model, relative to its position.

        """

    measurement_model:  MeasurementModel= Property(
        doc="Measurement model that creates Detections and Clutter")

    detection_probability: float= Property(
        default=0.8,
        doc="Probability that a detection will be received"
    )

    probability_estimation: bool= Property(
        default=False
    )



    def is_detectable(self, state, measurement_model=None) -> bool:
        return True if np.random.rand() <= self.detection_probability else False

    def is_clutter_detectable(self, state) -> bool:
        return True

# %%
"""
==============
GOSPA PLOTTING
==============
"""

def plot_gospa(gospa_metrics, gospa_gen_name: str | list[str]):

    if type(gospa_gen_name) != list:
        gospa_gen_name = [gospa_gen_name]
    for gen_name in gospa_gen_name:
        plt.figure()
        gospa_timesteps = []
        gospa_distances = []
        gospa_localisation = []
        gospa_missed = []
        gospa_false = []
        gospa_switching = []

        gospa_time_range_metrics = gospa_metrics[gen_name]['GOSPA Metrics']
        for single_time_metric in gospa_time_range_metrics.value:
            gospa_timesteps.append(single_time_metric.timestamp)
            gospa_distances.append(single_time_metric.value['distance'])
            gospa_localisation.append(single_time_metric.value['localisation'])
            gospa_missed.append(single_time_metric.value['missed'])
            gospa_false.append(single_time_metric.value['false'])
            gospa_switching.append(single_time_metric.value['switching'])

        ax1 = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=2)
        ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
        ax3 = plt.subplot2grid((3, 4), (2, 1), colspan=2)
        ax4 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
        ax5 = plt.subplot2grid((3, 4), (1, 0), colspan=2)


        def format_date(a, b):
            t=num2date(a)
            ms = str(t.microsecond)[:1]
            res = f"{t.second}.{ms}"
            return res

        plt.suptitle(f"GOSPA Metrics {gen_name}")

        ax1.set_title("Distance")
        ax1.xaxis.set_major_locator(SecondLocator(interval=1))
        ax1.xaxis.set_major_formatter(FuncFormatter(format_date))
        ax1.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax1.set_xlabel("t in seconds")
        ax1.plot(gospa_timesteps, gospa_distances)
        # plt.xticks(np.arange(start=start_time, stop=start_time + timedelta(seconds=stop), step=timedelta(seconds=1)))

        ax2.set_title("Localisation")
        ax2.xaxis.set_major_locator(SecondLocator(interval=1))
        ax2.xaxis.set_major_formatter(FuncFormatter(format_date))
        ax2.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax2.set_xlabel("t in seconds")
        ax2.plot(gospa_timesteps, gospa_localisation)

        ax3.set_title("Switching")
        ax3.xaxis.set_major_locator(SecondLocator(interval=1))
        ax3.xaxis.set_major_formatter(FuncFormatter(format_date))
        ax3.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax3.set_xlabel("t in seconds")
        ax3.plot(gospa_timesteps, gospa_switching)

        ax4.set_title("Missed")
        ax4.xaxis.set_major_locator(SecondLocator(interval=1))
        ax4.xaxis.set_major_formatter(FuncFormatter(format_date))
        ax4.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax4.set_xlabel("t in seconds")
        ax4.plot(gospa_timesteps, gospa_missed)

        ax5.set_title("False")
        ax5.xaxis.set_major_locator(SecondLocator(interval=1))
        ax5.xaxis.set_major_formatter(FuncFormatter(format_date))
        ax5.xaxis.set_minor_locator(MicrosecondLocator(100000))
        ax5.set_xlabel("t in seconds")
        ax5.plot(gospa_timesteps, gospa_false)

        plt.tight_layout(pad=0, h_pad=-1.4)

    return plt

def plot_all_gospas(gospa_metrics, gospa_gen_name: str | list[str]):
    if type(gospa_gen_name) != list:
        gospa_gen_name = [gospa_gen_name]

    plt.figure()
    gospa_metrics_dict = {
        "gospa_timesteps" : [],
        "gospa_distances" : [],
        "gospa_localisation" : [],
        "gospa_missed" : [],
        "gospa_false" : [],
        "gospa_switching" : [],
        "gospa_gen_names": gospa_gen_name
    }


    for gen_name in gospa_gen_name:

        gospa_distances = []
        gospa_localisation = []
        gospa_missed = []
        gospa_false = []
        gospa_switching = []

        c = gospa_metrics[gen_name]['GOSPA Metrics'].generator.c
        p = gospa_metrics[gen_name]['GOSPA Metrics'].generator.p
        alpha = 2
        cost_factor = c**p / alpha

        gospa_time_range_metrics = gospa_metrics[gen_name]['GOSPA Metrics']
        for single_time_metric in gospa_time_range_metrics.value:
            gospa_metrics_dict["gospa_timesteps"].append(single_time_metric.timestamp) if single_time_metric.timestamp not in gospa_metrics_dict["gospa_timesteps"] else None
            gospa_distances.append(single_time_metric.value['distance'])
            gospa_localisation.append(single_time_metric.value['localisation'])
            gospa_missed.append(single_time_metric.value['missed']/cost_factor)
            gospa_false.append(single_time_metric.value['false']/cost_factor)
            gospa_switching.append(single_time_metric.value['switching'])

        gospa_metrics_dict["gospa_distances"].append(gospa_distances)
        gospa_metrics_dict["gospa_localisation"].append(gospa_localisation)
        gospa_metrics_dict["gospa_missed"].append(gospa_missed)
        gospa_metrics_dict["gospa_false"].append(gospa_false)
        gospa_metrics_dict["gospa_switching"].append(gospa_switching)

    ax1 = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((3, 4), (2, 1), colspan=2)
    ax4 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    ax5 = plt.subplot2grid((3, 4), (1, 0), colspan=2)


    def format_date(a, b):
        t=num2date(a)
        ms = str(t.microsecond)[:1]
        res = f"{t.second}.{ms}"
        return res

    plt.suptitle(f"All GOSPA Metrics")

    ax1.set_title("Distance")
    ax1.xaxis.set_major_locator(SecondLocator(interval=1))
    ax1.xaxis.set_major_formatter(FuncFormatter(format_date))
    ax1.xaxis.set_minor_locator(MicrosecondLocator(100000))
    ax1.set_xlabel("t in seconds")
    for idx, data in enumerate(gospa_metrics_dict["gospa_distances"]):
        ax1.plot(gospa_metrics_dict["gospa_timesteps"], data, label=gospa_metrics_dict["gospa_gen_names"][idx])
    ax1.legend(loc="best")

    ax2.set_title("Localisation")
    ax2.xaxis.set_major_locator(SecondLocator(interval=1))
    ax2.xaxis.set_major_formatter(FuncFormatter(format_date))
    ax2.xaxis.set_minor_locator(MicrosecondLocator(100000))
    ax2.set_xlabel("t in seconds")
    for idx, data in enumerate(gospa_metrics_dict["gospa_localisation"]):
        ax2.plot(gospa_metrics_dict["gospa_timesteps"], data, label=gospa_metrics_dict["gospa_gen_names"][idx])
    ax2.legend(loc="best")

    ax3.set_title("Switching")
    ax3.xaxis.set_major_locator(SecondLocator(interval=1))
    ax3.xaxis.set_major_formatter(FuncFormatter(format_date))
    ax3.xaxis.set_minor_locator(MicrosecondLocator(100000))
    ax3.set_xlabel("t in seconds")
    for idx, data in enumerate(gospa_metrics_dict["gospa_switching"]):
        ax3.plot(gospa_metrics_dict["gospa_timesteps"], data, label=gospa_metrics_dict["gospa_gen_names"][idx])
    ax3.legend(loc="best")

    ax4.set_title("Missed")
    ax4.xaxis.set_major_locator(SecondLocator(interval=1))
    ax4.xaxis.set_major_formatter(FuncFormatter(format_date))
    ax4.xaxis.set_minor_locator(MicrosecondLocator(100000))
    ax4.set_xlabel("t in seconds")
    ax4.set_ylabel("Number of missed tracks")
    ax4.yaxis.set_major_locator(MultipleLocator(1))
    for idx, data in enumerate(gospa_metrics_dict["gospa_missed"]):
        ax4.plot(gospa_metrics_dict["gospa_timesteps"], data, label=gospa_metrics_dict["gospa_gen_names"][idx])
    ax4.legend(loc="best")

    ax5.set_title("False")
    ax5.xaxis.set_major_locator(SecondLocator(interval=1))
    ax5.xaxis.set_major_formatter(FuncFormatter(format_date))
    ax5.xaxis.set_minor_locator(MicrosecondLocator(100000))
    ax5.set_xlabel("t in seconds")
    ax5.set_ylabel("Number of false tracks")
    ax5.yaxis.set_major_locator(MultipleLocator(1))
    for idx, data in enumerate(gospa_metrics_dict["gospa_false"]):
        ax5.plot(gospa_metrics_dict["gospa_timesteps"], data, label=gospa_metrics_dict["gospa_gen_names"][idx])
    ax5.legend(loc="best")

    plt.tight_layout(pad=0, h_pad=-1.4, w_pad=-2)


    return plt
# %%
"""
==============================
MAPPING FROM STONE SOUP TO TTB
==============================
"""
def get_components_via_number_of_models(number_of_models):
    position_components = ["X", "Y", "Z"]
    return position_components[:number_of_models]


def get_components_of_stonesoup_models(transition_model, measurement_model):
    mapping = measurement_model.mapping
    components_list = []

    # check if transition_model is a single model from mapping dict
    if type(transition_model) in _stonesoup_model_mapping_to_component_dict:
        components_list = _stonesoup_model_mapping_to_component_dict[type(transition_model)]

    # check if transition_model is a combined model
    elif hasattr(transition_model, "model_list"):

        number_of_models = len(transition_model.model_list)
        components = get_components_via_number_of_models(number_of_models)

        if isinstance(transition_model, CombinedGaussianTransitionModel):
            for i in range(number_of_models):
                new_components = _stonesoup_model_mapping_to_component_dict[type(transition_model.model_list[i])].copy() # copy is important!

                for j in range(len(new_components)):
                    if "X" in new_components[j]:
                        new_components[j] = new_components[j].replace("X", components[i])

                components_list.extend(new_components)

    components_eval = [comp for comp in components_list if components_list.index(comp) in mapping]
    components_state = tuple([eval(comp) for comp in components_list])
    # components_eval.sort()
    components = tuple([eval(comp) for comp in components_eval])
    # print("Components", components)
    return components, components_state

# Attention! lists are mutable, they have to be copied!
_stonesoup_model_mapping_to_component_dict = {
    RandomWalk : ["_api.Component.POS_X"],
    ConstantVelocity : ["_api.Component.POS_X", "_api.Component.VEL_X"],
    ConstantAcceleration : ["_api.Component.POS_X", "_api.Component.VEL_X", "_api.Component.ACC_X"],
    OrnsteinUhlenbeck : ["_api.Component.POS_X", "_api.Component.VEL_X"],
    Singer : ["_api.Component.POS_X", "_api.Component.VEL_X", "_api.Component.ACC_X"],
    KnownTurnRate : ["_api.Component.POS_X", "_api.Component.VEL_X", "_api.Component.POS_Y", "_api.Component.VEL_Y"],
    ConstantTurn : ["_api.Component.POS_X", "_api.Component.VEL_X", "_api.Component.POS_Y", "_api.Component.VEL_Y", "_api.Component.VEL_ROT_Z"]
}


"""
============
TTB Tracking
============
"""

_config_defaults = {
    "<ID_SIM0>" : "sim0",
    "<SIM0_CLUTTER_RATE>" : 2,
    "<SIM0_DETECTION_PROB>" : 0.99,
    "<SIM0_PROB_ESTIMATION>" : "false",
    "<USE_STATE_MODELS>" : 0, # CV
    "<USE_MEAS_MODELS>" : "- sim0",
    "<CV_EXTENT>" : "NONE",
    "<CTP_EXTENT>" : "RECTANGULAR",
    "<CP_EXTENT>" : "NONE",
    "<CTRV_EXTENT>" : "RECTANGULAR",
    "<CV_NOISE_ACC_X>" : 0.2,
    "<CV_NOISE_ACC_Y>" : 0.2,
    "<CV_NOISE_VEL_Z>" : 0.001,
    "<CTRV_NOISE_VEL_Z>" : 0.1,
    "<CTRV_NOISE_ACC_ABS>" : 5,
    "<CTRV_NOISE_ACC_ROT_Z>" : 0.5,
    "<CTP_NOISE_VEL_X>" : 1,
    "<CTP_NOISE_VEL_Y>" : 1,
    "<CTP_NOISE_VEL_Z>" : 1,
    "<CP_NOISE_VEL_X>" : 5,
    "<CP_NOISE_VEL_Y>" : 5,
    "<CP_NOISE_VEL_Z>" : 1
}

# %%
# deleting config_file.yaml
def delete_config_file():
    os.remove("config_file.yaml")

def create_config_file_and_ttb_manager(transition_model: stonesoup.models.transition,
                       sensors: list[Sensor_TTB],
                       transition_model_config = "CV",
                       extent = None,
                       delete_config_file_after_use = False):

    sensor_ids=[]
    for i in range(len(sensors)):
        sensor_ids.append("sim"+str(i))

    config_dict = _config_defaults.copy()

    # configure state model(s)
    relevant_keys = [key for key in _config_defaults.keys() if transition_model_config in key]
    if not relevant_keys:
        raise ValueError("Invalid transition model! Choose between 'CV' (default), 'CTRV', 'CP', 'CTP'")
    # print(relevant_keys)
    if extent:
        config_dict[relevant_keys[0]] = extent

    if hasattr(transition_model, "model_list"):
        for i in range(len(transition_model.model_list)):
            config_dict[relevant_keys[i+1]] = transition_model.model_list[i].noise_diff_coeff
    # elif hasattr(transition_model, "noise_diff_coeff"):
    #     for i in range(1, transition_model.ndim + 1):
    #         config_dict[relevant_keys[i]] = transition_model.noise_diff_coeff
    else: raise ValueError("Invalid or incompatible stone soup transition model chosen!")

    # TODO other transition model configs

    relevant_keys = [key for key in _config_defaults.keys() if sensor_ids[0].upper() in key] * len(sensors)
    # print("Meas Model Keys", relevant_keys)
    number_of_keys_per_sensor = int(len(relevant_keys)/len(sensors))
    # print(number_of_keys_per_sensor)

    for s in range(len(sensors)):
        for key in range(number_of_keys_per_sensor):
            old = relevant_keys[number_of_keys_per_sensor*s + key]
            # print("old:", old)
            new = old.replace("0", str(s))
            # print("new:", new)
            relevant_keys[number_of_keys_per_sensor*s + key] = new



    # configure sensor(s)
    for s in range(len(sensors)):
        for key in range(number_of_keys_per_sensor):
            if key % number_of_keys_per_sensor == 0:
                config_dict[relevant_keys[number_of_keys_per_sensor*s + key]] = sensor_ids[s]
            elif key % number_of_keys_per_sensor == 1:
                config_dict[relevant_keys[number_of_keys_per_sensor * s + key]] = sensors[s].clutter_model.clutter_rate
            elif key % number_of_keys_per_sensor == 2:
                config_dict[relevant_keys[number_of_keys_per_sensor * s + key]] = sensors[s].detection_probability
            elif key % number_of_keys_per_sensor == 3:
                config_dict[relevant_keys[number_of_keys_per_sensor * s + key]] = str.lower(str(sensors[s].probability_estimation))


    # read yaml template into string variable
    with open(pathlib.Path(__file__).parent.resolve() / "config_template.yaml") as yaml_file:
        yaml_string = yaml_file.readlines()

    # extract meas_models part
    meas_models_string = ""
    extract_string = False
    all_string = ""
    for line in yaml_string:
        all_string += line
        if "###EXTRACT_START###" in line:
            extract_string = True
            continue
        if "###EXTRACT_STOP###" in line:
            extract_string = False
            # line_to_insert = yaml_string.index(line)-1
        if extract_string:
            meas_models_string += line

    meas_models = ""
    for s in range(len(sensors)):
        temp_string = meas_models_string.replace("SIM0", "SIM"+str(s))
        meas_models+=temp_string
    # print(meas_models)

    all_string = all_string.replace(meas_models_string, meas_models)
    all_string = all_string.splitlines(keepends=True)
    # print(all_string)

    for index in range(len(all_string)):
        start = all_string[index].find("<")
        end = all_string[index].find(">")
        if start != end:
            parameter = all_string[index][start:end+1]
            all_string[index] = all_string[index].replace(parameter, str(config_dict[parameter]))


    with open(pathlib.Path(__file__).parent.resolve() /"config_file.yaml", "w") as config_file:
        for line in all_string:
            config_file.write(line)

    # create TTB Manager
    ttb = TTB(pathlib.Path(pathlib.Path(__file__).parent.resolve() /"config_file.yaml"))

    if delete_config_file_after_use:
        delete_config_file()

    return ttb


def get_fov_sensor(sensor: SimpleSensor ,
                   position_mapping = (0, 1)):
    try:
        x_values = sensor.clutter_model.dist_params[position_mapping[0]]
        y_values = sensor.clutter_model.dist_params[position_mapping[1]]
    except IndexError:
        if len(sensor.clutter_model.dist_params) == len(position_mapping):
            x_values = sensor.clutter_model.dist_params[0]
            y_values = sensor.clutter_model.dist_params[1]
        else:
            raise

    x_buffer = (max(x_values) - min(x_values))*0.05
    y_buffer = (max(y_values) - min(y_values))*0.05
    fov_sensor = np.array([[max(x_values)+x_buffer, min(y_values)-y_buffer],
                           [max(x_values)+x_buffer, max(y_values)+y_buffer],
                           [min(x_values)-x_buffer, max(y_values)+y_buffer],
                           [min(x_values)-x_buffer, min(y_values)-y_buffer]])
    return fov_sensor


# %%

"""
====================================
TRANSFORMING DATA: STONE SOUP -> TTB
====================================
"""
# Stone Soup Detections (Measurements) to TTB MeasurementContainer

# Transforms list of Stone Soup Detections of ONE TIMESTEP to a Measurement Container for TTB
def transform_dets_to_meas_container(detections: set[stonesoup.types.detection], time: datetime.datetime,
                                     sensor_id: str, fov_polygon: np.ndarray, state_model: stonesoup.models.transition,
                                     sensor_model: stonesoup.models.measurement, sensor_alignment = None):
    container = _api.MeasurementContainer()
    container._id = sensor_id
    container._timestamp = time
    sensor_info = _api.SensorInformation()
    sensor_info._sensor_fov = _api.FieldOfView()
    sensor_info._sensor_fov.set_covered_area(fov_polygon)
    container._sensor_info = sensor_info
    container._measurements = [transform_det_to_meas(detection, time, state_model, sensor_model) for detection in detections]
    return container

# Transforms one Stone Soup Detection to a TTB Measurement
def transform_det_to_meas(detection: stonesoup.types.detection.Detection, time: datetime.datetime,
                          state_model: stonesoup.models.transition, sensor_model: stonesoup.models.measurement):

    comp, _ = get_components_of_stonesoup_models(state_model, sensor_model)
    components = _api.Components(comp)

    distribution = _api.GaussianDistribution()
    mapping = map_stonesoup_to_ttb_components(comp)
    # print("Mapping StSo to TTB:", mapping)
    mean_tmp = detection.state_vector.ravel()
    mean = np.choose(mapping, mean_tmp)
    cov = np.array(detection.measurement_model.noise_covar)
    distribution.set(mean.astype(np.float64))
    distribution.set(np.asfortranarray(cov.astype(np.float64)))
    distribution.set(1.0)

    measurement = _api.Measurement(distribution, time, components)
    measurement._ref_point_measured = True # is center by default, other values are also possible but not implemented here
    measurement._classification.probs = {_api.ClassLabel.UNKNOWN : 0.95}
    return measurement

# %%
def map_stonesoup_to_ttb_components(components):
    mapping = []
    for comp in components:
        mapping.append(_stonesoup_component_index_mapping[comp])
    # print(mapping)
    map_array = np.argsort(np.array(mapping))
    map_array = [int(x) for x in map_array]
    return map_array

_stonesoup_component_index_mapping = {
    Component.POS_X : 0,
    Component.POS_Y : 1,
    Component.POS_Z : 2,
    Component.VEL_X : 3,
    Component.VEL_Y : 4,
    Component.VEL_Z : 5
}

def map_ttb_components_to_stonesoup(components):
    mapping = []
    for idx, comp in enumerate(components):
        mapping.append(_stonesoup_component_index_mapping[comp])
    map_array = np.array(mapping)
    return map_array


# %%
"""
====================================
TRANSFORMING DATA: TTB -> STONE SOUP
====================================
"""

def transform_estimations_to_tracks(estimations: list[dict],
                                    transition_model,
                                    measurement_model) -> set[Track]:
    tracks_ttb = set()
    _, mapping_comp = get_components_of_stonesoup_models(transition_model, measurement_model)
    new_order = map_ttb_components_to_stonesoup(mapping_comp)
    for estimation in estimations:
        if len(estimation['estimation']) > 0:
            estimation_list = estimation['estimation']
            for est in estimation_list:
                state_vector = StateVector(est.mean[new_order])
                covar_matrix = CovarianceMatrix(est.cov[np.ix_(new_order, new_order)])
                track = GaussianState(state_vector=state_vector,
                                      covar=covar_matrix,
                                      timestamp=estimation['time'])
                is_new = True
                for t in tracks_ttb:
                    if est.label == t.id:
                        is_new = False
                        t.append(track)

                if is_new:
                    new_track = Track(id=est.label)
                    new_track.append(track)
                    tracks_ttb.add(new_track)
    return tracks_ttb


"""
====================================
FUNCTION TO REALIZE TRACKING VIA TTB
====================================
"""
# %%
def track_via_ttb(transition_model,
                  measurement_model,
                  timesteps: list[datetime],
                  all_measurements: list[list[set]],
                  sensors: list[Sensor_TTB],
                  position_mapping: tuple):

    sensor_ids = []
    for i in range(len(sensors)):
        sensor_ids.append("sim" + str(i))

    # Create TTB Manager and Config File dynamically out of transition and measurement models
    ttb = create_config_file_and_ttb_manager(transition_model, sensors)
    # ttb = TTB(pathlib.Path(pathlib.Path(__file__).parent.resolve() /"config_file.yaml"))
    _api.set_log_level("Error")
    estimation_results = []
    fovs = []
    for i in range(len(sensors)):
        fovs.append(get_fov_sensor(sensors[i], position_mapping))
    rt = []
    for idx, measurements in enumerate(all_measurements):
        # ^(idx, measurements)
        sim_time = timesteps[idx]
        meas = []
        for sensor_id in range(len(sensors)):
            meas.append(transform_dets_to_meas_container(measurements[sensor_id], sim_time,
                                                        sensor_id=sensor_ids[sensor_id],
                                                        fov_polygon=fovs[sensor_id],
                                                        state_model=transition_model,
                                                        sensor_model=measurement_model))
        start = timeit.default_timer()
        ttb.cycle_tracking(sim_time, meas)
        rt.append(timeit.default_timer()-start)
        est = dict(time=sim_time,
                   estimation=[get_estimate_from_state(obj, ttb.manager) for obj in ttb.getEstimate()])

        estimation_results.append(est)


    estimations_ttb = transform_estimations_to_tracks(estimations=estimation_results,
                                                      transition_model=transition_model,
                                                      measurement_model=measurement_model)
    return estimations_ttb, np.mean(rt)