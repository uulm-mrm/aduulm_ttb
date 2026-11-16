import datetime
from typing import List, Optional

import numpy as np
from dataclasses import dataclass

from tracking_lib import _tracking_lib_python_api as _api

@dataclass(slots=True)
class Detection:
    mean: np.array # n x 1 vector
    cov: np.array # n x n matrix
    components: tuple[_api.Component]  # Measured components of the sensor. The order must be the same as the order in mean and cov., e.g., (_api.Component.POS_X,_api.Component.POS_Y).
    class_label: _api.ClassLabel = _api.ClassLabel.UNKNOWN
    classification_prob: float = 0.95
    reference_point: _api.ReferencePoint = _api.ReferencePoint.CENTER
    time: datetime.datetime = datetime.datetime.now()

@dataclass(slots=True)
class Estimation:
    label: int
    mean: np.array
    cov: np.array
    components: tuple[_api.Component]
    existence_probability: float
    class_label: _api.ClassLabel = _api.ClassLabel.UNKNOWN
    time: datetime.datetime = datetime.datetime.now()


def create_sensor_information(fov_polygon, sensor_alignment):
    sensor_info = _api.SensorInformation()
    if fov_polygon is not None:
        sensor_info._sensor_fov = _api.Region()
        sensor_info._sensor_fov.set_covered_area(fov_polygon)
    if sensor_alignment is not None:
        sensor_info.set_sensor_pose(sensor_alignment)
    return sensor_info

def to_gt_container(scenario, time_step, time, manager, gt_tracks):
    track_list = []
    for mean, label in gt_tracks:

        np_mean = np.zeros((2))
        cov = 1e-7*np.eye(2)

        for idx in range(0,2):
            np_mean[idx] = mean[idx]
        if isinstance(manager, _api.TTBManagerAD):
            dist = _api.GaussianDistributionAD(np_mean.astype(np.float64), np.asfortranarray(cov.astype(np.float64)))
            state = _api.create_measurement_ad(manager, dist,
                                            time, _api.Components([_api.Component.POS_X, _api.Component.POS_Y]))
        else:
            dist = _api.GaussianDistribution(np_mean.astype(np.float64), np.asfortranarray(cov.astype(np.float64)))
            state = _api.create_measurement(manager, dist,
                                               time, _api.Components([_api.Component.POS_X, _api.Component.POS_Y]))
        state._label = label
        if isinstance(manager, _api.TTBManagerAD):
            bernoulli = _api.BernoulliStateAD(state, 1)
        else:
            bernoulli = _api.BernoulliState(state, 1)
        track_list.append(bernoulli)
    if isinstance(manager, _api.TTBManagerAD):
        container = _api.MeasurementContainerAD()
    else:
        container = _api.MeasurementContainer()
    container._id = "GROUND_TRUTH"
    container._timestamp = time
    container._tracks = track_list
    if 'persistence_prob' in scenario and 'birth' in scenario and 'state_model_noise' in scenario:
        truth_info = _api.GroundTruthInfo()
        state_info = _api.StateModelInfo()
        state_info.persistence_prob = scenario["persistence_prob"](time_step)
        state_info.noise = scenario['state_model_noise'](time_step)
        truth_info.state_model_info = state_info
        birth_model_info = _api.BirthModelInfo()
        birth_model_info.prob = scenario["birth"](time_step)[0]
        poss_means = scenario["birth"](time_step)[1]
        var = scenario["birth"](time_step)[2]
        for poss_mean in poss_means:
            birth_state = {}
            birth_state[_api.Component.POS_X] = (poss_mean[0], var[0])
            birth_state[_api.Component.VEL_X] = (poss_mean[1], var[1])
            birth_state[_api.Component.POS_Y] = (poss_mean[2], var[2])
            birth_state[_api.Component.VEL_Y] = (poss_mean[3], var[3])
            birth_state[_api.Component.POS_Z] = (0, 1)
            _api.add_gt_birth_pos(birth_model_info, birth_state)
        truth_info.birth_model_info = birth_model_info
        container._ground_truth_info = truth_info
    return container


def dets_to_meas_container(dets: List[Detection], time: datetime.datetime, sensor_id: str, detection_prob: float, clutter_rate: float,
			   fov_polygon: Optional[np.ndarray]=None, sensor_alignment: Optional[np.ndarray]=None,  manager: _api.TTBManager=None):
    container = _det_to_meas_container_impl(dets, time, sensor_id, fov_polygon, sensor_alignment, manager)
    meas_model_info = _api.MeasModelInfo()
    meas_model_info.detection_prob = 0.666
    meas_model_info.clutter_rate = clutter_rate
    truth_info = _api.GroundTruthInfo()
    truth_info.meas_model_info = meas_model_info
    container._ground_truth_info = truth_info
    return container

def _det_to_meas_container_impl(dets: List[Detection], time: datetime.datetime, model_id: str,
				fov_polygon: Optional[np.ndarray]=None, sensor_alignment: Optional[np.ndarray]=None,  manager: _api.TTBManager=None):
    if isinstance(manager, _api.TTBManagerAD):
        container = _api.MeasurementContainerAD()
        container._id = model_id
        container._timestamp = time
        container._sensor_info = create_sensor_information(fov_polygon, sensor_alignment)
        container._measurements = [_det_to_meas_impl(det, time, model_id, manager) for det in dets]
    else:
        container = _api.MeasurementContainer()
        container._id = model_id
        container._timestamp = time
        container._sensor_info = create_sensor_information(fov_polygon, sensor_alignment)
        container._measurements = [_det_to_meas_impl(det, time, model_id, manager) for det in dets]
    return container

def _det_to_meas_impl(det: Detection, time: datetime.datetime, model_id: str,  manager: _api.TTBManager):
    state_count = len(det.mean)
    mean = np.zeros((state_count))
    cov = np.zeros((state_count, state_count))

    for idx in range(0,state_count):
        mean[idx] = det.mean[idx]
        for idy in range(0,state_count):
            cov[idx,idy] = det.cov[idx,idy]
    if isinstance(manager, _api.TTBManagerAD):
        dist = _api.GaussianDistributionAD(mean.astype(np.float64), np.asfortranarray(cov.astype(np.float64)))
    else:
        dist = _api.GaussianDistribution(mean.astype(np.float64), np.asfortranarray(cov.astype(np.float64)))
    dist.set(_api.ReferencePoint.CENTER)
    components = _api.Components(det.components)
    if isinstance(manager, _api.TTBManagerAD):
        return _api.create_measurement_ad(manager, dist, time, components)
    else:
        return _api.create_measurement(manager, dist, time, components)

def get_estimate_from_state(state: _api.State, manager: _api.TTBManager):

    return _state_to_estimate_impl(state, manager)

def _state_to_estimate_impl(state: _api.State, manager: _api.TTBManager):
    label = state._label
    ex_prob = state._existenceProbability

    clazz = state._classification.get_estimate() if state._classification is not None else "UNKNOWN"
    clazz = clazz if clazz is not None else "UNKNOWN"

    comps, mean, cov = state.get_estimate()
    comp_names = [str(comp).split(".")[-1] for comp in comps._comps]
    estimation = Estimation(label, mean, cov, comp_names, ex_prob, clazz)
    return estimation
