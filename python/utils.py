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
        sensor_info._sensor_fov = _api.FieldOfView()
        sensor_info._sensor_fov.set_covered_area(fov_polygon)
    if sensor_alignment is not None:
        sensor_info.set_sensor_pose(sensor_alignment)
    return sensor_info

def dets_to_meas_container(dets: List[Detection], time: datetime.datetime, sensor_id: str, 
			   fov_polygon: Optional[np.ndarray]=None, sensor_alignment: Optional[np.ndarray]=None):
    container = _det_to_meas_container_impl(dets, time, sensor_id, fov_polygon, sensor_alignment)
    return container

def _det_to_meas_container_impl(dets: List[Detection], time: datetime.datetime, model_id: str, 
				fov_polygon: Optional[np.ndarray]=None, sensor_alignment: Optional[np.ndarray]=None):
    container = _api.MeasurementContainer()
    container._id = model_id
    container._timestamp = time
    container._sensor_info = create_sensor_information(fov_polygon, sensor_alignment)
    container._measurements = [_det_to_meas_impl(det, time, model_id) for det in dets]
    return container

def _det_to_meas_impl(det: Detection, time: datetime.datetime, model_id: str):
    dist = _api.GaussianDistribution()
    state_count = len(det.mean)
    mean = np.zeros((state_count))
    cov = np.zeros((state_count, state_count))

    for idx in range(0,state_count):
        mean[idx] = det.mean[idx]
        for idy in range(0,state_count):
            cov[idx,idy] = det.cov[idx,idy]
    dist.set(mean.astype(np.float64))
    dist.set(np.asfortranarray(cov.astype(np.float64)))
    dist.set(1.0)
    ref_pt = _api.ReferencePoint.CENTER
    dist.set(ref_pt)

    components = _api.Components(det.components)

    meas = _api.Measurement(dist, time, components)
    meas._ref_point_measured = True # is center by default, other values are also possible but not implemented here
    meas._classification.probs = {det.class_label: det.classification_prob}
    return meas

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
