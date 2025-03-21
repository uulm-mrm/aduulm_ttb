import matplotlib
matplotlib.use('tkagg')

import os
import json
import argparse
import pathlib
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import numpy as near
import importlib
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict
import math

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion

# from nuscenes.eval.common.loaders import load_gt
# from nuscenes.eval.tracking.data_classes import TrackingBox
# from nuscenes.eval.detection.data_classes import DetectionBox

import create_scenario
import get_measurements
from gospa import GOSPA
importlib.reload(get_measurements)
#importlib.reload(gospa)

from tracking_lib import _tracking_lib_python_api as _api
from tracking_lib.utils import dets_to_meas_container,get_estimate_from_state
from tracking_lib import TTB
from tracking_lib.utils import Detection

nusc_to_ttb_class_label_map = {
    "barrier"               : _api.ClassLabel.UNKNOWN,
    "traffic_cone"          : _api.ClassLabel.UNKNOWN,
    "bicycle"               : _api.ClassLabel.UNKNOWN,
    "motorcycle"            : _api.ClassLabel.BIKE_UNION,
    "pedestrian"            : _api.ClassLabel.PEDESTRIAN,
    "car"                   : _api.ClassLabel.CAR_UNION,
    "bus"                   : _api.ClassLabel.UNKNOWN,
    "construction_vehicle"  : _api.ClassLabel.UNKNOWN,
    "trailer"               : _api.ClassLabel.UNKNOWN,
    "truck"                 : _api.ClassLabel.TRUCK_UNION
}

ttb_to_nusc_class_label_map = {
#    _api.ClassLabel.UNKNOWN     : "trailer",
    _api.ClassLabel.BIKE_UNION  : "motorcycle",
    _api.ClassLabel.PEDESTRIAN  : "pedestrian",
    _api.ClassLabel.CAR_UNION   : "car",
    _api.ClassLabel.TRUCK_UNION : "truck"
}

ttb_params = {

    "seed_number" : 10,
    "detection_prob" : 0.5,
    "meas_pos_variance" : 1.5,
    "clutter_rate" : 0.001,
    "num_sensors" : 1,
    "measurement_type" : 1,
    "sensor_id_list" : ["sim0"],
    # "components_sensors" : [(_api.Component.POS_X, _api.Component.POS_Y,_api.Component.POS_Z, _api.Component.ROT_Z, _api.Component.LENGTH, _api.Component.WIDTH, _api.Component.HEIGHT)],
    "components_sensors" : [(_api.Component.POS_X, _api.Component.POS_Y,_api.Component.POS_Z, _api.Component.VEL_X, _api.Component.VEL_Y, _api.Component.ROT_Z, _api.Component.LENGTH, _api.Component.WIDTH, _api.Component.HEIGHT)],

    "cov_matrix" : np.array([[0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0.5, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0.01, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0.5, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0.5, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0.05, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0.01, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0.01, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0.01]]),
    "cov_matrix_pedestrian" : np.array([[0.1, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0.1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0.01, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 1.0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1.0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0.5, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0.01, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0.01, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0.01]]),

    "fov_sensor" : np.array([[320, 860],
                             [320, 970],
                             [195, 970],
                             [195, 860]]),
}

def preprocess(nusc, my_scene, detections, ttb_params, score):

    sample_token = my_scene['first_sample_token']

    k = 0

    saved_pose = False

    ego_x_min = ego_x_max = ego_y_min = ego_y_max = 0

    annotations = []
    annotations_plotting = []
    x_pos_annotations = []
    y_pos_annotations = []
    annotation_class = []

    measurements = []

    # iterate over the annotated detections
    while sample_token != '':

        sample = nusc.get('sample', sample_token)
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        current_sample_token = sample['token']

        ego_pose_token = sample_data['ego_pose_token']
        ego_pose = nusc.get('ego_pose', ego_pose_token)
        ego_x, ego_y, ego_z = ego_pose['translation']
        ego_x_min = ego_x
        ego_x_max = ego_x
        ego_y_min = ego_y
        ego_y_max = ego_y

        # get detections at current timestep
        detection = detections[current_sample_token]

        # get the current time
        timestamp = sample['timestamp']

        # create fov
        ego_pose_token = sample_data['ego_pose_token']
        ego_pose = nusc.get('ego_pose', ego_pose_token)
        ego_x, ego_y, ego_z = ego_pose['translation']
        x_max = ego_x + 60
        x_min = ego_x - 60
        y_max = ego_y + 60
        y_min = ego_y -60
        fov = np.array([[x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                        [x_min, y_min]])
        if ego_x < ego_x_min:
            ego_x_min = ego_x
        if ego_x > ego_x_max:
            ego_x_max = ego_x
        if ego_y < ego_y_min:
            ego_y_min = ego_y
        if ego_y > ego_y_max:
            ego_y_max = ego_y

        Z = []
        # create measurement dictionaries for detections 
        for entry in detection:
            if entry['detection_score'] > score:
                x, y, z = entry['translation']
                quaternion = Quaternion(entry['rotation'])
                yaw = quaternion.angle if quaternion.axis[2] > 0 else -quaternion.angle
                width, length, height = entry['size']
                vel_x, vel_y = entry['velocity']
                mean = np.zeros((9))
                mean[0] = x
                mean[1] = y
                mean[2] = z
                mean[3] = vel_x
                mean[4] = vel_y
                mean[5] = yaw
                mean[6] = length
                mean[7] = width
                mean[8] = height
                if nusc_to_ttb_class_label_map[entry['detection_name']] == _api.ClassLabel.PEDESTRIAN:
                    cov = ttb_params["cov_matrix_pedestrian"]
                else:
                    cov = ttb_params["cov_matrix"]

                meas = Detection(mean,cov,ttb_params["components_sensors"][0],nusc_to_ttb_class_label_map[entry['detection_name']],0.95)
                Z.append(meas)

        measurements.append([dict(id=0, measurements = Z, timestamp=timestamp, fov=fov)])

        annotations_timestep = []
        #annotations_timestep = {}
        #annotations_timestep['sample_token'] =
        #annotations_timestep['translations'] = []
        # get annotation for those detections (GT)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(ns_loaders.load_gt_of_sample_tokens(nusc,[current_sample_token],'TrackingBox'))
        for anno_token in sample['anns']:

            sample_annotation = nusc.get('sample_annotation', anno_token)
            instance_token = sample_annotation['instance_token']
            #print(annotation)
            x_pos = sample_annotation['translation'][0]
            y_pos = sample_annotation['translation'][1]

            x_pos_annotations.append(x_pos)
            y_pos_annotations.append(y_pos)

            annotation_class.append(sample_annotation['category_name'])

            #current_anno['translation'] = [x_pos, y_pos] # translation
            label_exists = False
            for anno in annotations_plotting:
                if anno['label'] == instance_token:
                    label_exists = True
                    anno['x'].append(x_pos)
                    anno['y'].append(y_pos)

            annotations_timestep.append([x_pos, y_pos])
            if not label_exists:
                new_anno = dict(label = instance_token, x = [x_pos], y = [y_pos])
                annotations_plotting.append(new_anno)


        annotations.append(annotations_timestep)

        sample_token = sample['next']
        k += 1

    # annotations in different format with classes, used it for generating plot
    gt = (x_pos_annotations, y_pos_annotations, annotation_class)

    x_max = ego_x_max + 60
    x_min = ego_x_min - 60
    y_max = ego_y_max + 60
    y_min = ego_y_min-60
    delta = 10
    fov_plot = np.array([[x_max+delta, y_min-delta],
                         [x_max+delta, y_max+delta],
                         [x_min-delta, y_max+delta],
                         [x_min-delta, y_min-delta]])

    # print(f"measurements: {measurements}")
    # print(f"annotations: {annotations}")
    # print(f"gt: {gt}")
    return measurements, annotations, gt, fov_plot, annotations_plotting

def print_gt_trajectories(gt, fov, save_plot, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_max = fov[0,0]
    x_min = fov[2,0]
    y_min = fov[0,1]
    y_max = fov[1,1]
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_title('sample annotations (GT)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    for target in gt:
        line = ax.plot(target['x'], target['y'])
        #line.set_label('Track ' + str(label))
    #ax.legend()

    if save_plot:
        plt.savefig(str(pathlib.Path(save_path)) + '/gt.pdf')

    plt.close()

def print_estimated_trajectories(estimation_results, save_plot, save_path, fov):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_max = fov[0,0]
    x_min = fov[2,0]
    y_min = fov[0,1]
    y_max = fov[1,1]
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_title('Estimated tracks')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    # get labels of all estimated tracks
    labels_list = []

    for time in range(len(estimation_results)):
        if estimation_results[time]:
            for est in estimation_results[time]["estimation"]:
                labels_list.append(est.label)

    labels_set = set(labels_list)
    # print(labels_set)

    for label in labels_set:
        x = []
        y = []
        for time in range(len(estimation_results)):
            if estimation_results[time]:
                for est in estimation_results[time]["estimation"]:
                    if label == est.label:
                        x.append(est.mean[0])
                        y.append(est.mean[1])

        line = ax.plot(x, y)
        # line.set_label('Track ' + str(label))
    #ax.legend()
    if save_plot:
        plt.savefig(str(pathlib.Path(save_path)) + '/estimated_tracks.pdf')
    # plt.show()
    plt.close()

def print_estimated_trajectories_ref(nusc, my_scene, tracks, fov, save_plot, save_path, score):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_max = fov[0,0]
    x_min = fov[2,0]
    y_min = fov[0,1]
    y_max = fov[1,1]
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_title('Estimated reference tracks')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    first_sample_token = my_scene['first_sample_token']
    first_sample = nusc.get('sample', first_sample_token)
    current_sample = first_sample
    current_sample_token = first_sample['token']

    has_next = True
    k = 0

    extracted_tracks = []

    while has_next:
        tracks_val = tracks[current_sample_token]

        # get the current time
        timestamp = current_sample['timestamp']
        for track in tracks_val:
            if track['tracking_score'] > score:
                label = track['tracking_id']
                x, y, z = track['translation']
                quaternion = Quaternion(track['rotation'])
                yaw = quaternion.angle if quaternion.axis[2] > 0 else -quaternion.angle
                width, length, height = track['size']
                vel_x, vel_y = track['velocity']
                mean = np.zeros((7))
                mean[0] = x
                mean[1] = y
                mean[2] = z
                # mean[3] = vel_x
                # mean[4] = vel_y
                mean[3] = yaw
                mean[4] = length
                mean[5] = width
                mean[6] = height

                track_exists = False
                for track in extracted_tracks:
                    if track['label'] == label:
                        # Append values
                        track_exists = True
                        track['x'].append(mean[0])
                        track['y'].append(mean[1])

                if not track_exists:
                    new_track = dict(label = label, x = [mean[0]], y = [mean[1]])
                    extracted_tracks.append(new_track)

        # check for next sample, terminate if theres none
        next_sample_token = current_sample['next']

        #print("----------------------------------------------------------")

        if (next_sample_token == ""):
            print("num steps = ", k)
            break

        current_sample_token = next_sample_token
        current_sample = nusc.get('sample', current_sample_token)
        sample_data = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
        k += 1


    for track in extracted_tracks:
        line = ax.plot(track['x'], track['y'])
        #line.set_label('Track ' + str(label))
    #ax.legend()
    if save_plot:
        plt.savefig(str(pathlib.Path(save_path)) + '/estimated_reference_tracks.pdf')
    # plt.show()
    plt.close()

def format_result(sample_token: str,
                  translation: List[float],
                  size: List[float],
                  yaw: float,
                  velocity: List[float],
                  tracking_id: int,
                  tracking_name: str,
                  tracking_score: float) -> Dict:
    """
    Format tracking result for 1 single target as following
    sample_result {
        "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
        "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
        "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
        "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
        "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
        "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
        "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                           Note that the tracking_name cannot change throughout a track.
        "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                           We average over frame level scores to compute the track level score.
                                           The score is used to determine positive and negative tracks via thresholding.
    }
    """
    sample_result = {}
    sample_result['sample_token'] = sample_token
    sample_result['translation'] = translation
    sample_result['size'] = size
    sample_result['rotation'] = Quaternion(angle=yaw, axis=[0, 0, 1]).elements.tolist()
    sample_result['velocity'] = velocity
    sample_result['tracking_id'] = tracking_id
    sample_result['tracking_name'] = tracking_name
    sample_result['tracking_score'] = tracking_score
    return sample_result

def get_gospa(targets, tracks, time, gospa_metric):
    rel_target_vals = []
    # get relevant gt targets
    for target in targets:
        rel_target_vals.append(target)

    rel_track_vals = []
    # get right format of estimation
    for track in tracks:
        rel_track_vals.append(track)

    gospa = gospa_metric.compute_gospa_metric(rel_track_vals, rel_target_vals)

    return gospa

def get_gospa_reference_tracks(current_sample_token, tracks_ref, score, detections=False):
    tracks_adapted = []
    try:
        tracks = tracks_ref[current_sample_token]
        for track in tracks:
            if (not detections) and track['tracking_score'] > score:
                x, y, z = track['translation']
                quaternion = Quaternion(track['rotation'])
                yaw = quaternion.angle if quaternion.axis[2] > 0 else -quaternion.angle
                width, length, height = track['size']
                vel_x, vel_y = track['velocity']
                val = [x,y]
                tracks_adapted.append(val)
            elif detections and track['detection_score'] > score:
                x, y, z = track['translation']
                quaternion = Quaternion(track['rotation'])
                yaw = quaternion.angle if quaternion.axis[2] > 0 else -quaternion.angle
                width, length, height = track['size']
                vel_x, vel_y = track['velocity']
                val = [x,y]
                tracks_adapted.append(val)

    except:
        print("no reference tracks or detections for gospa")

    return tracks_adapted

def eucl_dist(a, b):
    return np.linalg.norm(np.array(a[:]) - np.array(b[:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='PythonScenarioGenerator'
    )
    parser.add_argument('--tracking_config_path', type=pathlib.Path, default=pathlib.Path(os.path.dirname(os.path.realpath(__file__))+'/../tracking_configs/ic_lmb.yaml'))
    parser.add_argument('--dataset_path', type=pathlib.Path, default='/home/thomas/workspace/nuScenes/train_val_dataset/v1.0-trainval_meta')
    parser.add_argument('--detections_path', type=pathlib.Path, default='/home/thomas/mount_points/hd1/sequences/nuscenes/2024-10/sep 24/odet')
    parser.add_argument('--detection_score_thresh', type=float, default=0.2) # for centerpoint 0.3 # for odet (felicia) 0.2
    parser.add_argument('--ref_tracking_path', type=pathlib.Path, default='/home/thomas/mount_points/hd1/sequences/nuscenes/2024-10/sep 24/tracking')
    parser.add_argument('--ref_tracking_score_thresh', type=float, default=0.2)
    parser.add_argument('--save_plots_path', type=pathlib.Path, default=pathlib.Path(os.path.dirname(os.path.realpath(__file__))+'/../tracking_configs/'))
    parser.add_argument('--all_sequences', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # alignment_sensor = np.array([[1.0, 0.0, 0.0, -600.0],
    #                              [0.0, 1.0, 0.0, -600.0],
    #                              [0.0, 0.0, 1.0, 0.0],
    #                              [0.0, 0.0, 0.0, 1.0]])

    # get metadata of validation dataset
    # supposing the directory contains at least metadata of the dataset
    dataset_version = 'v1.0-trainval'
    nusc = NuScenes(version=dataset_version, dataroot=args.dataset_path)
    # print("nuscenes list: ",nusc.list_scenes())

    # get training and validation data splits
    split_data = splits.create_splits_scenes()
    train_scenes  = split_data['train']
    val_scenes = split_data['val']
    detection_score_thresh = args.detection_score_thresh
    reference_tracking_score_thresh = args.ref_tracking_score_thresh
    print("val_scenes: ", val_scenes)
    # gt_ns = load_gt(nusc,'val', DetectionBox)
    # print("gt loaded!")
    # print("results: ",gt_ns)

    if args.all_sequences:
        scenes = val_scenes
    else:
        scenes = ['scene-0003']

    # load detections
    detections_path = os.path.join(args.detections_path, "results_nusc.json")

    reference_tracking_path = os.path.join(args.ref_tracking_path, "results_nusc.json")
    print("load_detections!")

    with open(detections_path, 'r') as f:
        detections_json = json.load(f)

    with open(reference_tracking_path, 'r') as f:
        ref_tracks_json = json.load(f)

    detections = detections_json['results']
    ref_tracks = ref_tracks_json['results']

    tracking_results = {}

    for scene in scenes:
        print("processing scene: ", scene)

        #[TODO] select a scene
        # useful command to find a scene: nusc.list_scenes()
        my_scene_token = nusc.field2token('scene', 'name', scene)[0]
        my_scene = nusc.get('scene', my_scene_token)

        # preprocess
        measurements, annotations, gt, fov_plot, annotations_plotting = preprocess(nusc, my_scene, detections, ttb_params, detection_score_thresh)

        print("preprocessing done!")
        # perform simulation
        # -------------------------------------------------

        # move measurements to C++ TTB [TODO] change path
        config_file = pathlib.Path(os.path.dirname(os.path.realpath(__file__))+'/../tracking_configs/ic_lmb_nuscenes.yaml')
        #config_file = pathlib.Path("/home/hermann/workspace/jazzy_sandbox/src/tracking/library/python/python_tracking_simulator/tracking_configs/ic_lmb_nuscenes.yaml")
        ttb = TTB.from_yaml(config_file)
        # _api.set_log_level("Debug")
        # _api.set_log_level("Debug")
        # manager = _api.TTBManager(str(config_file))

        current_sample_token = nusc.get('scene', my_scene_token)['first_sample_token']
        # init tracking results for the whole val set
        estimation_results = []
        gospa_results = {}
        gospa_results_reference_tracking = {}
        gospa_results_reference_detection = {}
        tracking_results_gospa = []

        # gospa configurations
        c = 3
        p = 2
        alpha = 2
        gospa_metric = GOSPA(c, p, alpha, mapping=[0, 1])

        k = 0

        # initial_time = time(0, 0)
        # dummy_date = datetime.today()
        # init_time = datetime.combine(dummy_date, initial_time)
        # sim_time = init_time

        print("starting simulation ...")
        cycle = 0
        #get_measurements.plot_measurements_of_sensor(measurements,0, True, args.save_plots_path)
        last_time_stamp = 0
        for measurement in measurements:
            # initialize tracking results for this sample
            tracking_results[current_sample_token] = []
            tracking_results_gospa = []
            if cycle>0:
                diff = datetime.fromtimestamp( measurement[0]['timestamp'] / 1000000.0) - last_time_stamp
                print("diff between times: ", diff)
            #     sim_time = sim_time + timedelta(0, time_step)
            sim_time = datetime.fromtimestamp( measurement[0]['timestamp'] / 1000000.0)
            last_time_stamp =  datetime.fromtimestamp( measurement[0]['timestamp'] / 1000000.0)
            print("sim_time: ", sim_time)
            print("cycle: ", cycle)
            cycle = cycle + 1

            for sensor in measurement:

                meas_container_sensor = dets_to_meas_container(sensor["measurements"],
                                                               sim_time,
                                                               ttb_params["sensor_id_list"][sensor["id"]],
                                                               sensor["fov"])
                # print(meas_container_sensor)
                ttb.addMeasurement(meas_container_sensor,sim_time)

            ttb.manager.cycle(sim_time+timedelta(0,0.075))

            est = dict(time=time, estimation=[get_estimate_from_state(obj, ttb.manager) for obj in ttb.getEstimate()])

            est_eval = [get_estimate_from_state(obj, ttb.manager) for obj in ttb.getEstimate()]

            # print("est: ", est_eval)
            # print("")
            # print("")
            # print("")
            # print("")

            if est_eval:
                for target_est in est_eval:
                    if target_est.class_label != _api.ClassLabel.UNKNOWN:
                        sample_result = format_result(current_sample_token,
                                              #target_est['translation'] + [target_est['height']],
                                                      [target_est.mean[target_est.components.index('POS_X')], # translation
                                                       target_est.mean[target_est.components.index('POS_Y')],
                                                       target_est.mean[target_est.components.index('POS_Z')]],
                                              #target_est['size'],
                                                      [target_est.mean[target_est.components.index('WIDTH')], # translation
                                                       target_est.mean[target_est.components.index('LENGTH')],
                                                       target_est.mean[target_est.components.index('HEIGHT')]],
                                              # target_est['orientation'],
                                                      target_est.mean[target_est.components.index('ROT_Z')],
                                              #target_est['velocity'],
                                                      [target_est.mean[target_est.components.index('VEL_ABS')]*math.cos(target_est.mean[target_est.components.index('ROT_Z')]),
                                                       target_est.mean[target_est.components.index('VEL_ABS')]*math.sin(target_est.mean[target_est.components.index('ROT_Z')])],
                                                      target_est.label,
                                                      ttb_to_nusc_class_label_map[target_est.class_label],
                                                      target_est.existence_probability)
                        tracking_results[current_sample_token].append(sample_result)
                    sample_gospa = [target_est.mean[target_est.components.index('POS_X')], # translation
                                    target_est.mean[target_est.components.index('POS_Y')]]
                    tracking_results_gospa.append(sample_gospa)

            else:
                print("!!no estimation in first time step!!")
                sample_result = format_result(current_sample_token,
                                              [0, 0, 0],
                                              [0, 0, 0],
                                              0,
                                              [0, 0],
                                              0,
                                              'car',
                                              0)
                tracking_results[current_sample_token].append(sample_result)

                #sample_gospa = []
                #tracking_results_gospa.append(sample_gospa)
            detections_reference_results_gospa = get_gospa_reference_tracks(current_sample_token, detections, detection_score_thresh, detections=True)
            tracking_reference_results_gospa = get_gospa_reference_tracks(current_sample_token, ref_tracks, reference_tracking_score_thresh)
            # gospa calculation
            if k not in gospa_results:
                gospa_results[k] = []
                gospa_results_reference_tracking[k] = []
                gospa_results_reference_detection[k] = []
            gospa_results[k].append(get_gospa(annotations[k], tracking_results_gospa, time, gospa_metric))
            gospa_results_reference_tracking[k].append(get_gospa(annotations[k], tracking_reference_results_gospa, time, gospa_metric))
            gospa_results_reference_detection[k].append(get_gospa(annotations[k], detections_reference_results_gospa, time, gospa_metric))

            # move on
            current_sample_token = nusc.get('sample', current_sample_token)['next']
            estimation_results.append(est)
            k = k + 1



        gospa_plots_path = os.path.join(args.save_plots_path, "gospa/" + scene)
        print(gospa_plots_path)
        if not os.path.exists(gospa_plots_path):
            os.makedirs(gospa_plots_path)

        print("plotting and saving gospas")
        #eval_gospa.printGOSPA(gospa_results, True, gospa_plots_path, 'ic_lmb')
        #eval_gospa.printGOSPA(gospa_results_reference_tracking, True, gospa_plots_path, 'reference_tracking')
        #eval_gospa.printGOSPA(gospa_results_reference_detection, True, gospa_plots_path, 'reference_detection')
        print_estimated_trajectories(estimation_results, True, gospa_plots_path, fov_plot)
        print_estimated_trajectories_ref(nusc, my_scene, ref_tracks, fov_plot, True, gospa_plots_path, reference_tracking_score_thresh)
        print_gt_trajectories(annotations_plotting, fov_plot, True, gospa_plots_path)
        get_measurements.plot_measurements_of_sensor(measurements,0, True, gospa_plots_path, fov_plot)
        # plot ground truth
        #plot_gt(annotations, save=True, path=gospa_plots_path)

    # save tracking result
    meta = {'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external': False}
    output_data = {'meta': meta, 'results': tracking_results}
    estimation_results_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__))+'/../tracking_configs/estimation-result')
    print(estimation_results_path)
    if not os.path.exists(estimation_results_path):
        os.makedirs(estimation_results_path)
    with open(pathlib.Path(os.path.dirname(os.path.realpath(__file__))+'/../tracking_configs/estimation-result/all-results-validation-set-{}.json'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))), 'w') as outfile:
        json.dump(output_data, outfile)
