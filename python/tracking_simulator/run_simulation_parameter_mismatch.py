#!/usr/bin/env python3
import create_scenario
import get_measurements
import argparse
import pathlib
import datetime
import numpy as np
import os
from tqdm import tqdm
import sys

from tracking_lib import _tracking_lib_python_api as _api
from tracking_lib.utils import dets_to_meas_container, get_estimate_from_state, Estimation, to_gt_container
import pickle


def run_mc(args):
    config_file = pathlib.Path(args.tracking_config_path)
    ttb_manager = _api.TTBManager(config_file)
    _api.set_log_level("Fatal")
    print(f'MC Run')
    for mc_run in tqdm(range(args.num_mc_runs)):
        try:
            scenario = eval(f"create_scenario.scenario_{args.scenario}(args)")
        except:
            raise Exception(f"Unknown Scenario: create_scenario.scenario_{args.scenario}")
        measurement_types = [1]*args.num_sensors
        sensor_id_list = ["sim0", "sim1", "sim2", "sim3", "sim4"]
        range_x = scenario['region_x']
        range_y = scenario['region_y']
        poly = np.array([[range_x[0], range_y[0]],
                [range_x[0], range_y[1]],
                [range_x[1], range_y[1]],
                [range_x[1], range_y[0]]])
        fovs = [poly] * args.num_sensors
        alignment_sensors = [np.array([[1.0, 0.0, 0.0, range_x[0]],
                                      [0.0, 1.0, 0.0, range_y[0]],
                                      [0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]])] * args.num_sensors


        components_sensors = [(_api.Component.POS_X, _api.Component.POS_Y)] * args.num_sensors

        measurements = get_measurements.get_measurements(scenario, args.num_sensors, args.detection_probs, args.clutter_rates,
                                                         measurement_types, args.meas_pos_variances, components_sensors)


        time_step = 0
        time = datetime.datetime.fromtimestamp(0, datetime.timezone.utc)
        counter = 1
        for meas_time in measurements:
            counter = counter + 1
            for sensor in meas_time:
                meas_container_sensor = dets_to_meas_container(sensor["measurements"], time,
                                                               sensor_id_list[sensor["id"]],
                                                               args.detection_probs[sensor["id"]],
                                                               args.clutter_rates[sensor["id"]],
                                                               fovs[sensor["id"]],
                                                               alignment_sensors[sensor["id"]], ttb_manager)
                ttb_manager.addData(meas_container_sensor, time)
                del meas_container_sensor
            gt = create_scenario.get_gt_tracks(scenario, time_step)
            ttb_manager.addData(to_gt_container(gt, time, ttb_manager), time)
            del gt
            ttb_manager.cycle(time)
            time += datetime.timedelta(0, args.dt)
            time_step += 1
        ttb_manager.reset()
        del measurements
    print("MC runs finished")

class Scratch(object):
    pass

if __name__ == '__main__':
    args = Scratch()
    args.tracking_config_path = "/home/alex/gitlab/aduulm_sandbox/src/tracking/library/python/scheible/glmb_ms_so_gibbs.yaml"
    args.num_mc_runs = 50
    args.scenario = "one_track"
    args.sim_time_steps=1000
    args.dt=0.1
    args.num_sensors=3
    args.mismatch_sensors=3
    rope_results = {}
    template_path  = args.tracking_config_path
    for detection_prob_filter in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
        for detection_prob_scenario in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
    # for detection_prob_filter in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    #     for detection_prob_scenario in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
            for clutter_rate in [30]:
                clutter_rate_filter = clutter_rate
                clutter_rate_scenario = clutter_rate
                print(f"detection rate {detection_prob_filter}, {detection_prob_scenario}")
                ttb_file = pathlib.Path(template_path)
                with ttb_file.open('r') as file:
                    data = file.read()
                for s in range(args.mismatch_sensors):
                    data = data.replace(f'<DETECTION_PROB{s}>', str(detection_prob_filter)).replace(f'<CLUTTER_RATE{s}>', str(clutter_rate_filter))
                for s in range(args.mismatch_sensors, args.num_sensors):
                    data = data.replace(f'<DETECTION_PROB{s}>', str(detection_prob_scenario)).replace(f'<CLUTTER_RATE{s}>', str(clutter_rate_scenario))
                args.clutter_rates = [clutter_rate_scenario]* args.num_sensors
                args.detection_probs = [detection_prob_scenario]*args.num_sensors
                args.meas_pos_variances = [1]*args.num_sensors
                path = f"{ttb_file.parent.resolve()}/tmp_detection.yaml"
                try:
                    with open(path, 'x') as new_file:
                        new_file.write(data)
                except:
                    with open(path, 'w') as new_file:
                        new_file.write(data)

                args.tracking_config_path = path

                run_mc(args)

