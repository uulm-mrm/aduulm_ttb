#!/usr/bin/env python3
import create_scenario
import get_measurements
import argparse
import pathlib
import datetime
import numpy as np
import os
import tqdm
tqdm.monitor_interval = 0
from tracking_lib import _tracking_lib_python_api as _api
from tracking_lib.utils import dets_to_meas_container, get_estimate_from_state, Estimation, to_gt_container


import pickle
from gospa import GOSPA


def get_gospa(targets, tracks, gospa_metric):
    rel_track_vals = []
    # get right format of estimation
    for track in tracks["estimation"]:
        rel_track_vals.append(track.mean)
    gospa = gospa_metric.compute_gospa_metric(rel_track_vals, targets)
    return gospa

CLUTTER_FILE = pathlib.Path("/tmp/clutter_rate_estimation.stats")
DETECTION_FILE = pathlib.Path("/tmp/detection_probability_estimation.stats")
MEAS_LIKELIHOODS_FILE = pathlib.Path("/tmp/meas_log_likelihoods.stats")

def read_estimations():
    clutter_estimation = {}
    if CLUTTER_FILE.is_file():
        print("Found clutter rate estimation results")
        clutter_rate_data = CLUTTER_FILE.read_text()
        import datetime
        import scipy
        for estimation_txt in [d for d in clutter_rate_data.split('\n') if len(d) > 0]:
            model_id, time, alpha, beta, bayes_factor = estimation_txt.split(', ')
            time = datetime.datetime.fromtimestamp(int(time) / int(1e9), datetime.timezone.utc)
            alpha = float(alpha)
            beta = float(beta)
            bayes_factor = float(bayes_factor)
            if not model_id in clutter_estimation:
                clutter_estimation[model_id] = {}
                clutter_estimation[model_id]['est'] = {}
                clutter_estimation[model_id]['bf'] = {}
            if not time in clutter_estimation[model_id]['est']:
                clutter_estimation[model_id]['est'][time] = []
                clutter_estimation[model_id]['bf'][time] = []
            clutter_estimation[model_id]['est'][time].append(scipy.stats.gamma(alpha, scale=1 / beta))
            clutter_estimation[model_id]['bf'][time].append(bayes_factor)
    detection_estimation = {}
    if DETECTION_FILE.is_file():
        print("Found detection prob estimation results")
        detection_prob_estimation_data = DETECTION_FILE.read_text()
        import datetime
        import scipy
        for estimation_txt in [d for d in detection_prob_estimation_data.split('\n') if len(d) > 0]:
            model_id, time, alpha, beta, bayes_factor = estimation_txt.split(', ')
            time = datetime.datetime.fromtimestamp(int(time) / int(1e9), datetime.timezone.utc)
            alpha = float(alpha)
            beta = float(beta)
            bayes_factor = float(bayes_factor)
            if not model_id in detection_estimation:
                detection_estimation[model_id] = {}
                detection_estimation[model_id]['est'] = {}
                detection_estimation[model_id]['bf'] = {}
            if not time in detection_estimation[model_id]['est']:
                detection_estimation[model_id]['est'][time] = []
                detection_estimation[model_id]['bf'][time] = []
            detection_estimation[model_id]['est'][time].append(scipy.stats.beta(alpha, beta))
            detection_estimation[model_id]['bf'][time].append(bayes_factor)
    meas_likelihoods = {}
    if MEAS_LIKELIHOODS_FILE.is_file():
        print("Found MeasLikelihood data")
        data = MEAS_LIKELIHOODS_FILE.read_text()
        import datetime
        for line in [d for d in data.split('\n') if len(d) > 0]:
            model_id, time, pdata, soft_clutter, soft_detection = line.split(', ')
            time = datetime.datetime.fromtimestamp(int(time) / int(1e9), datetime.timezone.utc)
            pdata = float(pdata)
            soft_clutter = float(soft_clutter)
            soft_detection = float(soft_detection)
            if not model_id in meas_likelihoods:
                meas_likelihoods[model_id] = {}
                meas_likelihoods[model_id]['meas_likelihood'] = {}
                meas_likelihoods[model_id]['soft_clutter_meas_likelihood'] = {}
                meas_likelihoods[model_id]['soft_detection_meas_likelihood'] = {}
            if not time in meas_likelihoods[model_id]['meas_likelihood']:
                meas_likelihoods[model_id]['meas_likelihood'][time] = []
                meas_likelihoods[model_id]['soft_clutter_meas_likelihood'][time] = []
                meas_likelihoods[model_id]['soft_detection_meas_likelihood'][time] = []
            meas_likelihoods[model_id]['meas_likelihood'][time].append(pdata)
            meas_likelihoods[model_id]['soft_clutter_meas_likelihood'][time].append(soft_clutter)
            meas_likelihoods[model_id]['soft_detection_meas_likelihood'][time].append(soft_detection)
    # CLUTTER_FILE.unlink(missing_ok=True)
    # DETECTION_FILE.unlink(missing_ok=True)
    # MEAS_LIKELIHOODS_FILE.unlink(missing_ok=True)
    return clutter_estimation, detection_estimation, meas_likelihoods
import time



class HPX:
    def __init__(self, api, n_threads):
        self.api = api
        self.n_threads = n_threads
    def __enter__(self):
        self.api.start_hpx(self.n_threads)
    def __exit__(self, exc_type, exc_value, traceback):
        self.api.stop_hpx()

def run_mc(args):
    scenarios = {}
    config_file = pathlib.Path(args.tracking_config_path)
    if args.ad:
        ttb_manager = _api.TTBManagerAD(config_file)
    else:
        ttb_manager = _api.TTBManager(config_file)
    # _api.set_log_level("Debug")
    # _api.set_log_level("Debug")
    for mc_run in range(args.num_mc_runs):
        try:
            scenario = eval(f"create_scenario.scenario_{args.scenario}(args)")
        except:
            raise Exception(f"Unknown Scenario: create_scenario.scenario_{args.scenario}")
        scenarios[mc_run] = scenario
        clutter_rate_estimation_path = CLUTTER_FILE
        if clutter_rate_estimation_path.is_file():
            print("remove old clutter data")
            clutter_rate_estimation_path.unlink(missing_ok=True)
        detection_estimation_path = DETECTION_FILE
        if detection_estimation_path.is_file():
            print("remove old detection data")
            detection_estimation_path.unlink(missing_ok=True)
        print(f'MC Run {mc_run}')
        detection_prob = float(args.detection_prob)
        meas_pos_variance = float(args.meas_pos_variance)
        clutter_rate = float(args.clutter_rate)
        num_sensors = args.num_sensors
        measurement_type = 1
        sensor_id_list = ["sim0", "sim1", "sim2", "sim3", "sim4"]
        range_x = scenario['region_x']
        range_y = scenario['region_y']
        poly = np.array([[range_x[0], range_y[0]],
                [range_x[0], range_y[1]],
                [range_x[1], range_y[1]],
                [range_x[1], range_y[0]]])
        fovs = [poly]
        fovs = fovs + [fovs[-1], ] * (num_sensors - len(fovs))
        alignment_sensor = np.array([[1.0, 0.0, 0.0, range_x[0]],
                                      [0.0, 1.0, 0.0, range_y[0]],
                                      [0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]])
        alignments_sensors = [alignment_sensor]*num_sensors

        components_sensors = [(_api.Component.POS_X, _api.Component.POS_Y)]*num_sensors
        clutter_rates = [clutter_rate]*num_sensors
        detection_probs = [detection_prob]*num_sensors
        measurement_types = [measurement_type]*num_sensors
        meas_pos_variances = [meas_pos_variance]*num_sensors
        measurements = get_measurements.get_measurements(scenario, num_sensors, detection_probs, clutter_rates,
                                                         measurement_types, meas_pos_variances, components_sensors)

        detection_probs = [0.4, 0.4, 0.4]

        time_step = 0
        time = datetime.datetime.fromtimestamp(0, datetime.timezone.utc)

        for meas_time in tqdm.tqdm(measurements):
            meas_containers = []
            for sensor in meas_time:
                if "detection" not in scenario:
                    meas_container_sensor = dets_to_meas_container(sensor["measurements"], time,
                                                                   sensor_id_list[sensor["id"]],
                                                                   detection_probs[sensor["id"]],
                                                                   clutter_rates[sensor["id"]],
                                                                   fovs[sensor["id"]],
                                                                   alignments_sensors[sensor["id"]], ttb_manager)
                else:
                    meas_container_sensor = dets_to_meas_container(sensor["measurements"], time,
                                                                   sensor_id_list[sensor["id"]],
                                                                   lambda target: scenario["detection"](sensor["id"], time_step, target),
                                                                   scenario["clutter"](sensor["id"], time_step),
                                                                   fovs[sensor["id"]],
                                                                   alignments_sensors[sensor["id"]], ttb_manager)
                ttb_manager.addData(meas_container_sensor, time)
                meas_containers.append(meas_container_sensor)
            ttb_manager.addData(to_gt_container(scenario, time_step, time, ttb_manager, create_scenario.get_gt_tracks(scenario, time_step)), time)
            # run tracking
            # ttb_manager.cycle(time, meas_containers, True) # dont use buffer
            ttb_manager.cycle(time)
            time += datetime.timedelta(0, args.dt)
            time_step += 1
        if mc_run+1<args.num_mc_runs:
            print("Reset Tracking")
            ttb_manager.reset()
    if args.wait:
        input("Press Enter to continue...")
    del ttb_manager
    print("deleted ttb manager")
    print("MC runs finished")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PythonScenarioGenerator'
    )
    parser.add_argument('--scenario', help='scenario number', type=str, default="1")  # 1=standard 10 tracks scenario
    parser.add_argument('--plot_scenario', help='Plot the map of the scenario?', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--save_plots', help='Save all plots?', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--save_results', help='Save and process results', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--tracking_config_path', type=pathlib.Path, default=pathlib.Path(
        os.path.dirname(os.path.realpath(__file__)) + '/../../tutorials/lmb_ic_lbp.yaml'))
    parser.add_argument('--save_plots_path', type=pathlib.Path, default=pathlib.Path('.'))
    parser.add_argument('--num_mc_runs', help='number of Monte Carlo runs', default=1, type=int)
    parser.add_argument('--num_sensors', help='number of sensors', default=1, type=int)
    parser.add_argument('--meas_pos_variance', help='set the measurement variance of the position', default=1, type=float)
    parser.add_argument('--clutter_rate', help='set the clutter rate for simulation', default=1, type=float)
    parser.add_argument('--detection_prob', help='set the detection probability for simulation', default=0.9, type=float)
    parser.add_argument('--sim_time_steps', default=500, type=int)
    parser.add_argument('--wait', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--ad', action=argparse.BooleanOptionalAction,
                        default=False, help='Enable AD computation')
    parser.add_argument('--dt', help='time step duration', default=0.1, type=float)
    args = parser.parse_args()
    if args.ad:
        n_threads = 2
    else:
        n_threads = 12

    with HPX(_api, n_threads):
        run_mc(args)

    print("All done")
    import os
    os._exit(0)