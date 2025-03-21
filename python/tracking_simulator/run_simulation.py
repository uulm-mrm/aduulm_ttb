#!/usr/bin/env python3
import create_scenario
import get_measurements
import argparse
import pathlib
import datetime
import numpy as np
import os

from tracking_lib import _tracking_lib_python_api as _api
from tracking_lib.utils import dets_to_meas_container, get_estimate_from_state, Estimation
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


def clutter_detection_estimation():
    clutter_rate_estimation_stats = CLUTTER_FILE
    clutter_estimation = {}
    if clutter_rate_estimation_stats.is_file():
        print("Found clutter rate estimation results")
        clutter_rate_data = clutter_rate_estimation_stats.read_text();
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
    detection_prob_estimation = DETECTION_FILE
    detection_estimation = {}
    if detection_prob_estimation.is_file():
        print("Found detection prob estimation results")
        detection_prob_estimation_data = detection_prob_estimation.read_text()
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
    return clutter_estimation, detection_estimation

def run_mc(args, send):
    # gospa configurations
    c = 10
    p = 2
    alpha = 2
    gospa_metric = GOSPA(c, p, alpha, mapping=[0, 1])
    gospa = {}
    gt_tracks = {}
    tracks = {}
    scenarios = {}
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
        alignments_sensors = [alignment_sensor]
        alignments_sensors = alignments_sensors + [alignments_sensors[-1], ] * (num_sensors - len(alignments_sensors))

        components_sensors = [(_api.Component.POS_X, _api.Component.POS_Y)]
        components_sensors = components_sensors + [components_sensors[-1], ] * (num_sensors - len(components_sensors))

        measurements = get_measurements.get_measurements(scenario, num_sensors, detection_prob, clutter_rate,
                                                         measurement_type, meas_pos_variance, components_sensors)

        config_file = pathlib.Path(args.tracking_config_path)
        ttb_manager = _api.TTBManager(config_file)
        _api.set_log_level("Warning")
        #_api.set_log_level("Debug")

        time_step = 0
        time = datetime.datetime.fromtimestamp(0, datetime.timezone.utc)
        counter = 1
        for meas_time in measurements:
            print("cycle: ", counter)
            counter = counter + 1
            meas_containers = []
            for sensor in meas_time:
                meas_container_sensor = dets_to_meas_container(sensor["measurements"], time,
                                                               sensor_id_list[sensor["id"]], fovs[sensor["id"]],
                                                               alignments_sensors[sensor["id"]])
                meas_containers.append(meas_container_sensor)
            # run tracking
            ttb_manager.cycle(time, meas_containers, True)
            est = dict(time=time,
                       estimation=[get_estimate_from_state(obj, ttb_manager) for obj in ttb_manager.getEstimate()])
            # run evaluation for this cycle
            if time not in gospa:
                gospa[time] = []
                tracks[time] = []
                gt_tracks[time] = []
            gospa[time].append(get_gospa(create_scenario.get_gt_tracks(scenario, time_step), est, gospa_metric))
            tracks[time].append(est)
            gt_tracks[time].append(create_scenario.get_gt_tracks(scenario, time_step))
            time += datetime.timedelta(0, args.dt)
            time_step += 1

        del ttb_manager
    print("MC runs finished")
    if args.save_results:
        results = {}
        results['gospa'] = gospa
        results['tracks'] = tracks
        results['gt_tracks'] = gt_tracks
        results['scenario'] = scenarios
        send.send(results)
        return
    print("returning")
    return

import multiprocessing as mp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PythonScenarioGenerator'
    )
    parser.add_argument('--scenario', help='scenario number', type=int, default=1)  # 1=standard 10 tracks scenario
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
    parser.add_argument('--num_sensors', help='number of sensors', default=2, type=int)
    parser.add_argument('--meas_pos_variance', help='set the measurement variance of the position', default=1.5, type=float)
    parser.add_argument('--clutter_rate', help='set the clutter rate for simulation', default=1, type=float)
    parser.add_argument('--detection_prob', help='set the detection probability for simulation', default=0.9, type=float)
    parser.add_argument('--sim_time_steps', default=500, type=int)
    parser.add_argument('--dt', help='time step duration', default=0.1, type=float)

    args = parser.parse_args()
    if args.save_results:
        print("save_results")
        mp.set_start_method('spawn')
        recv, send = mp.Pipe(False)
        p = mp.Process(target=run_mc, args=(args, send))
        p.start()
        rec = recv.recv()
        mc_results = {'gospa': rec['gospa'], 'tracks': rec['tracks'], 'gt_tracks': rec['gt_tracks'], 'scenario': rec['scenario']}
        p.join()
        print("Simulations done")
        print("Processing results")
        clutter_estimation, detection_estimation = clutter_detection_estimation()
        mc_results['clutter_estimation'] = clutter_estimation
        mc_results['detection_estimation'] = detection_estimation
        with open('mc_data.pkl', 'wb') as file:
            pickle.dump(mc_results, file)
        print("Saved Results")
    else:
        run_mc(args, None)
        print("All done")
    exit(0)
