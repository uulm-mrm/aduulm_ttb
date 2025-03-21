#!/usr/bin/env python3


import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def plot_mc_estimation(mc_results, what):
    p = 0.13
    mean_values = {}
    lower_values = {}
    higher_values = {}
    bf = {}
    for model_id, data in mc_results[what].items():
        if not model_id in mean_values:
            mean_values[model_id] = {}
            lower_values[model_id] = {}
            higher_values[model_id] = {}
            bf[model_id] = {}
        for time, mc_runs in data['est'].items():
            mean_values[model_id][time] = np.mean([e.mean() for e in mc_runs])
            lower_values[model_id][time] = np.mean([e.ppf(p) for e in mc_runs])
            higher_values[model_id][time] = np.mean([e.ppf(1-p) for e in mc_runs])
            bf[model_id][time] = np.mean([e for e in data['bf'][time]])
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(what)
    for model_id, data in mean_values.items():
        ts = [time for time in data.keys()]
        ax1.plot(ts, [mean for mean in data.values()],
                 label=f'{model_id} Mean: {np.mean([mean for mean in data.values()])}')
        ax1.fill_between(ts, [v for v in lower_values[model_id].values()], [h for h in higher_values[model_id].values()], alpha=0.2)
        ax2.plot(ts, [val for val in bf[model_id].values()], label=model_id)
    ax2.set_yscale('log')
    ax1.legend()
    ax2.legend()

def plot_mc_gospa(mc_results):
    gospa = mc_results['gospa']
    fig, ax = plt.subplots()
    fig.suptitle("GOSPA")
    mean_gospa = [ np.mean([v['distance'] for v in gospa[time]]) for time in gospa]
    print(f'Mean Gospa {np.mean(mean_gospa)}')
    ax.plot([t for t in gospa.keys()], mean_gospa, label='Gospa')
    gospa_loc = [ np.mean([v['localization'] for v in gospa[time]]) for time in gospa]
    print(f'Mean Gospa Loc {np.mean(gospa_loc)}')
    ax.plot([t for t in gospa.keys()], gospa_loc, label='Gospa Loc')
    gospa_missed = [ np.mean([v['num_missed'] for v in gospa[time]]) for time in gospa]
    print(f'Mean Gospa Missed {np.mean(gospa_missed)}')
    ax.plot([t for t in gospa.keys()], gospa_missed, label='#Missed')
    gospa_false = [ np.mean([v['num_false'] for v in gospa[time]]) for time in gospa]
    print(f'Mean Gospa False {np.mean(gospa_false)}')
    ax.plot([t for t in gospa.keys()], gospa_false, label='#False')
    ax.legend()

def plot_scenario(mc_results, mc_run=0):
    fig, ax = plt.subplots()
    fig.suptitle("Ground Truth")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    for target in mc_results['scenario'][mc_run]['targets']:
        created_path = target["path"]
        ax.plot(created_path[0], created_path[1])

def plot_tracks(mc_results, mc_run=0):
    trajs = {}
    for time in mc_results['tracks']:
        for track in mc_results['tracks'][time][mc_run]['estimation']:
            label = track.label
            if label not in trajs:
                trajs[label] = []
            trajs[label].append(track.mean[:2])
    fig, ax = plt.subplots()
    fig.suptitle("Tracks")
    for label, traj in trajs.items():
        ax.plot([p[0] for p in traj], [p[1] for p in traj], label=label)
    ax.legend()

def save_all(filename):
    pp = PdfPages(filename)
    for fig_n in plt.get_fignums():
        fig = plt.figure(fig_n)
        fig.savefig(pp, format='pdf')
    pp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='MC data visualizer'
    )
    parser.add_argument('--file', type=str, default='mc_data.pkl')
    args = parser.parse_args()
    with open(args.file, 'rb') as f:
        mc_results = pickle.load(f)

    plot_mc_estimation(mc_results, 'clutter_estimation')
    plot_mc_estimation(mc_results, 'detection_estimation')
    plot_mc_gospa(mc_results)
    plot_tracks(mc_results)
    plot_scenario(mc_results)
    save_all('results.pdf')
    plt.show()