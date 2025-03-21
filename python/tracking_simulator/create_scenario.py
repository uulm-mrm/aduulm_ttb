#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pathlib


def cv_path(x0, acc_x_dev, acc_y_dev, dt, n_steps):
    """
    Create a random path according to the CV State Model
    X = [POS_X, VEL_X, POS_Y, VEL_Y]
    """
    xs = []
    ys = []
    for i in range(n_steps):
        xs.append(x0[0])
        ys.append(x0[2])
        x_noise = acc_x_dev * np.random.randn()
        y_noise = acc_y_dev * np.random.randn()
        xnew = x0[0] + dt * x0[1] + dt * dt / 2 * x_noise
        vxnew = x0[1] + dt * x_noise
        ynew = x0[2] + dt * x0[3] + dt * dt / 2 * y_noise
        vynew = x0[3] + dt * y_noise
        x0 = [xnew, vxnew, ynew, vynew]
    return xs, ys

def scenario_1(args):
    # scenario informations...
    sim_time = args.sim_time_steps
    dt = args.dt
    x_dev = 0.2
    y_dev = 0.2
    #Track1
    created_path = cv_path([-400,3,0,4], x_dev, y_dev, dt, 1000)
    target = dict(birth_time=0, last_alive_time=990, path=created_path)
    targets = [target]
    #Track2
    created_path = cv_path([-400, 4, -400, 6], x_dev, y_dev, dt, 700)
    target = dict(birth_time=100, last_alive_time=790, path=created_path)
    targets.append(target)
    #Track3
    created_path = cv_path([-400,5,400,-6], x_dev, y_dev, dt, 700)
    target = dict(birth_time=100, last_alive_time=790, path=created_path)
    targets.append(target)
    #Track4
    created_path = cv_path([400,-6,400,-3], x_dev, y_dev, dt, 400)
    target = dict(birth_time=200, last_alive_time=590, path=created_path)
    targets.append(target)
    #Track5
    created_path = cv_path([400,-9,0,0], x_dev, y_dev, dt, 500)
    target = dict(birth_time=500, last_alive_time=990, path=created_path)
    targets.append(target)
    #Track6
    created_path = cv_path([400,-12,-400,1.5], x_dev, y_dev, dt, 700)
    target = dict(birth_time=0, last_alive_time=690, path=created_path)
    targets.append(target)
    #Track7
    created_path = cv_path([-400,8,-400,5], x_dev, y_dev, dt, 600)
    target = dict(birth_time=400, last_alive_time=990, path=created_path)
    targets.append(target)
    #Track8
    created_path = cv_path([-400,12,0,-7], x_dev, y_dev, dt, 400)
    target = dict(birth_time=500, last_alive_time=890, path=created_path)
    targets.append(target)
    #Track9
    created_path = cv_path([-400,12,400,-2], x_dev, y_dev, dt, 400)
    target = dict(birth_time=300, last_alive_time=690, path=created_path)
    targets.append(target)
    #Track10
    created_path = cv_path([400,-10,0,9], x_dev, y_dev, dt, 300)
    target = dict(birth_time=300, last_alive_time=590, path=created_path)
    targets.append(target)
    scenario = dict(total_sim_time=sim_time, region_x = [-500,500], region_y = [-500,500], targets = targets)
    return scenario

def scenario_2(args):
    # scenario informations...
    sim_time_steps = args.sim_time_steps
    dt = args.dt
    x_dev = 0.1
    y_dev = 0.1
    plus_x = 100
    plus_y = 100
    region_x = [-plus_x,plus_x]
    region_y = [-plus_y,plus_y]

    targets = []

    for track in range(np.random.randint(4, 10)):
        bt = np.random.randint(0, sim_time_steps)
        bt = 0
        lat = min(bt + np.random.randint(min(30, sim_time_steps-1), sim_time_steps), sim_time_steps)
        # lat = sim_time_steps
        path =  cv_path([(np.random.random()-0.5)*1000, 10*(np.random.random()-0.5), (np.random.random()-0.5)*1000, 10*(np.random.random()-0.5)], x_dev, y_dev, dt, lat-bt+1)
        target = dict(birth_time=bt, last_alive_time=lat, path=path)
        targets.append(target)
        region_x[0] = min(region_x[0], min(path[0])-plus_x)
        region_x[1] = max(region_x[1], max(path[0])+plus_x)
        region_y[0] = min(region_y[0], min(path[1])-plus_y)
        region_y[1] = max(region_y[1], max(path[1])+plus_y)

    scenario = dict(total_sim_time=sim_time_steps, region_x = region_x, region_y = region_y, targets = targets)

    return scenario
def scenario_3(args):
    # scenario informations...
    sim_time = 1000
    dt = args.dt
    x_dev = 0
    y_dev = 0
    #Track1
    created_path = cv_path([-400,3,0,4], x_dev, y_dev, dt, 1000)
    target = dict(birth_time=0, last_alive_time=990, path=created_path)
    targets = [target]
    #Track2
    created_path = cv_path([-400, 3, 2, 4], x_dev, y_dev, dt, 1000)
    target = dict(birth_time=0, last_alive_time=990, path=created_path)
    targets.append(target)
    #Track3
    created_path = cv_path([-400,3,10,4], x_dev, y_dev, dt, 1000)
    target = dict(birth_time=0, last_alive_time=990, path=created_path)
    targets.append(target)
    #Track4
    created_path = cv_path([400,-6,400,-3], x_dev, y_dev, dt, 600)
    target = dict(birth_time=200, last_alive_time=790, path=created_path)
    targets.append(target)
    #Track5
    created_path = cv_path([400,-6,398,-3], x_dev, y_dev, dt, 600)
    target = dict(birth_time=200, last_alive_time=790, path=created_path)
    targets.append(target)
    #Track6
    created_path = cv_path([400,-12,-400,1.5], x_dev, y_dev, dt, 500)
    target = dict(birth_time=0, last_alive_time=490, path=created_path)
    targets.append(target)
    #Track7
    created_path = cv_path([400,-11.9,-399,1.5], x_dev, y_dev, dt, 500)
    target = dict(birth_time=0, last_alive_time=490, path=created_path)
    targets.append(target)
    scenario = dict(total_sim_time=sim_time, region_x = [-500,500], region_y = [-500,500], targets = targets)
    return scenario

def get_gt_tracks(scenario, time):
    gt_tracks = []
    # get relevant gt targets
    for target in scenario["targets"]:
        if target["birth_time"] <= time <= target["last_alive_time"]:
            mean = target["path"]
            val = [m[time - target["birth_time"]] for m in mean]
            gt_tracks.append(val)
    return gt_tracks
