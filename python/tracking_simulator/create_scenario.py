#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import scipy


def cv_path(x0, acc_x_dev, acc_y_dev, dt, n_steps, region_x=None, region_y=None):
    """
    Create a random path according to the CV State Model
    X = [POS_X, VEL_X, POS_Y, VEL_Y]
    """
    xs = []
    ys = []
    for i in range(n_steps):
        if region_x is not None and region_y is not None:
            if x0[0]-10<region_x[0] or x0[0]+10>region_x[1] or x0[2]-10<region_y[0] or x0[2]+10>region_y[1]:
                return xs, ys, i-1
        xs.append(x0[0])
        ys.append(x0[2])
        x_noise = acc_x_dev * np.random.randn()
        y_noise = acc_y_dev * np.random.randn()
        xnew = x0[0] + dt * x0[1] + dt * dt / 2 * x_noise
        vxnew = x0[1] + dt * x_noise
        ynew = x0[2] + dt * x0[3] + dt * dt / 2 * y_noise
        vynew = x0[3] + dt * y_noise
        x0 = [xnew, vxnew, ynew, vynew]

    return xs, ys, n_steps-1

def scenario_1(args):
    # scenario informations...
    sim_time = args.sim_time_steps
    dt = args.dt
    x_dev = 0.2
    y_dev = 0.2
    region_x = [-500, 500]
    region_y = [-500, 500]
    #Track1
    xs, ys, lat = cv_path([-400,3,0,4], x_dev, y_dev, dt, 100, region_x, region_y)
    target = dict(birth_time=0, last_alive_time=lat, path=(xs, ys))
    targets = [target]
    latest_lat = lat
    #Track2
    xs, ys, lat = cv_path([-400, 4, -400, 6], x_dev, y_dev, dt, sim_time, region_x, region_y)
    target = dict(birth_time=10, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    latest_lat = max(latest_lat, lat)
    #Track3
    xs, ys, lat = cv_path([-400,5,400,-6], x_dev, y_dev, dt, 50, region_x, region_y)
    target = dict(birth_time=20, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    latest_lat = max(latest_lat, lat)
    #Track4
    xs, ys, lat = cv_path([400,-6,400,-3], x_dev, y_dev, dt, sim_time, region_x, region_y)
    target = dict(birth_time=30, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    latest_lat = max(latest_lat, lat)
    #Track5
    xs, ys, lat = cv_path([400,-9,0,0], x_dev, y_dev, dt, 10, region_x, region_y)
    target = dict(birth_time=40, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    latest_lat = max(latest_lat, lat)
    #Track6
    xs, ys, lat = cv_path([400,-12,-400,1.5], x_dev, y_dev, dt, sim_time, region_x, region_y)
    target = dict(birth_time=50, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    latest_lat = max(latest_lat, lat)
    #Track7
    xs, ys, lat = cv_path([-400,8,-400,5], x_dev, y_dev, dt, sim_time, region_x, region_y)
    target = dict(birth_time=10, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    latest_lat = max(latest_lat, lat)
    #Track8
    xs, ys, lat = cv_path([-400,12,0,-7], x_dev, y_dev, dt, sim_time, region_x, region_y)
    target = dict(birth_time=70, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    latest_lat = max(latest_lat, lat)
    #Track9
    xs, ys, lat = cv_path([-400,12,400,-2], x_dev, y_dev, dt, sim_time, region_x, region_y)
    target = dict(birth_time=80, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    latest_lat = max(latest_lat, lat)
    #Track10
    xs, ys, lat = cv_path([400,-10,0,9], x_dev, y_dev, dt, sim_time, region_x, region_y)
    target = dict(birth_time=90, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    latest_lat = max(latest_lat, lat)
    scenario = dict(total_sim_time=sim_time, region_x = region_x, region_y = region_y, targets = targets)
    print("Latest track alive: ", latest_lat)
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
    region_x = [-100, 100]
    region_y = [-100, 100]

    targets = []

    for track in range(np.random.randint(20, 50)):
        bt = np.random.randint(0, sim_time_steps)
        bt = 0
        lat = min(bt + np.random.randint(min(min(100, args.sim_time_steps-1), sim_time_steps-1), sim_time_steps), sim_time_steps)
        lat = sim_time_steps
        xs, ys, lat =  cv_path([(np.random.random()-0.5)*200, 10*(np.random.random()-0.5), (np.random.random()-0.5)*200, 10*(np.random.random()-0.5)], x_dev, y_dev, dt, lat-bt+1, region_x, region_y)
        target = dict(birth_time=bt, last_alive_time=lat, path=(xs, ys))
        if len(xs)>50:
            targets.append(target)
        # region_x[0] = min(region_x[0], min(path[0])-plus_x)
        # region_x[1] = max(region_x[1], max(path[0])+plus_x)
        # region_y[0] = min(region_y[0], min(path[1])-plus_y)
        # region_y[1] = max(region_y[1], max(path[1])+plus_y)

    scenario = dict(total_sim_time=sim_time_steps, region_x = region_x, region_y = region_y, targets = targets)

    return scenario
def scenario_3(args):
    # scenario informations...
    sim_time = 1000
    dt = args.dt
    x_dev = 0
    y_dev = 0
    region_x = [-500,500]
    region_y = [-500,500]
    #Track1
    xs, ys, lat = cv_path([-400,3,0,4], x_dev, y_dev, dt, 1000, region_x, region_y)
    target = dict(birth_time=0, last_alive_time=lat, path=(xs, ys))
    targets = [target]
    #Track2
    xs, ys, lat = cv_path([-400, 3, 2, 4], x_dev, y_dev, dt, 1000, region_x, region_y)
    target = dict(birth_time=0, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    #Track3
    xs, ys, lat = cv_path([-400,3,10,4], x_dev, y_dev, dt, 1000, region_x, region_y)
    target = dict(birth_time=0, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    #Track4
    xs, ys, lat = cv_path([400,-6,400,-3], x_dev, y_dev, dt, 600, region_x, region_y)
    target = dict(birth_time=200, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    #Track5
    xs, ys, lat = cv_path([400,-6,398,-3], x_dev, y_dev, dt, 600, region_x, region_y)
    target = dict(birth_time=200, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    #Track6
    xs, ys, lat = cv_path([400,-12,-400,1.5], x_dev, y_dev, dt, 500, region_x, region_y)
    target = dict(birth_time=0, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    #Track7
    xs, ys, lat = cv_path([400,-11.9,-399,1.5], x_dev, y_dev, dt, 500, region_x, region_y)
    target = dict(birth_time=0, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    scenario = dict(total_sim_time=sim_time, region_x = region_x, region_y = region_y, targets = targets)
    return scenario

def scenario_4(args):
    # scenario informations...
    sim_time = 500
    dt = args.dt
    x_dev = 0
    y_dev = 0
    region_x = [-100,100]
    region_y = [-100,100]
    targets = []
    for y in [80, 50, 20,  -10, -40, -70]:
        #Track1
        xs, ys, lat = cv_path([-80,5,y,0], x_dev, y_dev, dt, sim_time, region_x, region_y)
        target = dict(birth_time=10, last_alive_time=lat, path=(xs, ys))
        targets.append(target)
        #Track2
        xs, ys, lat = cv_path([-70, 4, y-20/3, 0], x_dev, y_dev, dt, sim_time, region_x, region_y)
        target = dict(birth_time=20, last_alive_time=lat, path=(xs, ys))
        targets.append(target)
        #Track3
        xs, ys, lat = cv_path([80,-3,y-2*20/3,0], x_dev, y_dev, dt, sim_time, region_x, region_y)
        target = dict(birth_time=30, last_alive_time=lat, path=(xs, ys))
        targets.append(target)
        #Track4
        xs, ys, lat = cv_path([80,-3,y-20,0], x_dev, y_dev, dt, sim_time, region_x, region_y)
        target = dict(birth_time=30, last_alive_time=lat, path=(xs, ys))
        targets.append(target)

    scenario = dict(total_sim_time=sim_time, region_x = region_x, region_y = region_y, targets = targets)
    return scenario

def scenario_one_track(args):
    # scenario informations...
    sim_time = 500
    dt = args.dt
    x_dev = 0
    y_dev = 0
    region_x = [80,200]


    region_y = [-20,20]
    targets = []

    xs, ys, lat = cv_path([150,0,0,0], x_dev, y_dev, dt, sim_time, region_x, region_y)
    target = dict(birth_time=10, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    # xs, ys, lat = cv_path([120,0,0.3,0], x_dev, y_dev, dt, sim_time, region_x, region_y)
    # target = dict(birth_time=10, last_alive_time=lat, path=(xs, ys))
    # targets.append(target)
    xs, ys, lat = cv_path([100,5,1.3,0], x_dev, y_dev, dt, sim_time, region_x, region_y)
    target = dict(birth_time=10, last_alive_time=lat, path=(xs, ys))
    targets.append(target)
    # xs, ys, lat = cv_path([106,4,0.15,0], x_dev, y_dev, dt, sim_time, region_x, region_y)
    # target = dict(birth_time=10, last_alive_time=lat, path=(xs, ys))
    # targets.append(target)
    # xs, ys, lat = cv_path([180,-5,0.2,0], x_dev, y_dev, dt, sim_time, region_x, region_y)
    # target = dict(birth_time=10, last_alive_time=lat, path=(xs, ys))
    # targets.append(target)
    sim_time = lat+10

    scenario = dict(total_sim_time=sim_time, region_x = region_x, region_y = region_y, targets = targets)
    return scenario

def scenario_more_tracks(args):
    # scenario informations...
    sim_time = 500
    dt = args.dt
    x_dev = 0
    y_dev = 0
    region_x = [-100,100]

    num_tracks = 10
    space = 5
    region_y = [-30,num_tracks*space+30]
    targets = []
    for i in range(num_tracks):
        xs, ys, lat = cv_path([-90,10,i*space,0], x_dev, y_dev, dt, sim_time, region_x, region_y)
        target = dict(birth_time=10, last_alive_time=lat, path=(xs, ys))
        targets.append(target)
    sim_time = lat+10
    num_sensors = 3
    def clutter(sensor, tpoint):
        return 7
    def detection(sensor, tpoint):
        return 0.8

    scenario = dict(total_sim_time=sim_time, region_x = region_x, region_y = region_y, targets = targets, num_sensors=num_sensors, clutter=clutter, detection=detection)
    return scenario


def cv_path_2(x0, time_step_0, acc_x_dev, acc_y_dev, PS, dt, region_x, region_y):
    """
    Create a random path according to the CV State Model
    X = [POS_X, VEL_X, POS_Y, VEL_Y]
    """
    xs = []
    ys = []
    for i in range(1000000):
        if x0[0]-10<region_x[0] or x0[0]+10>region_x[1] or x0[2]-10<region_y[0] or x0[2]+10>region_y[1]:
            return xs, ys, i-1
        time_step = time_step_0+i
        xs.append(x0[0])
        ys.append(x0[2])
        if scipy.stats.bernoulli(1-PS(time_step)**dt).rvs():
           return xs, ys, i
        x_noise = acc_x_dev(time_step) * np.random.randn()
        y_noise = acc_y_dev(time_step) * np.random.randn()
        xnew = x0[0] + dt * x0[1] + dt * dt / 2 * x_noise
        vxnew = x0[1] + dt * x_noise
        ynew = x0[2] + dt * x0[3] + dt * dt / 2 * y_noise
        vynew = x0[3] + dt * y_noise
        x0 = [xnew, vxnew, ynew, vynew]
    return None


def to_tikz_plot(x, y, note="", tol=0.1):
    import shapely
    line = shapely.LineString([(x, y) for x, y in zip(x, y)])
    line = line.simplify(tol)
    s = ""
    s += "\\addplot [black]\ntable {%" + note + "\n"
    for x, y in line.coords:
        s += f"{x} {y}\n"
    s += "};\n"
    return s

def scenario_bob(args):
    # scenario informations...
    sim_time = 1500
    dt = 0.1
    region_x = [-500,500]
    region_y = [-500,500]

    # state model dev
    def state_model_dev(tpoint):
        return 5
        values = [(0, 0.1), (100, 0.1), (101, 3), (250, 3), (251, 1),  (400, 1)]
        return np.interp(tpoint, [f for f, s in values], [s for f, s in values])

    def meas_model_dev(tpoint):
        return 1

    def persistence_prob(tpoint):
        if tpoint==sim_time:
            return 0.0
        # if tpoint < 200:
        #     return 0.99
        # return 0.95
        values = [(0, 0.95), (100, 0.95), (101, 0.99), (150, 0.99), (250, 0.999), (251, 0.9), (300, 0.85), (350, 0.85), (351, 0.95), (400, 0.95)]
        return np.interp(tpoint, [f for f, s in values], [s for f, s in values])
    targets = []


    random_state = 77
    ## birth_pos


    def possible_birth_pos(time):
        x_rv = scipy.stats.uniform(-5*80, 5*160)
        x_rv.random_state=time
        vx_rv = scipy.stats.uniform(-5, 10)
        vx_rv.random_state=time+774
        y_rv = scipy.stats.uniform(-5*80, 5*160)
        y_rv.random_state=time+993
        vy_rv = scipy.stats.uniform(-5, 10)
        vy_rv.random_state=time+343
        return [(x_rv.rvs(), vx_rv.rvs(), y_rv.rvs(), vy_rv.rvs()) for i in range(1)]

    def birth_var(time):
        return (0.5, 1, 0.5, 1)
    def birth_prob(time):
        if time%2==0:
            return 0.5
        return 0
    def birth_parameters(time):
        return (birth_prob(time), possible_birth_pos(time), birth_var(time))
    def birth(time):
        def birth_pos(mean, var):
            return [scipy.stats.uniform(loc=m, scale=np.sqrt(v)).rvs() for m, v in zip(mean, var)]

        pos = []
        for poss_mean in possible_birth_pos(time):
            if scipy.stats.bernoulli(birth_prob(time)).rvs():
                pos.append(birth_pos(poss_mean, birth_var(time)))

        return pos

    for time in range(sim_time):
        for pos in birth(time):
            xs, ys, alive_steps = cv_path_2(pos, time, state_model_dev, state_model_dev, persistence_prob, dt, region_x, region_y)
            target = dict(birth_time=time, last_alive_time=min(time+alive_steps, sim_time), path=(xs, ys))
            targets.append(target)
    num_sensors = 1


    def clutter(sensor, tpoint):
        return 30
        if True or sensor==0:
            values = [(0, 7), (10, 7), (30, 50), (40, 50), (50, 30), (80, 30), (1450, 15), (1500, 15), (1501, 7), (1550, 0.1), (1600, 0.1), (1601, 7), (1700, 7), (1701, 20), (1800, 20), (1801, 0.1), (1900, 0.1)]
            return np.interp(tpoint, [f for f, s in values], [s for f, s in values])
        elif sensor==1:
            return 7
        elif sensor==2:
            return 7
    def detection(sensor, tpoint, target_mean):
        if target_mean[0]>0 and target_mean[1]>0:
            return 0.999
        elif target_mean[0]<-100 and target_mean[1]<-100:
            return 0.8
        elif target_mean[0] < -300 and target_mean[1]>300:
            return 0.8
        elif target_mean[0]>0:
            return 0.5
        else:
            return 0.3
        pd = 1/(1+np.exp(-np.linalg.norm(target_mean)/np.sqrt(5000)))
        return pd
        if True or sensor==0:
            values = [(0, 0.8), (10, 0.7), (30, 0.95), (40, 0.95), (100, 0.4), (550, 0.4), (551, 0.6), (600, 0.6), (620, 0.7), (750, 0.7), (800, 0.5), (801, 0.8), (1700, 0.8), (1701, 0.95), (1750, 0.95), (1751, 0.6), (1850, 0.6), (1851, 0.95), (1900, 0.95)]
            return np.interp(tpoint, [f for f, s in values], [s for f, s in values])
        elif sensor==1:
            return 0.8
        elif sensor==2:
            return 0.8

    scenario = dict(total_sim_time=sim_time, region_x = region_x, region_y = region_y, targets = targets, num_sensors=num_sensors, clutter=clutter, detection=detection, persistence_prob=persistence_prob, birth=birth_parameters, state_model_noise=state_model_dev)
    num_surv = {}
    num_die = {}
    tot_num_die = 0
    tot_surv = 0
    for time in range(sim_time):
        num_surv[time] = 0
        num_die[time] = 0
    tracks_before = set()
    for time in range(sim_time):
        tracks_now = set([t[1] for t in get_gt_tracks(scenario, time)])
        num_surv[time] = len(tracks_before.intersection(tracks_now))
        num_die[time] = len(tracks_before-tracks_now)
        tot_num_die += num_die[time]
        tot_surv += num_surv[time]
        tracks_before = tracks_now
    # print([(num_surv[time], num_die[time]) for time in num_surv])
    # print(f"Tot die {tot_num_die}")
    # print(f"Tot surv {tot_surv}")
    # exit(0)
    # for target in scenario['targets']:
    #     created_path = target["path"]
    #     print(to_tikz_plot(created_path[0], created_path[1]))
    # exit(0)
    return scenario


def scenario_single_object_prediction(args):
    # scenario informations...
    sim_time = 400
    dt = 0.1
    region_x = [-500,500]
    region_y = [-500,500]

    # state model dev
    def state_model_dev(tpoint):
        return 1
    def meas_model_dev(tpoint):
        return 1

    def persistence_prob(tpoint):
        if tpoint==sim_time:
            return 0.0
        values = [(0, 0.95), (100, 0.95), (101, 0.99), (150, 0.99), (250, 0.999), (251, 0.9), (300, 0.85), (350, 0.85), (351, 0.95), (400, 0.95)]
        return np.interp(tpoint, [f for f, s in values], [s for f, s in values])
    targets = []


    random_state = 77
    ## birth_pos


    def possible_birth_pos(time):
        x_rv = scipy.stats.uniform(-80, 160)
        x_rv.random_state=time
        vx_rv = scipy.stats.uniform(-5, 10)
        vx_rv.random_state=time+774
        y_rv = scipy.stats.uniform(-80, 160)
        y_rv.random_state=time+993
        vy_rv = scipy.stats.uniform(-5, 10)
        vy_rv.random_state=time+343

        if time%2==0:
            return [(x_rv.rvs(), vx_rv.rvs(), y_rv.rvs(), vy_rv.rvs()) for i in range(1)]
        else:
            return []
    def birth_var(time):
        return (0.5, 1, 0.5, 1)
    def birth_prob(time):
        return 0.5
    def birth_parameters(time):
        return (birth_prob(time), possible_birth_pos(time), birth_var(time))
    def birth(time):
        def birth_pos(mean, var):
            return [scipy.stats.uniform(loc=m, scale=np.sqrt(v)).rvs() for m, v in zip(mean, var)]

        pos = []
        for poss_mean in possible_birth_pos(time):
            if scipy.stats.bernoulli(birth_prob(time)).rvs():
                pos.append(birth_pos(poss_mean, birth_var(time)))

        return pos

    for time in range(sim_time):
        for pos in birth(time):
            xs, ys, alive_steps = cv_path_2(pos, time, state_model_dev, state_model_dev, persistence_prob, dt, region_x, region_y)
            target = dict(birth_time=time, last_alive_time=min(time+alive_steps, sim_time), path=(xs, ys))
            targets.append(target)
    num_sensors = 3


    def clutter(sensor, tpoint):
        if True or sensor==0:
            values = [(0, 7), (10, 7), (30, 50), (40, 50), (50, 30), (80, 30), (1450, 15), (1500, 15), (1501, 7), (1550, 0.1), (1600, 0.1), (1601, 7), (1700, 7), (1701, 20), (1800, 20), (1801, 0.1), (1900, 0.1)]
            return np.interp(tpoint, [f for f, s in values], [s for f, s in values])
        elif sensor==1:
            return 7
        elif sensor==2:
            return 7
    def detection(sensor, tpoint):
        if True or sensor==0:
            values = [(0, 0.8), (10, 0.7), (30, 0.95), (40, 0.95), (100, 0.4), (550, 0.4), (551, 0.6), (600, 0.6), (620, 0.7), (750, 0.7), (800, 0.5), (801, 0.8), (1700, 0.8), (1701, 0.95), (1750, 0.95), (1751, 0.6), (1850, 0.6), (1851, 0.95), (1900, 0.95)]
            return np.interp(tpoint, [f for f, s in values], [s for f, s in values])
        elif sensor==1:
            return 0.8
        elif sensor==2:
            return 0.8

    scenario = dict(total_sim_time=sim_time, region_x = region_x, region_y = region_y, targets = targets, num_sensors=num_sensors, clutter=clutter, detection=detection, persistence_prob=persistence_prob, birth=birth_parameters)
    num_surv = {}
    num_die = {}
    tot_num_die = 0
    tot_surv = 0
    for time in range(sim_time):
        num_surv[time] = 0
        num_die[time] = 0
    tracks_before = set()
    for time in range(sim_time):
        tracks_now = set([t[1] for t in get_gt_tracks(scenario, time)])
        num_surv[time] = len(tracks_before.intersection(tracks_now))
        num_die[time] = len(tracks_before-tracks_now)
        tot_num_die += num_die[time]
        tot_surv += num_surv[time]
        tracks_before = tracks_now
    return scenario

def get_gt_tracks(scenario, time):
    gt_tracks = []
    # get relevant gt targets
    for label, target in enumerate(scenario["targets"]):
        if target["birth_time"] <= time <= target["last_alive_time"]:
            mean = target["path"]
            val = [m[time - target["birth_time"]] for m in mean]
            gt_tracks.append((val, label))
    return gt_tracks
