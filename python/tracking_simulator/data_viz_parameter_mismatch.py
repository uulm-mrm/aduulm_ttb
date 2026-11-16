#!/usr/bin/env python3


import matplotlib.pyplot as plt
import numpy as np
import pathlib
import itertools
import scipy as sc


baseline = "lmb_oracle_100"


comparison = "lmb_false_100"


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


def read_gospa_model_params(file):

    num_sensors = 1
    gospa_datas = file.read_text()

    gospas = {}
    model_params = {}
    model_params["persistence"] = {}
    for estimation_txt in [d for d in gospa_datas.split('\n') if len(d) > 0]:
        time, missed, false, loc, gospa = estimation_txt.split(", ")[:5]
        params = {}
        model_params_txt = estimation_txt.split(", ")[5:5+5*num_sensors]
        for source, gtd, d, gtc, c in  itertools.batched(model_params_txt, 5):

            d = float(d)
            gtd = float(gtd)
            gtc = float(gtc)
            c = float(c)
            params[source] = (gtd, d, gtc, c)
        time = float(time)

        if not time in gospas:
            gospas[time] = []
            model_params[time] = params
        persistence_gt, p_filter = estimation_txt.split(", ")[5+5*num_sensors:5+5*num_sensors+2]
        model_params["persistence"][time] = (float(persistence_gt), float(p_filter))


        assert model_params[time] == params
        gospas[time].append((float(missed), float(false), float(loc), float(gospa)))
    return gospas, model_params

gospa_baseline_file = pathlib.Path(f"/home/alex/gitlab/aduulm_sandbox/src/tracking/library/python/scheible/{baseline}_gospa.stats")
gospa_comparison_file = pathlib.Path(f"/home/alex/gitlab/aduulm_sandbox/src/tracking/library/python/scheible/{comparison}_gospa.stats")

gospa_baseline, model_params_baseline = read_gospa_model_params(gospa_baseline_file)
gospa_comparison, model_params_comparison = read_gospa_model_params(gospa_comparison_file)



def plot_gospa(gospa_values):
    mean_gospas = {}
    for time in gospa_values:
        mean_gospas[time] = np.array(gospa_values[time])
    # false, missed, loc gospa
    fig, ax = plt.subplots(figsize=(400, 400))

    times = np.array(list(gospa_values.keys()))
    ts = times[TMIN:TMAX]
    false = [np.mean(mean_gospas[T], 0)[0] for T in times[TMIN:TMAX]]
    missed = [np.mean(mean_gospas[T], 0)[1] for T in times[TMIN:TMAX]]
    loc = [np.mean(mean_gospas[T], 0)[2] for T in times[TMIN:TMAX]]
    gospa = [np.mean(mean_gospas[T], 0)[3] for T in times[TMIN:TMAX]]
    # print(to_tikz_plot(ts, false, "false"))
    # print(to_tikz_plot(ts, missed, "missed"))
    # print(to_tikz_plot(ts, loc, "loc"))
    # print(to_tikz_plot(ts, gospa, "gospa"))
    # exit(0)

    p11 = ax.plot(times[TMIN:TMAX], false, label="False")
    p12 = ax.plot(times[TMIN:TMAX], missed, label="Missed")
    p13 = ax.plot(times[TMIN:TMAX], loc, label="Loc")
    p14 = ax.plot(times[TMIN:TMAX], gospa, label="Gospa")
    ax.legend(handles=p11+p12+p13+p14)
    return times

TMAX=-1
TMIN=0
times = plot_gospa(gospa_comparison)
plot_gospa(gospa_baseline)

def plot_gospa_comp(gospa_values_baseline, gospa_values_comparison):
    mean_gospas_baseline = {}
    for time in gospa_values_baseline:
        mean_gospas_baseline[time] = np.array(gospa_values_baseline[time])

    mean_gospas_comp = {}
    for time in gospa_values_comparison:
        mean_gospas_comp[time] = np.array(gospa_values_comparison[time])
    # false, missed, loc gospa
    fig, ax = plt.subplots(figsize=(400, 400))
    times = np.array(list(mean_gospas_baseline.keys()))
    ts = times[TMIN:TMAX]
    false = [(np.mean(mean_gospas_comp[T], 0)[0])-(np.mean(mean_gospas_baseline[T], 0)[0]) for T in times[TMIN:TMAX]]
    missed = [(np.mean(mean_gospas_comp[T], 0)[1])-(np.mean(mean_gospas_baseline[T], 0)[1]) for T in times[TMIN:TMAX]]
    loc = [(np.mean(mean_gospas_comp[T], 0)[2])-(np.mean(mean_gospas_baseline[T], 0)[2]) for T in times[TMIN:TMAX]]
    gospa =  [(np.mean(mean_gospas_comp[T], 0)[3])-(np.mean(mean_gospas_baseline[T], 0)[3]) for T in times[TMIN:TMAX]]

    # print(to_tikz_plot(ts, false, "false"))
    # print(to_tikz_plot(ts, missed, "missed"))
    # print(to_tikz_plot(ts, loc, "loc"))
    # print(to_tikz_plot(ts, gospa, "gospa"))
    # exit(0)
    p11 = ax.plot(times[TMIN:TMAX], false, label="False")
    p12 = ax.plot(times[TMIN:TMAX], missed, label="Missed")
    p13 = ax.plot(times[TMIN:TMAX], [(np.mean(mean_gospas_comp[T], 0)[2])-(np.mean(mean_gospas_baseline[T], 0)[2])  if True or (np.mean(mean_gospas_baseline[T], 0)[2])>0 else 1 for T in times[TMIN:TMAX]], label="Loc")
    p14 = ax.plot(times[TMIN:TMAX], [(np.mean(mean_gospas_comp[T], 0)[3])-(np.mean(mean_gospas_baseline[T], 0)[3])  if True or (np.mean(mean_gospas_baseline[T], 0)[3])>0 else 1 for T in times[TMIN:TMAX]], label="Gospa")
    ax.legend(handles=p11+p12+p13+p14)
    return times


plot_gospa_comp(gospa_baseline, gospa_comparison)


def compare_gospa(gospa_baseline, gospa_other):
    mean_gospas_baseline = {}
    mean_gospas_other = {}
    for time in gospa_baseline:
        mean_gospas_baseline[time] = np.array(gospa_baseline[time])
    for time in gospa_other:
        mean_gospas_other[time] = np.array(gospa_other[time])
    
        # false, missed, loc gospa
    fig, ax = plt.subplots(figsize=(400, 400))

def read_persistence_estimation(file, model_params):

    detection_datas = file.read_text()
    pers = {}
    for time in times:
        pers[time] = []
    for estimation_txt in [d for d in detection_datas.split('\n') if len(d) > 0]:
        time, mean, lower, upper = estimation_txt.split(", ")
        time = float(time)/1e9
        if time not in pers:
            pers[time] = []
        # if len(detection[source][time])>0:
        #     break
        pers[time].append((float(mean), float(lower), float(upper)))

    fig, ax = plt.subplots(figsize=(400, 400))
    gt_detection = [model_params["persistence"][t][0] for t in times[TMIN:TMAX]]
    ax.plot(times[TMIN:TMAX], gt_detection, label=f"GT Persistence Prob")
    ts = times[TMIN:TMAX]
    # print(to_tikz_plot(ts, gt_detection, "gt", 0.001))
    est_detection = [np.mean([d[0] for d in pers[time]]) for time in times[TMIN:TMAX]]
    print(to_tikz_plot(ts[1:], est_detection[1:], "est_detection", 0.001))
    exit(0)
    ax.plot(times[TMIN:TMAX], est_detection, label=f"Estimation")
    lower = [np.mean([d[1] for d in pers[time]]) for time in times[TMIN:TMAX]]
    upper = [np.mean([d[2] for d in pers[time]]) for time in times[TMIN:TMAX]]
    ax.plot(times[TMIN:TMAX], lower, label=f"Lower ")
    ax.plot(times[TMIN:TMAX], upper, label=f"Upper")

    filter = [model_params["persistence"][time][1] for time in times[TMIN:TMAX]]
    print(to_tikz_plot(ts[1:], lower[1:], "lower", 0.001))
    exit(0)

    print(to_tikz_plot(ts[1:], upper[1:], "upper", 0.001))
    exit(0)
    print(to_tikz_plot(ts, filter, "filter", 0.001))
    exit(0)

    ax.plot(times[TMIN:TMAX], filter, label=f"Filter Parameter")
    ax.legend()
    ax.set_ylim([0.8, 1.1])
    return pers




file = pathlib.Path(f"/home/alex/gitlab/aduulm_sandbox/src/tracking/library/python/scheible/{baseline}_persistence.stats")
# persistence_estimation_baseline = read_persistence_estimation(file, model_params_baseline)
file = pathlib.Path(f"/home/alex/gitlab/aduulm_sandbox/src/tracking/library/python/scheible/{comparison}_persistence.stats")
persistence_estimation_comparison = read_persistence_estimation(file, model_params_comparison)

plt.show()

exit(0)


###################### Detection
file = pathlib.Path(f"/home/alex/gitlab/aduulm_sandbox/src/tracking/library/python/scheible/{type}_detection.stats")
detection_datas = file.read_text()
detection = {}
for estimation_txt in [d for d in detection_datas.split('\n') if len(d) > 0]:
    source, time, alpha0, alpha1, _ = estimation_txt.split(", ")
    if source not in detection:
        detection[source] = {}
    time = float(time)/1e9
    if time not in detection[source]:
        detection[source][time] = []
    # if len(detection[source][time])>0:
    #     break
    detection[source][time].append(sc.stats.beta(float(alpha0), float(alpha1)))

fig, ax = plt.subplots(figsize=(400, 400))
for sim in ["sim0", "sim1", "sim2"]:
    gt_detection = [model_params[t][sim][0] for t in times[TMIN:TMAX]]
    ax.plot(times[TMIN:TMAX], gt_detection, label=f"GT Detection Prob {sim}")
    est_detection = [np.mean([d.mean() for d in detection[sim][time]]) for time in times[TMIN:TMAX]]
    ax.plot(times[TMIN:TMAX], est_detection, label=f"Estimation {sim}")
    ax.plot(times[TMIN:TMAX], [model_params[time][sim][1] for time in times[TMIN:TMAX]], label=f"Filter Parameter {sim}")
ax.legend()

###################### Clutter
file = pathlib.Path(f"/home/alex/gitlab/aduulm_sandbox/src/tracking/library/python/scheible/{type}_clutter.stats")

clutter_datas = file.read_text()

clutter = {}
for estimation_txt in [d for d in clutter_datas.split('\n') if len(d) > 0]:
    source, time, alpha, beta, _ = estimation_txt.split(", ")
    if source not in clutter:
        clutter[source] = {}
    time = float(time)/1e9
    if time not in clutter[source]:
        clutter[source][time] = []
    # if len(clutter[source][time])>0:
    #     break
    clutter[source][time].append(sc.stats.gamma(float(alpha), scale=1/float(beta)))

fig, ax = plt.subplots(figsize=(400, 400))
for sim in ["sim0", "sim1", "sim2"]:
    gt_clutter = [model_params[t][sim][2] for t in times[TMIN:TMAX]]
    ax.plot(times[TMIN:TMAX], gt_clutter, label=f"GT Clutter {sim}")
    est_clutter = [np.mean([d.mean() for d in clutter[sim][time]]) for time in times[TMIN:TMAX]]
    ax.plot(times[TMIN:TMAX], est_clutter, label=f"Estimation {sim}")
    ax.plot(times[TMIN:TMAX], [model_params[time][sim][3] for time in times[TMIN:TMAX]], label=f"Filter Parameter {sim}")
ax.legend()

######################
file = pathlib.Path(f"/home/alex/gitlab/aduulm_sandbox/src/tracking/library/python/scheible/{type}_bf.stats")

bf_datas = file.read_text()

bf = {}
for estimation_txt in [d for d in bf_datas.split('\n') if len(d) > 0]:
    source, time, clutter_bf, detection_bf = estimation_txt.split(", ")
    if source not in bf:
        bf[source] = {}
    time = float(time)/1e9
    if time not in bf[source]:
        bf[source][time] = []
    # if len(bf[source][time])>0:
    #     break
    bf[source][time].append((float(clutter_bf), float(detection_bf)))

fig, ax = plt.subplots(figsize=(400, 400))
# ax2 = ax.twinx()
# ax3 = ax.twinx()
# ax3.spines['right'].set_position(('outward', 60))
# p2 = ax2.plot(times[TMIN:TMAX], gt_detection, color="k", label="GT Detection")
# p3 = ax3.plot(times[TMIN:TMAX], gt_clutter, color="b", label="GT Clutter")
# ax.set_ylabel("Bayes Factor")
# ax2.set_ylabel("GT Detection")
# ax3.set_ylabel("GT Clutter")
# ax2.yaxis.label.set_color(p2[0].get_color())
# ax3.yaxis.label.set_color(p3[0].get_color())
handles = None
for sim in ["sim0", "sim1", "sim2"]:
    bf_clutter = [np.mean([d[0] for d in bf[sim][time]]) for time in times[TMIN:TMAX]]
    p = ax.plot(times[TMIN:TMAX], bf_clutter, label=f"BF Clutter {sim}")
    if handles is None:
        handles=p
    else:
        handles = handles + p
    bf_detection = [np.mean([d[1] for d in bf[sim][time]]) for time in times[TMIN:TMAX]]
    p = ax.plot(times[TMIN:TMAX], bf_detection, label=f"BF Detection {sim}")
    handles = handles + p
ax.legend(handles=handles)


plt.show()
