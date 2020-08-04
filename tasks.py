from invoke import task
import json
import os
from typing import Tuple, Sequence, Dict, Optional, List
import glob
from collections import OrderedDict
import random
import heapq

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
from scipy import interpolate
from scipy import stats

from utils.system import get_logger


@task
def make_results(ctx, output_json_file):
    folder = os.path.dirname(output_json_file)
    os.makedirs(folder, exist_ok=True)

    np.random.seed(23)

    experiments = {}
    for mit in range(4):
        model = {}
        for eit in range(5):
            experiment = []
            for step in range(0, int(1e8), int(5e6)):
                experiment.append(
                    dict(
                        steps=int(step + 1e4 * np.random.randn()),
                        # score=np.sin(2 * np.pi * step / 5e7)
                        # + 2 * mit
                        # + (mit + 1) * 0.25 * np.random.randn(),
                        score=np.clip(
                            (mit + 1) / 4 * (1 - np.exp(-step / 2e7))
                            - np.abs((mit + 1 + step / 5e7) * 0.03 * np.random.randn()),
                            0.0,
                            1.0,
                        ),
                    )
                )
            model["exp{}".format(eit)] = experiment
        experiments["model{}".format(mit)] = model

    with open(output_json_file, "w") as f:
        json.dump(experiments, f, indent=4, sort_keys=True)

    get_logger().info("Done")


@task
def plots(
    ctx, input_json_file, min_score=0.0, max_score=1.0, smooth_margin=0.1
):  # , output_template):
    make_results(ctx, input_json_file)

    with open(input_json_file, "r") as f:
        data = json.load(f)

    # folder = os.path.dirname(output_template)
    # os.makedirs(folder, exist_ok=True)

    for mod_name in data:
        model = {}

        minx = max([data[mod_name][e][0]["steps"] for e in data[mod_name]])
        maxx = min([data[mod_name][e][-1]["steps"] for e in data[mod_name]])
        get_logger().info("minx {} maxx {}".format(minx, maxx))
        xnew = np.arange(minx, maxx, (maxx - minx) / 1000)

        for exp_name in data[mod_name]:
            abs_ord = [(d["steps"], d["score"]) for d in data[mod_name][exp_name]]
            x = [d[0] for d in abs_ord]
            y = [d[1] for d in abs_ord]
            tck = interpolate.splrep(x, y, s=0)
            ynew = interpolate.splev(xnew, tck, der=0)
            model[exp_name] = dict(y=ynew, origx=x, origy=y)
        #     plt.plot(xnew, ynew, zorder=10)
        #     plt.plot(x, y, zorder=10)
        #     break
        # break

        seqs = np.stack([model[e]["y"] for e in model])

        model_mean = np.clip(np.mean(seqs, axis=0), 0.0, 1.0)
        model_err = np.std(seqs, axis=0)

        if smooth_margin > 0.0:
            mean_filt = interpolate.splrep(xnew, model_mean, s=smooth_margin)
            mean_filt = interpolate.splev(xnew, mean_filt, der=0)
            err_filt = interpolate.splrep(xnew, model_err, s=smooth_margin)
            err_filt = interpolate.splev(xnew, err_filt, der=0)
        else:
            mean_filt, err_filt = model_mean, model_err

        lower_filt, upper_filt = (
            mean_filt - (1 + smooth_margin) * err_filt,
            mean_filt + (1 + smooth_margin) * err_filt,
        )
        plt.fill_between(
            xnew,
            np.clip(lower_filt, min_score, max_score),
            np.clip(upper_filt, min_score, max_score),
            alpha=0.25,
            zorder=10,
        )

        plt.plot(xnew, model_mean, zorder=11)

        # # # show original data points:
        # measx = []
        # measy = []
        # for e in model:
        #     measx += model[e]["origx"]
        #     measy += model[e]["origy"]
        # plt.scatter(
        #     measx,
        #     measy,
        #     color=plt.gca().lines[-1].get_color(),
        #     facecolors="none",
        #     zorder=12,
        # )

    plt.legend([mod_name for mod_name in data])
    plt.grid(linewidth=0.2, zorder=0)
    plt.show()

    get_logger().info("Done")


def plot_max_hp_curves(
    subset_size_to_expected_mas_est_list: List[Dict[int, float]],
    subset_size_to_bootstrap_points_list: Sequence[Dict[int, Sequence[float]]],
    method_labels: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
    line_styles: Optional[Sequence] = None,
    line_markers: Optional[Sequence] = None,
    title: str = "",
    ylabel: str = "",
    fig_size=(4, 4 * 3.0 / 5.0),
    save_path: Optional[str] = None,
    put_legend_outside: bool = True,
    include_legend: bool = False,
):
    """Plots E[max()] curves over sampled hyperparameter values.
    For more information on studying sensitivity of methods to
    hyperparameter tuning, refer to Dodge et al. EMNLP 2019
    https://arxiv.org/abs/1909.03004
    """
    line_styles = ["solid"] * len(colors) if line_styles is None else line_styles
    line_markers = [""] * len(colors) if line_markers is None else line_markers

    plt.grid(
        b=True,
        which="major",
        color=np.array([0.93, 0.93, 0.93]),
        linestyle="-",
        zorder=-2,
    )
    plt.minorticks_on()
    plt.grid(
        b=True,
        which="minor",
        color=np.array([0.97, 0.97, 0.97]),
        linestyle="-",
        zorder=-2,
    )
    ax = plt.gca()
    ax.set_axisbelow(True)
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for (
        index,
        (
            subset_size_to_expected_max_est,
            subset_size_to_bootstrap_points,
            method_label,
            color,
            line_style,
            line_marker,
        ),
    ) in enumerate(
        zip(
            subset_size_to_expected_mas_est_list,
            subset_size_to_bootstrap_points_list,
            method_labels,
            colors,
            line_styles,
            line_markers,
        )
    ):
        xvals = list(sorted(subset_size_to_bootstrap_points.keys()))
        points_list = [subset_size_to_bootstrap_points[x] for x in xvals]
        points = [subset_size_to_expected_max_est[x] for x in xvals]

        try:
            lower, _, upper = unzip(
                [np.percentile(points, [25, 50, 75]) for points in points_list]
            )

        except Exception as _:
            print(
                "Could not generate max_hp_curve for {}, too few points".format(
                    method_label
                )
            )
            continue
        plt.gca().fill_between(
            xvals, lower, upper, color=np.array(color + (25,)) / 255, zorder=1
        )
        plt.plot(
            xvals,
            points,
            label=r"{}.{}".format(index + 1, "\ \ " if index + 1 < 10 else " ")
            + method_label,
            color=np.array(color) / 255,
            lw=1.5,
            linestyle=line_style,
            marker=line_marker,
            markersize=3,
            markevery=4,
            zorder=2,
        )

    plt.title(title)
    plt.xlabel("Hyperparam. Evals")
    plt.ylabel(ylabel)

    plt.tight_layout()

    if include_legend:
        if put_legend_outside:
            ax = plt.gca()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            plt.legend()

    set_size(*fig_size)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(
            save_path, bbox_inches="tight",
        )
        plt.close()


def set_size(w, h, ax=None):
    """Set figure axis sizes.
    Taken from the answer in
    https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    w, h: width, height in inches
    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def unzip(xs):
    a = None
    n = None
    for x in xs:
        if n is None:
            n = len(x)
            a = [[] for _ in range(n)]
        for i, y in enumerate(x):
            a[i].append(y)
    return a


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    if confidence == 0.95:
        h = se * 1.96
    else:
        h = se * stats.t.ppf(
            (1 + confidence) / 2.0, n - 1
        )  # approx 1.96 for confidence 0.95
    return m, m - h, m + h


def access(datum, path):
    for p in path:
        datum = datum[p]
    return datum


def tasks_to_score(tasks, path=("success",), confidence=0.95):
    data = [access(task, path) for task in tasks]
    return mean_confidence_interval(data, confidence)


@task
def make_comparison(
    ctx,
    folder="/Users/jordis/Downloads/metrics",
    target="success",
    min_score=0.0,
    max_score=1.0,
    prefix="BabyAIGoToLocal",
    title="BabyAI",
    ylabel="Success rate",
    map=r"",
):
    target = target.split(",")
    map = OrderedDict([tuple(io.split(",")) for io in map.split(";") if "," in io])

    METHOD_ORDER = [
        "BC",
        "Dagger",
        "BCTeacherForcing",
        "PPO",
        "A2C",
        "BCOffPolicy",
    ]
    METHOD_ORDER = [prefix + morder for morder in METHOD_ORDER]

    data = {}
    subs = glob.glob(os.path.join(folder, "*"))
    for sub in subs:
        name = os.path.basename(sub)
        jsonfile = glob.glob(os.path.join(sub, "*", "*.json"))[0]
        with open(jsonfile, "r") as f:
            data[name] = json.load(f)

    get_logger().info("Read {} models".format(len(data)))

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    formatter = FuncFormatter(
        lambda val, tick_pos: "0" if val == 0 else "%3.0fM" % (val * 1e-6)
    )

    # Hide the right and top spines
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_major_formatter(formatter)

    values = {}
    for name in METHOD_ORDER:
        values[name] = dict(x=[], y=[], low=[], high=[])
        for ckpt in data[name]:
            values[name]["x"].append(ckpt["training_steps"])
            y, low, high = tasks_to_score(ckpt["tasks"], target)
            values[name]["y"].append(np.clip(y, min_score, max_score))
            values[name]["low"].append(np.clip(low, min_score, max_score))
            values[name]["high"].append(np.clip(high, min_score, max_score))
        # get_logger().info("Scores {}".format(values))

        plt.fill_between(
            values[name]["x"],
            np.clip(values[name]["low"], min_score, max_score),
            np.clip(values[name]["high"], min_score, max_score),
            alpha=0.25,
            zorder=10,
        )

        plt.plot(values[name]["x"], values[name]["y"], zorder=11)

    legend = [mod_name.replace(prefix, "") for mod_name in METHOD_ORDER]
    legend = [map[name] if name in map else name for name in legend]
    plt.legend(legend)
    # plt.grid(linewidth=0.2, zorder=0)
    plt.grid(
        b=True,
        which="major",
        color=np.array([0.93, 0.93, 0.93]),
        linestyle="-",
        zorder=-2,
    )
    plt.minorticks_on()
    plt.grid(
        b=True,
        which="minor",
        color=np.array([0.97, 0.97, 0.97]),
        linestyle="-",
        zorder=-2,
    )

    # plt.title(title)
    plt.xlabel("Training steps", fontsize=MEDIUM_SIZE)
    if ylabel is not None:
        plt.ylabel("Success rate", fontsize=MEDIUM_SIZE)

    set_size(4, 5)

    plt.tight_layout()

    plt.show()

    get_logger().info("Done")


@task
def babyai(
    ctx,
    folder="/Users/jordis/Downloads/metrics",
    target="success",
    min_score=0.0,
    max_score=1.0,
    prefix="BabyAIGoToLocal",
    title="GoToLocal (BabyAI)",
    ylabel="Success rate",
    map=r"BCTeacherForcing,BC$^{\mathrm{tf}=1}$;BCOffPolicy,BC$^{\mathrm{off-policy}}$;Dagger,DAgger",
):
    make_comparison(
        ctx,
        folder=folder,
        target=target,
        min_score=min_score,
        max_score=max_score,
        prefix=prefix,
        title=title,
        ylabel=ylabel,
        map=map,
    )


@task
def robothor_objectnav(
    ctx,
    folder="/Users/jordis/Downloads/metrics",
    target="success",
    min_score=0.0,
    max_score=1.0,
    prefix="objectnav",
    title="RoboThor ObjectNav",
    ylabel=None,
    map=r"foo,bar;frobnicable,filter",
):
    make_comparison(
        ctx,
        folder=folder,
        target=target,
        min_score=min_score,
        max_score=max_score,
        prefix=prefix,
        title=title,
        ylabel=ylabel,
        map=map,
    )


@task
def successful_tasks(
    ctx,
    mfile="/Users/jordis/Desktop/all_val/metrics__test_2020-07-27_15-01-03.json",
    val_file="rl_robothor/data/val.json",
    nsamples=128,
    out_file="/Users/jordis/Desktop/successful_tasks.json",
):
    with open(val_file, "r") as f:
        all_eps = {ep["id"]: ep for ep in json.load(f)}
    get_logger().info("{} validation episodes".format(len(all_eps)))

    with open(mfile, "r") as f:
        metrics = json.load(f)
    for ckpt in metrics:
        ckpt["tasks"] = {t["task_info"]["id"]: t for t in ckpt["tasks"]}
    get_logger().info("metrics for {} checkpoints".format(len(metrics)))

    good_tasks = []
    for t in metrics[0]["tasks"]:
        if (
            metrics[1]["tasks"][t]["success"]
            and metrics[1]["tasks"][t]["ep_length"]
            < metrics[0]["tasks"][t]["ep_length"]
            and metrics[1]["tasks"][t]["ep_length"]
        ):
            good_tasks.append((metrics[1]["tasks"][t]["spl"], all_eps[t]))

    random.seed(23)

    random.shuffle(good_tasks)

    all_levels = dict(easy=0, medium=1, hard=2)

    good_tasks = sorted(
        good_tasks, key=lambda x: (all_levels[x[1]["difficulty"]], x[0]), reverse=True,
    )  # sort by level, spl
    get_logger().info("successful tasks: {}".format(len(good_tasks)))

    # all_targets = sorted(list(set([t[1]["object_type"] for t in good_tasks])))
    # all_targets = {tget: i for i, tget in enumerate(all_targets)}
    # get_logger().info("successful targets: {} {}".format(len(all_targets), all_targets))
    #
    # all_rooms = sorted(list(set([t[1]["scene"] for t in good_tasks])))
    # all_rooms = {room: i for i, room in enumerate(all_rooms)}
    # get_logger().info("successful rooms: {} {}".format(len(all_rooms), all_rooms))

    sample = []
    skipped = [t for spl, t in good_tasks]
    while len(skipped):
        new_skipped = []
        used_groups = set()
        for t in skipped:
            group = (t["scene"], t["object_type"], t["difficulty"])
            if group in used_groups:
                new_skipped.append(t)
            else:
                sample.append(t)
                used_groups.add(group)
        skipped = new_skipped

    with open(out_file, "w") as f:
        json.dump(sample[:nsamples], f, indent=4, sort_keys=True)

    get_logger().info("task_ids {}".format([t["id"] for t in sample[:nsamples]]))

    get_logger().info("Done")


@task
def lava_crossing(
    ctx,
    file="/Users/jordis/Downloads/checkpoint_test_with_searched_hp_LavaCrossingCorruptExpertS15N7_1000.tsv",
    title="LC Corrupt S15N7 (MiniGrid)",
    ylabel="Success rate",
    min_score=0.0,
    max_score=1.0,
):
    FIXED_ADVISOR_STR = "ADV"
    EXPERIMENT_TYPE_TO_LABEL_DICT = {
        "dagger_then_ppo": r"$\dagger \to$ PPO",
        "dagger_then_advisor_fixed_alpha_different_head_weights": r"$\dagger \to$ {}".format(
            FIXED_ADVISOR_STR
        ),
        "bc_then_ppo": r"BC$ \to$ PPO",
        "advisor_fixed_alpha_different_heads": r"{}".format(FIXED_ADVISOR_STR),
        "bc": r"BC",
        "dagger": r"DAgger $(\dagger)$",
        "ppo": r"PPO",
        "ppo_with_offpolicy_advisor_fixed_alpha_different_heads": r"ADV$^{\mathrm{demo}} +$ PPO",
        "ppo_with_offpolicy": r"BC$^{\mathrm{demo}} +$ PPO",
        "pure_offpolicy": r"BC$^{\mathrm{demo}}$",
        "bc_teacher_forcing": r"BC$^{\mathrm{tf}=1}$",
        "bc_teacher_forcing_then_ppo": r"BC$^{\mathrm{tf}=1} \to$ PPO",
        "bc_teacher_forcing_then_advisor_fixed_alpha_different_head_weights": r"BC$^{\mathrm{tf}=1} \to$ ADV",
    }
    METHOD_ORDER = [
        "bc",
        "dagger",
        "bc_teacher_forcing",
        "ppo",
        "bc_then_ppo",
        "dagger_then_ppo",
        "bc_teacher_forcing_then_ppo",
        "advisor_fixed_alpha_different_heads",
        "dagger_then_advisor_fixed_alpha_different_head_weights",
        "bc_teacher_forcing_then_advisor_fixed_alpha_different_head_weights",
        # "pure_offpolicy",
        # "ppo_with_offpolicy",
        # "ppo_with_offpolicy_advisor_fixed_alpha_different_heads",
    ]
    data = pd.read_csv(file, sep="\t")
    get_logger().info("Done")

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=10)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    formatter = FuncFormatter(
        lambda val, tick_pos: "0"
        if val == 0.0
        else "%3.0fk" % (val * 1e-3)
        if val < 1e6
        else "%3.0fM" % (val * 1e-6)
    )

    # Hide the right and top spines
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_major_formatter(formatter)

    series = []
    for method in METHOD_ORDER:
        get_logger().info("{}".format(method))
        method_data = {"exp_type": method}
        method_rows = data[data["exp_type"] == method]
        dates = list(set(method_rows["date"]))
        assert len(dates) == 10
        x = set(method_rows[method_rows["date"] == dates[0]]["iterations"])
        assert len(x) == 10
        method_series = np.zeros((len(dates), len(x)))
        for itdate, date in enumerate(dates):
            experiment_rows = method_rows[method_rows["date"] == date]
            iters = set(method_rows[method_rows["date"] == date]["iterations"])
            assert x.intersection(iters) == x and x.union(iters) == x
            # break
            # get_logger().info("{}".format(experiment_rows["iterations"]))
            curdata = []
            for idx, row in experiment_rows[["success", "iterations"]].iterrows():
                # print(idx, row["success"], row["iterations"])
                curdata.append((row["iterations"], row["success"]))
            curdata = sorted(curdata, key=lambda x: x[0])
            method_series[itdate, :] = [c[1] for c in curdata]
        method_data["x"] = sorted(list(x))
        lq, median, uq = np.percentile(method_series, [25, 50, 75], axis=0)
        method_data["y_median"] = median
        method_data["y_lq"] = lq
        method_data["y_uq"] = uq
        # break
        # lq, median, uq = np.percentile(points, [25, 50, 75])

        plt.fill_between(
            method_data["x"],
            np.clip(method_data["y_lq"], min_score, max_score),
            np.clip(method_data["y_uq"], min_score, max_score),
            alpha=0.25,
            zorder=10,
        )

        plt.plot(method_data["x"], method_data["y_median"], zorder=11)

    # legend = [mod_name.replace(prefix, "") for mod_name in data]
    # plt.xlim(-1e5, 1e6)
    legend = [EXPERIMENT_TYPE_TO_LABEL_DICT[method] for method in METHOD_ORDER]
    plt.legend(legend)
    # plt.grid(linewidth=0.2, zorder=0)
    plt.grid(
        b=True,
        which="major",
        color=np.array([0.93, 0.93, 0.93]),
        linestyle="-",
        zorder=-2,
    )
    plt.minorticks_on()
    plt.grid(
        b=True,
        which="minor",
        color=np.array([0.97, 0.97, 0.97]),
        linestyle="-",
        zorder=-2,
    )

    # plt.title(title)
    plt.xlabel("Training steps", fontsize=MEDIUM_SIZE)
    if ylabel is not None:
        plt.ylabel("Success rate", fontsize=MEDIUM_SIZE)

    set_size(4, 5)

    plt.tight_layout()

    plt.show()

    get_logger().info("Done")


@task
def thor_habitat_success(
    ctx,
    folder="/Users/jordis/Downloads/robothor_ithor",
    target="Success",
    min_score=0.0,
    max_score=1.0,
    prefix="",
    title="RoboTHOR, iTHOR, Habitat (DD-PPO)",
    ylabel="Success rate",
    map=r"iTHOR-ResNet-SimpleConv,iTHOR-SimpleConv-DDPPO",
):
    target = target.split(",")
    map = OrderedDict([tuple(io.split(",")) for io in map.split(";") if "," in io])

    METHOD_ORDER = [
        "Habitat-Pointnav-Depth-SimpleConv-DDPPO",
        "iTHOR-Pointnav-Depth-ResNet-SimpleConv",
        "RoboTHOR-Pointnav-Depth-SimpleConv-DDPPO",
        "iTHOR-Objectnav-RGB-ResNet-DDPPO",
        "RoboTHOR-Objectnav-RGB-ResNet-DDPPO",
    ]
    METHOD_ORDER = [prefix + morder for morder in METHOD_ORDER]

    data = {}
    subs = glob.glob(os.path.join(folder, "*"))
    for sub in subs:
        name = os.path.basename(sub).replace(".json", "")
        with open(sub, "r") as f:
            data[name] = json.load(f)

    get_logger().info("Read {} models".format(len(data)))

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    formatter = FuncFormatter(
        lambda val, tick_pos: "0" if val == 0 else "%3.0fM" % (val * 1e-6)
    )

    # Hide the right and top spines
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MultipleLocator(50e6))

    values = {}
    lhs = []
    for it, name in enumerate(METHOD_ORDER):
        # values[name] = dict(x=[], y=[], low=[], high=[])
        # for ckpt in data[name]:
        #     values[name]["x"].append(ckpt["training_steps"])
        #     y, low, high = tasks_to_score(ckpt["tasks"], target)
        #     values[name]["y"].append(np.clip(y, min_score, max_score))
        #     values[name]["low"].append(np.clip(low, min_score, max_score))
        #     values[name]["high"].append(np.clip(high, min_score, max_score))
        # # get_logger().info("Scores {}".format(values))
        #
        # plt.fill_between(
        #     values[name]["x"],
        #     np.clip(values[name]["low"], min_score, max_score),
        #     np.clip(values[name]["high"], min_score, max_score),
        #     alpha=0.25,
        #     zorder=10,
        # )

        values[name] = dict(x=[], y=[])
        for ckpt in data[name]:
            values[name]["x"].append(ckpt["Step"])
            y = ckpt["Success"]
            values[name]["y"].append(np.clip(y, min_score, max_score))
            #     y, low, high = tasks_to_score(ckpt["tasks"], target)

        if it > 2:
            (clhs,) = plt.plot(values[name]["x"], values[name]["y"], zorder=11)
            lhs.append(clhs)
        else:
            plt.plot(values[name]["x"], values[name]["y"], zorder=11)

        if it == 2:
            prefix = "-Pointnav-Depth"
            legend = [mod_name.replace(prefix, "") for mod_name in METHOD_ORDER[:3]]
            legend = [map[name] if name in map else name for name in legend]
            legend = [name.replace("-DDPPO", "") for name in legend]
            legend = [name.replace("-SimpleConv", "") for name in legend]
            legend = [name.replace("-", " ") for name in legend]
            leg1 = ax.legend(
                legend,
                loc="upper left",
                #             bbox_to_anchor=(0.5, 0.4, 0.5, 0.5),
                #             title="Point Nav",
                #         )
                #
                # prefix = "-Objectnav-RGB"
                # legend = [mod_name.replace(prefix, "") for mod_name in METHOD_ORDER[3:]]
                # legend = [map[name] if name in map else name for name in legend]
                # legend = [name.replace("-DDPPO", "") for name in legend]
                # legend = [name.replace("-ResNet", "") for name in legend]
                # legend = [name.replace("-", " ") for name in legend]
                # leg2 = ax.legend(
                #     lhs,
                #     legend,
                #     bbox_to_anchor=(0.5, 0.0, 0.5, 0.5),
                bbox_to_anchor=(0.35, 0.3, 0.5, 0.6),
                title="Point Nav",
            )

        prefix = "-Objectnav-RGB"
        legend = [mod_name.replace(prefix, "") for mod_name in METHOD_ORDER[3:]]
        legend = [map[name] if name in map else name for name in legend]
        legend = [name.replace("-DDPPO", "") for name in legend]
        legend = [name.replace("-ResNet", "") for name in legend]
        legend = [name.replace("-", " ") for name in legend]
        leg2 = ax.legend(
            lhs,
            legend,
            bbox_to_anchor=(0.35, 0.0, 0.5, 0.52),
            loc="lower left",
            title="Object Nav",
        )
    ax.add_artist(leg1)
    # plt.grid(linewidth=0.2, zorder=0)
    plt.grid(
        b=True,
        which="major",
        color=np.array([0.93, 0.93, 0.93]),
        linestyle="-",
        zorder=-2,
    )
    plt.minorticks_on()
    plt.grid(
        b=True,
        which="minor",
        color=np.array([0.97, 0.97, 0.97]),
        linestyle="-",
        zorder=-2,
    )

    # plt.title(title)
    plt.xlabel("Training steps", fontsize=MEDIUM_SIZE)
    if ylabel is not None:
        plt.ylabel("Success rate", fontsize=MEDIUM_SIZE)

    set_size(4, 5)

    plt.tight_layout()

    plt.show()

    get_logger().info("Done")


@task
def thor_habitat_spl(
    ctx,
    folder="/Users/jordis/Downloads/robothor_ithor",
    target="spl",
    min_score=0.0,
    max_score=1.0,
    prefix="",
    title="RoboTHOR, iTHOR, Habitat (DD-PPO)",
    ylabel="SPL",
    map=r"iTHOR-ResNet-SimpleConv,iTHOR-SimpleConv-DDPPO",
):
    target = target.split(",")
    map = OrderedDict([tuple(io.split(",")) for io in map.split(";") if "," in io])

    METHOD_ORDER = [
        "Habitat-Pointnav-Depth-SimpleConv-DDPPO",
        "iTHOR-Pointnav-Depth-ResNet-SimpleConv",
        "RoboTHOR-Pointnav-Depth-SimpleConv-DDPPO",
        "iTHOR-Objectnav-RGB-ResNet-DDPPO",
        "RoboTHOR-Objectnav-RGB-ResNet-DDPPO",
    ]
    METHOD_ORDER = [prefix + morder for morder in METHOD_ORDER]

    data = {}
    subs = glob.glob(os.path.join(folder, "*"))
    for sub in subs:
        name = os.path.basename(sub).replace(".json", "")
        with open(sub, "r") as f:
            data[name] = json.load(f)

    get_logger().info("Read {} models".format(len(data)))

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    formatter = FuncFormatter(
        lambda val, tick_pos: "0" if val == 0 else "%3.0fM" % (val * 1e-6)
    )

    # Hide the right and top spines
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MultipleLocator(50e6))

    values = {}
    lhs = []
    for it, name in enumerate(METHOD_ORDER):
        # values[name] = dict(x=[], y=[], low=[], high=[])
        # for ckpt in data[name]:
        #     values[name]["x"].append(ckpt["training_steps"])
        #     y, low, high = tasks_to_score(ckpt["tasks"], target)
        #     values[name]["y"].append(np.clip(y, min_score, max_score))
        #     values[name]["low"].append(np.clip(low, min_score, max_score))
        #     values[name]["high"].append(np.clip(high, min_score, max_score))
        # # get_logger().info("Scores {}".format(values))
        #
        # plt.fill_between(
        #     values[name]["x"],
        #     np.clip(values[name]["low"], min_score, max_score),
        #     np.clip(values[name]["high"], min_score, max_score),
        #     alpha=0.25,
        #     zorder=10,
        # )

        values[name] = dict(x=[], y=[])
        for ckpt in data[name]:
            values[name]["x"].append(ckpt["Step"])
            y = ckpt["spl"]
            values[name]["y"].append(np.clip(y, min_score, max_score))
            #     y, low, high = tasks_to_score(ckpt["tasks"], target)

        if it > 2:
            (clhs,) = plt.plot(values[name]["x"], values[name]["y"], zorder=11)
            lhs.append(clhs)
        else:
            plt.plot(values[name]["x"], values[name]["y"], zorder=11)

        if it == 2:
            prefix = "-Pointnav-Depth"
            legend = [mod_name.replace(prefix, "") for mod_name in METHOD_ORDER[:3]]
            legend = [map[name] if name in map else name for name in legend]
            legend = [name.replace("-DDPPO", "") for name in legend]
            legend = [name.replace("-SimpleConv", "") for name in legend]
            legend = [name.replace("-", " ") for name in legend]
            leg1 = ax.legend(
                legend,
                loc="upper left",
                bbox_to_anchor=(0.35, 0.3, 0.5, 0.6),
                title="Point Nav",
            )

    prefix = "-Objectnav-RGB"
    legend = [mod_name.replace(prefix, "") for mod_name in METHOD_ORDER[3:]]
    legend = [map[name] if name in map else name for name in legend]
    legend = [name.replace("-DDPPO", "") for name in legend]
    legend = [name.replace("-ResNet", "") for name in legend]
    legend = [name.replace("-", " ") for name in legend]
    leg2 = ax.legend(
        lhs,
        legend,
        bbox_to_anchor=(0.35, 0.0, 0.5, 0.52),
        loc="upper left",
        title="Object Nav",
    )
    ax.add_artist(leg1)
    # plt.grid(linewidth=0.2, zorder=0)
    plt.grid(
        b=True,
        which="major",
        color=np.array([0.93, 0.93, 0.93]),
        linestyle="-",
        zorder=-2,
    )
    plt.minorticks_on()
    plt.grid(
        b=True,
        which="minor",
        color=np.array([0.97, 0.97, 0.97]),
        linestyle="-",
        zorder=-2,
    )

    # plt.title(title)
    plt.xlabel("Training steps", fontsize=MEDIUM_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=MEDIUM_SIZE)

    set_size(4, 5)

    plt.tight_layout()

    plt.show()

    get_logger().info("Done")


@task
def thor_habitat_success2(
    ctx,
    folder="/Users/jordis/Downloads/robothor_ithor2",
    target="Success",
    min_score=0.0,
    max_score=1.0,
    prefix="",
    title="RoboTHOR, iTHOR, Habitat (DD-PPO)",
    ylabel="Success rate",
    map=r"iTHOR-ResNet-SimpleConv,iTHOR-SimpleConv-DDPPO",
):
    target = target.split(",")
    map = OrderedDict([tuple(io.split(",")) for io in map.split(";") if "," in io])

    METHOD_ORDER = [
        # "Habitat-Pointnav-Depth-SimpleConv-DDPPO",
        # "iTHOR-Pointnav-Depth-ResNet-SimpleConv",
        # "RoboTHOR-Pointnav-Depth-SimpleConv-DDPPO",
        # "iTHOR-Objectnav-RGB-ResNet-DDPPO",
        # "RoboTHOR-Objectnav-RGB-ResNet-DDPPO",
        "Summary-Habitat-Pointnav-Depth-SimpleConv-DDPPO",
        "Summary-iTHOR-Pointnav-Depth-SimpleConv-DDPPO",
        "Summary-RoboTHOR-Pointnav-Depth-SimpleConv-DDPPO",
        "Summary-iTHOR-Objectnav-RGB-DDPPO",
        "Summary-RoboTHOR-Objectnav-RGB-DDPPO",
    ]
    METHOD_ORDER = [prefix + morder for morder in METHOD_ORDER]

    data = {}
    subs = glob.glob(os.path.join(folder, "*"))
    for sub in subs:
        name = os.path.basename(sub).replace(".json", "")
        with open(sub, "r") as f:
            data[name] = json.load(f)

    get_logger().info("Read {} models".format(len(data)))

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    formatter = FuncFormatter(
        lambda val, tick_pos: "0" if val == 0 else "%3.0fM" % (val * 1e-6)
    )

    # Hide the right and top spines
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MultipleLocator(50e6))

    values = {}
    lhs = []
    for it, name in enumerate(METHOD_ORDER):
        # values[name] = dict(x=[], y=[], low=[], high=[])
        # for ckpt in data[name]:
        #     values[name]["x"].append(ckpt["training_steps"])
        #     y, low, high = tasks_to_score(ckpt["tasks"], target)
        #     values[name]["y"].append(np.clip(y, min_score, max_score))
        #     values[name]["low"].append(np.clip(low, min_score, max_score))
        #     values[name]["high"].append(np.clip(high, min_score, max_score))
        # # get_logger().info("Scores {}".format(values))
        #
        # plt.fill_between(
        #     values[name]["x"],
        #     np.clip(values[name]["low"], min_score, max_score),
        #     np.clip(values[name]["high"], min_score, max_score),
        #     alpha=0.25,
        #     zorder=10,
        # )

        values[name] = dict(x=[], y=[], ymin=[], ymax=[])
        for ckpt in data[name]:
            values[name]["x"].append(ckpt["Step"])
            y = ckpt["Success"]["Mean"]
            values[name]["y"].append(np.clip(y, min_score, max_score))
            ymin = ckpt["Success"]["Min"]
            values[name]["ymin"].append(np.clip(ymin, min_score, max_score))
            ymax = ckpt["Success"]["Max"]
            values[name]["ymax"].append(np.clip(ymax, min_score, max_score))
            #     y, low, high = tasks_to_score(ckpt["tasks"], target)

        plt.fill_between(
            values[name]["x"],
            np.clip(values[name]["ymin"], min_score, max_score),
            np.clip(values[name]["ymax"], min_score, max_score),
            alpha=0.25,
            zorder=10,
        )

        if it > 2:
            (clhs,) = plt.plot(values[name]["x"], values[name]["y"], zorder=11)
            lhs.append(clhs)
        else:
            plt.plot(values[name]["x"], values[name]["y"], zorder=11)

        if it == 2:
            prefix = "-Pointnav-Depth"
            legend = [mod_name.replace(prefix, "") for mod_name in METHOD_ORDER[:3]]
            legend = [map[name] if name in map else name for name in legend]
            legend = [name.replace("-DDPPO", "") for name in legend]
            legend = [name.replace("-SimpleConv", "") for name in legend]
            legend = [name.replace("Summary-", "") for name in legend]
            legend = [name.replace("-", " ") for name in legend]
            leg1 = ax.legend(
                legend,
                loc="upper left",
                #             bbox_to_anchor=(0.5, 0.4, 0.5, 0.5),
                #             title="Point Nav",
                #         )
                #
                # prefix = "-Objectnav-RGB"
                # legend = [mod_name.replace(prefix, "") for mod_name in METHOD_ORDER[3:]]
                # legend = [map[name] if name in map else name for name in legend]
                # legend = [name.replace("-DDPPO", "") for name in legend]
                # legend = [name.replace("-ResNet", "") for name in legend]
                # legend = [name.replace("-", " ") for name in legend]
                # leg2 = ax.legend(
                #     lhs,
                #     legend,
                #     bbox_to_anchor=(0.5, 0.0, 0.5, 0.5),
                bbox_to_anchor=(0.35, 0.3, 0.5, 0.6),
                title="Point Nav",
            )

        prefix = "-Objectnav-RGB"
        legend = [mod_name.replace(prefix, "") for mod_name in METHOD_ORDER[3:]]
        legend = [map[name] if name in map else name for name in legend]
        legend = [name.replace("-DDPPO", "") for name in legend]
        legend = [name.replace("-ResNet", "") for name in legend]
        legend = [name.replace("Summary-", "") for name in legend]
        legend = [name.replace("-", " ") for name in legend]
        leg2 = ax.legend(
            lhs,
            legend,
            bbox_to_anchor=(0.35, 0.0, 0.5, 0.52),
            loc="lower left",
            title="Object Nav",
        )
    ax.add_artist(leg1)
    # plt.grid(linewidth=0.2, zorder=0)
    plt.grid(
        b=True,
        which="major",
        color=np.array([0.93, 0.93, 0.93]),
        linestyle="-",
        zorder=-2,
    )
    plt.minorticks_on()
    plt.grid(
        b=True,
        which="minor",
        color=np.array([0.97, 0.97, 0.97]),
        linestyle="-",
        zorder=-2,
    )

    # plt.title(title)
    plt.xlabel("Training steps", fontsize=MEDIUM_SIZE)
    if ylabel is not None:
        plt.ylabel("Success rate", fontsize=MEDIUM_SIZE)

    set_size(4, 5)

    plt.tight_layout()

    plt.show()

    get_logger().info("Done")


@task
def thor_habitat_spl2(
    ctx,
    folder="/Users/jordis/Downloads/robothor_ithor2",
    target="spl",
    min_score=0.0,
    max_score=1.0,
    prefix="",
    title="RoboTHOR, iTHOR, Habitat (DD-PPO)",
    ylabel="SPL",
    map=r"iTHOR-ResNet-SimpleConv,iTHOR-SimpleConv-DDPPO",
):
    target = target.split(",")
    map = OrderedDict([tuple(io.split(",")) for io in map.split(";") if "," in io])

    METHOD_ORDER = [
        # "Habitat-Pointnav-Depth-SimpleConv-DDPPO",
        # "iTHOR-Pointnav-Depth-ResNet-SimpleConv",
        # "RoboTHOR-Pointnav-Depth-SimpleConv-DDPPO",
        # "iTHOR-Objectnav-RGB-ResNet-DDPPO",
        # "RoboTHOR-Objectnav-RGB-ResNet-DDPPO",
        "Summary-Habitat-Pointnav-Depth-SimpleConv-DDPPO",
        "Summary-iTHOR-Pointnav-Depth-SimpleConv-DDPPO",
        "Summary-RoboTHOR-Pointnav-Depth-SimpleConv-DDPPO",
        "Summary-iTHOR-Objectnav-RGB-DDPPO",
        "Summary-RoboTHOR-Objectnav-RGB-DDPPO",
    ]
    METHOD_ORDER = [prefix + morder for morder in METHOD_ORDER]

    data = {}
    subs = glob.glob(os.path.join(folder, "*"))
    for sub in subs:
        name = os.path.basename(sub).replace(".json", "")
        with open(sub, "r") as f:
            data[name] = json.load(f)

    get_logger().info("Read {} models".format(len(data)))

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    formatter = FuncFormatter(
        lambda val, tick_pos: "0" if val == 0 else "%3.0fM" % (val * 1e-6)
    )

    # Hide the right and top spines
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MultipleLocator(50e6))

    values = {}
    lhs = []
    for it, name in enumerate(METHOD_ORDER):
        # values[name] = dict(x=[], y=[], low=[], high=[])
        # for ckpt in data[name]:
        #     values[name]["x"].append(ckpt["training_steps"])
        #     y, low, high = tasks_to_score(ckpt["tasks"], target)
        #     values[name]["y"].append(np.clip(y, min_score, max_score))
        #     values[name]["low"].append(np.clip(low, min_score, max_score))
        #     values[name]["high"].append(np.clip(high, min_score, max_score))
        # # get_logger().info("Scores {}".format(values))
        #
        # plt.fill_between(
        #     values[name]["x"],
        #     np.clip(values[name]["low"], min_score, max_score),
        #     np.clip(values[name]["high"], min_score, max_score),
        #     alpha=0.25,
        #     zorder=10,
        # )

        # values[name] = dict(x=[], y=[])
        # for ckpt in data[name]:
        #     values[name]["x"].append(ckpt["Step"])
        #     y = ckpt["spl"]
        #     values[name]["y"].append(np.clip(y, min_score, max_score))
        #     #     y, low, high = tasks_to_score(ckpt["tasks"], target)

        values[name] = dict(x=[], y=[], ymin=[], ymax=[])
        for ckpt in data[name]:
            values[name]["x"].append(ckpt["Step"])
            y = ckpt["spl"]["Mean"]
            values[name]["y"].append(np.clip(y, min_score, max_score))
            ymin = ckpt["spl"]["Min"]
            values[name]["ymin"].append(np.clip(ymin, min_score, max_score))
            ymax = ckpt["spl"]["Max"]
            values[name]["ymax"].append(np.clip(ymax, min_score, max_score))
            #     y, low, high = tasks_to_score(ckpt["tasks"], target)

        plt.fill_between(
            values[name]["x"],
            np.clip(values[name]["ymin"], min_score, max_score),
            np.clip(values[name]["ymax"], min_score, max_score),
            alpha=0.25,
            zorder=10,
        )

        if it > 2:
            (clhs,) = plt.plot(values[name]["x"], values[name]["y"], zorder=11)
            lhs.append(clhs)
        else:
            plt.plot(values[name]["x"], values[name]["y"], zorder=11)

        if it == 2:
            prefix = "-Pointnav-Depth"
            legend = [mod_name.replace(prefix, "") for mod_name in METHOD_ORDER[:3]]
            legend = [map[name] if name in map else name for name in legend]
            legend = [name.replace("-DDPPO", "") for name in legend]
            legend = [name.replace("-SimpleConv", "") for name in legend]
            legend = [name.replace("Summary-", "") for name in legend]
            legend = [name.replace("-", " ") for name in legend]
            leg1 = ax.legend(
                legend,
                loc="upper left",
                bbox_to_anchor=(0.35, 0.3, 0.5, 0.6),
                title="Point Nav",
            )

    prefix = "-Objectnav-RGB"
    legend = [mod_name.replace(prefix, "") for mod_name in METHOD_ORDER[3:]]
    legend = [map[name] if name in map else name for name in legend]
    legend = [name.replace("-DDPPO", "") for name in legend]
    legend = [name.replace("-ResNet", "") for name in legend]
    legend = [name.replace("Summary-", "") for name in legend]
    legend = [name.replace("-", " ") for name in legend]
    leg2 = ax.legend(
        lhs,
        legend,
        bbox_to_anchor=(0.35, 0.0, 0.5, 0.52),
        loc="upper left",
        title="Object Nav",
    )
    ax.add_artist(leg1)
    # plt.grid(linewidth=0.2, zorder=0)
    plt.grid(
        b=True,
        which="major",
        color=np.array([0.93, 0.93, 0.93]),
        linestyle="-",
        zorder=-2,
    )
    plt.minorticks_on()
    plt.grid(
        b=True,
        which="minor",
        color=np.array([0.97, 0.97, 0.97]),
        linestyle="-",
        zorder=-2,
    )

    # plt.title(title)
    plt.xlabel("Training steps", fontsize=MEDIUM_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=MEDIUM_SIZE)

    set_size(4, 5)

    plt.tight_layout()

    plt.show()

    get_logger().info("Done")


@task
def update_checkpoints_for_test(ctx, folder):
    import shutil
    import torch

    files = glob.glob(os.path.join(folder, "*.pt"))
    for file in files:
        dest = file + ".backup"
        if not os.path.exists(dest):
            print("Copying file")
            shutil.copy(file, dest)
        data = torch.load(file, map_location="cpu")
        data["total_steps"] = data["total_steps"] + data["step_count"]
        torch.save(data, file)

    print("Done")


@task
def minigrid(ctx, dump=False):
    from extensions.rl_minigrid.minigrid_tasks import MiniGridTaskSampler, MiniGridTask
    from extensions.rl_minigrid.minigrid_sensors import EgocentricMiniGridSensor
    from extensions.rl_minigrid.minigrid_environments import FastCrossing
    from gym_minigrid.envs import LavaCrossingEnv, EmptyRandomEnv5x5
    from extensions.rl_minigrid.minigrid_models import MiniGridSimpleConvRNN
    import gym
    import torch
    import cv2

    sampler = MiniGridTaskSampler(
        # env_class=lambda *args, **kwargs: FastCrossing(*args, **kwargs),
        # env_class=lambda *args, **kwargs: LavaCrossingEnv(*args, **kwargs),
        env_class=lambda *args, **kwargs: EmptyRandomEnv5x5(),
        sensors=[EgocentricMiniGridSensor(agent_view_size=10, view_channels=3),],
        # sensors=[],
        env_info=dict(),
    )

    get_logger().info("{}".format(MiniGridTask.action_space))

    # model = MiniGridSimpleConvRNN(
    #     action_space=gym.spaces.Discrete(len(MiniGridTask.class_action_names())),
    #     observation_space=sampler.sensors.observation_spaces,
    #     num_objects=sampler.sensors.get("minigrid_ego_image").num_objects,
    #     num_colors=sampler.sensors.get("minigrid_ego_image").num_colors,
    #     num_states=sampler.sensors.get("minigrid_ego_image").num_states,
    # )

    for taskit in range(10):
        stepcount = 0
        task = sampler.next_task()
        while True:
            if not dump:
                task.env.render()
            else:
                img = task.env.render(mode="rgb")
                cv2.imwrite(
                    "/Users/jordis/Desktop/viz/task{}step{}.png".format(
                        taskit, stepcount
                    ),
                    img[:, :, ::-1],
                )
            stepcount += 1
            # while True:
            #     pass
            # obs = task.get_observations()
            # obs["minigrid_ego_image"] = torch.from_numpy(
            #     obs["minigrid_ego_image"]
            # ).unsqueeze(0)
            # get_logger().info(
            #     "obs {} {}".format(obs.keys(), obs["minigrid_ego_image"].shape)
            # )

            action = task.query_expert()[0]
            task.step(action)
            if task.is_done():
                if not dump:
                    task.env.render()
                else:
                    img = task.env.render(mode="rgb")
                    cv2.imwrite(
                        "/Users/jordis/Desktop/viz/task{}step{}.png".format(
                            taskit, stepcount
                        ),
                        img[:, :, ::-1],
                    )
                break

    get_logger().info("Done")


@task
def luca(ctx,):
    import torch.multiprocessing as mp

    class BaseBabyAIGoToObjExperimentConfig:
        MY_GREAT_VARIABLE = 3

    cfg1 = BaseBabyAIGoToObjExperimentConfig()
    print("main1", cfg1.MY_GREAT_VARIABLE)

    BaseBabyAIGoToObjExperimentConfig.MY_GREAT_VARIABLE = 5

    cfg = BaseBabyAIGoToObjExperimentConfig()

    def my_func(cfg2=None):
        cfg2 = BaseBabyAIGoToObjExperimentConfig()  # this will make 3?
        print(cfg2.MY_GREAT_VARIABLE)

    mp.Process(target=my_func, kwargs=dict(cfg2=None)).start()
    print("main", cfg.MY_GREAT_VARIABLE)
    print("main1", cfg1.MY_GREAT_VARIABLE)


@task
def make_grid(ctx, files="/Users/jordis/Desktop/viz/task{}step{}.png", task=4, steps=6):
    import cv2
    import numpy as np

    ims = [cv2.imread(files.format(task, step)) for step in range(steps)]

    space = 255 * np.ones_like(ims[0])[:, : int(round(ims[0].shape[1] / 8)), :]

    imspace = []
    for step in range(steps - 1):
        imspace.append(ims[step])
        imspace.append(space)
    imspace.append(ims[-1])

    im = np.concatenate(imspace, axis=1)

    cv2.imwrite("/Users/jordis/Desktop/viz/task{}tiled.png".format(task), im)
