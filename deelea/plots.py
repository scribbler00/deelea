from math import sqrt

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def setup_plots(FONT_SIZE=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("poster")
    sns.set_style("whitegrid")
    plt.rcParams["axes.grid"] = True
    plt.rc("font", family="Times New Roman")

    # latexify()
    if FONT_SIZE is not None:
        params = {
            "backend": "ps",
            #   'text.latex.preamble': ['\usepackage{gensymb}'],
            # "text.latex.preamble": r"\usepackage{gensymb}",
            "axes.labelsize": FONT_SIZE - 2,  # fontsize for x and y labels (was 10)
            "axes.titlesize": FONT_SIZE,
            "axes.labelweight": "bold",
            "font.size": FONT_SIZE,  # was 10
            "font.weight": "900",  # bold is 700, normal is 400
            "legend.fontsize": FONT_SIZE,  # was 10
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "text.usetex": True,
            # "figure.figsize": [fig_width, fig_height],
            # "font.family": "serif",
        }

        matplotlib.rcParams.update(params)


def latexify(fig_width=None, fig_height=None, columns=1, FONT_SIZE=8):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert columns in [1, 2]

    MAX_HEIGHT_INCHES = 8.0

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    if fig_height > MAX_HEIGHT_INCHES:
        print(
            "WARNING: fig_height too large:"
            + str(fig_height)
            + "so will reduce to"
            + str(MAX_HEIGHT_INCHES)
            + "inches."
        )
        fig_height = MAX_HEIGHT_INCHES

    params = {
        "backend": "ps",
        #   'text.latex.preamble': ['\usepackage{gensymb}'],
        "text.latex.preamble": r"\usepackage{gensymb}",
        "axes.labelsize": FONT_SIZE - 2,  # fontsize for x and y labels (was 10)
        "axes.titlesize": FONT_SIZE,
        "axes.labelweight": "bold",
        "font.size": FONT_SIZE,  # was 10
        "font.weight": "900",  # bold is 700, normal is 400
        "legend.fontsize": FONT_SIZE,  # was 10
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
    }

    matplotlib.rcParams.update(params)


def boxplot_target_season_num_days_training(
    model_name,
    dest_folder,
    data_type,
    cur_data,
    sim_measure="",
    ycol_name="test_rmse",
    save_fig=True,
    lim_y=False,
):
    file_name = f"{dest_folder}/{model_name}_target_{data_type}"
    if len(sim_measure) > 0:
        file_name = f"{file_name}_{sim_measure}"

    cur_data.to_csv(f"{file_name}.csv", sep=";")
    plt.figure(figsize=(16, 9))
    plt.grid()
    sns.boxplot(
        hue="NumDaysTraining", y=ycol_name, x="Season", data=cur_data, showmeans=True
    )
    # plt.yticks(np.linspace(0, 1, 21))
    plt.title(data_type + " " + ("".join(sim_measure.split("_"))).upper())
    plt.grid(True)
    if lim_y:
        if "PV" in data_type:
            plt.ylim((0, 0.2))
        else:
            plt.ylim((0, 0.3))
    if save_fig:
        plt.savefig(f"{file_name}.jpg")
    else:
        plt.close()


def boxplot_target_days_training(
    model_name,
    dest_folder,
    data_type,
    cur_data,
    sim_measure="",
    ycol_name="test_rmse",
    do_save=True,
):
    file_name = f"{dest_folder}/{model_name}_target_{data_type}"
    if len(sim_measure) > 0:
        file_name = f"{file_name}_{sim_measure}"

    cur_data.to_csv(f"{file_name}.csv", sep=";")
    plt.figure(figsize=(16, 9))
    plt.grid()
    sns.boxplot(x="NumDaysTraining", y=ycol_name, data=cur_data, showmeans=True)
    # plt.yticks(np.linspace(0, 1, 21))
    plt.title(data_type + " " + ("".join(sim_measure.split("_"))).upper())
    plt.grid(True)
    if do_save:
        plt.savefig(f"{file_name}.jpg")
