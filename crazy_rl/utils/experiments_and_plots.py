"""Utilities for running experiments and plotting results."""
import os

import expt
import pandas as pd
from expt import Hypothesis
from matplotlib import pyplot as plt


def save_results(returns, exp_name, seed):
    """Saves the results of an experiment to a csv file.

    Args:
        returns: a list of triples (timesteps, time, episodic_return)
        exp_name: experiment name
        seed: seed of the experiment
    """
    if not os.path.exists("results"):
        os.makedirs("results")
    filename = f"results/results_{exp_name}_{seed}.csv"
    print(f"Saving results to {filename}")
    df = pd.DataFrame(returns)
    df.columns = ["Total timesteps", "Time", "Episodic return"]
    df.to_csv(filename, index=False)


def _ci(hypothesis: Hypothesis):
    group = hypothesis.grouped
    mean, sem = group.mean(), group.sem()
    return (mean - 1.96 * sem, mean + 1.96 * sem)


def load_and_plot(exp_names, env_name):
    """Loads the results of multiples experiments and plots them.

    Args:
        exp_names: a dictionary mapping experiment names to file patterns
        env_name: the name of the environment
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=False)

    colors = [
        "#5CB5FF",
        "#D55E00",
        "#009E73",
        "#e6194b",
    ]
    colors = colors[: len(exp_names)]

    ex = expt.Experiment(env_name)
    print(os.getcwd())
    for exp_name, exp_file_pattern in exp_names.items():
        runs = expt.get_runs(exp_file_pattern)
        for run in runs:
            print(run.summary())
        h = runs.to_hypothesis(exp_name)
        ex.add_hypothesis(h)

    print(ex.summary())

    ex.plot(
        ax=ax[0],
        x="Total timesteps",
        y="Episodic return",
        err_style="fill",
        err_fn=_ci,
        legend=False,
        std_alpha=0.1,
        rolling=100,
        n_samples=10000,
        colors=colors,
        linewidth=2,
    )
    ex.plot(
        ax=ax[1],
        x="Time",
        y="Episodic return",
        err_style="fill",
        err_fn=_ci,
        legend=False,
        std_alpha=0.1,
        rolling=100,
        n_samples=10000,
        colors=colors,
        linewidth=2,
    )

    ax[0].set_title("")
    ax[1].set_title("")
    ax[0].set_xlabel("Total timesteps", fontsize=16)
    ax[1].set_xlabel("Time (seconds)", fontsize=16)
    ax[1].set_ylabel("")
    ax[0].set_ylabel("")
    fig.supylabel("Episodic return", fontsize=16)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=fig.transFigure,
        ncols=3,
        fontsize="15",
    )
    fig.tight_layout()
    fig.savefig(f"results/{env_name}.png", bbox_inches="tight")
    fig.savefig(f"results/{env_name}.pdf", bbox_inches="tight")


def plot_training_time_mo(exp_names):
    """Plot for training time when training multiple policies.

    Args:
        exp_names: a dictionary mapping experiment names to file patterns
        file_pattern: file pattern to match the results
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), sharex=True)

    colors = [
        "#5CB5FF",
        # "#D55E00",
        "#e6194b",
        "#009E73",
    ]

    ex = expt.Experiment("MOMAX")
    print(os.getcwd())
    for exp_name, exp_file_pattern in exp_names.items():
        runs = expt.get_runs(exp_file_pattern)
        for run in runs:
            print(run.summary())
        h = runs.to_hypothesis(exp_name)
        ex.add_hypothesis(h)

    # cpu_env_training_time = [
    #     7264.94,
    #     7239.02,
    #     7220.11,
    #     7194.72,
    #     7249.48,
    #     7215.46,
    #     7235.19,
    #     7210.7,
    #     7251.2,
    #     7205.2,
    # ]
    # mean = sum(cpu_env_training_time) / len(cpu_env_training_time)
    # df = pd.DataFrame({"Training time": [mean] * 7, "Number of policies": [1, 5, 10, 15, 20, 25, 30]})
    # runs_cpu = expt.Run("MAPPO CPU env (1 policy)", df)
    # print(runs_cpu)
    # h = runs_cpu.to_hypothesis()
    # print(h)
    # ex.add_hypothesis(h)

    print(ex.summary())
    ax.set_prop_cycle(linestyle=[(0, (3, 5, 1, 5, 1, 5)), (0, (3, 5, 1, 5)), (0, (5, 10))])
    # for ax in axs:
    ex.plot(
        ax=ax,
        x="Number of policies",
        y="Training time",
        err_style="fill",
        err_fn=_ci,
        legend=False,
        std_alpha=0.2,
        # rolling=100,
        # n_samples=10000,
        colors=colors,
        linewidth=2,
    )
    # Plots a line with the function x = y
    ax.plot([0, 30], [0, 10 * 30], ls="-.", c=".3", alpha=0.7, label="Linear scaling", linewidth=2)
    # ax.plot([0, 30], [7264, 7264], ls="--", c=".3", label="Training 1 MAPPO policy (CPU env)")

    # axs[0].set_ylim([7000, 7500])
    ax.set_ylim([0, 85])

    # ax = axs[0]
    # ax2 = axs[1]

    # ax.spines["bottom"].set_visible(False)
    # ax2.spines["top"].set_visible(False)
    # ax.xaxis.tick_top()
    # ax.tick_params(labeltop=False)  # don't put tick labels at the top
    # ax2.xaxis.tick_bottom()

    # d = 0.01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
    # ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    # ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # for ax in axs:
    ax.set_title("")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.supylabel("Training time (seconds)", fontsize=16)
    fig.supxlabel("Number of policies (3M steps per policy)", fontsize=16)
    h, l = ax.get_legend_handles_labels()
    # matplotlib.rcParams.update({"font.size": 25})
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure, ncols=4, fontsize="16")
    fig.tight_layout()
    fig.savefig("results/mo/training_time.png", bbox_inches="tight")
    fig.savefig("results/mo/training_time.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # load_and_plot(
    #     {
    #         "MAPPO CPU (1 env)": "results/results_MAPPO_CPU_*",
    #         "MAPPO Full GPU (1 env)": "results/results_MAPPO_GPU_Circle_(1env*",
    #         "MAPPO Full GPU (10 envs)": "results/results_MAPPO_GPU_Circle_(10envs*",
    #         # "MAPPO GPU (20 envs)": "results/results_MAPPO_GPU_Circle_(20envs*",
    #         # "MAPPO GPU (128 envs)": "results/results_MAPPO_GPU_Circle_(128envs*",
    #     },
    #     "Circle",
    # )

    plot_training_time_mo(
        {
            "Surround": "results/mo/training_time_surround_*",
            "Escort": "results/mo/training_time_surround_*",
            "Catch": "results/mo/training_time_surround_*",
        }
    )
