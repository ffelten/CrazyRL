"""Utilities for running experiments and plotting results."""
import os

import expt
import pandas as pd
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
        h = runs.to_hypothesis(exp_name)
        ex.add_hypothesis(h)

    print(ex.summary())
    ex.plot(
        ax=ax[0],
        x="Total timesteps",
        y="Episodic return",
        err_style="fill",
        legend=False,
        std_alpha=0.1,
        rolling=100,
        n_samples=10000,
        colors=colors,
    )
    ex.plot(
        ax=ax[1],
        x="Time",
        y="Episodic return",
        err_style="fill",
        legend=False,
        std_alpha=0.1,
        rolling=100,
        n_samples=10000,
        colors=colors,
    )

    ax[0].set_title("")
    ax[1].set_title("")
    ax[0].set_xlabel("Total timesteps")
    ax[1].set_xlabel("Time (seconds)")
    ax[1].set_ylabel("")
    ax[0].set_ylabel("")
    fig.supylabel("Episodic return")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure, ncols=len(exp_names))
    fig.tight_layout()
    fig.savefig(f"results/{env_name}.png", bbox_inches="tight")
    # fig.savefig(f"results/{env_name}.pdf", bbox_inches="tight")


def plot_training_time_mo(file_pattern: str = "results/mo/*"):
    """Plot for training time when training multiple policies.

    Args:
        file_pattern: file pattern to match the results
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), sharex=False)

    colors = [
        "#5CB5FF",
        # "#D55E00",
        # "#009E73",
        # "#e6194b",
    ]

    ex = expt.Experiment("MO Surround")
    print(os.getcwd())
    runs = expt.get_runs(file_pattern)
    h = runs.to_hypothesis("MO Surround")
    ex.add_hypothesis(h)

    print(ex.summary())
    ex.plot(
        ax=ax[0],
        x="Number of policies",
        y="Training time",
        err_style="fill",
        legend=False,
        std_alpha=0.1,
        rolling=100,
        n_samples=10000,
        colors=colors,
    )

    ax[0].set_title("")
    ax[0].set_xlabel("Number of policies")
    ax[0].set_ylabel("")
    fig.supylabel("Training time (seconds)")
    h, l = ax[0].get_legend_handles_labels()
    # fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure, ncols=len(exp_names))
    fig.tight_layout()
    fig.savefig("results/mo/training_time.png", bbox_inches="tight")
    # fig.savefig("results/mo/training_time.pdf", bbox_inches="tight")


if __name__ == "__main__":
    load_and_plot(
        {
            "MAPPO CPU (1 env)": "results/results_MAPPO_CPU_*",
            "MAPPO GPU (1 env)": "results/results_MAPPO_GPU_Circle_(1env*",
            # "MAPPO GPU (10 envs)": "results/results_MAPPO_GPU_Circle_(10envs*",
            # "MAPPO GPU (20 envs)": "results/results_MAPPO_GPU_Circle_(20envs*",
        },
        "Circle",
    )
