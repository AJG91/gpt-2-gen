import numpy as np
import matplotlib.pyplot as plt
from typing import Any

def plot_perplexity(
    log_history: list[dict[str, Any]], 
    path: str, 
    dpi: int,
    figsize: tuple = (8, 10)
    ):
    """
    Plots the training and validation perplexity over steps.

    Parameters
    ----------
    log_history : list[dict[str, Any]]
        A list of log dictionaries.
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    figsize : Tuple, optional (default=(10, 8))
        Specifies the figure size.
    """
    train_perplexity, valid_perplexity = [], []
    train_steps, valid_steps = [], []

    for log in log_history:
        if "loss" in log:
            train_perplexity.append(np.exp(log["loss"]))
            train_steps.append(log["step"])
        if "eval_loss" in log:
            valid_perplexity.append(np.exp(log["eval_loss"]))
            valid_steps.append(log["step"])

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=figsize)

    ax1.plot(train_steps, train_perplexity)
    ax1.set_xlabel("Steps", fontsize=16)
    ax1.set_ylabel("Training perplexity", fontsize=16)

    ax2.plot(valid_steps, valid_perplexity)
    ax2.set_xlabel("Steps", fontsize=16)
    ax2.set_ylabel("Validation perplexity", fontsize=16)
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"perplexity_curve.png", bbox_inches="tight", dpi=dpi)