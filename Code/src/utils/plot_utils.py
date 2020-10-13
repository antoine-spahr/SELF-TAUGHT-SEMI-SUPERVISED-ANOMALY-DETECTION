import numpy as np
import sys
sys.path.append('../')

import matplotlib
import matplotlib.pyplot as plt

def metric_barplot(metrics_scores, serie_names, group_names, colors=None, w=None,
                   ax=None, fontsize=12, jitter=False, jitter_color=None, gap=None,
                   legend_kwargs=None):
    """
    Plot a grouped barplot from the passed array, for various metrics.
    ----------
    INPUT
        |---- metric_scores (list of 2D np.array) the data to plot each element
        |           of the list is a np.array (N_replicats x N_group). The lenght
        |           of the lists gives the number of series plotted.
        |---- series_name (list of str) the names for each series (to appear in
        |           the legend).
        |---- group_names (list of str) the names of the groups (the x-ticks labels).
        |---- colors (list of str) the colors for each series. If None, colors
        |           are randomly picked.
        |---- w (float) the bar width. If None, w is automoatically computed.
        |---- ax (matplotlib Axes) the axes where to plot.
        |---- fontsize (int) the fontsize to use for the texts.
        |---- jitter (bool) whether to plot the points on top as a scatter plot.
        |---- jitter_color (matplotlib color) the color to use for the jitter points.
        |---- gap (float) the x position where to place a gap between the bars.
        |---- legend_kwargs (dict) dictionnary of keyboard arguments for the
        |           legend placement and properties.
    OUTPUT
        |---- None
    """
    # find axes
    ax = plt.gca() if ax is None else ax

    n = len(metrics_scores)
    if colors is None: colors = np.random.choice(list(matplotlib.colors.CSS4_COLORS.keys()), size=n)
    if jitter_color is None: jitter_color = np.random.choice(list(matplotlib.colors.CSS4_COLORS.keys()))

    offsets = list(np.arange(-(n-1),(n-1)+2, 2))
    if w is None: w = 0.9/n
    ind = np.arange(metrics_scores[0].shape[1]) # number of different groups
    if gap:
        ind = np.where(ind + 1 > gap, ind + 0.5, ind)

    for metric, offset, name, color in zip(metrics_scores, offsets, serie_names, colors):
        means = np.nanmean(metric, axis=0)
        stds = np.nanstd(metric, axis=0)
        ax.bar(ind + offset*w/2, means, width=w, yerr=1.96*stds,
               fc=color, ec='black', lw=1, label=name)

        for i, x in enumerate(ind):
            ax.text(x + offset*w/2, means[i]-0.03, f'{means[i]:.2%}', fontsize=fontsize, ha='center', va='top', rotation=90)

        if jitter:
            for j, x in enumerate(ind):
                ax.scatter(np.random.normal(x + offset*w/2, 0.00, metric.shape[0]),
                           metric[:,j], c=jitter_color, marker='o', s=30, lw=0, zorder=5)

    handles, labels = ax.get_legend_handles_labels()
    if jitter:
        handles += [matplotlib.lines.Line2D((0,0),(0,0), lw=0, marker='o',
                    markerfacecolor=jitter_color, markeredgecolor=jitter_color, markersize=7)]
        labels += ['Measures']

    if legend_kwargs is None:
        ax.legend(handles, labels, loc='upper right', ncol=1, frameon=False, framealpha=0.0,
                  fontsize=fontsize, bbox_to_anchor=(1, 1.1), bbox_transform=ax.transAxes)
    elif isinstance(legend_kwargs, dict):
        ax.legend(handles, labels, **legend_kwargs)

    ax.set_xticks(ind)
    ax.set_xticklabels(group_names)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('', fontsize=fontsize)
    ax.set_ylim([0,1])
