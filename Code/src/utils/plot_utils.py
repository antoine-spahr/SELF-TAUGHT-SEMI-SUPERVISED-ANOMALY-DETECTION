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

def add_stat_significance(pairs, data, serie_names, group_names, w=None, mode='adjusted',
    h_offset=0.06, h_gap=0.02, fontsize=12, stat_test='ttest', stat_test_param=dict(equal_var=False, nan_policy='omit'),
    stat_display='symbol', avoid_cross=True, link_color='lightgray', text_color='gray', ax=None, text_rot=0):
    """
    Compute and display significance comparison between two bars of a metric barplot.
    ----------
    INPUT
        |----
    OUTPUT
        |----
    """
    # find axes
    ax = plt.gca() if ax is None else ax
    # get current heights and other inputs
    if mode == 'adjusted':
        heights = [np.nanmean(arr, axis=0)+1.96*np.nanstd(arr, axis=0) for arr in data]
    elif mode == 'flat':
        h = [np.nanmean(arr, axis=0)+1.96*np.nanstd(arr, axis=0) for arr in data]
        max_h = np.concatenate(h, axis=0).max()
        heights = [np.ones(arr.shape[1])*max_h for arr in data]
    elif isinstance(mode, int) or isinstance(mode, float):
        heights = [np.ones(arr.shape[1])*mode for arr in data]
    else:
        raise ValueError(f'Height mode {mode} not supported. Please use flat or adjusted or a digit.')

    # get x data by series and group
    ind = np.arange(data[0].shape[1])
    n = len(data)
    if w is None: w = 0.9/n
    offsets = list(np.arange(-(n-1),(n-1)+2, 2))
    posx = [ind + offset*w/2 for offset in offsets]

    for p in pairs:
        # get index of pair
        s1, g1 = serie_names.index(p[0][1]), group_names.index(p[0][0])
        s2, g2 = serie_names.index(p[1][1]), group_names.index(p[1][0])
        # get data
        data1 = data[s1][:,g1]
        data2 = data[s2][:,g2]
        h1 = heights[s1][g1]
        h2 = heights[s2][g2]

        # get max height between the two bars
        if posx[s1][g1] < posx[s2][g2]:
            gl, sl, hl = g1, s1, h1
            gh, sh, hh = g2, s2, h2
        else:
            gl, sl, hl = g2, s2, h2
            gh, sh, hh = g1, s1, h1

        low = gl * len(serie_names) + sl
        high = gh * len(serie_names) + sh
        heights_arr = np.array(heights).transpose().ravel()[low:high+1]
        x = [posx[sl][gl]]*2 + [posx[sh][gh]]*2
        y = [hl + h_gap, heights_arr.max() + h_offset, heights_arr.max() + h_offset, hh + h_gap]

        # perform test
        if stat_test == 'ttest':
            pval = ttest_ind(data1, data2, **stat_test_param)[1]
        else:
            raise ValueError(f'Usupported statisical test {stat_test}. Supported: ttest.')

        # get string symbol : larger that 10% --> NS ; between 5 and 10% --> . ; between 1 and 5% --> * ; between 0.1 and 1% --> ** ; below 0.1% --> ***
        if stat_display == 'symbol':
            if pval > 0.1:
                significance = 'ns'
            elif pval > 0.05:
                significance = '.'
            elif pval > 0.01:
                significance = '*'
            elif pval > 0.001:
                significance = '**'
            else:
                significance = '***'
        elif stat_display == 'value':
            significance = f'{pval:.2g}'
        else:
            raise ValueError(f'Usupported statisical display type {stat_display}. Supported: symbol or value.')

        # update heights
        if avoid_cross:
            # update all columns between data1 and data2 to avoid any crossing
            if gl != gh:
                for s in range(sl, len(serie_names)):
                    heights[s][gl] = heights_arr.max() + h_offset
                for g in range(gl+1, gh):
                    for s in range(len(serie_names)):
                        heights[s][g] = heights_arr.max() + h_offset
                for s in range(sh+1):
                    heights[s][gh] = heights_arr.max() + h_offset
            else:
                for s in range(sl, sh+1):
                    heights[s][gl] = heights_arr.max() + h_offset
        else:
            # update only data1 and data2 alowing crossing
            heights[s1][g1] = heights_arr.max() + h_offset
            heights[s2][g2] = heights_arr.max() + h_offset

        # plot
        ax.plot(x, y, lw=2, color=link_color)
        ax.text((x[0]+x[-1])/2, y[1], significance, ha='center', va='bottom', fontsize=fontsize, color=text_color, rotation=text_rot)
        ax.set_ylim([0, heights_arr.max()+0.3])
