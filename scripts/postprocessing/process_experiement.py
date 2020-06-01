import numpy as np
import pandas as pd
import glob
import json
import os
import sys
sys.path.append('../../')
import click

from sklearn.metrics import roc_auc_score, roc_curve

from src.utils.plot_utils import metric_barplot
from src.datasets.MURADataset import MURA_TrainValidTestSplitter

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def plot_loss(epoch_loss, ax, title=''):
    """
    Plot loss evolutoin givenin epoch_loss.
    """
    ax.plot(epoch_loss[:,0], epoch_loss[:,1], color='darkgray', lw=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Epoch [-]', fontsize=12)
    ax.set_ylabel('Loss [-]', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(left=0)

def plot_tSNE_bodypart(tsne2D, body_part, ax, title='', legend=False):
    """
    plot a 2D t-SNE by body part.
    """
    cmap = matplotlib.cm.get_cmap('Set2')
    color_list = cmap(np.linspace(0.1,0.9,7))

    for bp, color in zip(np.unique(body_part), color_list):
        ax.scatter(tsne2D[body_part == bp, 0],
                   tsne2D[body_part == bp, 1],
                   s=10, color=color, marker='.', alpha=0.8)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight='bold')

    if legend:
        handles = [matplotlib.patches.Patch(facecolor=color) for color in color_list]
        leg_name = [bp.title() for bp in np.unique(body_part)]
        ax.legend(handles, leg_name, ncol=len(np.unique(body_part)), loc='upper center', frameon=False,
                  fontsize=12, bbox_to_anchor=(0.5, 0), bbox_transform=ax.transAxes)

def plot_tSNE_label(tsne2D, labels, ax, title='', legend=False):
    """
    plot a 2D t-SNE by labels.
    """
    color_dict = {1: 'coral', 0: 'limegreen'}

    for lab, color in color_dict.items():
        ax.scatter(tsne2D[labels == lab, 0],
                   tsne2D[labels == lab, 1],
                   s=10, color=color, marker='.', alpha=0.5)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight='bold')

    if legend:
        handles = [matplotlib.patches.Patch(facecolor=color, alpha=0.5) for color in color_dict.values()]
        leg_names = ['Normal' if lab == 0 else 'Abnormal' for lab in color_dict.keys()]
        ax.legend(handles, leg_names, ncol=2, loc='upper center', frameon=False,
                  fontsize=12, bbox_to_anchor=(0.5, 0), bbox_transform=ax.transAxes)

def plot_tSNE_sphere(tsne2D, sphere_idx, ax, title='', legend=False):
    """
    plot a 2D t-SNE by sphere.
    """
    N_sphere = len(np.unique(sphere_idx))
    cmap = matplotlib.cm.get_cmap('Set2')
    color_list = cmap(np.linspace(0.1,0.9,N_sphere))

    for idx, color in zip(np.unique(sphere_idx), color_list):
        ax.scatter(tsne2D[sphere_idx == idx, 0],
                   tsne2D[sphere_idx == idx, 1],
                   s=10, color=color, marker='.', alpha=0.5)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight='bold')

    if legend:
        handles = [matplotlib.patches.Patch(facecolor=color, alpha=0.5) for color in color_list]
        leg_names = [f'Sphere {idx+1}' for idx in np.unique(sphere_idx)]
        ax.legend(handles, leg_names, ncol=2, loc='upper center', frameon=False,
                  fontsize=12, bbox_to_anchor=(0.5, 0), bbox_transform=ax.transAxes)

def plot_sphere_dist(df_samples, ax, title=''):
    """
    Plot the distribution of body part by sphere.
    """
    # prepare data
    BodyPart_list = np.sort(df_samples.body_part.unique())
    sphere_list = np.sort(df_samples.sphere_idx.unique())
    idx_df = pd.MultiIndex.from_product([BodyPart_list, sphere_list])

    df_count = df_samples.groupby(by=['body_part', 'sphere_idx']).count() \
                         .reindex(idx_df, fill_value=0) \
                         .iloc[:,0] \
                         .reset_index(level=1) \
                         .rename(columns={'patientID':'Count', 'level_1':'Nsphere'})
    # plot
    # bottom of bar
    prev = [0]*len(df_count.Nsphere.unique())
    # colors
    cmap = matplotlib.cm.get_cmap('Spectral')
    color_list = cmap(np.linspace(0, 1, 7))

    for part, color in zip(BodyPart_list, color_list):
        ax.barh([f'Sphere {i+1}' for i in df_count.Nsphere.unique()], df_count.loc[part, 'Count'],
               height=0.8, left=prev, color=color, ec='k', lw=0, label=part.title())
        if len(sphere_list) > 1:
            prev += df_count.loc[part, 'Count'].values
        else:
            prev += df_count.loc[part, 'Count']

    for i, count in enumerate(prev):
        ax.text(count+50, i, str(count), va='center', fontsize=10, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=7, loc='lower center', bbox_to_anchor=(0.5, -0.25), frameon=False)
    ax.set_xlabel('Counts [-]', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')

def plot_score_dist(scores, labels, ax, title='', legend=False):
    """
    Plot the score distribution by labels.
    """
    ax.hist(scores[labels == 1],
            bins=40, density=False, log=True,
            range=(scores.min(), scores.max()),
            color='coral', alpha=0.4)

    # plot normal distribution
    ax.hist(scores[labels == 0],
            bins=40, density=False, log=True,
            range=(scores.min(), scores.max()),
            color='limegreen', alpha=0.4)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('anomaly score [-]', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if legend:
        handles = [matplotlib.patches.Patch(facecolor=color, alpha=0.4) for color in ['limegreen', 'coral']]
        leg_names = ['Normal', 'Abnormal']
        ax.legend(handles, leg_names, ncol=1, loc='upper right', frameon=False,
                  fontsize=12, bbox_to_anchor=(1.5, 1), bbox_transform=ax.transAxes)

def plot_ROC(scores, labels, set, ax, title=''):
    """
    Plot the test and validation ROC curves.
    """
    # ROC curve valid and test
    for set_i in np.unique(set):
        fpr, tpr, thres = roc_curve(labels[set == set_i], scores[set == set_i])
        ax.plot(fpr, tpr, color='coral', lw=1)
        ax.fill_between(fpr, tpr, facecolor='coral', alpha=0.05)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Posiitve Rate', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

def get_AUC_list(scores, labels, body_part):
    """
    """
    auc_list = [roc_auc_score(labels, scores)]
    name_list = ['All']
    for bp in np.unique(body_part):
        auc_list.append(roc_auc_score(labels[body_part == bp], scores[body_part == bp]))
        name_list.append(bp.title())

    return np.array(auc_list).reshape(1,-1), name_list

@click.command()
@click.argument('expe_folder', type=click.Path(exists=True))
@click.option('--rep', type=int, default=1, help='The replicate number to process. Default: 1.')
@click.option('--data_info_path', type=click.Path(exists=True), default='../../../data/data_info.csv',
              help='Where to find the info dataframe. Default: ../../../data/data_info.csv')
@click.option('--pretrain', type=click.Choice(['SimCLR', 'AE']), default='SimCLR',
              help='The pretraining method used. Default SimCLR.')
@click.option('--model', type=click.Choice(['DSAD', 'DMSAD']), default='DSAD',
              help='The anomaly detection model used. Default DSAD.')
@click.option('--frac_abnormal', type=float, default=0.05,
              help='The fraction of known abnormal samples. Default: 0.05.')
def main(expe_folder, rep, data_info_path, pretrain, model, frac_abnormal):
    """
    Generate a Figure for the results of the provided SimCLR-AD or AE-AD experiment.
    """
    if not os.path.isdir(expe_folder + 'analysis/'): os.makedirs(expe_folder + 'analysis/', exist_ok=True)
    ################################# LOAD DATA ################################
    # load data_info
    df_info = pd.read_csv(data_info_path)
    df_info = df_info.drop(df_info.columns[0], axis=1)
    df_info = df_info[df_info.low_contrast == 0]
    # Get valid and test set
    spliter = MURA_TrainValidTestSplitter(df_info, train_frac=0.5,
                                          ratio_known_normal=0.05,
                                          ratio_known_abnormal=frac_abnormal, random_state=42)
    spliter.split_data(verbose=False)
    valid_df = spliter.get_subset('valid')
    test_df = spliter.get_subset('test')

    # load results
    with open(expe_folder + f'results/results_{rep}.json', 'r') as f:
        results = json.load(f)

    # concat valid and test 512 and 128 embedding of SimCLR
    df = []
    for set, df_set in zip(['valid', 'test'], [valid_df, test_df]):
        df_tmp = df_set.copy() \
                       .drop(columns=['patient_any_abnormal', 'body_part_abnormal', 'low_contrast', 'semi_label'])

        cols = ['idx', '512_embed', '128_embed'] if pretrain == 'SimCLR' else ['idx', 'label', 'AE_score', '512_embed', '128_embed']

        df_scores = pd.DataFrame(data=results[pretrain][set]['embedding'], columns=cols) \
                      .set_index('idx')
        df_scores['set'] = set
        df.append(pd.merge(df_tmp, df_scores, how='inner', left_index=True, right_index=True))

    # concat valid and test
    df_sim = pd.concat(df, axis=0)

    # concat valid and test scores and embedding of DSAD
    df = []
    for set, df_set in zip(['valid', 'test'], [valid_df, test_df]):
        df_tmp = df_set.copy() \
                       .drop(columns=['patient_any_abnormal', 'body_part_abnormal', 'low_contrast', 'semi_label'])

        cols = ['idx', 'label', 'ad_score', 'sphere_idx','128_embed'] if model == 'DMSAD' else  ['idx', 'label', 'ad_score', '128_embed']

        df_scores = pd.DataFrame(data=results['AD'][set]['scores'], columns=cols) \
                      .set_index('idx') \
                      .drop(columns=['label'])
        df_scores['set'] = set
        df.append(pd.merge(df_tmp, df_scores, how='inner', left_index=True, right_index=True))

    # concat valid and test
    df_ad = pd.concat(df, axis=0)

    ############################# INITIALIZE FIGURE ############################
    if model == 'DSAD':
        fig = plt.figure(figsize=(24,33))
        gs = fig.add_gridspec(nrows=5, ncols=24, hspace=0.4, wspace=5, height_ratios=[2/15, 4/15, 4/15, 2/15, 3/15])
    else:
        fig = plt.figure(figsize=(24,40))
        gs = fig.add_gridspec(nrows=6, ncols=24, hspace=0.4, wspace=5, height_ratios=[2/19, 4/19, 4/19, 4/19, 2/19, 3/19])

    ################################ PLOT LOSS #################################
    ax_loss_sim = fig.add_subplot(gs[0, :12])
    epoch_loss = np.array(results[pretrain]['train']['loss'])
    plot_loss(epoch_loss, ax_loss_sim, title=f'{pretrain} Loss Evolution')

    ax_loss_ad = fig.add_subplot(gs[0, 12:])
    epoch_loss = np.array(results['AD']['train']['loss'])
    plot_loss(epoch_loss, ax_loss_ad, title=f'{model} Loss Evolution')

    ############################### PLOT T-SNE  ################################
    df_sim_val = df_sim[df_sim.set == 'valid']
    embed2D = np.stack(df_sim_val['512_embed'].values, axis=0)
    labels = df_sim_val.abnormal_XR.values
    body_part = df_sim_val.body_part.values
    # by body_part
    ax_repr_sim512 = fig.add_subplot(gs[1, :8])
    plot_tSNE_bodypart(embed2D, body_part, ax_repr_sim512,
              title=f't-SNE Representation of {pretrain} 512-Dimensional Space \nBy Body Part',
              legend=False)
    # by labels
    ax_repr_sim512 = fig.add_subplot(gs[2, :8])
    plot_tSNE_label(embed2D, labels, ax_repr_sim512,
              title=f't-SNE Representation of {pretrain} 512-Dimensional Space\nBy Labels',
              legend=False)

    embed2D = np.stack(df_sim_val['128_embed'].values, axis=0)
    labels = df_sim_val.abnormal_XR.values
    body_part = df_sim_val.body_part.values
    # by body_part
    ax_repr_sim128 = fig.add_subplot(gs[1, 8:16])
    plot_tSNE_bodypart(embed2D, body_part, ax_repr_sim128,
              title=f't-SNE Representation of {pretrain} 128-Dimensional Space \nBy Body Part',
              legend=True)
    # by labels
    ax_repr_sim128 = fig.add_subplot(gs[2, 8:16])
    plot_tSNE_label(embed2D, labels, ax_repr_sim128,
              title=f't-SNE Representation of {pretrain} 128-Dimensional Space\nBy Labels',
              legend=True)

    df_ad_val = df_ad[df_ad.set == 'valid']
    embed2D = np.stack(df_ad_val['128_embed'].values, axis=0)
    labels = df_ad_val.abnormal_XR.values
    body_part = df_ad_val.body_part.values
    # by body part
    ax_repr_AD128 = fig.add_subplot(gs[1, 16:])
    plot_tSNE_bodypart(embed2D, body_part, ax_repr_AD128,
              title=f't-SNE Representation of {model} 128-Dimensional Space \nBy Body Part',
              legend=False)
    # by labels
    ax_repr_AD128 = fig.add_subplot(gs[2, 16:])
    plot_tSNE_label(embed2D, labels, ax_repr_AD128,
              title=f't-SNE Representation of {model} 128-Dimensional Space\nBy Labels',
              legend=False)

    ########################## PLOT SPHERE DIAGNOSTIC ##########################
    if model == 'DMSAD':
        # distribution by sphere
        ax_sphere_dist = fig.add_subplot(gs[3, :16])
        plot_sphere_dist(df_ad_val[df_ad_val.abnormal_XR == 0], ax_sphere_dist, title='Body Part Distribution by Sphere')

        # tSNE by sphere
        embed2D = np.stack(df_ad_val['128_embed'].values, axis=0)
        sphere_index = df_ad_val.sphere_idx.values
        ax_sphere_tSNE = fig.add_subplot(gs[3, 16:])
        plot_tSNE_sphere(embed2D, sphere_index, ax_sphere_tSNE,
                         title=f't-SNE Representation of {model} 128-Dimensional Space\n by Sphere',
                         legend=True)

    ######################### PLOT SCORE DISTRIBUTION  #########################
    df_ad_val = df_ad[df_ad.set == 'valid']
    ad_scores = df_ad_val.ad_score.values
    labels = df_ad_val.abnormal_XR.values
    body_part = df_ad_val.body_part.values
    # all
    ax_score_all = fig.add_subplot(gs[-2, :3])
    plot_score_dist(ad_scores, labels, ax_score_all,
                    title='All Scores', legend=False)
    ax_score_all.set_ylabel('Count [-]', fontsize=12)

    # by body part
    for i, bp in enumerate(np.unique(body_part), start=1):
        ax_score = fig.add_subplot(gs[-2, 3*i:3*(i+1)], sharey=ax_score_all, sharex=ax_score_all)
        plot_score_dist(ad_scores[body_part == bp], labels[body_part == bp], ax_score,
                        title=f'{bp.title()} Scores', legend=False)

    ########################## PLOT AUC AND ROC CURVE  #########################
    # ROC
    ax_roc = fig.add_subplot(gs[-1, :6])
    ad_scores = df_ad.ad_score.values
    labels = df_ad.abnormal_XR.values
    set = df_ad.set.values
    plot_ROC(ad_scores, labels, set, ax_roc, title='ROC curve')

    # AUC Barplot
    ax_auc = fig.add_subplot(gs[-1, 6:])
    ad_scores = df_ad.ad_score.values
    labels = df_ad.abnormal_XR.values
    body_part = df_ad.body_part.values
    set = df_ad.set.values
    valid_auc, names = get_AUC_list(ad_scores[set == 'valid'], labels[set == 'valid'], body_part[set == 'valid'])
    test_auc, names = get_AUC_list(ad_scores[set == 'test'], labels[set == 'test'], body_part[set == 'test'])
    metric_barplot([valid_auc, test_auc], ['Validation', 'Test'], names, ['lightsalmon', 'peachpuff'], gap=1, ax=ax_auc, fontsize=12)
    ax_auc.set_title('Overall AUC scores and AUC by Body Part', fontsize=12, fontweight='bold')

    ################################ SAVE FIGURE ###################################
    fig.savefig(expe_folder + f'analysis/summary_{rep}.pdf', dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    main()
