import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import importlib
sys.path.append('../code_proc/')
import aisst
importlib.reload(aisst)

plt.switch_backend('Agg')

def plot_variance_correlation(region='NorthSea', csv='models.csv'):
    #==============================================================================
    # load validation data
    #==============================================================================

    tensor_in = aisst.load_tensor_from_csv('../data/ml_training/' + region + '/ml_input_data_valtest.csv')
    tensor_out = aisst.load_tensor_from_csv('../data/ml_training/' + region + '/ml_output_data_valtest.csv')

    input_mean, input_std, output_mean, output_std = aisst.load_normalization_parameters(region)

    tensor_in_n = aisst.normalize_data(tensor_in, input_mean, input_std)


    variance_ref = 2 * np.sum(np.abs(tensor_out.numpy()[:,1:]) ** 2, axis=1)

    #==============================================================================
    # Load models
    #==============================================================================

    models_df = pd.read_csv('../models/' + region + '/' + csv)

    N = len(models_df)
    figcols = int(np.ceil(N/2))
    if figcols == 1:
        figcols = 2
    fig, ax = plt.subplots(2, figcols , figsize=(3.0*figcols*.9, 3*2*.9))
    fig.subplots_adjust(hspace=0.04, wspace=0.04, bottom=0.1, top=0.99, left=0.1, right=0.99)

    xl = [-2.5, 0.4];

    abc='abcdefghijklmnopqrstuvwxyz'

    for i in range(N):

        # input data depndeing on feature set
        if 'feature set' in models_df.columns:    
            feature_set = models_df.loc[i, 'feature set']
        else:
            feature_set = 'full'

        # input data depending on feature set
        t_in = aisst.reduce_feature_set(tensor_in, feature_set)
        t_in_n = aisst.reduce_feature_set(tensor_in_n, feature_set)

        input_size = t_in.shape[1]
        output_size = tensor_out.shape[1]


        if models_df.loc[i, 'type'] == 'NN':
            loss_type = models_df.loc[i, 'loss fn']
            epochs = models_df.loc[i, 'epochs']
            drop_prop = models_df.loc[i, 'drop_out']
            shape = models_df.loc[i, 'shape']
            if 'x' in shape:
                number_hidden_layers = int(shape.split('x')[0])
                hidden_size = int(shape.split('x')[1])

            else: # this could later be used for more complex shapes
                continue

            model = aisst.load_model(number_hidden_layers, hidden_size, loss_type, epochs=epochs, input_size=input_size, output_size=output_size, drop_prop=drop_prop, idir='../models/' + region + '/' + feature_set + '/')
            if loss_type == 'fft':
                pre = model(t_in_n).detach().numpy()
            else:
                pre = model(t_in_n).detach()
                pre = aisst.denormalize_data(pre, output_mean, output_std).numpy()
        else : # for skit-learn models
            model = joblib.load(models_df.loc[i, 'file'])
            pre = model.predict(t_in.numpy())
        
        variance = 2 * np.sum(np.abs(pre[:,1:]) ** 2, axis=1)
        corr = np.corrcoef(variance_ref, variance)[0,1]

        # upper quartile of variance
        sort_idx = np.argsort(variance_ref)
        Npoints = len(variance_ref)
        idx = sort_idx[int(0.75 * Npoints):]
        corr_upper = np.corrcoef(variance_ref[idx], variance[idx])[0,1]
        idx = sort_idx[:int(0.25 * Npoints)]
        corr_lower = np.corrcoef(variance_ref[idx], variance[idx])[0,1]


        #---------------plotting-----------------

        cmap='RdPu'
        cmap='bone_r'
        cmap='binary'
        cmap='Greys'
        ax[i%2, i//2].hist2d(np.log10(variance_ref), np.log10(variance), bins=30, cmap=cmap, norm='log')
        # ax[i%2, i//2].hist2d(np.log10(variance_ref), np.log10(variance), bins=30, cmap=cmap)

        msize = 7; malpha = 1.0
        ax[i%2, i//2].plot(np.log10(np.median(variance_ref)), np.log10(np.median(variance)), 'o', markersize=msize, alpha=malpha)
        ax[i%2, i//2].plot(np.log10(np.mean(variance_ref)), np.log10(np.mean(variance)), 'x', markersize=msize, alpha=malpha)
        ax[i%2, i//2].plot(np.log10(np.percentile(variance_ref, 90)), np.log10(np.percentile(variance,90)), '^', markersize=msize, alpha=malpha)
        ax[i%2, i//2].plot(np.log10(np.percentile(variance_ref, 10)), np.log10(np.percentile(variance,10)), 'v', markersize=msize, alpha=malpha)
        ax[i%2, i//2].plot(np.log10(np.percentile(variance_ref, 99)), np.log10(np.percentile(variance,99)), '>', markersize=msize, alpha=malpha)
        ax[i%2, i//2].plot(np.log10(np.percentile(variance_ref,  1)), np.log10(np.percentile(variance, 1)), '<', markersize=msize, alpha=malpha)

        ax[i%2, i//2].set_aspect('equal')

        # ax[a].set_title(data['name'][i])
        ax[i%2, i//2].text(0.01, 0.9, models_df.loc[i, 'name'], transform=ax[i%2, i//2].transAxes)
        ax[i%2, i//2].text(0.99, 0.03, 'ERA5 reference', transform=ax[i%2, i//2].transAxes, ha='right', va='bottom')
        if i%2 == 1:
            ax[i%2, i//2].set_xlabel(r'log$_{10} \sigma^2_T$')
        else:
            ax[i%2, i//2].set_xticklabels([])
        if i//2 == 0:
            ax[i%2, i//2].set_ylabel(r'log$_{10} \sigma^2_T$')
        else:
            ax[i%2, i//2].set_yticklabels([])

        # cal correlation
        ax[i%2, i//2].text(0.99, 0.32, r'$r_u$={:.2f}'.format(corr_upper), ha='right', va='center', color=[0.6,0.,0.], transform=ax[i%2, i//2].transAxes, fontsize=12)
        ax[i%2, i//2].text(0.99, 0.25, r'$r_a$={:.2f}'.format(corr), ha='right', va='center', transform=ax[i%2, i//2].transAxes, fontsize=12)
        ax[i%2, i//2].text(0.99, 0.18, r'$r_l$={:.2f}'.format(corr_lower), ha='right', va='center', color=[0.,0.,0.6], transform=ax[i%2, i//2].transAxes, fontsize=12)
        # cal r^2


        # one-to-one line
        ax[i%2, i//2].plot(xl, xl, 'k--')
        

        ax[i%2, i//2].set_xlim(xl)
        ax[i%2, i//2].set_ylim(xl)

        ax[i%2, i//2].text(0.02, 0.5, '(' + abc[i] + ')', transform = ax[i%2, i//2].transAxes, fontsize=13, fontweight='bold', ha='left')


    # plt.tight_layout()

    fig.savefig('../figures/compare_variance_' + region + '_'  + csv.split('.')[0]  + '.png', dpi=200, facecolor='w', edgecolor='w')


if __name__ == '__main__':
    region = sys.argv[1]
    csv = sys.argv[2]
    print( "Plotting variance and correlation for region: ", region, " and csv: ", csv)
    plot_variance_correlation(region, csv)






