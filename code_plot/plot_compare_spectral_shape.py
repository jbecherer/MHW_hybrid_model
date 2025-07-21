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

def plot_variance_correlation(region='NWEuroShelf', csv='models_2plot.csv'):
    #==============================================================================
    # load validation data
    #==============================================================================

    # tensor_in = aisst.load_tensor_from_csv('../data/ml_training/' + region + '/ml_input_data_valtest.csv')
    # tensor_out = aisst.load_tensor_from_csv('../data/ml_training/' + region + '/ml_output_data_valtest.csv')
    tensor_in = aisst.load_tensor_from_csv('../data/ml_training/' + region + '/ml_input_data_val.csv')
    tensor_out = aisst.load_tensor_from_csv('../data/ml_training/' + region + '/ml_output_data_val.csv')

    input_mean, input_std, output_mean, output_std = aisst.load_normalization_parameters(region)

    tensor_in_n = aisst.normalize_data(tensor_in, input_mean, input_std)


    variance_ref = 2 * np.sum(np.abs(tensor_out.numpy()[:,1:]) ** 2, axis=1)

    # choose quartile of variance
    Npoints = len(variance_ref)
    sort_idx = np.argsort(variance_ref)
    ii = sort_idx[int(0.75 * Npoints):] # use the upper 25% of the data
    ii = sort_idx[:int(0.25 * Npoints)] # use the lower 25% of the data
    ii = sort_idx[int(0.25 * Npoints):int(0.75 * Npoints)]  # use the middle 50% of the data
    ii = sort_idx

    spec_shape_ref = np.log10(( tensor_out.numpy()[ii,]** 2 / variance_ref[ii, None] ).mean(axis=0))
    spec_std_ref = np.log10(( tensor_out.numpy()[ii,]** 2 / variance_ref[ii, None] )).std(axis=0)
    f = np.arange(0, spec_shape_ref.shape[0])
    f = f/ 30 # convert to frequency in 1/day


    #==============================================================================
    # Load models
    #==============================================================================

    models_df = pd.read_csv('../models/' + region + '/' + csv)

    N = len(models_df)

    fig = plt.figure( figsize = (8, 6), facecolor = (1, 1, 1))
    ax = fig.add_subplot(111)
    col = plt.get_cmap("tab10")

    msize = 7; malpha = 1.0

    # plot ERA reference
    ax.plot(f,spec_shape_ref, 'o', color='k', markersize=msize, alpha=malpha, label='ERA5 reference')
    ax.fill_between(f,spec_shape_ref - spec_std_ref, spec_shape_ref + spec_std_ref, alpha=0.3, color='k')

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

        # shape it
        spec_shape = np.log10(( pre[ii,] ** 2 / variance[ii, None] ).mean(axis=0))
        spec_std = np.log10(( pre[ii,] ** 2/ variance[ii, None] )).std(axis=0)


        ax.plot(f, spec_shape, '.', color=col(i), markersize=msize, alpha=malpha, label=models_df.loc[i, 'name'])
        ax.fill_between(f, spec_shape - spec_std, spec_shape + spec_std, alpha=0.3, color=col(i))
        


    # plt.tight_layout()
    ax.legend(loc='upper right', fontsize=10, framealpha=0.5)
    ax.set_xlabel('Frequency (1/day)', fontsize=12)
    ax.set_ylabel(r'$\Phi^2(f)/\sigma^2_T$ (log10)', fontsize=12)

    fig.savefig('../figures/spectral_shape_' + region + '_'  + csv.split('.')[0]  + '.png', dpi=200, facecolor='w', edgecolor='w')


if __name__ == '__main__':
    region = sys.argv[1]
    csv = sys.argv[2]
    print( "Plotting variance and correlation for region: ", region, " and csv: ", csv)
    plot_variance_correlation(region, csv)






