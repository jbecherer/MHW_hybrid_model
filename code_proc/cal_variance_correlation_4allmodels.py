import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib


import importlib
import aisst
importlib.reload(aisst)

import sys


def add_variance_correlation2csv(region='NorthSea', csv='models.csv'):
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


    for i in range(len(models_df)):

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


        r_median = np.median(variance)/np.median(variance_ref)
        r_mean = np.mean(variance)/np.mean(variance_ref)
        r_90 = np.percentile(variance, 90)/np.percentile(variance_ref, 90)
        r_10 = np.percentile(variance, 10)/np.percentile(variance_ref, 10)


        models_df.loc[i, 'correlation'] = f"{corr:.2f}"
        models_df.loc[i, 'r_median'] =  f"{r_median:.2f}"
        models_df.loc[i, 'r_mean'] =    f"{r_mean:.2f}"
        models_df.loc[i, 'r_90'] =      f"{r_90:.2f}"
        models_df.loc[i, 'r_10'] =      f"{r_10:.2f}"

    models_df.to_csv('../models/' + region + '/' + csv, index=False)


if __name__ == '__main__':
    region = sys.argv[1]
    csv = sys.argv[2]
    print( "Calculating variance and correlation for region: ", region, " and csv: ", csv)
    add_variance_correlation2csv(region, csv)
