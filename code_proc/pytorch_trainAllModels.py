import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd

import importlib
import aisst
importlib.reload(aisst)

from cal_variance_correlation_4allmodels import add_variance_correlation2csv

import sys



def train_models(region='NorthSea', csv='models.csv'):
    #==============================================================================
    # load data
    #==============================================================================

    batch_size = 32

    # ---------------load data ----------------
    train_in = aisst.load_tensor_from_csv('../data/ml_training/' + region + '/ml_input_data_train.csv')
    train_out = aisst.load_tensor_from_csv('../data/ml_training/' + region + '/ml_output_data_train.csv')


    test_in = aisst.load_tensor_from_csv('../data/ml_training/' + region + '/ml_input_data_test.csv')
    test_out = aisst.load_tensor_from_csv('../data/ml_training/' + region + '/ml_output_data_test.csv')

    # ------------normalize data----------------
    input_mean, input_std, output_mean, output_std = aisst.load_normalization_parameters(region)

    train_in = aisst.normalize_data(train_in, input_mean, input_std)
    test_in = aisst.normalize_data(test_in, input_mean, input_std)

    # train_out_fft = train_out.detach()
    # test_out_fft = test_out.detach()
    train_out_fft = train_out.clone()
    test_out_fft = test_out.clone()

    # for non-fft loss functions
    train_out = aisst.normalize_data(train_out, output_mean, output_std)
    test_out = aisst.normalize_data(test_out, output_mean, output_std)

    #------------- create dataset and data loader----------------
    # train_dataset = aisst.CustomDataset(train_in, train_out)
    # test_dataset = aisst.CustomDataset(test_in, test_out)
    #
    # train_dataset_fft = aisst.CustomDataset(train_in, train_out_fft)
    # test_dataset_fft = aisst.CustomDataset(test_in, test_out_fft)
    #
    # # create data loader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # train_loader_fft = DataLoader(train_dataset_fft, batch_size=batch_size, shuffle=True)
    # test_loader_fft = DataLoader(test_dataset_fft, batch_size=batch_size, shuffle=True)

    #==============================================================================
    # model architecture
    #==============================================================================

    models_df = pd.read_csv('../models/' + region + '/' + csv) 


    for i in range(len(models_df)):


        print('Training model: ', models_df.loc[i, 'name'])

        if models_df.loc[i, 'type'] != 'NN':
            continue

        # model parameters
        loss_type = models_df.loc[i, 'loss fn']
        learning_rate = models_df.loc[i, 'learning rate']
        epochs = int(models_df.loc[i, 'epochs'])

        drop_prop = models_df.loc[i, 'drop_out']

        # input data depndeing on feature set
        if 'feature set' in models_df.columns:    
            feature_set = models_df.loc[i, 'feature set']
        else:
            feature_set = 'full'

        # input data depending on feature set
        tr_in = aisst.reduce_feature_set(train_in, feature_set)
        te_in = aisst.reduce_feature_set(test_in, feature_set)

        # out data
        if loss_type == 'fft':
            tr_out = train_out_fft.detach()
            te_out = test_out_fft.detach()
        else:
            tr_out = train_out.detach()
            te_out = test_out.detach()


        shape = models_df.loc[i, 'shape']
        if 'x' in shape:
            number_hidden_layers = int(shape.split('x')[0])
            hidden_size = int(shape.split('x')[1])

        else: # this could later be used for more complex shapes
            continue

        input_size = tr_in.shape[1]
        output_size = tr_out.shape[1]

        model = aisst.NeuralNetwork(input_size, hidden_size, output_size, number_hidden_layers, drop_prop=drop_prop)

        loss_fn = aisst.get_loss_fn(loss_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        # create data loader
        train_loader = aisst.create_dataloader(tr_in, tr_out, batch_size=batch_size, shuffle=True)
        test_loader = aisst.create_dataloader(te_in, te_out, batch_size=batch_size, shuffle=True)

        # train model
        test_loss = aisst.train_loop(train_loader, test_loader, model, loss_fn, optimizer, epochs, print_loss=False)

        # save model
        fname = aisst.save_model(model, number_hidden_layers, hidden_size, loss_type, drop_prop=drop_prop, epochs=epochs, odir='../models/' + region + '/' + feature_set + '/' )

        models_df.loc[i, 'file'] = fname

        # find epoch with min loss
        epoch_minloss = np.argmin(np.array(test_loss))
        models_df.loc[i, 'epoch_minloss'] = epoch_minloss


    models_df.to_csv('../models/' + region + '/' + csv, index=False)


if __name__ == '__main__':
    region = sys.argv[1]
    csv = sys.argv[2]
    print( "Training models for region: " +   region +  " and csv: "  + csv)
    train_models(region, csv)
    print("Calculating variance and correlation")
    add_variance_correlation2csv(region, csv)
