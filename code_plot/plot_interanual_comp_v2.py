import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import cmocean

import importlib
sys.path.append('../code_proc/')
import aisst
importlib.reload(aisst)

plt.switch_backend('Agg')


def load_model(models_df, i):
            shape = models_df.loc[i, 'shape']
            number_hidden_layers = int(shape.split('x')[0])
            hidden_size = int(shape.split('x')[1])
            loss_type = models_df.loc[i, 'loss fn']
            epochs = models_df.loc[i, 'epochs']
            input_size = 6

            feature_set = models_df.loc[i, 'feature set']

            model = aisst.load_model(number_hidden_layers, hidden_size, loss_type, epochs=epochs, input_size=input_size, idir='../models/NWEuroShelf/' + feature_set + '/')

            return model

def cal_ml_variance(tensor_val):
    #--- ML predictions
    tensor_in = tensor_val[:,0:-4]

    input_mean, input_std, output_mean, output_std = aisst.load_normalization_parameters('NWEuroShelf')
    tensor_in_n = aisst.normalize_data(tensor_in, input_mean, input_std)

    feature_set = 'reduced'
    t_in = aisst.reduce_feature_set(tensor_in, feature_set)
    t_in_n = aisst.reduce_feature_set(tensor_in_n, feature_set)

    models_df = pd.read_csv('../models/NWEuroShelf/best_model.csv')
    i=0
    model = load_model(models_df, i)

    # predict with best model
    pre = model(t_in_n).detach()
    pre = aisst.denormalize_data(pre, output_mean, output_std).numpy()
    variance_ml = 2 * np.sum(np.abs(pre[:,1:]) ** 2, axis=1)

    return variance_ml

def cluster_by_latlon_month(variance, lats, lons, months):
    month_arr = np.unique(months)
    lat_arr = np.unique(lats)
    lon_arr = np.unique(lons)
    # print('lat arr: ', lat_arr)
    # print('lon arr: ', lon_arr)
    # print('month arr: ', month_arr)
    # print(len(variance), len(month_arr), len(lat_arr), len(lon_arr))


    N2 = np.round(len(variance)/(len(month_arr)*len(lat_arr)*len(lon_arr))).astype(int)
    N2 = N2 * 2 # dirty hack to account for nans in original grid
    variance_month = np.ones((len(month_arr), len(lat_arr), len(lon_arr), N2)) * np.nan

    for m in range(len(month_arr)):
        for i in range(len(lat_arr)):
            for j in range(len(lon_arr)):
                # print('Clustering month: ', m+1, ' lat: ', lat_arr[i], ' lon: ', lon_arr[j])
                find_loc = (months == m + 1) & (lats == lat_arr[i]) & (lons == lon_arr[j])
                var_m = variance[find_loc]
                var_tmp = np.ones(N2)*np.nan
                var_tmp[0:len(var_m)] = var_m
                variance_month[m,i,j,:] = var_tmp

    return variance_month, lat_arr, lon_arr, month_arr

def cal_statistics(xr, xm):
    ''' Calculate statistics between two arrays
    Parameters:
    xr: np.array
        Reference array
    xm: np.array
        Model array
    Returns:
    corr: float
        Correlation coefficient between the two arrays
    R: float
        Relative error between the two arrays
    '''
    corr = np.corrcoef(xr, xm)[0,1]
    r_lg2 = np.log2(xm / xr)
    r_med = np.median(r_lg2)
    r_var = np.var(r_lg2)
    r_std = np.std(r_lg2)
    r_mean = np.mean(xm) / np.mean(xr)

    return corr, r_med, r_mean, r_var, r_std

def cal_correlation_map(var_ref_grid, var_model_grid):
    ''' Calculate correlation map between two variance grids
    Parameters:
    var_ref_grid: np.array
        Reference variance grid
    var_model_grid: np.array
        Model variance grid
    Returns:
    corr_map: np.array
        Correlation map between the two variance grids
    '''
    nlat = var_ref_grid.shape[1]
    nlon = var_ref_grid.shape[2]
    corr_map = np.ones((nlat, nlon)) * np.nan

    for i in range(nlat):
        for j in range(nlon):
            ref_series = var_ref_grid[:,i,j,:].flatten()
            model_series = var_model_grid[:,i,j,:].flatten()
            # remove nans
            valid_idx = ~np.isnan(ref_series) & ~np.isnan(model_series)
            if np.sum(valid_idx) > 10:
                corr, _, _, _, _ = cal_statistics(ref_series[valid_idx], model_series[valid_idx])
                corr_map[i,j] = corr

    return corr_map

#===========================================================================

#-- load data--#

tensor_val = aisst.load_tensor_from_csv('../data/interanual_comp/NWEuroShelf/interanual_data_val.csv')
# tensor_val = aisst.load_tensor_from_csv('../data/interanual_comp/NWEuroShelf/interanual_data_valtest.csv')
# tensor_val = aisst.load_tensor_from_csv('../data/interanual_comp/NWEuroShelf/interanual_data.csv')


#____process variance data____#

# ERA5 reference variance
var_ref = tensor_val.numpy()[:,-1]

var_ref_grid, lats, lons, months = cluster_by_latlon_month(var_ref, tensor_val.numpy()[:,-5], tensor_val.numpy()[:,-4], tensor_val.numpy()[:,-3])

# ML variance
var_ml = cal_ml_variance(tensor_val)
var_ml_grid, _, _, _ = cluster_by_latlon_month(var_ml, tensor_val.numpy()[:,-5], tensor_val.numpy()[:,-4], tensor_val.numpy()[:,-3])



#__________plot interanual comparison maps__________#
plt.close('all')

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
cor_inter = []
for m in range(var_ref_grid.shape[0]):
    tmp1 = var_ref_grid[m,:,:,:].flatten()
    tmp2 = var_ml_grid[m,:,:,:].flatten()
    iinnan = ~np.isnan(tmp1) & ~np.isnan(tmp2)
    cor_inter.append(np.corrcoef(tmp1[iinnan], tmp2[iinnan])[0,1])


fig = plt.figure(figsize=(5,2.5))
ax = fig.add_subplot(1,1,1)

ax = [ax]

a=0
ax[a].plot(month_names, cor_inter, marker='o', color='b', linestyle=None, linewidth=.3, label='interannual correlation')
xl = ax[a].get_xlim()
ax[a].hlines(0, xl[0], xl[1], color='k', linestyle='-', linewidth=0.8)
ax[a].hlines(0.63, xl[0], xl[1], color='r', linestyle='--', linewidth=2.8, label='total correlation')
ax[a].hlines(np.mean(cor_inter), xl[0], xl[1], color='g', linestyle='--', linewidth=2.8, label='mean interannual correlation')
ax[a].set_xlim(xl)
lg = ax[a].legend(loc='lower right', fontsize=10)
ax[a].set_ylabel(r'corr($\sigma^2_{Tml}, \sigma^2_{Tera}$)', fontsize=12)

fig.savefig('../figures/correlation_interanual_NWEuroShelf_ML_vs_ERA5.png', dpi=300)
fig.savefig('../figures/correlation_interanual_NWEuroShelf_ML_vs_ERA5.pdf', dpi=300)


