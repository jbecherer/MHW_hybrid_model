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

# plot map of SST anomalies
minlat = lats.min()
maxlat = lats.max()
minlon = lons.min()
maxlon = lons.max()

z1 = np.nanmedian(var_ref_grid, axis=(0,3))
z2 = np.nanmedian(var_ml_grid, axis=(0,3))
z3 = np.nanmedian(np.log2(var_ml_grid/var_ref_grid), axis=(0,3))
z4 = cal_correlation_map(var_ref_grid, var_ml_grid)

z5 = np.nanmedian(var_ref_grid[[0,1,11],:,:,:], axis=(0,3))
z6 = np.nanmedian(var_ref_grid[2:5,:,:,:], axis=(0,3))
z7 = np.nanmedian(var_ref_grid[5:8,:,:,:], axis=(0,3))
z8 = np.nanmedian(var_ref_grid[9:11,:,:,:], axis=(0,3))

z9 = np.nanmedian(var_ml_grid[[0,1,11],:,:,:], axis=(0,3))
z10 = np.nanmedian(var_ml_grid[2:5,:,:,:], axis=(0,3))
z11 = np.nanmedian(var_ml_grid[5:8,:,:,:], axis=(0,3))
z12 = np.nanmedian(var_ml_grid[9:11,:,:,:], axis=(0,3))

z13 = np.nanmedian(np.log2(var_ml_grid[[0,1,11],:,:,:]/var_ref_grid[[0,1,11],:,:,:]), axis=(0,3))
z14 = np.nanmedian(np.log2(var_ml_grid[2:5,:,:,:]/var_ref_grid[2:5,:,:,:]), axis=(0,3))
z15 = np.nanmedian(np.log2(var_ml_grid[5:8,:,:,:]/var_ref_grid[5:8,:,:,:]), axis=(0,3))
z16 = np.nanmedian(np.log2(var_ml_grid[9:11,:,:,:]/var_ref_grid[9:11,:,:,:]), axis=(0,3))

Z = [z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16]


labels = [r'med$(\sigma^2_T)$ for ERA5', r'med$(\sigma^2_T)$ for ML model', r'med(log$_2(\sigma^2_{Tml}/\sigma^2_{Tera}))$', r'corr($\sigma^2_{Tml}, \sigma^2_{Tera}$)', 
          r'$\sigma_{r}$ (winter DJF)', r'$\sigma_{r}$ (spring MAM)', r'$\sigma_{r}$ (summer JJA)', r'$\sigma_{r}$ (autumn SON)',
            r'$\sigma_{m}$ (winter DJF)', r'$\sigma_{m}$ (spring MAM)', r'$\sigma_{m}$ (summer JJA)', r'$\sigma_{m}$ (autumn SON)',
          r'log$_2(\sigma_{m}/\sigma_r)$ (winter DJF)', r'log$_2(\sigma_{m}/\sigma_r)$ (spring MAM)', r'log$_2(\sigma_{m}/\sigma_r)$ (summer JJA)', r'log$_2(\sigma_{m}/\sigma_r)$ (autumn SON)']

cmap1 = cmocean.cm.dense
cmap2 = cmocean.cm.thermal
cmap3 = cmocean.cm.balance

cmap = [ cmap1, cmap1, cmap3, cmap2,
        cmap1, cmap1, cmap1, cmap1,
        cmap1, cmap1, cmap1, cmap1,
        cmap3, cmap3, cmap3, cmap3]

cl_sig1 = [0.01, .3]
cl_sig = [0.03, .6]
cl_ba  = [-1.8, 1.8]
cl = [cl_sig1, cl_sig1, cl_ba, [0, .8],
      cl_sig, cl_sig, cl_sig, cl_sig,
        cl_sig, cl_sig, cl_sig, cl_sig,
        cl_ba, cl_ba, cl_ba, cl_ba]


N=8
N=12

projection = ccrs.PlateCarree()
fig, ax = plt.subplots(np.round(N/4).astype(int), 4, figsize=(10*1.2, 1*1.7*N/4*1.2), subplot_kw={'projection': projection})

ax = ax.flatten()

fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.05, top=0.95, left=0.05, right=0.98)

abc = 'abcdefghijklmnopqrstuvwxyz'
for a in range(N):
    pc = ax[a].pcolor(lons, lats, Z[a], cmap=cmap[a], vmin=cl[a][0], vmax=cl[a][1], transform=projection)
    if ( cl[a] == cl_sig ) | ( cl[a] == cl_sig1 ):
        pc = ax[a].pcolor(lons, lats, Z[a], cmap=cmap[a], norm=colors.LogNorm(vmin=cl[a][0], vmax=cl[a][1]), transform=projection)
    
    axpos = ax[a].get_position()
    cax  = fig.add_axes([axpos.x1-.22*axpos.width, axpos.y0+.02*axpos.height, .22*axpos.width, .015 ])
    cb = fig.colorbar(pc , extend='both', cax=cax, orientation='horizontal') # ticks=[1,2,3])
    cb = fig.colorbar(pc , extend='both', cax=cax, orientation='horizontal') # ticks=[1,2,3])
    cax.xaxis.set_ticks_position('top')
    t = ax[a].text(0.01,  .95, labels[a], horizontalalignment='left', verticalalignment='top', transform=ax[a].transAxes)
    t.set_fontweight('bold')
    t.set_fontsize(10)
    t.set_backgroundcolor([1, 1, 1, .5])


    ax[a].tick_params(axis='x', labelsize=14)  # Change font size of x-tick labels
    ax[a].tick_params(axis='y', labelsize=14)
    ax[a].coastlines(zorder=2.1)    
    ax[a].add_feature(cfeature.LAND, color=[.7, .7, .7], zorder=2)
    tabc = ax[a].text(0.98,  .98, '('+abc[a]+')', ha='right', va='top', transform=ax[a].transAxes)
    tabc.set_fontweight('bold')
    tabc.set_fontsize(14)
    tabc.set_backgroundcolor([1, 1, 1, .5])

    if a in [0, 4, 8, 12]:
        ax[a].set_yticks(np.arange(np.floor(minlat)+2, maxlat, 5), crs=ccrs.PlateCarree())
        lat_formatter = LatitudeFormatter()
        ax[a].yaxis.set_major_formatter(lat_formatter)
    else:
        ax[a].set_yticks([])

    if a > N-5:
        ax[a].set_xticks(np.arange(np.floor(minlon)+2, maxlon, 5), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        ax[a].xaxis.set_major_formatter(lon_formatter)
    else:
        ax[a].set_xticks([])

figname = '../figures/interanual_comp_maps.png'
fig.savefig(figname, dpi=300)

N=4
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(np.round(N/2).astype(int), 2, figsize=(5.4*1.5, 1*1.7*N/2*1.5), subplot_kw={'projection': projection})

ax = ax.flatten()

fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.05, top=0.95, left=0.08, right=0.98)

abc = 'abcdefghijklmnopqrstuvwxyz'
for a in range(N):
    pc = ax[a].pcolor(lons, lats, Z[a], cmap=cmap[a], vmin=cl[a][0], vmax=cl[a][1], transform=projection)
    # if ( cl[a] == cl_sig ) | ( cl[a] == cl_sig1 ):
    #     pc = ax[a].pcolor(lons, lats, Z[a], cmap=cmap[a], norm=colors.LogNorm(vmin=cl[a][0], vmax=cl[a][1]), transform=projection)
    
    axpos = ax[a].get_position()
    cax  = fig.add_axes([axpos.x1-.22*axpos.width, axpos.y0+.02*axpos.height, .22*axpos.width, .015 ])
    cb = fig.colorbar(pc , extend='both', cax=cax, orientation='horizontal') # ticks=[1,2,3])
    cb = fig.colorbar(pc , extend='both', cax=cax, orientation='horizontal') # ticks=[1,2,3])
    cax.xaxis.set_ticks_position('top')
    t = ax[a].text(0.01,  .95, labels[a], horizontalalignment='left', verticalalignment='top', transform=ax[a].transAxes)
    # t.set_fontweight('bold')
    t.set_fontsize(10)
    t.set_backgroundcolor([1, 1, 1, .5])


    ax[a].tick_params(axis='x', labelsize=14)  # Change font size of x-tick labels
    ax[a].tick_params(axis='y', labelsize=14)
    ax[a].coastlines(zorder=2.1)    
    ax[a].add_feature(cfeature.LAND, color=[.7, .7, .7], zorder=2)
    tabc = ax[a].text(0.98,  .98, '('+abc[a]+')', ha='right', va='top', transform=ax[a].transAxes)
    tabc.set_fontweight('bold')
    tabc.set_fontsize(14)
    tabc.set_backgroundcolor([1, 1, 1, .5])

    if a in [0, 2]:
        ax[a].set_yticks(np.arange(np.floor(minlat)+2, maxlat, 5), crs=ccrs.PlateCarree())
        lat_formatter = LatitudeFormatter()
        ax[a].yaxis.set_major_formatter(lat_formatter)
    else:
        ax[a].set_yticks([])

    if a > N-3:
        ax[a].set_xticks(np.arange(np.floor(minlon)+2, maxlon, 5), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        ax[a].xaxis.set_major_formatter(lon_formatter)
    else:
        ax[a].set_xticks([])

fig.savefig('../figures/interanual_comp_maps_small.png', dpi=300)
fig.savefig('../figures/interanual_comp_maps_small.pdf')

N=2
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(np.round(N/2).astype(int), 2, figsize=(5.4*1.2, 1*1.7*N/2*1.2), subplot_kw={'projection': projection})

ax = ax.flatten()

fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.12, top=0.95, left=0.12, right=0.98)

abc = 'abcdefghijklmnopqrstuvwxyz'
for a in range(N):
    a1 = a+2
    pc = ax[a].pcolor(lons, lats, Z[a1], cmap=cmap[a1], vmin=cl[a1][0], vmax=cl[a1][1], transform=projection)
    # if ( cl[a] == cl_sig ) | ( cl[a] == cl_sig1 ):
    #     pc = ax[a].pcolor(lons, lats, Z[a], cmap=cmap[a], norm=colors.LogNorm(vmin=cl[a][0], vmax=cl[a][1]), transform=projection)
    
    axpos = ax[a].get_position()
    cax  = fig.add_axes([axpos.x1-.22*axpos.width, axpos.y0+.02*axpos.height, .22*axpos.width, .015 ])
    cb = fig.colorbar(pc , extend='both', cax=cax, orientation='horizontal') # ticks=[1,2,3])
    cb = fig.colorbar(pc , extend='both', cax=cax, orientation='horizontal') # ticks=[1,2,3])
    cax.xaxis.set_ticks_position('top')
    t = ax[a].text(0.01,  .95, labels[a1], horizontalalignment='left', verticalalignment='top', transform=ax[a].transAxes)
    # t.set_fontweight('bold')
    t.set_fontsize(12)
    t.set_backgroundcolor([1, 1, 1, .5])


    ax[a].tick_params(axis='x', labelsize=14)  # Change font size of x-tick labels
    ax[a].tick_params(axis='y', labelsize=14)
    ax[a].coastlines(zorder=2.1)    
    ax[a].add_feature(cfeature.LAND, color=[.7, .7, .7], zorder=2)
    tabc = ax[a].text(0.98,  .98, '('+abc[a]+')', ha='right', va='top', transform=ax[a].transAxes)
    tabc.set_fontweight('bold')
    tabc.set_fontsize(14)
    tabc.set_backgroundcolor([1, 1, 1, .5])

    if a in [0, 2]:
        ax[a].set_yticks(np.arange(np.floor(minlat)+2, maxlat, 5), crs=ccrs.PlateCarree())
        lat_formatter = LatitudeFormatter()
        ax[a].yaxis.set_major_formatter(lat_formatter)
    else:
        ax[a].set_yticks([])

    ax[a].set_xticks(np.arange(np.floor(minlon)+2, maxlon, 5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax[a].xaxis.set_major_formatter(lon_formatter)

fig.savefig('../figures/interanual_comp_maps_smaller.png', dpi=300)
fig.savefig('../figures/interanual_comp_maps_smaller.pdf')
sys.exit()


# put ticks to the top of axis
cb = fig.colorbar(pc_era , extend='both', cax=cax, orientation='horizontal') # ticks=[1,2,3])
cax.xaxis.set_ticks_position('top')
cax.tick_params(axis='x', labelsize=13)  # Y-axis tick labels
# cax.set_title(r'log$_2$(model/era5)')
t = ax[a,0].text(0.01,  .01, labels[a], horizontalalignment='left', verticalalignment='bottom', transform=ax[a,0].transAxes)
t.set_fontweight('bold')
t.set_fontsize(14)
t.set_backgroundcolor([1, 1, 1, .5])


cmap = cm.coolwarm
cmap = cmocean.cm.balance
clim = [-1.5, 1.5]
pc = ax[a, 1].pcolor(era.lon, era.lat, np.log2(empi[col].values/era[col].values), cmap=cmap, vmin=clim[0],  vmax=clim[1], transform=projection)
pc = ax[a, 2].pcolor(era.lon, era.lat, np.log2(ehyb[col].values/era[col].values), cmap=cmap, vmin=clim[0],  vmax=clim[1], transform=projection)

# plot the minlon, maxlon, minlat, maxlat box
ax[a, 0].plot([min_lon, max_lon, max_lon, min_lon, min_lon], [min_lat, min_lat, max_lat, max_lat, min_lat], color='w', transform=projection)
ax[a, 1].plot([min_lon, max_lon, max_lon, min_lon, min_lon], [min_lat, min_lat, max_lat, max_lat, min_lat], color='w', transform=projection)
ax[a, 2].plot([min_lon, max_lon, max_lon, min_lon, min_lon], [min_lat, min_lat, max_lat, max_lat, min_lat], color='w', transform=projection)

ax[0, 0].set_title('ERA5', fontsize=15)
ax[0, 1].set_title('MPIESM-d', fontsize=15)
ax[0, 2].set_title('hybrid model', fontsize=15)

axpos = ax[0, 1].get_position()
cax  = fig.add_axes([axpos.x1-.35*axpos.width, axpos.y1+.23*axpos.height, .8*axpos.width, .01 ])
# put ticks to the top of axis
cb = fig.colorbar(pc , extend='both', cax=cax, orientation='horizontal') # ticks=[1,2,3])
cax.xaxis.set_ticks_position('bottom')
cax.tick_params(axis='x', labelsize=14)  # Y-axis tick labels
cax.set_title(r'log$_2$(model/ERA5)', fontsize=14)

abc = 'abcdefghijklmnopqrstuvwxyz'
ax=ax.flatten()
for a in range(len(ax)):
    ax[a].tick_params(axis='x', labelsize=14)  # Change font size of x-tick labels
    ax[a].tick_params(axis='y', labelsize=14)
    ax[a].coastlines(zorder=2.1)    
    ax[a].add_feature(cfeature.LAND, color=[.7, .7, .7], zorder=2)
    # ax[a].add_feature(cfeature.BORDERS, linestyle='-', zorder=3)
    tabc = ax[a].text(0.98,  .98, '('+abc[a]+')', ha='right', va='top', transform=ax[a].transAxes)
    tabc.set_fontweight('bold')
    tabc.set_fontsize(14)
    tabc.set_backgroundcolor([1, 1, 1, .5])

    if a in [0, 3, 6, 9]:
        ax[a].set_yticks(np.arange(np.floor(minlat)+2, maxlat, 5), crs=ccrs.PlateCarree())
        lat_formatter = LatitudeFormatter()
        ax[a].yaxis.set_major_formatter(lat_formatter)
    else:
        ax[a].set_yticks([])

    if a in [9, 10, 11]:
        ax[a].set_xticks(np.arange(np.floor(minlon)+2, maxlon, 5), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        ax[a].xaxis.set_major_formatter(lon_formatter)
    else:
        ax[a].set_xticks([])
    #remove axis labels
    ax[a].set_xlabel('')
    ax[a].set_ylabel('')
    ax[a].contour(bat.lon, bat.lat, bat['elevation'], levels=[-200], colors='k', transform=ccrs.PlateCarree())
    ax[a].set_extent([hyb.lon.min().values, hyb.lon.max().values, hyb.lat.min().values, hyb.lat.max().values], crs=ccrs.PlateCarree())

fig, ax = plt.subplots(1, 3 , figsize=(9, 4.5))
fig.subplots_adjust(hspace=0.04, wspace=0.04, bottom=0.1, top=0.99, left=0.05, right=0.99)

a=0
im = ax[a].pcolor(np.log10(np.nanmedian(var_ref_grid[10:11,:,:,:],axis=(0,3))), vmin=-1.5, vmax=0.5, cmap='jet')

ax[a].set_title('Reference variance')
fig.colorbar(im, ax=ax[a], orientation='vertical', label='log10(variance)')
a=1
im = ax[a].pcolor(np.log10(np.nanmedian(var_ml_grid[10:11,:,:,:],axis=(0,3))), vmin=-1.5, vmax=0.5, cmap='jet')
ax[a].set_title('ML variance')
fig.colorbar(im, ax=ax[a], orientation='vertical', label='log10(variance)')

a=2
im = ax[a].pcolor(np.nanmedian(np.log10(var_ml_grid/var_ref_grid),axis=(0,3)), vmin=-1, vmax=1, cmap='bwr')
ax[a].set_title('ML / Reference variance')
fig.colorbar(im, ax=ax[a], orientation='vertical', label='log10(ML / Ref)')


figname = '../figures/interanual_comp_maps.png'
fig.savefig(figname, dpi=300)

