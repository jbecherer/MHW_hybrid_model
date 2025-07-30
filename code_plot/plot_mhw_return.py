import sys
import xarray as xr
import numpy as np
import pandas as pd

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
# backends
plt.switch_backend('agg')

import importlib
sys.path.append('../code_proc/')
import aisst
importlib.reload(aisst)


#==============================================================================
# load data
#==============================================================================

histssp585 = xr.open_dataset('../data/mhw/merged_return_histssp585.nc')

ssp126 = xr.open_dataset('../data/mhw/merged_return_ssp126.nc')
ssp245 = xr.open_dataset('../data/mhw/merged_return_ssp245.nc')
ssp370 = xr.open_dataset('../data/mhw/merged_return_ssp370.nc')

min_lat = histssp585.lat.min().values + 1
max_lat = histssp585.lat.max().values - 1
min_lon = histssp585.lon.min().values + 2
max_lon = histssp585.lon.max().values - 1

histssp585 = histssp585.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
ssp126 = ssp126.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
ssp245 = ssp245.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
ssp370 = ssp370.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

mSST_histssp585 = xr.open_dataset('../data/mhw/mSST_histssp585.nc')
mSST_histssp585 = mSST_histssp585.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
mSST = mSST_histssp585['mean_sst'].mean(axis=0).mean(axis=0)


plt.close('all')

data_sets = [histssp585, ssp126, ssp245, ssp370]
data_colors = ['black', 'blue', 'green', 'orange', 'red']
data_labels = ['historical + SSP5-8.5', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0']

feature_set = list(histssp585.data_vars.keys())
feature_labels = ['duration', 'mean intensity', 'peak intensity', 'cumulative intensity']

periods = ['1875', '1995', '2030' ,'2085']
p_colors = [cm.Dark2(i) for i in range(len(periods))]
p_labels = ['pre-industrial', 'climatology', 'present (SSP5-8.5)', 'future (SSP5-8.5)']
mSST_periods = np.array([mSST.sel(time=slice(p, p)).mean(axis=0).mean(axis=0).values for p in periods])

# pick specifig return periods
ii_rp = [449, 89, 8, 0] # corresponding to 2, 10, 100, 900 years return period
rp_colors = ['black', 'blue', 'green', 'orange']
rp_labels = ['2 year', '10 years', '100 years', '900 years']

N_features = len(feature_set)
fig = plt.figure( figsize = (3.5, 1.7*N_features ), facecolor = (1, 1, 1))
ax = fig.subplots(N_features, 1, sharex=True)

mSST_periods[1] = 0

for i, feature in enumerate(feature_set):
    txt = ax[i].text(0.01, 0.95, feature_labels[i], transform=ax[i].transAxes, fontsize=10, verticalalignment='top')
    txt.set_bbox(dict(facecolor='white', alpha=0.3, edgecolor='white'))
    for j, p in enumerate(periods):
        data=histssp585.sel(time=slice(p, p))
        data_flat = data[feature].stack(x=('lat', 'lon')).squeeze('time')
        data_min = data_flat.quantile(.05, dim='x').values
        # data_min[data_min  == 0] = 1e-6 # avoid log(0) for plotting pourpose in log scale
        data_max = data_flat.quantile(.95, dim='x').values
        # data_min = data_flat.min(dim='x').values
        # data_max = data_flat.max(dim='x').values
        ax[i].fill_between(data.r_period, data_min, data_max, color=p_colors[j], alpha=0.2)
    for j, p in enumerate(periods):
        data=histssp585.sel(time=slice(p, p))
        data_flat = data[feature].stack(x=('lat', 'lon')).squeeze('time')
        ax[i].plot(data.r_period, data_flat.mean(dim='x').values, label=p_labels[j], color=p_colors[j])
        # if j == 1 and (i == 1 or i == 2):
        if j == 1 and i == 2:
            # ax[i].plot([1000, 1000, 1000, 1000], mSST_periods +  data[feature].mean(axis=0).mean(axis=0).max(axis=0).values[0], '+', color='k')
            # ax[i].hlines(list(mSST_periods+  data[feature].mean(axis=0).mean(axis=0).max(axis=0).median(axis=0).values), 1, 1000, color='k', linestyle='dashed',linewidth=0.5)
            for k in range(4):
                ax[i].plot(data.r_period, data[feature].mean(axis=0).mean(axis=0).mean(axis=0).values + mSST_periods[k], linestyle='--', color=p_colors[k], linewidth=1)

    ax[i].set_xscale('log')
    yl = ax[i].get_ylim()

    ax[i].vlines(data.r_period[ii_rp], yl[0], yl[1], colors=np.array(rp_colors), linestyles='--', linewidth=1)

    ax[i].set_xlim(1, 1000)

legend = ax[0].legend(loc='lower left', shadow=True, fontsize=8, bbox_to_anchor=(0., 1.05), ncol=2)


ax[0].set_yscale('log')
ax[0].set_ylim(1, 5000)
ax[3].set_yscale('log')
ax[3].set_ylim(1, 15000)

ax[0].set_ylabel('days')
ax[1].set_ylabel(r'$^\circ$C')
ax[2].set_ylabel(r'$^\circ$C')
ax[3].set_ylabel(r'$^\circ$C$\cdot$days')
ax[3].set_xlabel('return period [years]')

fig.subplots_adjust(hspace=0.07, wspace=0.02, right=0.98, left=0.18, top=0.9, bottom=0.1)

abc = 'abcdefgh'
for i, axi in enumerate(ax):
    axi.text(0.95, 0.02, '(' + abc[i] + ')', transform=axi.transAxes, fontsize=12, va='bottom', ha='right')



fig.savefig('../figures/mhw_return_period.png', dpi=200, facecolor='w', edgecolor='w')
fig.savefig('../figures/mhw_return_period.pdf', facecolor='w', edgecolor='w')



#==============================================================================
# return period time series
#==============================================================================


fig = plt.figure( figsize = (3.5, 1.7*N_features ), facecolor = (1, 1, 1))
ax = fig.subplots(N_features, 1, sharex=True)

t0 = np.where(mSST.values > 0)[0][0]

for i, feature in enumerate(feature_set):
    txt = ax[i].text(0.01, 0.95, feature_labels[i], transform=ax[i].transAxes, fontsize=10, verticalalignment='top')
    txt.set_bbox(dict(facecolor='white', alpha=0.3, edgecolor='white'))
    data_flat = histssp585[feature].stack(x=('lat', 'lon'))
    data_min = data_flat.quantile(.05, dim='x').values
    data_max = data_flat.quantile(.95, dim='x').values
    data_mean = data_flat.mean(dim='x').values
    for j, ii in enumerate(ii_rp):
        ax[i].fill_between(data_flat.time, data_min[:,ii], data_max[:,ii], color=rp_colors[j], alpha=0.2)
    for j, ii in enumerate(ii_rp):
        ax[i].plot(data_flat.time, data_mean[:,ii], label=rp_labels[j], color=rp_colors[j])
        if i==2:
            ax[i].plot(mSST.time, mSST + data_mean[t0,ii], label='mean SST', color='black', linestyle='--', linewidth=1)

    yl = ax[i].get_ylim()
    ax[i].vlines( np.array(periods, dtype='datetime64[Y]'), yl[0], yl[1], colors=np.array(p_colors), linestyles='--', linewidth=1)
    # if i==2:
    #     ax[i].plot(mSST.time, mSST+3, label='mean SST', color='black', linestyle='--', linewidth=1)

    # ax[i].set_yscale('log')

legend = ax[0].legend(loc='lower left', shadow=True, fontsize=10, bbox_to_anchor=(0., 1.05), ncol=2)
ax[0].set_yscale('log')
ax[3].set_yscale('log')

ax[0].set_ylabel('days')
ax[1].set_ylabel(r'$^\circ$C')
ax[2].set_ylabel(r'$^\circ$C')
ax[3].set_ylabel(r'$^\circ$C$\cdot$days')
ax[3].set_xlabel('year')

fig.subplots_adjust(hspace=0.07, wspace=0.02, right=0.98, left=0.18, top=0.9, bottom=0.1)

abc = 'abcdefgh'
for i, axi in enumerate(ax):
    axi.text(0.95, 0.02, '(' + abc[i] + ')', transform=axi.transAxes, fontsize=12, va='bottom', ha='right')

fig.savefig('../figures/mhw_return_period_time_series.png', dpi=200, facecolor='w', edgecolor='w')
fig.savefig('../figures/mhw_return_period_time_series.pdf', facecolor='w', edgecolor='w')



