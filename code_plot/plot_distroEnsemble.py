import sys
import xarray as xr
import numpy as np
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

import matplotlib.pyplot as plt
from matplotlib import cm
import cmocean
import seaborn as sns
# backends
plt.switch_backend('agg')

import importlib
sys.path.append('../code_proc/')
import aisst
importlib.reload(aisst)

import marineHeatWaves as mhw
importlib.reload(mhw)

#==============================================================================
# load data
#==============================================================================

era = xr.open_dataset('../data/mhw/mhw_stats_era_1982-2023.nc')
mpi = xr.open_dataset('../data/mhw/mhw_stats_mpi_1982-2023.nc')
mlf = xr.open_dataset('../data/mhw/mhw_stats_mpilf_1982-2023.nc')
hyb = xr.open_dataset('../data/mhw/mhw_stats_hybrid_1982-2023.nc')

bat = xr.open_dataset('../data/bathymetry_1deg.nc')

# correct NpYear
era['NpYear'] = era['NpYear'] 
mpi['NpYear'] = mpi['NpYear'] 
mlf['NpYear'] = mlf['NpYear'] 
hyb['NpYear'] = hyb['NpYear'] 

# pick a point
lat = 55.5
lon = 5

min_lat = 54
max_lat = 58
min_lon = 0
max_lon = 7

min_lat = hyb.lat.min().values + 1
max_lat = hyb.lat.max().values - 1
min_lon = hyb.lon.min().values + 2
max_lon = hyb.lon.max().values - 1

aera = era.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
ampi = mpi.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
amlf = mlf.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
ahyb = hyb.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
bat = bat.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

# # ignoere points that are nan in hybrid model
# nonan = ~np.isnan(ahyb.NpYear) & ~np.isnan(ampi.NpYear) 
# aera = aera.where(nonan[:,:,0], drop=True)
# ampi = ampi.where(nonan, drop=True)
# amlf = amlf.where(nonan, drop=True)
# ahyb = ahyb.where(nonan, drop=True)


# check arrea average of era 
era_sst = xr.open_dataset('../data/NWEuroShelf_era5_1982_2023_allVars_1deg_daily.nc')
era_sst = era_sst.sst.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
era_sst = era_sst.to_dataset(name='sst')
# era_sst = era_sst.mean(dim='lat').mean(dim='lon')
period = [1982, 2011]
mhws, _ = mhw.detect(era_sst.time.values, era_sst.sst.mean(axis=2).mean(axis=1).values, climatologyPeriod=period.copy())


#==============================================================================

# marine heat wave stats plot
#==============================================================================
# features=['NpYear', 'mean_duration', 'max_duration', 'mean_mean_intensity', 'mean_mean_intensity','mean_peak_intensity', 'max_peak_intensity', 'tot_cum_intensity']

def bar_plot( features, aera, ahyb, ampi,  labels=None): 
    fig, ax = plt.subplots(1, 1, figsize=(1.8*6, 3.8))
    if labels is None:
        labels = features
    for a, col in enumerate(features):

        d1 = aera[col].values.flatten()
        d1 = d1[~np.isnan(d1)]
        data = [d1]
        colors = [ 'k']
        for m in range(30):
            d2 = ahyb[col].values[:,:,m].flatten()
            d2 = d2[(~np.isnan(d2)) & (bat['elevation'].values.flatten() > -300)]
            data.append(d2)
            colors.append(cm.tab10(1))
            d3 = ampi[col].values[:,:,m].flatten()
            d3 = d3[~np.isnan(d3) & (bat['elevation'].values.flatten() > -300)]
            data.append(d3)
            colors.append(cm.tab10(0))

        viola = ax.violinplot(data, showmeans=False, showmedians=True, 
                                 showextrema=False, widths=.9, bw_method=0.5)

        for i, vil in enumerate(viola['bodies']):
            vil.set_facecolor(colors[i])
        ax.boxplot(data, bootstrap=None, showfliers=True, 
                      flierprops={'marker':'o', 'markersize':2, 'color':'k', 'alpha':.3},
                      patch_artist=False, widths=0.4)


        # ax.set_xticks([1, 2, 3, 4], labels=[ '  ERA5   ','hybrid m.', 'MPIESM-d ', 'MPIESM-m '], rotation=30, fontsize=11, fontweight='bold')
        ax.tick_params( axis='y', labelsize=11)
        # plt.xticks(rotation=45)
        ax.set_title(labels[a])
    # change axis style to only show the bottom axis
    # for a in ax:
    #     a.set_ylim([0, a.get_ylim()[1]])
        

    plt.tight_layout()

    return fig, ax


# short_features = ['NpYear', 'mean_duration', 'max_duration', 'mean_mean_intensity', 'mean_cum_intensity', 'max_peak_intensity']
short_features = ['NpYear']
labels = ['frequency\n [events/year]', 
          'mean duration\n [days]', 
          'longest duration\n [days]', 
          'mean intensity\n [$^\circ$C]', 
          'cum. intensity\n [$^\circ$C]', 
          'peak intensity\n [$^\circ$C]']
fig, ax = bar_plot(short_features, aera, ahyb, ampi, labels=labels)
fig.subplots_adjust(hspace=0.05, wspace=0.4, bottom=0.20, top=0.85, left=0.05, right=0.98)
plt.savefig('../figures/distroEnsemble_frequency.png')

short_features = ['max_duration']
labels = [ 'longest duration\n [days]', 
          'mean intensity\n [$^\circ$C]', 
          'cum. intensity\n [$^\circ$C]', 
          'peak intensity\n [$^\circ$C]']
fig, ax = bar_plot(short_features, aera, ahyb, ampi, labels=labels)
fig.subplots_adjust(hspace=0.05, wspace=0.4, bottom=0.20, top=0.85, left=0.05, right=0.98)
plt.savefig('../figures/distroEnsemble_maxduration.png')
# area averaged MHW reference
# ax[0].plot([0, 5], [mhws['n_events']/42, mhws['n_events']/42], 'k--')
# ax[1].plot([0, 5], [np.mean(mhws['duration']), np.mean(mhws['duration'])], 'k--')
# ax[2].plot([0, 5], [np.max(mhws['duration']), np.max(mhws['duration'])], 'k--')
# ax[3].plot([0, 5], [np.mean(mhws['intensity_mean']), np.mean(mhws['intensity_mean'])], 'k--')
# ax[4].plot([0, 5], [np.mean(mhws['intensity_cumulative']), np.mean(mhws['intensity_cumulative'])], 'k--')
# ax[5].plot([0, 5], [np.max(mhws['intensity_max']), np.max(mhws['intensity_max'])], 'k--')

# ax[1].set_yscale('log')
# ax[1].set_ylim([10, 200])
# ax[2].set_yscale('log')
# ax[2].set_ylim([20, 2000])
# ax[4].set_yscale('log')
# ax[4].set_ylim([10, 200])

# abc = 'abcdefghijklmnopqrstuvwxyz'
# for a in range(len(ax)):
#     tabc = ax[a].text(0.01,  .98, '('+abc[a]+')', ha='left', va='top', transform=ax[a].transAxes)
#     tabc.set_fontweight('bold')
#     tabc.set_fontsize(11)




