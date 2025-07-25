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

bat = xr.open_dataset('../data/bathymetry_fullresolution.nc')
bat = bat.coarsen(lat=20, lon=20, boundary='trim').mean()

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

pera = era.sel(lat=lat, lon=lon, method='nearest')
pmpi = mpi.sel(lat=lat, lon=lon, method='nearest')
pmlf = mlf.sel(lat=lat, lon=lon, method='nearest')
phyb = hyb.sel(lat=lat, lon=lon, method='nearest')

aera = era.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
ampi = mpi.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
amlf = mlf.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
ahyb = hyb.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

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
sys.path.append('../external/marineHeatWaves/')
import marineHeatWaves as mhw
importlib.reload(mhw)
period = [1982, 2011]
mhws, _ = mhw.detect(era_sst.time.values, era_sst.sst.mean(axis=2).mean(axis=1).values, climatologyPeriod=period.copy())


#==============================================================================

# marine heat wave stats plot
#==============================================================================
# features=['NpYear', 'mean_duration', 'max_duration', 'mean_mean_intensity', 'mean_mean_intensity','mean_peak_intensity', 'max_peak_intensity', 'tot_cum_intensity']

def bar_plot( features, aera, ahyb, ampi, amlf, labels=None): 
    N = len(features)
    fig, ax = plt.subplots(1, N, figsize=(1.8*N, 3.8))
    if labels is None:
        labels = features
    for a, col in enumerate(features):

        d1 = aera[col].values.flatten()
        d1 = d1[~np.isnan(d1)]
        d2 = ahyb[col].values.flatten()
        d2 = d2[~np.isnan(d2)]
        d3 = ampi[col].values.flatten()
        d3 = d3[~np.isnan(d3)]
        d4 = amlf[col].values.flatten()
        d4 = d4[~np.isnan(d4)]
        data = [d1, d2, d3, d4]
        colors = [ 'k', cm.tab10(1), cm.tab10(0), cm.tab10(2)]

        viola = ax[a].violinplot(data, showmeans=False, showmedians=True, 
                                 showextrema=False, widths=.9, bw_method=0.5)

        for i, vil in enumerate(viola['bodies']):
            vil.set_facecolor(colors[i])
        ax[a].boxplot(data, bootstrap=None, showfliers=True, 
                      flierprops={'marker':'o', 'markersize':2, 'color':'k', 'alpha':.3},
                      patch_artist=False, widths=0.4)


        ax[a].set_xticks([1, 2, 3, 4], labels=[ '  ERA5   ','hybrid m.', 'MPIESM-d ', 'MPIESM-m '], rotation=30, fontsize=11, fontweight='bold')
        ax[a].tick_params( axis='y', labelsize=11)
        # plt.xticks(rotation=45)
        ax[a].set_title(labels[a])
    # change axis style to only show the bottom axis
    for a in ax:
        a.set_ylim([0, a.get_ylim()[1]])
        

    plt.tight_layout()

    return fig, ax


features = list(era.data_vars.keys())
fig, ax = bar_plot(features, aera, ahyb, ampi, amlf)
plt.savefig('../figures/intercomp_mhw_area_fullstats.png')
plt.savefig('../figures/intercomp_mhw_area_fullstats.pdf')

short_features = ['NpYear', 'mean_duration', 'max_duration', 'mean_mean_intensity', 'mean_cum_intensity', 'max_peak_intensity']
labels = ['frequency\n [events/year]', 
          'mean duration\n [days]', 
          'longest duration\n [days]', 
          'mean intensity\n [$^\circ$C]', 
          'cum. intensity\n [$^\circ$C]', 
          'peak intensity\n [$^\circ$C]']
fig, ax = bar_plot(short_features, aera, ahyb, ampi, amlf, labels=labels)
fig.subplots_adjust(hspace=0.05, wspace=0.4, bottom=0.20, top=0.85, left=0.05, right=0.98)
# area averaged MHW reference
# ax[0].plot([0, 5], [mhws['n_events']/42, mhws['n_events']/42], 'k--')
# ax[1].plot([0, 5], [np.mean(mhws['duration']), np.mean(mhws['duration'])], 'k--')
# ax[2].plot([0, 5], [np.max(mhws['duration']), np.max(mhws['duration'])], 'k--')
# ax[3].plot([0, 5], [np.mean(mhws['intensity_mean']), np.mean(mhws['intensity_mean'])], 'k--')
# ax[4].plot([0, 5], [np.mean(mhws['intensity_cumulative']), np.mean(mhws['intensity_cumulative'])], 'k--')
# ax[5].plot([0, 5], [np.max(mhws['intensity_max']), np.max(mhws['intensity_max'])], 'k--')

ax[1].set_yscale('log')
ax[1].set_ylim([10, 200])
ax[2].set_yscale('log')
ax[2].set_ylim([20, 2000])
ax[4].set_yscale('log')
ax[4].set_ylim([10, 200])

abc = 'abcdefghijklmnopqrstuvwxyz'
for a in range(len(ax)):
    tabc = ax[a].text(0.01,  .98, '('+abc[a]+')', ha='left', va='top', transform=ax[a].transAxes)
    tabc.set_fontweight('bold')
    tabc.set_fontsize(11)



plt.savefig('../figures/intercomp_mhw_area_shortstats.png')
plt.savefig('../figures/intercomp_mhw_area_shortstats.pdf')




#==============================================================================
# intercomparision maps
#==============================================================================

empi = mpi.median(dim='member')
ehyb = hyb.median(dim='member')

# plot map of SST anomalies
minlat = era['lat'].min().values
maxlat = era['lat'].max().values
minlon = era['lon'].min().values
maxlon = era['lon'].max().values


# plot map of SST anomalies in a subpannel of the figure 

# fig = plt.figure(figsize=(7, len(feature_set)*3))
# ax = fig.add_subplot(len(feature_set), 2, 1, projection=ccrs.PlateCarree())
# ax.set_extent([minlon, maxlon, minlat, maxlat], crs=ccrs.PlateCarree())

feature_set = ['NpYear', 'mean_duration', 'mean_mean_intensity', 'max_peak_intensity']

projection = ccrs.PlateCarree()
fig, ax = plt.subplots(len(feature_set), 3, figsize=(10*1.2, len(feature_set)*2*1.2), subplot_kw={'projection': projection})
clim = [-1.5, 1.5]

fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.05, top=0.90, left=0.05, right=0.98)
labels = ['freq. [events/year]', 'mean duration [days]', r'mean intensity [$^\circ$C]', 'peak event [$^\circ$C]']

min_lon = min_lon - 0.5
max_lon = max_lon + 0.5
min_lat = min_lat - 0.5
max_lat = max_lat + 0.5

for a, col in enumerate(feature_set):
    clim_era = [era[col].min().values , era[col].max().values]
    if a == 0:
        cmap1 = cmocean.cm.amp
    elif a == 1:
        cmap1 = cmocean.cm.tempo
    else:
        cmap1 = cmocean.cm.thermal
    pc_era = ax[a, 0].pcolor(era.lon, era.lat, era[col].values, cmap=cmap1, vmin=clim_era[0], vmax=clim_era[1], transform=projection)

    axpos = ax[a, 0].get_position()
    cax  = fig.add_axes([axpos.x1-.22*axpos.width, axpos.y0+.02*axpos.height, .22*axpos.width, .01 ])
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
axf=ax.flatten()
for a in range(len(axf)):
    axf[a].tick_params(axis='x', labelsize=14)  # Change font size of x-tick labels
    axf[a].tick_params(axis='y', labelsize=14)
    axf[a].coastlines(zorder=2.1)    
    axf[a].add_feature(cfeature.LAND, color=[.7, .7, .7], zorder=2)
    # axf[a].add_feature(cfeature.BORDERS, linestyle='-', zorder=3)
    tabc = axf[a].text(0.98,  .98, '('+abc[a]+')', ha='right', va='top', transform=axf[a].transAxes)
    tabc.set_fontweight('bold')
    tabc.set_fontsize(14)
    tabc.set_backgroundcolor([1, 1, 1, .5])

    if a in [0, 3, 6, 9]:
        axf[a].set_yticks(np.arange(np.floor(minlat)+2, maxlat, 5), crs=ccrs.PlateCarree())
        lat_formatter = LatitudeFormatter()
        axf[a].yaxis.set_major_formatter(lat_formatter)
    else:
        axf[a].set_yticks([])

    if a in [9, 10, 11]:
        axf[a].set_xticks(np.arange(np.floor(minlon)+2, maxlon, 5), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        axf[a].xaxis.set_major_formatter(lon_formatter)
    else:
        axf[a].set_xticks([])
    #remove axis labels
    axf[a].set_xlabel('')
    axf[a].set_ylabel('')
    axf[a].contour(bat.lon, bat.lat, bat['elevation'], levels=[-200], colors='k', transform=ccrs.PlateCarree())
    axf[a].set_extent([hyb.lon.min().values, hyb.lon.max().values, hyb.lat.min().values, hyb.lat.max().values], crs=ccrs.PlateCarree())

plt.savefig('../figures/intercomp_mhw_maps.png', dpi=400)
plt.savefig('../figures/intercomp_mhw_maps.pdf')
