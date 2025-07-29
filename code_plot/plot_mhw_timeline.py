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

min_lat = 54
max_lat = 58
min_lon = 0
max_lon = 7
histssp585 = xr.open_dataset('../data/mhw/merged_histssp585.nc')
histssp585 = histssp585.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

ssp126 = xr.open_dataset('../data/mhw/merged_ssp126.nc')
ssp245 = xr.open_dataset('../data/mhw/merged_ssp245.nc')
ssp370 = xr.open_dataset('../data/mhw/merged_ssp370.nc')
ssp126 = ssp126.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
ssp245 = ssp245.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
ssp370 = ssp370.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

# ignore cells deeper that 200 m
batlow = xr.open_dataset('../data/bathymetry_1deg.nc')
batlow = batlow.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
deepwater_mask_map = (batlow['elevation'].values > -200)
deepwater_mask_ssp = np.tile(
    deepwater_mask_map[:, :, np.newaxis, np.newaxis],
    (1, 1, ssp126.sizes['member'], ssp126.sizes['time'])
)
deepwater_mask_hist = np.tile(
    deepwater_mask_map[:, :, np.newaxis, np.newaxis],
    (1, 1, histssp585.sizes['member'], histssp585.sizes['time'])
)
for var in list(histssp585.data_vars.keys()):
    histssp585[var].values = np.where(deepwater_mask_hist,  histssp585[var].values, np.nan)
for var in list(ssp126.data_vars.keys()):
    ssp126[var].values = np.where(deepwater_mask_ssp,  ssp126[var].values, np.nan)
    ssp245[var].values = np.where(deepwater_mask_ssp,  ssp245[var].values, np.nan)
    ssp370[var].values = np.where(deepwater_mask_ssp,  ssp370[var].values, np.nan)



hist = histssp585.sel(time=slice('1850', '2000'))
ssp585 = histssp585.sel(time=slice('2030', '2100'))
hist_ssp585 = histssp585.sel(time=slice('2000', '2030'))

# mean SST
mSST_histssp585 = xr.open_dataset('../data/mhw/mSST_histssp585.nc')
mSST_ssp126 = xr.open_dataset('../data/mhw/mSST_ssp126.nc')
mSST_ssp245 = xr.open_dataset('../data/mhw/mSST_ssp245.nc')
mSST_ssp370 = xr.open_dataset('../data/mhw/mSST_ssp370.nc')
mSST_hist = mSST_histssp585.sel(time=slice('1850', '2000'))
mSST_ssp585 = mSST_histssp585.sel(time=slice('2030', '2100'))
mSST_ssp126 = mSST_ssp126.sel(time=slice('2030', '2100'))
mSST_ssp245 = mSST_ssp245.sel(time=slice('2030', '2100'))
mSST_ssp370 = mSST_ssp370.sel(time=slice('2030', '2100'))
mSST_hist_ssp585 = mSST_histssp585.sel(time=slice('2000', '2030'))

def add_fraction_of_heatdays(ds):
    ds['fraction_of_heatdays'] = ds['NpYear'] * ds['mean_duration'] / 365
    return ds

hist = add_fraction_of_heatdays(hist)
hist_ssp585 = add_fraction_of_heatdays(hist_ssp585)
ssp126 = add_fraction_of_heatdays(ssp126)
ssp245 = add_fraction_of_heatdays(ssp245)
ssp370 = add_fraction_of_heatdays(ssp370)
ssp585 = add_fraction_of_heatdays(ssp585)


def plot_line(ax, ds, feature, color, label):
    ax.fill_between(ds.time, ds[feature].mean(axis=0).mean(axis=0).max(axis=0).values, ds[feature].mean(axis=0).mean(axis=0).min(axis=0).values, color=color, alpha=0.3)
    ax.plot(ds.time, ds[feature].mean(axis=0).mean(axis=0).mean(axis=0).values, label=label, color=color)

def year2dt(year):
    return np.datetime64(f'{int(year)}-01-01')

def plot_feature_set(feature_set, feature_labels, data_sets, data_colors, data_labels):
    N_features = len(feature_set)
    fig = plt.figure( figsize = (9, 1.8*.9*N_features ), facecolor = (1, 1, 1))
    ax = fig.subplots(N_features, 2, sharex='col', sharey='row', width_ratios=[2,1])
    # am = [ [0+2*i, 1+2*i] for i in range(N_features)]
    # ax = fig.subplot_mosaic(am, sharex='col', sharey='row', width_ratios=[2.5,1])
    ax

    for i, feature in enumerate(feature_set):
        for j, ds in enumerate(data_sets):
            plot_line(ax[i,0], ds, feature, data_colors[j], data_labels[j])
        # ax[i,0].set_ylabel(feature_labels[i])
        ax[i,0].text(0.08, .95, feature_labels[i], transform=ax[i,0].transAxes, fontsize=12, va='top', ha='left')
        ax[i,0].set_xlim(year2dt(1840), year2dt(2100))

    legend = ax[0,0].legend(loc='lower left', shadow=True, fontsize=11, bbox_to_anchor=(0.01, 1.02), ncol=3)

    periods = [[1860,1890], [1982,2012], [2015,2045], [2070,2100]]
    p_colors = ['green', 'black', 'blue', 'red']
    p_colors = [cm.Dark2(i) for i in range(len(periods))]
    p_labels = ['pre-industrial', 'clima.', 'present', 'future']
    for i, p in enumerate(periods):
        year = int(.5*(p[0]+p[1]))

        if i < 2:
            ds = data_sets[0].sel(time=slice(str(year-2), str(year+2)))
            for j, feature in enumerate(feature_set):
                data = ds[feature].values.flatten()
                data = data[~np.isnan(data)]
                viola = ax[j,1].violinplot([data], positions=[i*2], showmeans=False, showmedians=True, showextrema=False, widths=0.25)
                viola['bodies'][0].set_facecolor(p_colors[i])
                ax[j,1].boxplot([data], positions=[i*2], bootstrap=None, showfliers=True, patch_artist=False, widths=0.25, flierprops={'marker':'+', 'markersize':2, 'markeredgecolor':(.2,.2,.2,.5)}) 
        else:
            for j, feature in enumerate(feature_set):
                d1 = data_sets[2].sel(time=slice(str(year), str(year)))[feature].values.flatten()
                d1 = d1[~np.isnan(d1)]
                d2 = data_sets[3].sel(time=slice(str(year), str(year)))[feature].values.flatten()
                d2 = d2[~np.isnan(d2)]
                d3 = data_sets[4].sel(time=slice(str(year), str(year)))[feature].values.flatten()
                d3 = d3[~np.isnan(d3)]
                d4 = data_sets[5].sel(time=slice(str(year), str(year)))[feature].values.flatten()
                d4 = d4[~np.isnan(d4)]
                data = [d1, d2, d3, d4]
                pos = [-.6 +i*2, -.2+i*2, .2+i*2, .6+i*2]
                viola = ax[j,1].violinplot(data, positions=pos, showmeans=False, showmedians=True, showextrema=False, widths=0.25)
                for k, vil in enumerate(viola['bodies']):
                    vil.set_facecolor(data_colors[k+2])
                ax[j,1].boxplot(data, positions=pos, bootstrap=None, showfliers=True, patch_artist=False, widths=0.25, flierprops={'marker':'+', 'markersize':2, 'markeredgecolor':(.2,.2,.2,.5)}) 


    y_pos = 10
    for i, p in enumerate(periods):
        year = int(.5*(p[0]+p[1]))
        # labels of periods in time plot
        ax[0,0].text(year2dt(year), y_pos-1, p_labels[i], va='top', ha='center', color=p_colors[i], fontsize=11)
        ax[0,0].plot([year2dt(p[0]), year2dt(p[1])], [y_pos, y_pos], color=p_colors[i], linewidth=2)

    ax[N_features-1,1].set_xticks([0, 2, 3.4, 3.8, 4.2, 4.6, 5.4, 5.8,6.2,6.6], 
                                  labels=[ 'pre-ind.','clima.', 
                                          'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 
                                          'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5'], 
                                  rotation=45, fontsize=9)
    ax[N_features-1,1].text(4, 0.5, 'present', ha='center', va='bottom', fontsize=10) 
    ax[N_features-1,1].text(6, 0.5, 'future', ha='center', va='bottom', fontsize=10)
    ax[N_features-1,0].set_xlabel('Year')

    # ax[0,0].set_yscale('log')
    # ax[0,1].set_yscale('log')
    ax[1,0].set_yscale('log')
    ax[1,1].set_yscale('log')
    ax[2,0].set_yscale('log')
    ax[2,1].set_yscale('log')

    ax[0,0].set_ylabel('events/year')
    ax[1,0].set_ylabel('days')
    ax[3,0].set_ylabel(r'$^\circ C$')
    ax[4,0].set_ylabel(r'$^\circ C$')

    ax[0,0].set_ylim(0, 13)
    ax[2,0].set_ylim(0.005, 2.)
    ax[3,0].set_ylim(0, 3)
    ax[4,0].set_ylim(0, 11)



    # vertical line deviding historical from scenario data
    for i, feature in enumerate(feature_set):
        yl = ax[i,0].get_ylim()
        ax[i,0].vlines(year2dt(2015), yl[0], yl[1], color='black', linestyle='--', linewidth=1)    
        for j, p in enumerate(periods):
            year = int(.5*(p[0]+p[1]))
            if i == 0:
                ax[i,0].vlines(year2dt(year), yl[0], 7, color=p_colors[j], linestyle='--', linewidth=1)
            else:
                ax[i,0].vlines(year2dt(year), yl[0], yl[1], color=p_colors[j], linestyle='--', linewidth=1)
        ax[i,0].set_ylim(yl)

    return fig

plt.close('all')

data_sets = [hist, hist_ssp585, ssp126, ssp245, ssp370, ssp585]
data_colors = ['black', [.3,0,0], 'blue', 'green', 'orange', 'red']
data_labels = ['historical', 'historical + SSP5-8.5', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']

# small feature set
feature_set = ['NpYear', 'mean_duration', 'fraction_of_heatdays', 'mean_mean_intensity', 'max_peak_intensity']
feature_labels = ['frequency', 'mean duration', 'fraction of time', 'mean intensity', 'max intensity of peak event']
N_features = len(feature_set)

fig = plot_feature_set(feature_set, feature_labels, data_sets, data_colors, data_labels)

# extra labels for fraction of heatdays 
ax = fig.get_axes()
xlim = ax[4].get_xlim()
ax[4].hlines(0.1, xlim[0], xlim[1], color='black', linestyle='--', linewidth=1)
ax[4].hlines(1, year2dt(1930), xlim[1], color='black', linestyle='-', linewidth=2)
ax[4].set_ylim(0, 1.1)
ax[4].set_xlim(xlim[0], xlim[1])
xlim = ax[5].get_xlim()
ax[5].hlines(0.1, xlim[0], xlim[1], color='black', linestyle='--', linewidth=1)
ax[5].hlines(1, 1, xlim[1], color='black', linestyle='-', linewidth=2)
ax[5].set_ylim(0, 1.1)
ax[5].set_xlim(xlim[0], xlim[1])

# add mean SST to bottom panel
a=8 
mSST_dss = [mSST_hist, mSST_hist_ssp585, mSST_ssp126, mSST_ssp245, mSST_ssp370, mSST_ssp585]
off_set=7
for j, ds in enumerate(mSST_dss):
    ax[a].plot(ds.time, ds['mean_sst'].mean(axis=0).mean(axis=0).values + off_set, label=data_labels[j], color=data_colors[j], linestyle='--', linewidth=1)
a=6
off_set=1
for j, ds in enumerate(mSST_dss):
    ax[a].plot(ds.time, ds['mean_sst'].mean(axis=0).mean(axis=0).values + off_set, label=data_labels[j], color=data_colors[j], linestyle='--', linewidth=1)



fig.subplots_adjust(hspace=0.1, wspace=0.02, right=0.98, left=0.08, top=0.92, bottom=0.07)

abc='afbgchdiej'
for i, a in enumerate(ax):
    a.text(0.01, .95, f'({abc[i]})', transform=a.transAxes, fontsize=12, va='top', ha='left')

fig.savefig('../figures/mhw_timeline_shortlist.png', dpi=200, facecolor='w', edgecolor='w')
# fig.savefig('../figures/mhw_timeline_shortlist.pdf', facecolor='w', edgecolor='w')


features_set = list(hist.data_vars.keys())
feature_labels = features_set
fig = plot_feature_set(features_set, feature_labels, data_sets, data_colors, data_labels)
fig.savefig('../figures/mhw_timeline_fulllist.png', dpi=200, facecolor='w', edgecolor='w')




