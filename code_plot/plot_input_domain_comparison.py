#==============================================================================
# Import libaries
#==============================================================================

import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import datetime 

import os, sys
from importlib import reload
import warnings
warnings.filterwarnings('ignore')

import xarray as xr
import pandas as pd

plt.switch_backend('Agg')
#==============================================================================
# load data
#============================================================================== load data
era = pd.read_csv('../data/ml_training/NWEuroShelf/ml_input_data.csv', index_col=0, parse_dates=True)
era = era.to_xarray()
# rename fiels
era = era.rename({'relative SST': 'sst_rel', 'SST monthly delta': 'sst_slope', 'wind speed': 'wsp', 'surface heat flux': 'hfds'})

# load MPI data
mpi_era = xr.open_dataset('../data/mpi/mon/aisst_histssp585_1982_2023_allvars.nc')
mpi_585 = xr.open_dataset('../data/mpi/mon/aisst_histssp585_2015_2100_allvars.nc')
mpi_126 = xr.open_dataset('../data/mpi/mon/aisst_ssp126_2015_2100_allvars.nc')
mpi_245 = xr.open_dataset('../data/mpi/mon/aisst_ssp245_2015_2100_allvars.nc')
mpi_370 = xr.open_dataset('../data/mpi/mon/aisst_ssp370_2015_2100_allvars.nc')
mpi_his = xr.open_dataset('../data/mpi/mon/aisst_histssp585_1850_1982_allvars.nc')

# load normalizations
norm_mpi = pd.read_csv('../models/NWEuroShelf/ml_norm_mpi.csv', index_col=0)
norm_era = pd.read_csv('../data/ml_training/NWEuroShelf/ml_norm_params_input.csv', index_col=0)


#==============================================================================
# normalize data
#==============================================================================
def normalize_data(ds: xr.Dataset, norm: pd.DataFrame) -> xr.Dataset:
    """Normalize the dataset using the provided normalization parameters."""
    ds['sst_rel_norm'] = (ds.sst_rel - norm['mean']['relative SST']) / norm['std']['relative SST']
    ds['sst_slope_norm'] = (ds.sst_slope - norm['mean']['SST monthly delta']) / norm['std']['SST monthly delta']
    ds['wsp_norm'] = (ds.wsp - norm['mean']['wind speed']) / norm['std']['wind speed']
    ds['hfds_norm'] = (ds.hfds - norm['mean']['surface heat flux']) / norm['std']['surface heat flux']
    return ds

era = normalize_data(era, norm_era)
mpi_era = normalize_data(mpi_era, norm_mpi)
mpi_585 = normalize_data(mpi_585, norm_mpi)
mpi_126 = normalize_data(mpi_126, norm_mpi)
mpi_245 = normalize_data(mpi_245, norm_mpi)
mpi_370 = normalize_data(mpi_370, norm_mpi)
mpi_his = normalize_data(mpi_his, norm_mpi)


#==============================================================================
# plot 2d histograms with 1d edges 
#==============================================================================

plt.close('all')


fig = plt.figure(figsize=(10, 10))
outer = GridSpec(2, 2, wspace=0.3, hspace=0.3)

axes_main = []
axes_xhist = []
axes_yhist = []

for i in range(2):
    for j in range(2):
        inner = GridSpecFromSubplotSpec(
            2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
            wspace=0.00, hspace=0.00, subplot_spec=outer[i, j]
        )
        ax_main = plt.subplot(inner[1, 0])
        ax_xhist = plt.subplot(inner[0, 0], sharex=ax_main)
        ax_yhist = plt.subplot(inner[1, 1], sharey=ax_main)
        # Remove tick labels on marginal plots
        plt.setp(ax_xhist.get_xticklabels(), visible=False)
        plt.setp(ax_yhist.get_yticklabels(), visible=False)
        ax_xhist.axis('off')
        ax_yhist.axis('off')

        axes_main.append(ax_main)
        axes_xhist.append(ax_xhist)
        axes_yhist.append(ax_yhist)



data_sets = [era, mpi_era,  mpi_585, mpi_his]
cmaps = [cm.Greys, cm.Blues, cm.Reds, cm.Greens, cm.Purples]

Xs = ['sst_rel', 'wsp', 'sst_rel_norm', 'wsp_norm']
Ys = ['sst_slope', 'hfds', 'sst_slope_norm', 'hfds_norm']
Xlabels = ['Relative SST [K]', 'Wind speed [m/s]', 'Normalized relative SST', 'Normalized wind speed']
Ys_labels = ['SST monthly delta [K/month]', 'Surface heat flux [W/m²]', 'Normalized SST monthly delta', 'Normalized surface heat flux']



for i in range(len(data_sets)):
    ds = data_sets[i]
    cmap = cmaps[i]

    for j in range(len(Xs)):

    
        X = ds[Xs[j]].values.ravel()
        Y = ds[Ys[j]].values.ravel()
        ii= ~np.isnan(X) & ~np.isnan(Y)
        X = X[ii]
        Y = Y[ii]
        # h, xedges, yedges, img = ax[0, 0].hist2d(X, Y, bins=30, cmap=cmap, norm='log')
        # X, Y = np.meshgrid(0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1]))
        # ax[0, 0].contour(X, Y, h.T, levels=5, colors='blue', linewidths=1)


        # 2D histogram
        # h, xedges, yedges, img = ax_main.hist2d(X, Y, bins=30, cmap=cmap, norm='log')
        h, xedges, yedges = np.histogram2d(X, Y, bins=50, density=True)
        Xc, Yc = np.meshgrid(0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1]))
        Nlevels = 5
        levels = np.logspace(np.log10(0.00001), np.log10(1), Nlevels)

        colors= cmap(np.linspace(0, 1, Nlevels))
        axes_main[j].contour(Xc, Yc, h.T, levels=levels, colors=colors, linewidths=1)
        # contourf = ax_main.contourf(Xc, Yc, h.T, levels=levels, cmap=cmap, alpha=0.3)

        # 1D histograms
        axes_xhist[j].hist(X, bins=xedges, color=cmap(0.5), alpha=0.3, density=True)
        axes_yhist[j].hist(Y, bins=yedges, orientation='horizontal', color=cmap(0.5), alpha=0.3 , density=True)

        axes_main[j].set_xlabel(Xlabels[j], fontsize=12)
        axes_main[j].set_ylabel(Ys_labels[j], fontsize=12)


#fig.savefig('../figures/input_range_comparison_2d.pdf', dpi=200, facecolor='w', edgecolor='w')
fig.savefig('../figures/input_range_comparison_2d.png', dpi=200, facecolor='w', edgecolor='w')

#==============================================================================
# plot 1d box plots 
#==============================================================================


def bar_plot( features, aera, ahyb, ampi, labels=None): 
    N = len(features)
    fig, ax = plt.subplots(1, N, figsize=(2.8*N, 3.8))

    # vertically squees the ax by .9
    for a in ax:
        pos = a.get_position()
        new_pos = [pos.x0, pos.y0 + 0.1, pos.width, pos.height*.9]
        a.set_position(new_pos)

    if labels is None:
        labels = features
    for a, col in enumerate(features):

        d1 = aera[col].values.flatten()
        d1 = d1[~np.isnan(d1)]
        d2 = ahyb[col].values.flatten()
        d2 = d2[~np.isnan(d2)]
        d3 = ampi[col].values.flatten()
        d3 = d3[~np.isnan(d3)]
        data = [d1, d2, d3]
        colors = [ 'k', cm.tab10(0), cm.tab10(1)]

        viola = ax[a].violinplot(data, showmeans=False, showmedians=True, 
                                 showextrema=False, widths=.9)
                                 #showextrema=False, widths=.9, bw_method=0.5)
        # legend on the bottom outside over the entire figure


        for i, vil in enumerate(viola['bodies']):
            vil.set_facecolor(colors[i])
        if a == 0:
            lg = fig.legend([vil for vil in viola['bodies']], ['ERA5 1982-2023', 'MPIESM 1982-2023', 'MPIESM 2015-2100 ssp585'], 
                          loc='lower center', bbox_to_anchor=(0.5, 0.), ncol=3, fontsize=14, frameon=False)
                          #loc='lower left', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11, frameon=False)
            #move_legend(ax[a], lg, 1.3,-.1) # 1,0 move legend to right outside edge
        
        ax[a].boxplot(data, bootstrap=None, showfliers=True, 
                      flierprops={'marker':'o', 'markersize':1, 'color':'k', 'alpha':.1},
                      patch_artist=False, widths=0.4, whis=(1.0,99.0),)


        ax[a].set_xticks([1, 2, 3], labels=[ ' ', ' ', ' '], rotation=30, fontsize=11, fontweight='bold')

        # remove ticks from x axis
        ax[a].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # ax[a].set_xticks([1, 2, 3], labels=[ ' ERA5  1982-2023', 'MPIESM 1982-2023', 'SSP585 2015-2100'], rotation=30, fontsize=11, fontweight='bold')
        # ax[a].tick_params( axis='y', labelsize=11)
        # plt.xticks(rotation=45)
        ax[a].set_title(labels[a], fontsize=14, fontweight='bold')

    plt.tight_layout()
    abc = 'abcdefghijklmnopqrstuvwxyz'
    for a in range(len(ax)):
        tabc = ax[a].text(0.01, 0.95, '(' + abc[a] + ')', transform=ax[a].transAxes)
        tabc.set_fontweight('bold')
        tabc.set_fontsize(11)

        pos = ax[a].get_position()
        new_pos = [pos.x0, pos.y0 + 0.1, pos.width, pos.height*.9]
        ax[a].set_position(new_pos)
    # change axis style to only show the bottom axis
    # for a in ax:
    #     a.set_ylim([0, a.get_ylim()[1]])
        


    return fig, ax



features = list(['sst_rel_norm', 'sst_slope_norm', 'wsp_norm', 'hfds_norm'])
fig, ax = bar_plot(features, era, mpi_era, mpi_585, labels=[r'$\overline{T_{\text{rel}}}$', r'$\overline{\Delta T}$', r'$\overline{U_{10}}$', r'$\overline{Q_{\text{net}}}$'])


# fig.savefig('../figures/input_range_comparison.pdf', dpi=200, facecolor='w', edgecolor='w')
fig.savefig('../figures/input_range_comparison.png', dpi=200, facecolor='w', edgecolor='w')

#==============================================================================
# create a csv table with statistics
#==============================================================================

stats = pd.DataFrame(index=['sst_rel mean', 'sst_rel std', 'sst_rel range', 'sst_rel percentile', 'sst_rel normlized range', 'sst_rel normlized percentile',
                             'sst_slope mean', 'sst_slope std', 'sst_slope range', 'sst_slope percentile' , 'sst_slope normlized range', 'sst_slope normlized percentile',
                             'wsp mean', 'wsp std', 'wsp range', 'wsp percentile', 'wsp normalized range', 'wsp normalized percentile',
                             'hfds mean', 'hfds std', 'hfds range', 'hfds percentile', 'hfds normalized range', 'hfds normalized percentile'], 
                     columns=['ERA5 1982-2023', 'MPIESM 1982-2023', 'MPIESM 2015-2100 ssp126', 'MPIESM 2015-2100 ssp245', 'MPIESM 2015-2100 ssp370', 'MPIESM 2015-2100 ssp585', 'MPIESM 1850-1982'])


def get_stats(ds: xr.Dataset, norm: pd.DataFrame) -> pd.Series:
    """Calculate statistics for a given dataset and normalization parameters."""
    stats = pd.Series(index=['sst_rel mean', 'sst_rel std', 'sst_rel range', 'sst_rel percentile', 'sst_rel normlized range', 'sst_rel normlized percentile',
                             'sst_slope mean', 'sst_slope std', 'sst_slope range', 'sst_slope percentile' , 'sst_slope normlized range', 'sst_slope normlized percentile',
                             'wsp mean', 'wsp std', 'wsp range', 'wsp percentile', 'wsp normalized range', 'wsp normalized percentile',
                             'hfds mean', 'hfds std', 'hfds range', 'hfds percentile', 'hfds normalized range', 'hfds normalized percentile'])
    
    stats['sst_rel mean']   = f"{ds.sst_rel.mean().values:.2f}"
    stats['sst_rel std']    = f"{ds.sst_rel.std().values:.2f}"
    stats['sst_rel range']  = f"{ds.sst_rel.min().values:.2f} - {ds.sst_rel.max().values:.2f}"
    percentile = [ds.sst_rel.quantile(0.001).values, ds.sst_rel.quantile(0.05).values, ds.sst_rel.quantile(0.95).values, ds.sst_rel.quantile(0.999).values]
    stats['sst_rel percentile'] = f"{percentile[0]:.2f} - {percentile[1]:.2f} | {percentile[2]:.2f} - {percentile[3]:.2f}"
    stats['sst_rel normlized range'] = f"{(ds.sst_rel.min().values - norm['mean']['relative SST']) / norm['std']['relative SST']:.2f} - {(ds.sst_rel.max().values - norm['mean']['relative SST']) / norm['std']['relative SST']:.2f}"
    stats['sst_rel normlized percentile'] = f"{(percentile[0] - norm['mean']['relative SST']) / norm['std']['relative SST']:.2f} - {(percentile[1] - norm['mean']['relative SST']) / norm['std']['relative SST']:.2f} | {(percentile[2] - norm['mean']['relative SST']) / norm['std']['relative SST']:.2f} - {(percentile[3] - norm['mean']['relative SST']) / norm['std']['relative SST']:.2f}"

    stats['sst_slope mean'] = f"{ds.sst_slope.mean().values:.2f}"
    stats['sst_slope std']  = f"{ds.sst_slope.std().values:.2f}"
    stats['sst_slope range'] = f"{ds.sst_slope.min().values:.2f} - {ds.sst_slope.max().values:.2f}"
    stats['sst_slope percentile'] = f"{ds.sst_slope.quantile(0.001).values:.2f} - {ds.sst_slope.quantile(0.05).values:.2f} | {ds.sst_slope.quantile(0.95).values:.2f} - {ds.sst_slope.quantile(0.999).values:.2f}"
    stats['sst_slope normlized range'] = f"{(ds.sst_slope.min().values - norm['mean']['SST monthly delta']) / norm['std']['SST monthly delta']:.2f} - {(ds.sst_slope.max().values - norm['mean']['SST monthly delta']) / norm['std']['SST monthly delta']:.2f}"
    stats['sst_slope normlized percentile'] = f"{(ds.sst_slope.quantile(0.001).values - norm['mean']['SST monthly delta']) / norm['std']['SST monthly delta']:.2f} - {(ds.sst_slope.quantile(0.05).values - norm['mean']['SST monthly delta']) / norm['std']['SST monthly delta']:.2f} | {(ds.sst_slope.quantile(0.95).values - norm['mean']['SST monthly delta']) / norm['std']['SST monthly delta']:.2f} - {(ds.sst_slope.quantile(0.999).values - norm['mean']['SST monthly delta']) / norm['std']['SST monthly delta']:.2f}"


    stats['wsp mean']       = f"{ds.wsp.mean().values:.2f}"
    stats['wsp std']        = f"{ds.wsp.std().values:.2f}"
    stats['wsp range']      = f"{ds.wsp.min().values:.2f} - {ds.wsp.max().values:.2f}"
    stats['wsp percentile'] = f"{ds.wsp.quantile(0.001).values:.2f} - {ds.wsp.quantile(0.05).values:.2f} | {ds.wsp.quantile(0.95).values:.2f} - {ds.wsp.quantile(0.999).values:.2f}"
    stats['wsp normalized range'] = f"{(ds.wsp.min().values - norm['mean']['wind speed']) / norm['std']['wind speed']:.2f} - {(ds.wsp.max().values - norm['mean']['wind speed']) / norm['std']['wind speed']:.2f}"
    stats['wsp normalized percentile'] = f"{(ds.wsp.quantile(0.001).values - norm['mean']['wind speed']) / norm['std']['wind speed']:.2f} - {(ds.wsp.quantile(0.05).values - norm['mean']['wind speed']) / norm['std']['wind speed']:.2f} | {(ds.wsp.quantile(0.95).values - norm['mean']['wind speed']) / norm['std']['wind speed']:.2f} - {(ds.wsp.quantile(0.999).values - norm['mean']['wind speed']) / norm['std']['wind speed']:.2f}"

    stats['hfds mean']      = f"{ds.hfds.mean().values:.2f}"
    stats['hfds std']       = f"{ds.hfds.std().values:.2f}"
    stats['hfds range']     = f"{ds.hfds.min().values:.2f} - {ds.hfds.max().values:.2f}"
    stats['hfds percentile'] = f"{ds.hfds.quantile(0.001).values:.2f} - {ds.hfds.quantile(0.05).values:.2f} | {ds.hfds.quantile(0.95).values:.2f} - {ds.hfds.quantile(0.999).values:.2f}"
    stats['hfds normalized range'] = f"{(ds.hfds.min().values - norm['mean']['surface heat flux']) / norm['std']['surface heat flux']:.2f} - {(ds.hfds.max().values - norm['mean']['surface heat flux']) / norm['std']['surface heat flux']:.2f}"
    stats['hfds normalized percentile'] = f"{(ds.hfds.quantile(0.001).values - norm['mean']['surface heat flux']) / norm['std']['surface heat flux']:.2f} - {(ds.hfds.quantile(0.05).values - norm['mean']['surface heat flux']) / norm['std']['surface heat flux']:.2f} | {(ds.hfds.quantile(0.95).values - norm['mean']['surface heat flux']) / norm['std']['surface heat flux']:.2f} - {(ds.hfds.quantile(0.999).values - norm['mean']['surface heat flux']) / norm['std']['surface heat flux']:.2f}"
    
    return stats

def get_range(era, mpi_era, mpi_126, mpi_245, mpi_370, mpi_585, mpi_his):

    range_stats = pd.DataFrame(
                                index=['ERA5 1982-2023', 'MPIESM 1982-2023', 'MPIESM 2015-2100 ssp126', 'MPIESM 2015-2100 ssp245', 'MPIESM 2015-2100 ssp370', 'MPIESM 2015-2100 ssp585', 'MPIESM 1850-1982'])

    vars = ['sst_rel_norm', 'sst_slope_norm', 'wsp_norm', 'hfds_norm']

    for var,i in zip(vars, range(len(vars))):
        era_range = [ era[var].min().values, era[var].max().values ]

        range_stats.loc['ERA5 1982-2023', var] = f"{era_range[0]:.2f} - {era_range[1]:.2f}"
        data = mpi_era[var].values.flatten()
        data = data[~np.isnan(data)]
        mpi_era_range = np.sum((data < era_range[0]) | (data > era_range[1]))/ data.size 
        range_stats.loc['MPIESM 1982-2023', var] = f"{mpi_era_range:.4%}"
        data = mpi_126[var].values.flatten()
        data = data[~np.isnan(data)]
        mpi_126_range = np.sum((data < era_range[0]) | (data > era_range[1]))/ data.size
        range_stats.loc['MPIESM 2015-2100 ssp126', var] = f"{mpi_126_range:.4%}"
        data = mpi_245[var].values.flatten()
        data = data[~np.isnan(data)]
        mpi_245_range = np.sum((data < era_range[0]) | (data > era_range[1]))/ data.size
        range_stats.loc['MPIESM 2015-2100 ssp245', var] = f"{mpi_245_range:.4%}"
        data = mpi_370[var].values.flatten()
        data = data[~np.isnan(data)]
        mpi_370_range = np.sum((data < era_range[0]) | (data > era_range[1]))/ data.size
        range_stats.loc['MPIESM 2015-2100 ssp370', var] = f"{mpi_370_range:.4%}"
        data = mpi_585[var].values.flatten()
        data = data[~np.isnan(data)]
        mpi_585_range = np.sum((data < era_range[0]) | (data > era_range[1]))/ data.size
        range_stats.loc['MPIESM 2015-2100 ssp585', var] = f"{mpi_585_range:.4%}"
        data = mpi_his[var].values.flatten()
        data = data[~np.isnan(data)]
        mpi_his_range = np.sum((data < era_range[0]) | (data > era_range[1]))/ data.size
        range_stats.loc['MPIESM 1850-1982', var] = f"{mpi_his_range:.4%}"
    return range_stats
    
range_stats = get_range(era, mpi_era, mpi_126, mpi_245, mpi_370, mpi_585, mpi_his)
range_stats.to_csv('../data/input_domain_range_comparison.csv')

stats.loc[:, 'ERA5 1982-2023'] = get_stats(era, norm_era)
stats.loc[:, 'MPIESM 1982-2023'] = get_stats(mpi_era, norm_mpi)
stats.loc[:, 'MPIESM 2015-2100 ssp126'] = get_stats(mpi_126, norm_mpi)
stats.loc[:, 'MPIESM 2015-2100 ssp245'] = get_stats(mpi_245, norm_mpi)
stats.loc[:, 'MPIESM 2015-2100 ssp370'] = get_stats(mpi_370, norm_mpi)
stats.loc[:, 'MPIESM 2015-2100 ssp585'] = get_stats(mpi_585, norm_mpi)
stats.loc[:, 'MPIESM 1850-1982'] = get_stats(mpi_his, norm_mpi)
stats.to_csv('../data/ml_input_domain_comparison.csv')
stats.T.iloc[[0,1,5],[4,5,9,10,14,15,19,23]].to_csv('../data/ml_input_domain_comparison_small.csv')
