import xarray as xr
import numpy as np
import pandas as pd
import os
import sys

import importlib
import aisst
importlib.reload(aisst)

import marineHeatWaves as mhw
importlib.reload(mhw)

import warnings
warnings.filterwarnings('ignore')


def calculate_mhw_stats_forlat(la):
    ds_histssp585 = xr.open_dataset('../data/hybrid_model/sst_hybrid_histssp585_1850-2100.nc')
    ds_ssp126 = xr.open_dataset('../data/hybrid_model/sst_hybrid_ssp126_2015-2100.nc')
    ds_ssp245 = xr.open_dataset('../data/hybrid_model/sst_hybrid_ssp245_2015-2100.nc')
    ds_ssp370 = xr.open_dataset('../data/hybrid_model/sst_hybrid_ssp370_2015-2100.nc')

    # clim period
    period = [1982, 2011]

    t_step = 5 # time step in years

    # full time period
    p_starts = np.arange(1850, 2071, t_step)
    p_ends = np.arange(1880, 2101, t_step)
    p_center = 0.5*(p_ends+p_starts)
    t_center = np.array([np.datetime64(str(int(p_center[i]))) for i in range(p_center.size)])

    # future periods for scenarios
    p_starts_scen = np.arange(2015, 2071, t_step)
    p_ends_scen = np.arange(2045, 2101, t_step)
    p_center_scen = 0.5*(p_ends_scen+p_starts_scen)
    t_center_scen = np.array([np.datetime64(str(int(p_center_scen[i]))) for i in range(p_center_scen.size)])


    mhw_stats_full_time = xr.Dataset( {'NpYear' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center.size))),
                                'mean_duration' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center.size))),
                                'max_duration' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center.size))),
                                'mean_mean_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center.size))),
                                'max_mean_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center.size))),
                                'mean_cum_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center.size))),
                                'max_cum_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center.size))),
                                'mean_peak_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center.size))),
                                'max_peak_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center.size)))},
                                coords = { 'lat' : ds_histssp585.lat, 'lon' : ds_histssp585.lon, 'member' : ds_histssp585.member, 'time' : t_center})

    mhw_stats_scen = xr.Dataset( {'NpYear' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center_scen.size))),
                                'mean_duration' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center_scen.size))),
                                'max_duration' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center_scen.size))),
                                'mean_mean_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center_scen.size))),
                                'max_mean_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center_scen.size))),
                                'mean_cum_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center_scen.size))),
                                'max_cum_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center_scen.size))),
                                'mean_peak_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center_scen.size))),
                                'max_peak_intensity' : (('lat', 'lon', 'member', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, ds_histssp585.member.size, t_center_scen.size)))},
                                coords = { 'lat' : ds_histssp585.lat, 'lon' : ds_histssp585.lon, 'member' : ds_histssp585.member, 'time' : t_center_scen})


    import time
    start = time.time()

    ssp126_stats = mhw_stats_scen.copy(deep=True)
    ssp245_stats = mhw_stats_scen.copy(deep=True)
    ssp370_stats = mhw_stats_scen.copy(deep=True)
    histssp585_stats = mhw_stats_full_time
    # for la in range(ds_histssp585.lat.size):
    for lo in range(ds_histssp585.lon.size):
        for m in range(ds_histssp585.member.size):

            print(time.time() - start)
            start = time.time()
            print(la, lo, m)

            sst_full = ds_histssp585.sst.isel(lat=la, lon=lo, member=m)
            sst_ssp126 = ds_ssp126.sst.isel(lat=la, lon=lo, member=m)
            sst_ssp245 = ds_ssp245.sst.isel(lat=la, lon=lo, member=m)
            sst_ssp370 = ds_ssp370.sst.isel(lat=la, lon=lo, member=m)

            # climatology over reference period
            sst_clim = sst_full.sel(time=slice(str(period[0]), str(period[1])))
            _ , clim_long = mhw.detect(sst_clim.time.values, sst_clim.values)
            clim = {}
            clim['thresh'] = clim_long['thresh'][0:365]
            clim['seas'] = clim_long['seas'][0:365]
            clim['missing'] = clim_long['missing']

            print('     cal full period 1850 2100:')
            for t in range(t_center.size):
            #    print('         ' +  str(p_center[t]))
                sst = sst_full.sel(time=slice(str(p_starts[t]), str(p_ends[t])))
                mhws, _ = mhw.detect(sst.time.values, sst.values, externalClimatology=clim)
                if mhws['n_events'] == 0:
                    continue
                stats = aisst.calculate_mhw_stats(mhws)
                histssp585_stats.NpYear[la, lo, m, t] = stats['n_events']/30
                histssp585_stats.mean_duration[la, lo, m, t] = stats['mean_duration']
                histssp585_stats.max_duration[la, lo, m, t] = stats['max_duration']
                histssp585_stats.mean_mean_intensity[la, lo, m, t] = stats['mean_mean_intensity']
                histssp585_stats.max_mean_intensity[la, lo, m, t] = stats['max_mean_intensity']
                histssp585_stats.mean_cum_intensity[la, lo, m, t] = stats['mean_cum_intensity']
                histssp585_stats.max_cum_intensity[la, lo, m, t] = stats['max_cum_intensity']
                histssp585_stats.mean_peak_intensity[la, lo, m, t] = stats['mean_peak_intensity']
                histssp585_stats.max_peak_intensity[la, lo, m, t] = stats['max_peak_intensity']

            print('     ssp126 cal future period 2015 2100:')
            for t in range(t_center_scen.size):
            #    print('         ' +  str(p_center_scen[t]))
                sst = sst_ssp126.sel(time=slice(str(p_starts_scen[t]), str(p_ends_scen[t])))
                mhws, _ = mhw.detect(sst.time.values, sst.values, externalClimatology=clim)
                if mhws['n_events'] == 0:
                    continue
                stats = aisst.calculate_mhw_stats(mhws)
                ssp126_stats.NpYear[la, lo, m, t] = stats['n_events']/30
                ssp126_stats.mean_duration[la, lo, m, t] = stats['mean_duration']
                ssp126_stats.max_duration[la, lo, m, t] = stats['max_duration']
                ssp126_stats.mean_mean_intensity[la, lo, m, t] = stats['mean_mean_intensity']
                ssp126_stats.max_mean_intensity[la, lo, m, t] = stats['max_mean_intensity']
                ssp126_stats.mean_cum_intensity[la, lo, m, t] = stats['mean_cum_intensity']
                ssp126_stats.max_cum_intensity[la, lo, m, t] = stats['max_cum_intensity']
                ssp126_stats.mean_peak_intensity[la, lo, m, t] = stats['mean_peak_intensity']
                ssp126_stats.max_peak_intensity[la, lo, m, t] = stats['max_peak_intensity']

            print('     ssp245 cal future period 2015 2100:')
            for t in range(t_center_scen.size):
            #    print('         ' + str(p_center_scen[t]))
                sst = sst_ssp245.sel(time=slice(str(p_starts_scen[t]), str(p_ends_scen[t])))
                mhws, _ = mhw.detect(sst.time.values, sst.values, externalClimatology=clim)
                if mhws['n_events'] == 0:
                    continue
                stats = aisst.calculate_mhw_stats(mhws)
                ssp245_stats.NpYear[la, lo, m, t] = stats['n_events']/30
                ssp245_stats.mean_duration[la, lo, m, t] = stats['mean_duration']
                ssp245_stats.max_duration[la, lo, m, t] = stats['max_duration']
                ssp245_stats.mean_mean_intensity[la, lo, m, t] = stats['mean_mean_intensity']
                ssp245_stats.max_mean_intensity[la, lo, m, t] = stats['max_mean_intensity']
                ssp245_stats.mean_cum_intensity[la, lo, m, t] = stats['mean_cum_intensity']
                ssp245_stats.max_cum_intensity[la, lo, m, t] = stats['max_cum_intensity']
                ssp245_stats.mean_peak_intensity[la, lo, m, t] = stats['mean_peak_intensity']
                ssp245_stats.max_peak_intensity[la, lo, m, t] = stats['max_peak_intensity']

            print('     ssp370 cal future period 2015 2100:')
            for t in range(t_center_scen.size):
            #    print('         ' + str(p_center_scen[t]))
                sst = sst_ssp370.sel(time=slice(str(p_starts_scen[t]), str(p_ends_scen[t])))
                mhws, _ = mhw.detect(sst.time.values, sst.values, externalClimatology=clim)
                if mhws['n_events'] == 0:
                    continue
                stats = aisst.calculate_mhw_stats(mhws)
                ssp370_stats.NpYear[la, lo, m, t] = stats['n_events']/30
                ssp370_stats.mean_duration[la, lo, m, t] = stats['mean_duration']
                ssp370_stats.max_duration[la, lo, m, t] = stats['max_duration']
                ssp370_stats.mean_mean_intensity[la, lo, m, t] = stats['mean_mean_intensity']
                ssp370_stats.max_mean_intensity[la, lo, m, t] = stats['max_mean_intensity']
                ssp370_stats.mean_cum_intensity[la, lo, m, t] = stats['mean_cum_intensity']
                ssp370_stats.max_cum_intensity[la, lo, m, t] = stats['max_cum_intensity']
                ssp370_stats.mean_peak_intensity[la, lo, m, t] = stats['mean_peak_intensity']
                ssp370_stats.max_peak_intensity[la, lo, m, t] = stats['max_peak_intensity']




    histssp585_stats.to_netcdf('../data/mhw/mhw_stats_histssp585_la' + str(la) + '.nc')
    ssp126_stats.to_netcdf('../data/mhw/mhw_stats_ssp126_la' + str(la) + '.nc')
    ssp245_stats.to_netcdf('../data/mhw/mhw_stats_ssp245_la' + str(la) + '.nc')
    ssp370_stats.to_netcdf('../data/mhw/mhw_stats_ssp370_la' + str(la) + '.nc')


if __name__ == '__main__':
    la_str = sys.argv[1]
    la = int(la_str)
    print("Calculating MHW stats for latitude index: ", la)

    if la > -1 and la < 12: 
        calculate_mhw_stats_forlat(la)
    else:
        for la in range(0, 12):
            calculate_mhw_stats_forlat(la)

