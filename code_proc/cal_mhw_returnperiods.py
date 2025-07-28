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

def get_member_joined_sst(ds, period):
    """ generates a flat sst time series, joining all ensemble members
    over a given time period, and returns the time and sst arrays, where 
    the time array is an artifical time array starting at 1982-01-01
    """

    sst = ds.sel(time=slice(str(period[0]), str(period[1])))
    # Create a boolean mask that selects all times that are not February 29
    non_leap_day_mask = ~((sst['time'].dt.month == 2) & (sst['time'].dt.day == 29))

    # Apply the mask to the dataset to remove February 29
    sst_filtered = sst.sel(time=non_leap_day_mask)
    sst_flat = sst_filtered.values.T.flatten()

    # generate 900 year long time series without leap days
    time_start = np.datetime64('2000-01-01')
    time_stop = np.datetime64('2900-01-01')
    time_flat = pd.to_datetime(np.arange(time_start, time_stop, np.timedelta64(1, 'D')))

    non_leap_day_mask = ~((time_flat.month == 2) & (time_flat.day == 29))
    time_flat = time_flat[non_leap_day_mask].values

    if time_flat.size != sst_flat.size:
        print('time and sst arrays do not have the same size')
        print('sst size: ', sst_flat.size)
        print('time size: ', time_flat.size)

    return time_flat, sst_flat

def sort_events(mhws, Nyears):

    duration = np.zeros(Nyears)
    mean_i = np.zeros(Nyears)
    peak_i = np.zeros(Nyears)
    cum_i = np.zeros(Nyears)

    if mhws['n_events'] < Nyears : 
        Nyears = mhws['n_events']
        print('Number of events is less than Nyears')

    duration[:Nyears] = np.sort(mhws['duration'])[::-1][:Nyears]
    mean_i[:Nyears] = np.sort(mhws['intensity_mean'])[::-1][:Nyears]
    peak_i[:Nyears] = np.sort(mhws['intensity_max'])[::-1][:Nyears]
    cum_i[:Nyears] = np.sort(mhws['intensity_cumulative'])[::-1][:Nyears]

    return duration, mean_i, peak_i, cum_i


def calculate_mhw_stats_forlat(la):
    ds_histssp585 = xr.open_dataset('../data/hybrid_model/sst_hybrid_histssp585_1850-2100.nc')
    ds_ssp126 = xr.open_dataset('../data/hybrid_model/sst_hybrid_ssp126_2015-2100.nc')
    ds_ssp245 = xr.open_dataset('../data/hybrid_model/sst_hybrid_ssp245_2015-2100.nc')
    ds_ssp370 = xr.open_dataset('../data/hybrid_model/sst_hybrid_ssp370_2015-2100.nc')

    # clim period
    clim_period = [1982, 2011]

    t_step = 5 # time step in years

    # full time period
    p_starts = np.arange(1850, 2071, t_step)
    p_ends = np.arange(1879, 2100, t_step)
    p_center = 0.5*(p_ends+p_starts+1)
    t_center = np.array([np.datetime64(str(int(p_center[i]))) for i in range(p_center.size)])

    # future periods for scenarios
    p_starts_scen = np.arange(2015, 2071, t_step)
    p_ends_scen = np.arange(2044, 2100, t_step)
    p_center_scen = 0.5*(p_ends_scen+p_starts_scen+1)
    t_center_scen = np.array([np.datetime64(str(int(p_center_scen[i]))) for i in range(p_center_scen.size)])


    return_period = 900/np.arange(1,901,1)
    return_full_time = xr.Dataset( {'duration' : (('lat', 'lon', 'time', 'r_period'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, t_center.size, return_period.size))),
                                'mean_intensity' : (('lat', 'lon', 'time', 'r_period'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, t_center.size, return_period.size))),
                                'peak_intensity' : (('lat', 'lon', 'time', 'r_period'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, t_center.size, return_period.size))),
                                'cum_intensity' : (('lat', 'lon', 'time', 'r_period'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, t_center.size, return_period.size)))},
                                     coords = { 'lat' : ds_histssp585.lat, 'lon' : ds_histssp585.lon, 'member' : ds_histssp585.member, 'time' : t_center, 'r_period' : return_period})
    return_part_time = xr.Dataset( {'duration' : (('lat', 'lon', 'time', 'r_period'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, t_center_scen.size, return_period.size))),
                                'mean_intensity' : (('lat', 'lon', 'time', 'r_period'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, t_center_scen.size, return_period.size))),
                                'peak_intensity' : (('lat', 'lon', 'time', 'r_period'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, t_center_scen.size, return_period.size))),
                                'cum_intensity' : (('lat', 'lon', 'time', 'r_period'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, t_center_scen.size, return_period.size)))},
                                     coords = { 'lat' : ds_histssp585.lat, 'lon' : ds_histssp585.lon, 'member' : ds_histssp585.member, 'time' : t_center_scen, 'r_period' : return_period})


    import time
    start = time.time()

    ssp126_stats = return_part_time.copy(deep=True)
    ssp245_stats = return_part_time.copy(deep=True)
    ssp370_stats = return_part_time.copy(deep=True)
    histssp585_stats = return_full_time
    # for la in range(ds_histssp585.lat.size):
    for lo in range(ds_histssp585.lon.size):

        print(time.time() - start)
        start = time.time()
        print(la, lo)

        sst_full = ds_histssp585.sst.isel(lat=la, lon=lo)
        sst_ssp126 = ds_ssp126.sst.isel(lat=la, lon=lo)
        sst_ssp245 = ds_ssp245.sst.isel(lat=la, lon=lo)
        sst_ssp370 = ds_ssp370.sst.isel(lat=la, lon=lo)


        # climatology over reference period
        print('cal climatology over reference period: 1982 2011')
        time_flat_clim, sst_clim_flat = get_member_joined_sst(sst_full, clim_period)
        
        _ , clim_long = mhw.detect(time_flat_clim, sst_clim_flat)
        clim = {}
        clim['thresh'] = clim_long['thresh'][0:365]
        clim['seas'] = clim_long['seas'][0:365]
        clim['missing'] = clim_long['missing']

        print('     cal full period 1850 2100:')
        for t in range(t_center.size):
            print('    :', [p_starts[t], p_ends[t]])
            time_flat, sst_flat = get_member_joined_sst(sst_full, [p_starts[t], p_ends[t]])
            mhws, _ = mhw.detect(time_flat, sst_flat, externalClimatology=clim)
            if mhws['n_events'] == 0:
                continue
            duration, mean_i, peak_i, cum_i = sort_events(mhws, 900)
            histssp585_stats.duration[la, lo, t, :]       = duration 
            histssp585_stats.mean_intensity[la, lo, t, :] = mean_i
            histssp585_stats.peak_intensity[la, lo, t, :] = peak_i
            histssp585_stats.cum_intensity[la, lo, t, :]  = cum_i
     

        print('     ssp126 cal future period 2015 2100:')
        for t in range(t_center_scen.size):
            print('    :', [p_starts_scen[t], p_ends_scen[t]])
            time_flat, sst_flat = get_member_joined_sst(sst_ssp126, [p_starts_scen[t], p_ends_scen[t]])
            mhws, _ = mhw.detect(time_flat, sst_flat, externalClimatology=clim)
            if mhws['n_events'] == 0:
                continue
            duration, mean_i, peak_i, cum_i = sort_events(mhws, 900)
            ssp126_stats.duration[la, lo, t, :]       = duration 
            ssp126_stats.mean_intensity[la, lo, t, :] = mean_i
            ssp126_stats.peak_intensity[la, lo, t, :] = peak_i
            ssp126_stats.cum_intensity[la, lo, t, :]  = cum_i

        print('     ssp245 cal future period 2015 2100:')
        for t in range(t_center_scen.size):
            print('    :', [p_starts_scen[t], p_ends_scen[t]])
            time_flat, sst_flat = get_member_joined_sst(sst_ssp245, [p_starts_scen[t], p_ends_scen[t]])
            mhws, _ = mhw.detect(time_flat, sst_flat, externalClimatology=clim)
            if mhws['n_events'] == 0:
                continue
            duration, mean_i, peak_i, cum_i = sort_events(mhws, 900)
            ssp245_stats.duration[la, lo, t, :]       = duration 
            ssp245_stats.mean_intensity[la, lo, t, :] = mean_i
            ssp245_stats.peak_intensity[la, lo, t, :] = peak_i
            ssp245_stats.cum_intensity[la, lo, t, :]  = cum_i


        print('     ssp370 cal future period 2015 2100:')
        for t in range(t_center_scen.size):
            print('    :', [p_starts_scen[t], p_ends_scen[t]])
            time_flat, sst_flat = get_member_joined_sst(sst_ssp370, [p_starts_scen[t], p_ends_scen[t]])
            mhws, _ = mhw.detect(time_flat, sst_flat, externalClimatology=clim)
            if mhws['n_events'] == 0:
                continue
            duration, mean_i, peak_i, cum_i = sort_events(mhws, 900)
            ssp370_stats.duration[la, lo, t, :]       = duration 
            ssp370_stats.mean_intensity[la, lo, t, :] = mean_i
            ssp370_stats.peak_intensity[la, lo, t, :] = peak_i
            ssp370_stats.cum_intensity[la, lo, t, :]  = cum_i




    histssp585_stats.to_netcdf('../data/mhw/return_histssp585_la' + str(la) + '.nc')
    ssp126_stats.to_netcdf('../data/mhw/return_ssp126_la' + str(la) + '.nc')
    ssp245_stats.to_netcdf('../data/mhw/return_ssp245_la' + str(la) + '.nc')
    ssp370_stats.to_netcdf('../data/mhw/return_ssp370_la' + str(la) + '.nc')


if __name__ == '__main__':
    la_str = sys.argv[1]
    la = int(la_str)
    print("Calculating MHW stats for latitude index: ", la)

    if la > -1 and la < 12: 
        calculate_mhw_stats_forlat(la)
    else:
        for la in range(0, 12):
            calculate_mhw_stats_forlat(la)

