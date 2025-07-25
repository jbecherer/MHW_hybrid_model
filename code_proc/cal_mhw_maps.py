import xarray as xr
import numpy as np
import pandas as pd
import os
import sys

import importlib
import aisst
importlib.reload(aisst)

sys.path.append('../external/marineHeatWaves/')
import marineHeatWaves as mhw
importlib.reload(mhw)

import warnings
warnings.filterwarnings('ignore')

# Load the data
min_lat = 49
max_lat = 61
min_lon = -11.5
max_lon = 9.5

#mpi  = xr.open_dataset('./data/cmip6/data/EuroShelf_1deg/MPI/day/tos/ensemble_1982_2023.nc')
mpi  = xr.open_dataset('../data/mpi/day/ensemble_1982_2023.nc')
# rename sfc to member
mpi = mpi.rename({'sfc': 'member'})
mpi = mpi.assign_coords(member=('member', np.arange(1,31)))

mpilf = xr.open_dataset('../data/mpi/mon/aisst_histssp585_1982_2023_allvars.nc')
mpilf = mpilf.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
# interpolate to daily
# mpilf = mpilf.interp(time=mpi.time)


era = xr.open_dataset('../data/NWEuroShelf_era5_1982_2023_allVars_1deg_daily.nc')
era = era.sst.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
era = era.to_dataset(name='sst')

#hybrid = xr.open_dataset('../data/hybrid_model/sst_hybrid_1982-2023.nc')
hybrid = xr.open_dataset('../data/hybrid_model/sst_hybrid_histssp585_1850-2100.nc')
# select same time period as the other datasets
hybrid = hybrid.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon), time=slice('1982-01-01', '2023-12-31'))

# clim period
period = [1982, 2011]

# create DataSet to store the mhw stats for each grid point and member
mhw_stats = xr.Dataset( { 'NpYear' : (('lat', 'lon', 'member'), np.nan * np.ones((era.lat.size, era.lon.size, mpi.member.size))), 
                            'mean_duration' : (('lat', 'lon', 'member'), np.nan * np.ones((era.lat.size, era.lon.size, mpi.member.size))),
                            'max_duration' : (('lat', 'lon', 'member'), np.nan * np.ones((era.lat.size, era.lon.size, mpi.member.size))),
                            'mean_mean_intensity' : (('lat', 'lon', 'member'), np.nan * np.ones((era.lat.size, era.lon.size, mpi.member.size))),
                            'max_mean_intensity' : (('lat', 'lon', 'member'), np.nan * np.ones((era.lat.size, era.lon.size, mpi.member.size))),
                            'mean_cum_intensity' : (('lat', 'lon', 'member'), np.nan * np.ones((era.lat.size, era.lon.size, mpi.member.size))),
                            'max_cum_intensity' : (('lat', 'lon', 'member'), np.nan * np.ones((era.lat.size, era.lon.size, mpi.member.size))),
                            'mean_peak_intensity' : (('lat', 'lon', 'member'), np.nan * np.ones((era.lat.size, era.lon.size, mpi.member.size))),
                            'max_peak_intensity' : (('lat', 'lon', 'member'), np.nan * np.ones((era.lat.size, era.lon.size, mpi.member.size))),}, 
                            coords = { 'lat' : era.lat, 'lon' : era.lon, 'member' : mpi.member})


print('Hybrid')
hybrid_stats = mhw_stats.copy(deep=True)
for la in range(hybrid.lat.size):
    for lo in range(hybrid.lon.size):
        for m in range(hybrid.member.size):
            print(la, lo)
            sst = hybrid.sst.isel(lat=la, lon=lo, member=m)
            mhws, _ = mhw.detect(sst.time.values, sst.values, climatologyPeriod=period.copy())
            if mhws['n_events'] == 0:
                continue
            stats = aisst.calculate_mhw_stats(mhws)
            hybrid_stats.NpYear[la, lo, m] = stats['n_events']/42
            hybrid_stats.mean_duration[la, lo, m] = stats['mean_duration']
            hybrid_stats.max_duration[la, lo, m] = stats['max_duration']
            hybrid_stats.mean_mean_intensity[la, lo, m] = stats['mean_mean_intensity']
            hybrid_stats.max_mean_intensity[la, lo, m] = stats['max_mean_intensity']
            hybrid_stats.mean_cum_intensity[la, lo, m] = stats['mean_cum_intensity']
            hybrid_stats.max_cum_intensity[la, lo, m] = stats['max_cum_intensity']
            hybrid_stats.mean_peak_intensity[la, lo, m] = stats['mean_peak_intensity']
            hybrid_stats.max_peak_intensity[la, lo, m] = stats['max_peak_intensity']

hybrid_stats.to_netcdf('../data/mhw/mhw_stats_hybrid.nc')



mpilf_stats = mhw_stats.copy(deep=True)
print('MPI low frequency')
for la in range(mpi.lat.size):
    for lo in range(mpi.lon.size):
        for m in range(mpi.member.size):
            print(la, lo, m)
            sst = mpilf.sst.isel(lat=la, lon=lo, member=m)
            # interpolate to daily
            sst = sst.interp(time=mpi.time)
            mhws, _ = mhw.detect(sst.time.values, sst.values, climatologyPeriod=period.copy())
            if mhws['n_events'] == 0:
                continue
            stats = aisst.calculate_mhw_stats(mhws)
            mpilf_stats.NpYear[la, lo, m] = stats['n_events']/42
            mpilf_stats.mean_duration[la, lo, m] = stats['mean_duration']
            mpilf_stats.max_duration[la, lo, m] = stats['max_duration']
            mpilf_stats.mean_mean_intensity[la, lo, m] = stats['mean_mean_intensity']
            mpilf_stats.max_mean_intensity[la, lo, m] = stats['max_mean_intensity']
            mpilf_stats.mean_cum_intensity[la, lo, m] = stats['mean_cum_intensity']
            mpilf_stats.max_cum_intensity[la, lo, m] = stats['max_cum_intensity']
            mpilf_stats.mean_peak_intensity[la, lo, m] = stats['mean_peak_intensity']
            mpilf_stats.max_peak_intensity[la, lo, m] = stats['max_peak_intensity']

mpilf_stats.to_netcdf('../data/mhw/mhw_stats_mpilf.nc')


era_stats = mhw_stats.copy(deep=True)
era_stats = era_stats.sel(member=1)

print('ERA')
for la in range(era.lat.size):
    for lo in range(era.lon.size):
        # for m in range(mpi.member.size):
            print(la, lo)
            sst = era.sst.isel(lat=la, lon=lo)
            mhws, _ = mhw.detect(sst.time.values, sst.values, climatologyPeriod=period.copy())
            if mhws['n_events'] == 0:
                continue
            stats = aisst.calculate_mhw_stats(mhws)
            era_stats.NpYear[la, lo,] = stats['n_events']/42
            era_stats.mean_duration[la, lo] = stats['mean_duration']
            era_stats.max_duration[la, lo] = stats['max_duration']
            era_stats.mean_mean_intensity[la, lo] = stats['mean_mean_intensity']
            era_stats.max_mean_intensity[la, lo] = stats['max_mean_intensity']
            era_stats.mean_cum_intensity[la, lo] = stats['mean_cum_intensity']
            era_stats.max_cum_intensity[la, lo] = stats['max_cum_intensity']
            era_stats.mean_peak_intensity[la, lo] = stats['mean_peak_intensity']
            era_stats.max_peak_intensity[la, lo] = stats['max_peak_intensity']

era_stats.to_netcdf('../data/mhw/mhw_stats_era.nc')


mpi_stats = mhw_stats.copy(deep=True)
print('MPI')
for la in range(mpi.lat.size):
    for lo in range(mpi.lon.size):
        for m in range(mpi.member.size):
            sst = mpi.sst.isel(lat=la, lon=lo, member=m)
            mhws, _ = mhw.detect(sst.time.values, sst.values, climatologyPeriod=period.copy())
            if mhws['n_events'] == 0:
                continue
            stats = aisst.calculate_mhw_stats(mhws)
            mpi_stats.NpYear[la, lo, m] = stats['n_events']/42
            mpi_stats.mean_duration[la, lo, m] = stats['mean_duration']
            mpi_stats.max_duration[la, lo, m] = stats['max_duration']
            mpi_stats.mean_mean_intensity[la, lo, m] = stats['mean_mean_intensity']
            mpi_stats.max_mean_intensity[la, lo, m] = stats['max_mean_intensity']
            mpi_stats.mean_cum_intensity[la, lo, m] = stats['mean_cum_intensity']
            mpi_stats.max_cum_intensity[la, lo, m] = stats['max_cum_intensity']
            mpi_stats.mean_peak_intensity[la, lo, m] = stats['mean_peak_intensity']
            mpi_stats.max_peak_intensity[la, lo, m] = stats['max_peak_intensity']

mpi_stats.to_netcdf('../data/mhw/mhw_stats_mpi.nc')



