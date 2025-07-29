# Description: Calculate mean SST for historical and future periods for the hybrid model
import xarray as xr
import numpy as np
import pandas as pd
import os
import sys

import importlib
import aisst
importlib.reload(aisst)




ds_histssp585 = xr.open_dataset('../data/hybrid_model/sst_hybrid_histssp585_1850-2100.nc')
ds_ssp126 = xr.open_dataset('../data/hybrid_model/sst_hybrid_ssp126_2015-2100.nc')
ds_ssp245 = xr.open_dataset('../data/hybrid_model/sst_hybrid_ssp245_2015-2100.nc')
ds_ssp370 = xr.open_dataset('../data/hybrid_model/sst_hybrid_ssp370_2015-2100.nc')

# define time stepping for nc file averages
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

mSST = xr.Dataset( {'mean_sst' : (('lat', 'lon', 'time'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size, t_center.size))),
                    'mean_sst_clim' : (('lat', 'lon'), np.nan * np.ones((ds_histssp585.lat.size, ds_histssp585.lon.size)))},
                              coords = { 'lat' : ds_histssp585.lat, 'lon' : ds_histssp585.lon, 'time' : t_center})

ssp126_mSST = mSST.copy(deep=True)
ssp245_mSST = mSST.copy(deep=True)
ssp370_mSST = mSST.copy(deep=True)
histssp585_mSST = mSST

sst_clima = ds_histssp585.sel(time=slice('1982', '2011')).mean(dim='time', skipna=True ).mean(dim='member', skipna=True).sst.values
mSST.mean_sst_clim[:,:] = sst_clima

for t in range(t_center.size):
    print(p_starts[t], p_ends[t])

    histssp585_mSST.mean_sst[:,:,t] = ds_histssp585.sel(time=slice(str(p_starts[t]), str(p_ends[t]))).mean(dim='time', skipna=True ).mean(dim='member', skipna=True).sst.values - sst_clima
    ssp126_mSST.mean_sst[:,:,t] = ds_ssp126.sel(time=slice(str(p_starts[t]), str(p_ends[t]))).mean(dim='time', skipna=True ).mean(dim='member', skipna=True).sst.values - sst_clima
    ssp245_mSST.mean_sst[:,:,t] = ds_ssp245.sel(time=slice(str(p_starts[t]), str(p_ends[t]))).mean(dim='time', skipna=True ).mean(dim='member', skipna=True).sst.values - sst_clima
    ssp370_mSST.mean_sst[:,:,t] = ds_ssp370.sel(time=slice(str(p_starts[t]), str(p_ends[t]))).mean(dim='time', skipna=True ).mean(dim='member', skipna=True).sst.values - sst_clima




histssp585_mSST.to_netcdf('../data/mhw/mSST_histssp585.nc')
ssp126_mSST.to_netcdf('../data/mhw/mSST_ssp126.nc')
ssp245_mSST.to_netcdf('../data/mhw/mSST_ssp245.nc')
ssp370_mSST.to_netcdf('../data/mhw/mSST_ssp370.nc')
