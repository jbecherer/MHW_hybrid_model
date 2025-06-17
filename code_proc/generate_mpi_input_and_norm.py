#==============================================================================
# Import libaries
#==============================================================================

import numpy as np
import datetime 
import scipy.io

import os, sys
import warnings
warnings.filterwarnings('ignore')

import xarray as xr
import pandas as pd

import aisst

#==============================================================================
# load data
#==============================================================================

def generate_mpi_eraperiod_ncfile() -> None:
    generate_mpi_period_ncfile(start='1982', stop='2023')


def generate_mpi_period_ncfile(start='1982', stop='2023', scenario='histssp585') -> None:    
    ''' This function generates the input data for the hybrid model from the MPI model

    Parameters:
    start (str): start year of the data
    stop (str): stop year of the data

    Returns:
    None
    '''
    nc_file = '../data/mpi/mon/aisst_' + scenario + '.nc'
    ds = xr.open_dataset(nc_file)

    # rename sfc to member
    ds = ds.rename({'sfc': 'member'})

    # take care of the height dimension
    ds = ds.assign_coords(member=('height', np.arange(1,31)))
    ds = ds.swap_dims({'height': 'member'})
    ds = ds.drop('height')


    # selct time period of era data
    ds_eperiod = ds.sel(time=slice(start, stop))

    # select the region NWEuroShelf
    minlat = 49
    maxlat = 61
    minlon = -11
    maxlon = 9
    ds_eperiod = ds_eperiod.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))



    #==============================================================================
    # cal sst relative to the min sst of the year
    #==============================================================================
    sst_min = ds_eperiod.sst.resample(time='Y', loffset='-183D').min(dim='time')
    temp_ref = sst_min.interp(time=ds_eperiod.time, method='nearest')
    # fix the edges
    temp_ref[0:6,:,:,:] = sst_min[0,:,:,:]
    temp_ref[-6:,:,:,:] = sst_min[-1,:,:,:]
    
    ds_eperiod['sst_rel'] = ds_eperiod.sst - temp_ref

    #---------------cal SST slope-----------------
    ds_eperiod['sst_slope'] = ds_eperiod.sst*0
    ds_eperiod['sst_slope'].values = .5*(ds_eperiod.sst.shift(time=-1).values - ds_eperiod.sst.shift(time=1).values)

    ds_eperiod.to_netcdf('../data/mpi/mon/aisst_' + scenario + '_' + start + '_' + stop + '_allvars.nc')

#==============================================================================
# cal normalization
#==============================================================================

def create_mpi_norm(ds_eperiod: xr.Dataset) -> None:
    # calculate the mean and std of the data
    # sst_mean = np.nanmean(ds_eperiod.sst.values)
    # sst_std = np.nanstd(ds_eperiod.sst.values)
    sst_rel_mean = np.nanmean(ds_eperiod.sst_rel.values)
    sst_rel_std = np.nanstd(ds_eperiod.sst_rel.values)
    sst_slope_mean = np.nanmean(ds_eperiod.sst_slope.values)
    sst_slope_std = np.nanstd(ds_eperiod.sst_slope.values)
    hfds_mean = np.nanmean(ds_eperiod.hfds.values)
    hfds_std = np.nanstd(ds_eperiod.hfds.values)
    wsp_mean = np.nanmean(ds_eperiod.wsp.values)
    wsp_std = np.nanstd(ds_eperiod.wsp.values)

    # load training nomalization
    norm = pd.read_csv('../data/ml_training/NWEuroShelf/ml_norm_params_input.csv', index_col=0)

    norm.at['relative SST', 'mean'] = sst_rel_mean
    norm.at['relative SST', 'std'] = sst_rel_std
    norm.at['SST monthly delta', 'mean'] = sst_slope_mean
    norm.at['SST monthly delta', 'std'] = sst_slope_std
    norm.at['surface heat flux', 'mean'] = hfds_mean
    norm.at['surface heat flux', 'std'] = hfds_std
    norm.at['wind speed', 'mean'] = wsp_mean
    norm.at['wind speed', 'std'] = wsp_std

    norm.to_csv('../models/NWEuroShelf/ml_norm_mpi.csv')


if __name__ == '__main__':
    start = sys.argv[1]
    stop = sys.argv[2]
    scenario = sys.argv[3]
    if start == 'era':
        generate_mpi_eraperiod_ncfile()
        ds = xr.open_dataset('../data/mpi/mon/aisst_histssp585_1982_2023_allvars.nc')
        create_mpi_norm(ds)
    else: 
        generate_mpi_period_ncfile(start, stop, scenario)



