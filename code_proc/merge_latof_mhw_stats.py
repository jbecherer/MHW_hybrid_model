# This script merges the mhw stats for the different latitude bands for each scenario


import numpy as np
import pandas as pd
import os
import glob
import sys
import xarray as xr

# Load in the data
idir = '../data/mhw/'

scenarios = ['histssp585', 'ssp126', 'ssp245', 'ssp370']

for i in range(len(scenarios)):
    scenario = scenarios[i]

    # loop through all files in the directory
    files = glob.glob(idir + 'mhw_stats_' +  scenario + '_la*.nc')
    ds = xr.open_dataset(files[0])
    for j in range(1, len(files)):
        ds2 = xr.open_dataset(files[j])
        ds = xr.merge([ds, ds2])

    ds.to_netcdf('../data/mhw/merged_' + scenario + '.nc')


