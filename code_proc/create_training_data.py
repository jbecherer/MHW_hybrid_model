#==============================================================================
# This script creates the training data for the machine learning model
# It loads the ERA5 data, bathymetry and tidal current data, calculates the
# monthly mean and anomaly of the sea surface temperature, calculates the
# truncated Fourier transform of the daily mean sea surface temperature,
# calculates the yearly minimum sea surface temperature as reference, and
# creates the input and output data for the machine learning model.
# The input data consists of the relative sea surface temperature, the
# relative 2m air temperature, the air pressure, the wind speed, the surface
# heat flux, the M2 tidal amplitude, the S2/M2 ratio, the water depth, and
# the latitude. The output data consists of the truncated Fourier transform
# coefficients of the sea surface temperature.
# The data is saved in CSV files for training, validation and test sets.
# The data is also normalized and the normalization parameters are saved in CSV files.
#
# INPUT:
# - ERA5 data: ../data/NWEuroShelf_era5_1982_2023_allVars_1deg_daily.nc
# - Bathymetry data: ../data/bathymetry_1deg.nc
# - Tidal current data: ../data/tidal_current_amplitude_1deg.nc
# OUTPUT:
# - Input data: ../data/ml_training/<region>/ml_input_data.csv
# - Output data: ../data/ml_training/<region>/ml_output_data.csv
# - Training input data: ../data/ml_training/<region>/ml_input_data_train.csv
# - Training output data: ../data/ml_training/<region>/ml_output_data_train.csv
# - Validation input data: ../data/ml_training/<region>/ml_input_data_val.csv
# - Validation output data: ../data/ml_training/<region>/ml_output_data_val.csv
# - Test input data: ../data/ml_training/<region>/ml_input_data_test.csv
# - Test output data: ../data/ml_training/<region>/ml_output_data_test.csv
# - Validation and test input data: ../data/ml_training/<region>/ml_input_data_valtest.csv
# - Validation and test output data: ../data/ml_training/<region>/ml_output_data_valtest.csv
#
# - Normalization parameters for input data: ../data/ml_training/<region>/ml_norm_params_input.csv
# - Normalization parameters for output data: ../data/ml_training/<region>/ml_norm_params_output.csv
#==============================================================================
#==============================================================================

import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import datetime 
import scipy.io

import os, sys
from importlib import reload
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import xarray as xr

from sklearn.model_selection import train_test_split

import importlib
import aisst
importlib.reload(aisst)


def create_data(region='NorthSea'):
    #==============================================================================
    # Load data
    #==============================================================================

    era = xr.open_dataset('../data/NWEuroShelf_era5_1982_2023_allVars_1deg_daily.nc')
    bat = xr.open_dataset('../data/bathymetry_1deg.nc')
    tide = xr.open_dataset('../data/tidal_current_amplitude_1deg.nc')



    #==============================================================================
    # select a region
    #==============================================================================

    maxdepth = 200 # exclude deep water

    if region == 'NorthSea':
        minlat = 53
        maxlat = 60
        minlon = -3
        maxlon = 9
    elif region == 'NorthSeaPoint':
        minlat = 56.
        maxlat = 57.
        minlon = 3.0
        maxlon = 4.0
    else: # NWEuroShelf
        minlat = 49
        maxlat = 61
        minlon = -11
        maxlon = 9

    era = era.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))
    bat = bat.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))
    tide = tide.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))



    # cal the monthly mean and the anomaly
    sst_monthly, sst_monthly_day, sst_anom = aisst.calculateSSTanomaly_monthly(era.sst)
    # cal fft for each month
    ds_fft = aisst.cal_monthly_truncated_fft(sst_monthly, sst_anom, 10)

    # cal yearly minimum sst as reference
    sst_min = sst_monthly.resample(time='Y', loffset='-183D').min() # 183D is half a year
    # reference temperature
    temp_ref = sst_min.interp(time=sst_monthly.time, method='nearest')
    # fix the edges
    temp_ref[0:6,:,:] = sst_min[0,:,:]
    temp_ref[-6:,:,:] = sst_min[-1,:,:]

    sst_relative = (sst_monthly - temp_ref)


    t2_monthly = era.t2m.resample(time='1MS', loffset='15D' ).mean()
    t2_relative = (t2_monthly - temp_ref)

    press = era.msl.resample(time='1MS', loffset='15D' ).mean()
    wspd = era.wspd.resample(time='1MS', loffset='15D' ).mean()
    hf = era.hf.resample(time='1MS', loffset='15D' ).mean()

    in_data  = pd.DataFrame({'relative SST':[], 'SST monthly delta':[], 'relative T2m':[], 'air pressure':[], 'wind speed':[], 'surface heat flux':[], 'M2 tidal amplitude':[], 'S2/M2 ratio':[], 'water depth':[], 'latitude':[], 'year':[]})


    out_data = []

    cnt = 0
    for lat in era.lat.values:
        for lon in era.lon.values:
            sst = sst_relative.sel(lat=lat, lon=lon).values
            time = sst_relative.sel(lat=lat, lon=lon).time.values
            t2 = t2_relative.sel(lat=lat, lon=lon).values
            p = press.sel(lat=lat, lon=lon).values
            w = wspd.sel(lat=lat, lon=lon).values
            h = hf.sel(lat=lat, lon=lon).values
            b = -bat.elevation.sel(lat=lat, lon=lon).values.item()
            m2 = tide.sel(lat=lat, lon=lon).Um2.values.item()
            rsm = tide.sel(lat=lat, lon=lon).Rs2m2.values.item()

            fft_amp = np.abs(ds_fft.sel(lat=lat, lon=lon).fft.values)
            
            if np.any(np.isnan([b, m2, rsm])):
                continue

            if b > maxdepth: # exclude deep water there is one point with <1000m depth
                continue

            for t in range(1,len(sst)-1):

                year = np.datetime64(time[t], 'Y').astype(str).astype(int)

                sst_slope = .5*(sst[t+1] - sst[t-1])
                in_lst = [sst[t], sst_slope, t2[t], p[t], w[t], h[t], m2, rsm, b, lat, year]

                out_arr = list(fft_amp[t,:])
                if ~np.any(np.isnan(np.array(in_lst))) & ~np.any(np.isnan(out_arr)):
                    in_data.loc[cnt] = in_lst
                    out_data.append(out_arr)
                    cnt += 1

    out_data = pd.DataFrame(out_data)


    # years reserved for validation data picked at random one year from each decade
    #validation_years = [1989, 1995, 2001, 2015]
    # larger test set for statistical robustness
    validation_years = [2009, 2003, 1994, 1989, 2019, 1984, 2010, 2017]

    in_data_val = in_data[in_data['year'].isin(validation_years)]
    out_data_val = out_data[in_data['year'].isin(validation_years)]
    in_data_tr = in_data[~in_data['year'].isin(validation_years)]
    out_data_tr = out_data[~in_data['year'].isin(validation_years)]
            
    # drop the year column
    in_data = in_data.drop(columns=['year'])
    in_data_tr = in_data_tr.drop(columns=['year'])
    in_data_val = in_data_val.drop(columns=['year'])




    # split the data into training and test set
    x_train, _, y_train, _ = train_test_split(in_data_tr, out_data_tr, test_size=1, random_state=42)

    # randomize validation data
    x_valtest, _, y_valtest, _ = train_test_split(in_data_val, out_data_val, test_size=1, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(in_data_val, out_data_val, test_size=.5, random_state=42)

    # split the data into training, validation and test set
    # x_train, x_test, y_train, y_test = train_test_split(in_data, out_data, test_size=0.2, random_state=42)
    # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    #

    # save the data

    directory = '../data/ml_training/' + region
    if not os.path.exists(directory):
        os.mkdir(directory)

    in_data.to_csv('../data/ml_training/' + region + '/ml_input_data.csv', index=False)
    out_data.to_csv('../data/ml_training/' + region + '/ml_output_data.csv', index=False)

    x_train.to_csv('../data/ml_training/' + region + '/ml_input_data_train.csv', index=False)
    y_train.to_csv('../data/ml_training/' + region + '/ml_output_data_train.csv', index=False)

    x_val.to_csv('../data/ml_training/' + region + '/ml_input_data_val.csv', index=False)
    y_val.to_csv('../data/ml_training/' + region + '/ml_output_data_val.csv', index=False)

    x_test.to_csv('../data/ml_training/' + region + '/ml_input_data_test.csv', index=False)
    y_test.to_csv('../data/ml_training/' + region + '/ml_output_data_test.csv', index=False)

    x_valtest.to_csv('../data/ml_training/' + region + '/ml_input_data_valtest.csv', index=False)
    y_valtest.to_csv('../data/ml_training/' + region + '/ml_output_data_valtest.csv', index=False)

    # cal normalization parameters
    input_mean = in_data.mean(axis=0)
    input_std = in_data.std(axis=0)
    output_mean = out_data.mean(axis=0)
    output_std = out_data.std(axis=0)
    # save in csv file
    norm_inputs = pd.DataFrame({'mean': input_mean, 'std': input_std})
    norm_outputs = pd.DataFrame({'mean': output_mean, 'std': output_std})
    norm_inputs.to_csv('../data/ml_training/' + region + '/ml_norm_params_input.csv')
    norm_outputs.to_csv('../data/ml_training/' + region + '/ml_norm_params_output.csv')


if __name__ == '__main__':

    for region in ['NorthSea', 'NWEuroShelf', 'NorthSeaPoint']:
        print('Creating data for region: ', region)
        create_data(region)
