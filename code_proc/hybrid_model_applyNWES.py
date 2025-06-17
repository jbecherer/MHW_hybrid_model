# This script applies the hybrid model to the entire NWEuropean Shelf


import xarray as xr
import numpy as np
import torch
import pandas as pd
import aisst

import importlib
importlib.reload(aisst)


def cal_hybrid_model4year(ds_org, bat, tides, year, scenario):
    ''' Calculate the hybrid model for a given year and save the results in a netcdf file

    Parameters:
    ds_org: xarray.Dataset
        Dataset containing the input variables for the hybrid model
    bat: xarray.Dataset
        Dataset containing the bathymetry data
    tides: xarray.Dataset
        Dataset containing the tidal current amplitude data
    year: str
        Year for which the hybrid model is calculated
    scenario: str
        Scenario for which the hybrid model is calculated
    '''

    ds = ds_org.sel(time=year).copy()

    # bat and tides lat-lon dimension are a bit larger than ds
    bat = bat.sel(lat=ds.lat, lon=ds.lon)
    tides = tides.sel(lat=ds.lat, lon=ds.lon)

    H = -bat.elevation.values
    Um2 = tides.Um2.values

    #==============================================================================
    # create input tensor
    #==============================================================================

    # add two extra dimension to H and Um2 repeating the same values along the time dimension of ds and the member dimension of ds
    H = H[np.newaxis, np.newaxis, :, :]
    Um2 = Um2[np.newaxis, np.newaxis, :, :]
    H = np.repeat(H, ds.time.size, axis=0)
    H = np.repeat(H, ds.member.size, axis=1)
    Um2 = np.repeat(Um2, ds.time.size, axis=0)
    Um2 = np.repeat(Um2, ds.member.size, axis=1)

    # number of total samples
    N = np.prod(H.shape)

    input_array: np.array = np.zeros((N, 6))

    input_array[:, 0] = ds.sst_rel.values.flatten()
    input_array[:, 1] = ds.sst_slope.values.flatten()
    input_array[:, 2] = ds.wsp.values.flatten()
    input_array[:, 3] = ds.hfds.values.flatten()
    input_array[:, 4] = Um2.flatten()
    input_array[:, 5] = H.flatten()

    input_tensor: torch.Tensor = torch.tensor(input_array, dtype=torch.float32)

    # load normalization parameters
    norm_par: pd.DataFrame = pd.read_csv('./models/NWEuroShelf/ml_norm_mpi.csv', index_col=0)
    input_mean: np.array = norm_par['mean'].values[[0,1,4,5,6,8]]
    input_std: np.array = norm_par['std'].values[[0,1,4,5,6,8]]

    # normalize input data
    input_n: torch.Tensor = aisst.normalize_data(input_tensor, input_mean, input_std)


    #==============================================================================
    # load model
    #==============================================================================\

    region = 'NWEuroShelf'
    model_csv = '../models/' + region  + '/best_model.csv'

    model = aisst.load_model_from_csv(model_csv, region, input_size=input_n.shape[1])


    #==============================================================================
    # apply model
    #==============================================================================
    _, _, output_mean, output_std = aisst.load_normalization_parameters(region)
    pre_n = model(input_n).detach()
    prediction = aisst.denormalize_data(pre_n, output_mean, output_std).numpy()


    # reshape predictiion 
    shape = H.shape + (prediction.shape[1],)
    prediction = prediction.reshape(shape)


    fft_cord = np.arange(prediction.shape[4])
    ds_fft = xr.DataArray(prediction+0j, coords=[ds.time, ds.member, ds.lat, ds.lon, fft_cord], dims=['time', 'member', 'lat', 'lon', 'freq'])
    ds_fft = ds_fft.to_dataset(name='fft')

    ds_fft = aisst.randomize_phase_of_dsfft(ds_fft)

    # create monthly sst data
    time_d = np.array(pd.date_range(start= year+'-01-01', end= year+'-12-31', freq='D'))
    sst_monthly_day = ds_org.sst.interp(time=time_d)
    sst_reconstructed = sst_monthly_day.copy()

    # loop through each month
    day_cnt = 0
    for t in ds_fft.time:
        month = t.dt.month
        year_date = t.dt.year

        sst_lf = sst_monthly_day.sel(time = (sst_monthly_day['time.year'] == year_date) & (sst_monthly_day['time.month'] == month) ).values.copy()
        fft_tr = ds_fft.sel(time=t).fft.values.copy()
        fft_tr = np.transpose(fft_tr, (3,0,1,2))

        n_days = sst_lf.shape[0]
        n_freq = fft_tr.shape[0]
        fft_m = np.zeros_like(sst_lf, dtype=complex)
        fft_m[:n_freq,:,:,:] = fft_tr
        fft_m[1:,:,:,:] = fft_m[1:,:,:,:]*2 # double the amplitude of the positive frequencies to account for missing negative ones
        # re-normalize
        fft_m = fft_m*n_days

        # sst_hf = np.real(np.fft.ifftn(fft_m, axes=(0,1,2)))
        sst_hf = np.real(np.fft.ifft(fft_m, axis=0))

        sst_reconstructed.values[day_cnt:(day_cnt+n_days), :,:,:] = sst_lf + sst_hf
        day_cnt += n_days

    sst_reconstructed = sst_reconstructed.to_dataset(name='sst')

    sst_reconstructed.to_netcdf('../data/hybrid_model/' + scenario + '/sst_hybrid_'+year+'.nc')

def main(start_year=1982, end_year=2023, scenario='histssp585'):
    #==============================================================================
    # load data
    #==============================================================================
    if start_year == 1982 and end_year == 2023 and scenario == 'histssp585':
        ds: xr.Dataset = xr.open_dataset('../data/mpi/mon/aisst_histssp585_1982_2023_allvars.nc')
    elif start_year == 1850 and end_year == 1982 and scenario == 'histssp585':
        ds: xr.Dataset = xr.open_dataset('../data/mpi/mon/aisst_histssp585_1850_1982_allvars.nc')
    else:
        ds: xr.Dataset = xr.open_dataset('../data/mpi/mon/aisst_'+scenario+'_2015_2100_allvars.nc')

    bat: xr.Dataset = xr.open_dataset('./data/bathymetry_1deg.nc')
    tides: xr.Dataset = xr.open_dataset('./data/tidal_current_amplitude_1deg.nc')

    years = [str(y) for y in range(start_year, end_year+1)]
    for year in years:
        print(scenario + ':' + str(year))
        cal_hybrid_model4year(ds, bat, tides, year, scenario)


if __name__ == '__main__':

    main(start_year=1982, end_year=2023, scenario='histssp585')
    main(start_year=1850, end_year=1982, scenario='histssp585')

    scenarios = ['ssp126', 'ssp245', 'ssp370','histssp585']
    for scenario in scenarios:
        main(start_year=2015, end_year=2100, scenario=scenario)



