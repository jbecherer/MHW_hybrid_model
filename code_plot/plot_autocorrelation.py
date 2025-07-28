import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import pandas as pd


era = xr.open_dataset('../data/NWEuroShelf_era5_1982_2023_allVars_1deg_daily.nc')
era = era['sst']
mpi = xr.open_dataset('../data/mpi/day/ensemble_1982_2023.nc')
mpi = mpi['sst']
hyb = xr.open_dataset('../data/hybrid_model/sst_hybrid_histssp585_1850-2100.nc')
hyb = hyb['sst']
# select just the reference period
hyb = hyb.sel(time=slice('1982-01-01', '2023-12-31'))

bat = xr.open_dataset('../data/bathymetry_1deg.nc')
bat = bat.sel(lat=slice(hyb.lat.min().values, hyb.lat.max().values),
            lon=slice(hyb.lon.min().values, hyb.lon.max().values))
deepwater_mask_map = (bat['elevation'].values > -200)
deepwater_mask_map = np.repeat(deepwater_mask_map[np.newaxis, :, :], hyb.sizes['member'], axis=0)
deepwater_mask_map = np.repeat(deepwater_mask_map[np.newaxis, :, :], hyb.sizes['time'], axis=0)
# apply the deepwater mask to the hybrid model
hyb.values = np.where(deepwater_mask_map,  hyb.values, np.nan)

def high_pass_filter(ts, cutoff_days=30):
    """
    High pass filter the time series to remove low frequency signals.
    """
    return ts - ts.rolling(time=cutoff_days, center=True).mean()

def cal_autocorrelation_length(sst):
    """
    Calculates the autocorrelation length for all time series in the member-lat-lon grid.
    Ignores NaNs in the time series.
    sst (xarray.DataArray): SST data with dimensions (time, member, lat, lon)
    Returns:
        xarray.DataArray: Autocorrelation length for each grid point (member, lat, lon)
    """
    def acl(ts):
        ts = ts[~np.isnan(ts)]
        if ts.size < 2:
            return np.nan
        ac = np.correlate(ts - np.mean(ts), ts - np.mean(ts), mode='full')
        ac = ac[ac.size // 2:ac.size // 2 + 100] / ac[ac.size // 2]
        return np.argmax(ac < 1/np.e)
    return xr.apply_ufunc(
        acl,
        sst,
        input_core_dims=[['time']],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )

# Calculate autocorrelation length for each model
era_acl = cal_autocorrelation_length(era)
print('ERA5 autocorrelation length: {:.2f} +/- {:.2f}'.format( np.nanmean(era_acl.values), np.nanstd(era_acl.values)))

mpi_acl = cal_autocorrelation_length(mpi[:,:,:,:])
print('MPIESM-d autocorrelation length: {:.2f} +/- {:.2f}'.format( np.nanmean(mpi_acl.values), np.nanstd(mpi_acl.values)))

hyb_acl = cal_autocorrelation_length(hyb[:,:,:,:])
print('Hybrid model autocorrelation length: {:.2f} +/- {:.2f}'.format( np.nanmean(hyb_acl.values), np.nanstd(hyb_acl.values)))

# now the same for high pass filtered data
era_hp = high_pass_filter(era)
era_acl_hp = cal_autocorrelation_length(era_hp)
print('ERA5 autocorrelation length after 30day high pass filter: {:.2f} +/- {:.2f}'.format( np.nanmean(era_acl_hp.values), np.nanstd(era_acl_hp.values)))

mpi_hp = high_pass_filter(mpi[:,:,:,:])
mpi_acl_hp = cal_autocorrelation_length(mpi_hp)
print('MPIESM-d autocorrelation length after 30day high pass filter: {:.2f} +/- {:.2f}'.format( np.nanmean(mpi_acl_hp.values), np.nanstd(mpi_acl_hp.values)))
hyb_hp = high_pass_filter(hyb[:,:,:,:])
hyb_acl_hp = cal_autocorrelation_length(hyb_hp)
print('Hybrid model autocorrelation length after 30day high pass filter: {:.2f} +/- {:.2f}'.format( np.nanmean(hyb_acl_hp.values), np.nanstd(hyb_acl_hp.values)))


