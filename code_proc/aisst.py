#==============================================================================
# Import libaries
#==============================================================================

import numpy as np
import datetime 
import scipy.io
from scipy.optimize import curve_fit

import os, sys

import xarray as xr
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

# add the path of the heat ewave module module to the path 
#sys.path.append('../external/marineHeatWaves/')
# Load marineHeatWaves definition module
import marineHeatWaves as mhw


#==============================================================================
# Pytorch helper functions
#==============================================================================

#---------------data-----------------
# define custom dataset
class CustomDataset(Dataset):
    ''' Custom dataset class for pytorch dataloader
    '''
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def create_dataloader(input_data, output_data, batch_size=32, shuffle=True):
    ''' Create a dataloader

    Parameters
    ----------
    input_data : torch tensor
        input data
    output_data : torch tensor
        output data
    batch_size : int (default 32)
        batch size
    shuffle : bool (default True)
        shuffle the data

    Returns
    -------
    dataloader : torch DataLoader
        dataloader
    '''
    dataset = CustomDataset(input_data, output_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def load_tensor_from_csv(file_path):
    ''' Load a csv file and convert it to a pytorch tensor

    Parameters
    ----------
    file_path : str
        path to the csv file
        
    Returns
    -------
    torch_data : torch tensor
        pytorch tensor
    '''
    df = pd.read_csv(file_path)
    np_data = df.to_numpy()
    torch_data = torch.from_numpy(np_data).float()

    return torch_data

def load_normalization_parameters(region):
    ''' get normalization parameters from csv file'''

    norm_in = pd.read_csv('./data/ml_training/' + region + '/ml_norm_params_input.csv')
    norm_out = pd.read_csv('./data/ml_training/' + region + '/ml_norm_params_output.csv')

    input_mean = norm_in['mean'].values
    input_std = norm_in['std'].values
    output_mean = norm_out['mean'].values
    output_std = norm_out['std'].values

    return input_mean, input_std, output_mean, output_std

def normalize_data(data, mean, std):
    """ Normalize the data

    Parameters
    ----------
    data : torch tensor
        data to normalize
    mean : numpy array
        mean of the data
    std : numpy array
        standard deviation of the data

    Returns
    -------
    normalized_data : torch tensor
        normalized data
    """

    mean = torch.from_numpy(mean).float()
    std = torch.from_numpy(std).float()

    # make there is no division by zero
    std[std == 0] = 1
    # std = torch.tensor(std)

    data_normalized = (data.detach() - mean) / std

    return data_normalized


def denormalize_data(data, mean, std):
    """ Denormalize the data

    Parameters
    ----------
    data : torch tensor
        data to denormalize
    mean : numpy array
        mean of the data
    std : numpy array
        standard deviation of the data

    Returns
    -------
    denormalized_data : torch tensor
        denormalized data
    """

    mean = torch.from_numpy(mean).float()
    std = torch.from_numpy(std).float()

    return data * std + mean


def reduce_feature_set(input_data, set='full'):
    ''' reduce the input feature subset of  features
    option :
        'full' : all features
        'reduced' : reduced set of features (SST, Dsst, wind, heat flux, tidal amplitude, water depth)
        'point' : point data (SST, Dsst, T2m, press,wind, heat flux)
        'point_reduced' : point data (SST, Dsst, wind, heat flux)

    Parameters
    ----------
    input_data : torch tensor or nparray
        input data
    set : str (default 'full')

    Returns
    -------
    reduced_input_data : torch tensor or nparray
        reduced input data
    '''

    if set == 'full':
        data = input_data
    elif set == 'reduced':
        data = input_data[:,[0,1,4,5,6,8]]
    elif set == 'point':
        data = input_data[:,[0,1,2,3,4,5]]
    elif set == 'point_reduced':
        data = input_data[:,[0,1,4,5]]
    else:
        raise ValueError('Unknown input set')

    if type(data) == torch.Tensor:
        reduced_input_data = data.detach()
    elif type(data) == np.ndarray:
        reduced_input_data = data

    return reduced_input_data

#---------------model architecture -----------------
class NeuralNetwork(nn.Module):
    ''' Neural network class for pytorch

    Parameters
    ----------
    input_size : int
        size of the input
    hidden_size : int
        size of the hidden layers
    output_size : int
        size of the output
    n_layers : int (default 3)
        number of hidden layers
    drop_prop : float (default 0.0)
        dropout probability

    '''
    def __init__(self, input_size, hidden_size, output_size, n_layers=3, drop_prop=0.0):
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prop = drop_prop

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(n_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = nn.ReLU()(x)
            x = nn.Dropout(p=self.drop_prop)(x)

        x = self.output_layer(x)
        return x


#---------------loss function -----------------

class fft_loss(nn.Module):
    ''' Custom loss function for pytorch
    This loss function calculates the mean absolute error between the ifft of the predicted and target fft
    '''
    def __init__(self):
        super(fft_loss, self).__init__()

    def forward(self, predicted, target):
        pre_ifft = torch.real(torch.fft.ifft(predicted))
        tar_ifft = torch.real(torch.fft.ifft(target))
        errors = torch.abs(pre_ifft - tar_ifft)
        loss = torch.mean(errors)
        return loss


class WeightedLoss(nn.Module):
    ''' Custom loss function for pytorch
    This loss function calculates the mean absolute error between the predicted and target values with weights
    '''
    def __init__(self, weights):
        super(WeightedLoss, self).__init__()
        self.weights = weights

    def forward(self, predicted, target):
        errors = torch.abs(predicted - target)
        weighted_errors = errors * self.weights
        loss = torch.mean(weighted_errors)
        return loss

def get_loss_fn(loss_type):
    ''' Return loss function based on loss_type
    input:
    loss_type: str, type of loss function. Options: 'mse', 'smooth_l1', 'fft', 'weighted'

    output:
    loss_fn: loss function
    '''
    if loss_type == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type == 'smooth_l1':
        loss_fn = nn.SmoothL1Loss()
    elif loss_type == 'fft':
        loss_fn = fft_loss()
    elif loss_type == 'weighted':
        weights = torch.arange(10, 0,-1, dtype=torch.float32)
        loss_fn = WeightedLoss(weights)
    else:
        raise ValueError('Unknown loss type')

    return loss_fn

#---------------training -----------------

def train(dataloader, model, loss_fn, optimizer, print_loss=False):
    ''' Train the model

    Parameters
    ----------
    dataloader : torch DataLoader
        dataloader
    model : pytorch model
        model
    loss_fn : pytorch loss function
        loss function
    optimizer : pytorch optimizer
        optimizer
    print_loss : bool (default False)
    '''
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if print_loss:
            if batch % 100 == 0:
                print(f"loss: {loss.item():>7f}  [{(batch * len(X)):>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    ''' Test the model

    Parameters
    ----------
    dataloader : torch DataLoader
        dataloader
    model : pytorch model
        model
    loss_fn : pytorch loss function
        loss function

    Returns
    -------
    test_loss : float
        test loss
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg loss for test set: {test_loss:>8f} \n")

    return test_loss

def train_loop(train_loader, test_loader, model, loss_fn, optimizer, epochs, print_loss=False): 
    ''' Training loop

    Parameters
    ----------
    train_loader : torch DataLoader
        dataloader for training data
    test_loader : torch DataLoader
        dataloader for test data
    model : pytorch model
        model
    loss_fn : pytorch loss function
        loss function
    optimizer : pytorch optimizer
        optimizer
    epochs : int
        number of epochs
    print_loss : bool (default False)

    Returns
    -------
    test_loss : list
        test loss for each epoch
    '''
    test_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, print_loss)
        t_loss = test(test_loader, model, loss_fn)
        test_loss.append(t_loss)
    print("Done!")

    return test_loss


#____________________ save and load model ______________________
def save_model(model, number_hidden_layers, hidden_size, loss_type, epochs, drop_prop=0.0, odir='./models/'):
    ''' Save the model

    Parameters
    ----------
    model : pytorch model
        model
    number_hidden_layers : int
        number of hidden layers
    hidden_size : int
        size of the hidden layers
    loss_type : str
        type of loss function
    epochs : int
        number of epochs
    drop_prop : float (default 0.0)
        dropout probability
    odir : str (default './models/')
        directory where the model is saved
    Returns
    -------
    fname : str
    '''
    fname = odir + f'torch_HL{number_hidden_layers}x{hidden_size}d{int(str(drop_prop).split(".")[1][0])}_LF{loss_type}_epoch{epochs}.pth'
    torch.save(model.state_dict(), fname)

    return fname


def load_model(number_hidden_layers, hidden_size, loss_type, epochs, input_size=10, output_size=10, drop_prop=0.0, idir='./models/'):
    ''' Load the model

    Parameters
    ----------
    model : pytorch model
        model
    number_hidden_layers : int
        number of hidden layers
    hidden_size : int
        size of the hidden layers
    loss_type : str
        type of loss function
    epochs : int
        number of epochs
    input_size : int (default 10)
        size of the input
    output_size : int (default 10)
        size of the output
    drop_prop : float (default 0.0)
        dropout probability
    idir : str (default './models/')
        directory where the model is saved
    '''
    model = NeuralNetwork(input_size, hidden_size, output_size, number_hidden_layers, drop_prop=drop_prop) 
    model.load_state_dict(torch.load(idir + f'torch_HL{number_hidden_layers}x{hidden_size}d{int(str(drop_prop).split(".")[1][0])}_LF{loss_type}_epoch{int(epochs)}.pth'))
    model.eval()

    return model


def load_model_from_csv(model_csv: str, region: str, input_size: int) -> torch.nn.Module:
    ''' Load the model from a csv file with model parameters

    Parameters
    ----------
    model_csv : str
        path to the csv file
    region : str
        region of the model
    input_size : int
        size of the input

    Returns
    -------
    model : pytorch model
        model
    '''
    models_df: pd.DataFrame = pd.read_csv(model_csv)

    output_size: int = 10
    feature_set: str = models_df.loc[0, 'feature set']
    loss_type = models_df.loc[0, 'loss fn']
    epochs = models_df.loc[0, 'epochs']
    drop_prop = models_df.loc[0, 'drop_out']
    shape = models_df.loc[0, 'shape']
    number_hidden_layers = int(shape.split('x')[0])
    hidden_size = int(shape.split('x')[1])

    model = load_model(number_hidden_layers, hidden_size, loss_type, epochs=epochs, input_size=input_size, output_size=output_size, drop_prop=drop_prop, idir='./models/' + region + '/' + feature_set + '/')

    return model

#==============================================================================
# decompose a time series into monthly means and anomalies
#==============================================================================
def calculateSSTanomaly_monthly(sst):
    '''
    Calculate the monthly mean and the anomaly of a daily sst time series
    
    Parameters
    ----------
    sst : xarray dataset
        daily sst time series

    Returns
    -------
    sst_monthly : xarray dataset
        monthly mean
    sst_monthly_day : xarray dataset
        monthly mean interpolated to daily time steps
    sst_anom : xarray dataset
        anomaly
    '''
    # cal the monthly mean put the time in the middle of the month
    sst_monthly = sst.resample(time='1MS', loffset='15D' ).mean()

    # interpolate the monthly mean to the daily time steps
    sst_monthly_day = sst_monthly.interp(time=sst.time, method='linear')

    # calculate anomaly
    sst_anom = sst - sst_monthly_day

    return sst_monthly, sst_monthly_day, sst_anom



#==============================================================================
# calculate the fft of a time series
#==============================================================================

def fft_truncatated_normalized(x, n):
    '''
    Truncate the fft of a time series to n terms and normalize the result

    Parameters
    ----------
    x : np.array
        time series
    n : int
        number of terms to keep

    Returns
    -------
    fft_tr : np.array
        truncated and normalized fft

    '''
    N = len(x)
    fft_x = np.fft.fft(x, axis=0)
    fft_tr = fft_x[:n]
    fft_tr = fft_tr/N
    return fft_tr

def cal_monthly_truncated_fft(sst_monthly, sst_anom, n=10):
    '''
    Calculate the truncated fft for each month in sst_anom

    Parameters
    ----------
    sst_monthly : xarray dataset
        monthly mean (mpnthly data)
    sst_anom : xarray dataset
        anomaly (daily data)
    n : int (default 10)
        number of terms to truncate the fft

    Returns
    -------
    ds_fft : xarray dataset
        fft for each month
    '''

    if type(sst_anom) == xr.core.dataarray.Dataset:
        sst_anom = sst_anom.sst

    fft_results = []
    for t in sst_monthly.time:
        month = t.dt.month
        year = t.dt.year
        sst_bit = sst_anom.sel(time = (sst_anom['time.year'] == year) & (sst_anom['time.month'] == month) ).values
        fft_sst = fft_truncatated_normalized(sst_bit, n)

        fft_results.append(fft_sst)

    fft_results_array = np.array(fft_results)
    time_coords = sst_monthly.time.values
    fft_coords = np.arange(n)
    if len(fft_results_array.shape) == 4:
        ds_fft = xr.DataArray(fft_results_array, coords={'time': time_coords, 'freq': fft_coords, 'lat':sst_anom.lat.values, 'lon':sst_anom.lon.values}, dims=('time','freq', 'lat','lon' ))
    else:
        ds_fft = xr.DataArray(fft_results_array, coords={'time': time_coords, 'freq': fft_coords}, dims=('time', 'freq'))
    ds_fft = ds_fft.to_dataset(name='fft')
    return ds_fft


def calculate_variance_from_truncated_fft(ds_fft):
    """ Calculate the variance of the signal from the truncated fft

    Parameters
    ----------
    ds_fft : xarray dataset
        dataset containing the fft values

    Returns
    -------
    variance : numpy array
        variance of the signal for each month

    Comments
    --------
    The fft is symmetric and we only have the positive frequencies so we multiply by 2
    The first frequency is the mean and we remove it
    The fft is already normalized so we do not need to divide by N**2
    """
    variance = 2 * np.sum(np.abs(ds_fft.fft.values[:,1:]) ** 2, axis=1)

    return variance

#==============================================================================
# reconstruct a time series from the fft
#==============================================================================

def ifft_truncatated(fft_tr, N):
    '''
    Inverse fft of a truncated fft

    Parameters
    ----------
    fft_tr : np.array
        truncated fft 
    N : int 
        length of the time series

    Returns
    -------
    ifft_x : np.array
        inverse fft (time series)
    '''
    n = len(fft_tr)
    fft_x = np.zeros(N, dtype=complex)
    fft_x[:n] = fft_tr
    fft_x[1:] = fft_x[1:]*2 # double the amplitude of the positive frequencies to account for missing negative ones
    # re-normalize
    fft_x = fft_x*N
    ifft_x = np.fft.ifft(fft_x)
    ifft_x = np.real(ifft_x)
    return ifft_x

def reconstruct_from_fft(ds_fft, sst_monthly_day):
    '''
    Reconstruct the time series from the fft in a xarray dataset with monthly fft.

    Parameters
    ----------
    ds_fft : xarray dataset
        fft
    sst_monthly_day : xarray dataset
        monthly mean interpolated to daily time steps

    Returns
    -------
    sst_reconstruct : xarray dataset
        reconstructed time series

    '''
    sst_reconstruct = np.zeros(sst_monthly_day['sst'].values.shape)
    cnt=0
    for t in ds_fft.time:
        fft = ds_fft.sel(time=t).fft.values
        month = t.dt.month
        year = t.dt.year

        sst_bit = sst_monthly_day.sel(time = (sst_monthly_day['time.year'] == year) & (sst_monthly_day['time.month'] == month) ).sst.values.copy()
        n = len(sst_bit)
        ifft = ifft_truncatated(fft, n)
        sst_reconstruct[cnt:cnt+n] = sst_bit + ifft
        cnt+=n

    sst_reconstruct = xr.DataArray(sst_reconstruct, coords={'time': sst_monthly_day.time}, dims=('time'))
    sst_reconstruct = sst_reconstruct.to_dataset(name='sst')

    return sst_reconstruct




#==============================================================================
# randomize the phase of a fft
#==============================================================================

def randomize_fft_phase(fft):
    '''
    Randomize the phase of a fft

    Parameters
    ----------
    fft : np.array
        fft

    Returns
    -------
    fft_r : np.array
        fft with randomized phase

    '''
    n = len(fft)
    phase_r = np.random.rand(n)*2*np.pi
    fft_r = np.abs(fft
                   )*np.exp(1j*phase_r)
    # restore offset
    fft_r[0] = fft[0]

    return fft_r

def randomize_phase_of_dsfft(ds_fft):
    '''
    Randomize the phase of the fft in a xarray dataset with monthly fft. 
    Each month is randomized independently.

    Parameters
    ----------
    ds_fft : xarray dataset
        fft

    Returns
    -------
    ds_fft_r : xarray dataset
        fft with randomized phase

    '''

    ds_fft_r = ds_fft.copy()

    rand_phase = np.random.rand(*ds_fft.fft.shape) * 2 * np.pi
    ds_fft_r.fft.values = np.abs(ds_fft.fft.values) * np.exp(1j*rand_phase)

    return ds_fft_r


def create_random_phase_ensemble(sst_monthly_day, ds_fft, n=30):
    '''
    Calculate an ensemble of time series with randomized phase of the fft

    Parameters
    ----------
    sst_monthly_day : xarray dataset
        monthly mean interpolated to daily time steps
    ds_fft : xarray dataset
        fft
    n : int (default 30)
        number of time series in the ensemble

    Returns
    -------
    sst_random_ensemble : xarray dataset
        ensemble of time series with randomized phase of the fft

    '''
    sst_ensmble = np.zeros((n, len(sst_monthly_day.time)))
    for i in range(n):
        ds_fft_r = randomize_phase_of_dsfft(ds_fft)
        sst_rr = reconstruct_from_fft(ds_fft_r, sst_monthly_day)
        sst_ensmble[i, :] = sst_rr.sst.values

    # save in xarray dataset
    sst_random_ensemble = xr.Dataset()
    sst_random_ensemble['sst'] = (('enseble','time'), sst_ensmble)
    sst_random_ensemble['time'] = sst_rr.time
    sst_random_ensemble['ensemble'] = np.arange(n)
    sst_random_ensemble = sst_random_ensemble.assign_coords(time=sst_monthly_day.time.values)
    sst_random_ensemble = sst_random_ensemble.assign_coords(ensemble=np.arange(n))

    return sst_random_ensemble



#==============================================================================
# marine heat wave functions
#==============================================================================

def detect_mhw_ensemble(sst_random_ensemble):
    """
    detect marine heatwaves for an ensemble of time series

    Parameters
    ----------
    sst_random_ensemble : xarray dataset
        ensemble of time series

    Returns
    -------
    mhws_ensemble : list of dicts
        marine heat wave dictionaries
    clim_ensemble : list of dicts
        climatology dictionaries

    """
    mhws_ensemble = []
    clim_ensemble = []
    for i in sst_random_ensemble.ensemble.values:
        mhws_rr, clim_rr = mhw.detect(sst_random_ensemble.time.values, sst_random_ensemble.sst.values[i,:])
        mhws_ensemble.append(mhws_rr)
        clim_ensemble.append(clim_rr)

    return mhws_ensemble, clim_ensemble

def calculate_mhw_stats(mhw):
    """
    calculate the marine heat wave stats

    Parameters
    ----------
    mhw : dict
        marine heat wave dictionary

    Returns
    -------
    mhw_stats : dict
        marine heat wave stats
    """
    mhw_stats = {}
    mhw_stats['n_events'] = mhw['n_events']
    mhw_stats['mean_duration'] = np.array(mhw['duration']).mean()
    mhw_stats['max_duration'] = np.array(mhw['duration']).max()
    mhw_stats['tot_duration'] = np.array(mhw['duration']).sum()
    mhw_stats['mean_mean_intensity'] = np.array(mhw['intensity_mean']).mean()
    mhw_stats['max_mean_intensity'] = np.array(mhw['intensity_mean']).max()
    mhw_stats['mean_peak_intensity'] = np.array(mhw['intensity_max']).mean()
    mhw_stats['max_peak_intensity'] = np.array(mhw['intensity_max']).max()
    mhw_stats['mean_cum_intensity'] = np.array(mhw['intensity_cumulative']).mean()
    mhw_stats['max_cum_intensity'] = np.array(mhw['intensity_cumulative']).max()
    mhw_stats['tot_cum_intensity'] = np.array(mhw['intensity_cumulative']).sum()
    return mhw_stats

def calculate_mhw_stats_ensemble(mhw_list):
    """
    calculate the marine heat wave stats for an ensemble

    Parameters
    ----------
    mhw_list : list of dicts
        marine heat wave dictionaries

    Returns
    -------
    mhw_stats : dict
        marine heat wave stats

    """
    mhw_stats = {}
    mhw_stats['n_events'] = np.array([mhw['n_events'] for mhw in mhw_list]).mean()
    mhw_stats['n_events_std'] = np.array([mhw['n_events'] for mhw in mhw_list]).std()
    mhw_stats['mean_duration'] = np.array([np.array(mhw['duration']).mean() for mhw in mhw_list]).mean()
    mhw_stats['mean_duration_std'] = np.array([np.array(mhw['duration']).mean() for mhw in mhw_list]).std()
    mhw_stats['max_duration'] = np.array([np.array(mhw['duration']).max() for mhw in mhw_list]).mean()
    mhw_stats['max_duration_std'] = np.array([np.array(mhw['duration']).max() for mhw in mhw_list]).std()
    mhw_stats['tot_duration'] = np.array([np.array(mhw['duration']).sum() for mhw in mhw_list]).mean()
    mhw_stats['tot_duration_std'] = np.array([np.array(mhw['duration']).sum() for mhw in mhw_list]).std()
    mhw_stats['mean_intensity'] = np.array([np.array(mhw['intensity_mean']).mean() for mhw in mhw_list]).mean()
    mhw_stats['mean_intensity_std'] = np.array([np.array(mhw['intensity_mean']).mean() for mhw in mhw_list]).std()
    mhw_stats['mean_peak_intensity'] = np.array([np.array(mhw['intensity_max']).mean() for mhw in mhw_list]).mean()
    mhw_stats['mean_peak_intensity_std'] = np.array([np.array(mhw['intensity_max']).mean() for mhw in mhw_list]).std()
    mhw_stats['max_peak_intensity'] = np.array([np.array(mhw['intensity_max']).max() for mhw in mhw_list]).mean()
    mhw_stats['max_peak_intensity_std'] = np.array([np.array(mhw['intensity_max']).max() for mhw in mhw_list]).std()
    mhw_stats['mean_cum_intensity'] = np.array([np.array(mhw['intensity_cumulative']).mean() for mhw in mhw_list]).mean()
    mhw_stats['mean_cum_intensity_std'] = np.array([np.array(mhw['intensity_cumulative']).mean() for mhw in mhw_list]).std()
    mhw_stats['max_cum_intensity'] = np.array([np.array(mhw['intensity_cumulative']).max() for mhw in mhw_list]).mean()
    mhw_stats['max_cum_intensity_std'] = np.array([np.array(mhw['intensity_cumulative']).max() for mhw in mhw_list]).std()
    mhw_stats['tot_cum_intensity'] = np.array([np.array(mhw['intensity_cumulative']).sum() for mhw in mhw_list]).mean()
    mhw_stats['tot_cum_intensity_std'] = np.array([np.array(mhw['intensity_cumulative']).sum() for mhw in mhw_list]).std()
    return mhw_stats

def convert_mhw_list_to_pandas(mhw_list):
    """
    convert the mhw list to a pandas dataframe

    Parameters
    ----------
    mhw_list : list of dicts
        marine heat wave dictionaries

    Returns
    -------
    mhw_pd : pandas dataframe
        marine heat wave stats

    """
    mhw_pd = pd.DataFrame()
    mhw_pd['n_events'] = [mhw['n_events'] for mhw in mhw_list]
    mhw_pd['mean_duration'] = [np.array(mhw['duration']).mean() for mhw in mhw_list]
    mhw_pd['max_duration'] = [np.array(mhw['duration']).max() for mhw in mhw_list]
    mhw_pd['tot_duration'] = [np.array(mhw['duration']).sum() for mhw in mhw_list]
    mhw_pd['mean_intensity'] = [np.array(mhw['intensity_mean']).mean() for mhw in mhw_list]
    mhw_pd['mean_peak_intensity'] = [np.array(mhw['intensity_max']).mean() for mhw in mhw_list]
    mhw_pd['max_peak_intensity'] = [np.array(mhw['intensity_max']).max() for mhw in mhw_list]
    mhw_pd['mean_cum_intensity'] = [np.array(mhw['intensity_cumulative']).mean() for mhw in mhw_list]
    mhw_pd['max_cum_intensity'] = [np.array(mhw['intensity_cumulative']).max() for mhw in mhw_list]
    mhw_pd['tot_cum_intensity'] = [np.array(mhw['intensity_cumulative']).sum() for mhw in mhw_list]

    return mhw_pd

def print_mhw_stats(mhw, label):
    """
    print the marine heat wave stats

    Parameters
    ----------
    mhw : dict
        marine heat wave dictionary
    label : str
        label for the stats

    """
    mhw_stats = calculate_mhw_stats(mhw)
    print(label)
    print('Number of MHWs:', mhw_stats['n_events'])
    print('Mean duration: {:3.2f}'.format(mhw_stats['mean_duration']))
    print('Max duration:', mhw_stats['max_duration'])
    print('Total duration:', mhw_stats['tot_duration'])
    print('Mean intensity: {:3.2f}'.format(mhw_stats['mean_mean_intensity']))
    print('Mean peak intensity: {:3.2f}'.format(mhw_stats['mean_peak_intensity']))
    print('Max  peak intensity: {:3.2f}'.format(mhw_stats['max_peak_intensity']))
    print('Mean cumulative intensity: {:3.2f}'.format(mhw_stats['mean_cum_intensity']))
    print('Max cumulative intensity: {:3.2f}'.format(mhw_stats['max_cum_intensity']))
    print('Total cumulative intensity: {:3.2f}'.format(mhw_stats['tot_cum_intensity']))
    print(' ')


def print_mhw_stats_ensemble(mhw_list, label):
    """
    print the marine heat wave stats for an ensemble

    Parameters
    ----------
    mhw_list : list of dicts
        marine heat wave dictionaries
    label : str
        label for the stats

    """

    mhw_stats = calculate_mhw_stats_ensemble(mhw_list)
    print(label)
    print('Number of MHWs: {:3.2f} +/- {:3.2f}'.format(mhw_stats['n_events'], mhw_stats['n_events_std']))
    print('Mean duration: {:3.2f} +/- {:3.2f}'.format(mhw_stats['mean_duration'], mhw_stats['mean_duration_std']))
    print('Max duration: {:3.2f} +/- {:3.2f}'.format(mhw_stats['max_duration'], mhw_stats['max_duration_std']))
    print('Total duration: {:3.2f} +/- {:3.2f}'.format(mhw_stats['tot_duration'], mhw_stats['tot_duration_std']))
    print('Mean intensity: {:3.2f} +/- {:3.2f}'.format(mhw_stats['mean_intensity'], mhw_stats['mean_intensity_std']))
    print('Mean peak intensity: {:3.2f} +/- {:3.2f}'.format(mhw_stats['mean_peak_intensity'], mhw_stats['mean_peak_intensity_std']))
    print('Max  peak intensity: {:3.2f} +/- {:3.2f}'.format(mhw_stats['max_peak_intensity'], mhw_stats['max_peak_intensity_std']))
    print('Mean cumulative intensity: {:3.2f} +/- {:3.2f}'.format(mhw_stats['mean_cum_intensity'], mhw_stats['mean_cum_intensity_std']))
    print('Max cumulative intensity: {:3.2f} +/- {:3.2f}'.format(mhw_stats['max_cum_intensity'], mhw_stats['max_cum_intensity_std']))
    print('Total cumulative intensity: {:3.2f} +/- {:3.2f}'.format(mhw_stats['tot_cum_intensity'], mhw_stats['tot_cum_intensity_std']))
    print(' ')



def compare_mhw_stats(mhw_sets):
    """
    compare the marine heat wave stats

    Parameters
    ----------
    mhw_sets : list of dicts
        marine heat wave dictionaries

    """
    for i, mhw_set in enumerate(mhw_sets):
        print_mhw_stats(mhw_set['mhws'], mhw_set['name'])


def cal_histograms(mhw, Nbins=50):
    """
    calculate histograms of duration, intensity_mean, intensity_max, intensity_cumulative

    Parameters
    ----------
    mhw : dict
        marine heat wave dictionary
    Nbins : int (default 50)
        number of bins

    Returns
    -------
    histograms : dict
        histograms of duration, intensity_mean, intensity_max, intensity_cumulative

    """
    keys = ['duration', 'intensity_mean', 'intensity_max', 'intensity_cumulative']
    bin_ranges = [(0, 200), (0, 5), (0, 10), (0, 400)]
    histograms = {}
    for i, key in enumerate(keys):
        hist, bins = np.histogram(mhw[key], bins=Nbins, range=bin_ranges[i])
        bin_centers = (bins[:-1] + bins[1:]) / 2
        histograms[key] = [hist, bin_centers]
    return histograms

def cal_histograms_ensemble(mhws_ensemble, Nbins=50):
    """
    calculate histograms of duration, intensity_mean, intensity_max, intensity_cumulative for an ensemble and merges the results binwise

    Parameters
    ----------
    mhws_ensemble : list of dicts
        marine heat wave dictionaries
    Nbins : int (default 50)
        number of bins

    Returns
    -------
    histograms_ens : dict
        histograms of duration, intensity_mean, intensity_max, intensity_cumulative for the ensemble

    """
    for i, mhw in enumerate(mhws_ensemble):
        histograms = cal_histograms(mhw, Nbins=Nbins)
        if i == 0:
            histograms_ens = histograms
        else:
            for key in histograms.keys():
                histograms_ens[key][0] += histograms[key][0]

    return histograms_ens

def cal_normalized_histograms(mhw, Nbins=50):
    """
    calculate normalized histograms of duration, intensity_mean, intensity_max, intensity_cumulative

    Parameters
    ----------
    mhw : dict
        marine heat wave dictionary
    Nbins : int (default 50)
        number of bins

    Returns
    -------
    histograms : dict
        normalized histograms of duration, intensity_mean, intensity_max, intensity_cumulative

    """
    histograms = cal_histograms(mhw, Nbins=Nbins)
    for key in histograms.keys():
        histograms[key][0] = histograms[key][0] / np.sum(histograms[key][0])
    return histograms

def cal_normalized_histograms_ensemble(mhws_ensemble, Nbins=50):
    """
    calculate normalized histograms of duration, intensity_mean, intensity_max, intensity_cumulative for an ensemble and merges the results binwise

    Parameters
    ----------
    mhws_ensemble : list of dicts
        marine heat wave dictionaries
    Nbins : int (default 50)
        number of bins

    Returns
    -------
    histograms_ens : dict
        normalized histograms of duration, intensity_mean, intensity_max, intensity_cumulative for the ensemble

    """
    histograms_ens = cal_histograms_ensemble(mhws_ensemble, Nbins=Nbins)
    for key in histograms_ens.keys():
        histograms_ens[key][0] = histograms_ens[key][0] / np.sum(histograms_ens[key][0])
    return histograms_ens


def cal_mhw_set(sst, period=[None,None], externalClimatology=False, Nbins=30):
    """
    detects marine heat waves and the meta statistics

    Parameters
    ----------
    sst : xarray dataset
        sst time series
    Period : list (default [None,None])
        climatology period
    externalClimatology : bool (default False)
        use external climatology
    Nbins : int (default 30)
        number of bins for the histograms

    Returns
    -------
    mhws : dict
        marine heat wave dictionary
    clim : dict
        climatology dictionary
    stats : dict
        marine heat wave stats
    hist : dict
        normalized histograms of duration, intensity_mean, intensity_max, intensity_cumulative
    """

    mhws, clim = mhw.detect(sst.time.values, sst.sst.values, climatologyPeriod=period.copy(), 
                            externalClimatology=externalClimatology)
    stats = calculate_mhw_stats(mhws)
    hist = cal_normalized_histograms(mhws, Nbins)
    return mhws, clim, stats, hist

def generate_mhw_set(sst, name, color):
    """
    generate a marine heat wave set

    Parameters
    ----------
    sst : xarray dataset
        sst time series
    name : str
        name of the set
    color : str
        color for plotting

    Returns
    -------
    mhw_set : dict
        marine heat wave set

    """
    mhws, clim, stats, hist = cal_mhw_set(sst)
    mhw_set = {'sst': sst, 'name': name, 'color': color, 'mhws': mhws, 'clim': clim, 'stats': stats, 'hist': hist}
    return mhw_set

def cal_mhw_set_ensemble(sst_random_ensemble, climatologyPeriod=[None,None], externalClimatology=False, Nbins=30):
    """
    detects marine heat waves and the meta statistics for an ensemble

    Parameters
    ----------
    sst_random_ensemble : xarray dataset
        ensemble of time series
    climatologyPeriod : list (default [None,None])
        climatology period
    externalClimatology : bool (default False)
        use external climatology
    Nbins : int (default 30)
        number of bins for the histograms

    Returns
    -------
    mhws_ensemble : list of dicts
        marine heat wave dictionaries
    clim_ensemble : list of dicts
        climatology dictionaries
    stats_ensemble : dict
        marine heat wave stats
    hist_ensemble : dict
        normalized histograms of duration, intensity_mean, intensity_max, intensity_cumulative
    """
    mhws_ensemble, clim_ensemble = detect_mhw_ensemble(sst_random_ensemble)
    stats_ensemble = calculate_mhw_stats_ensemble(mhws_ensemble)
    hist_ensemble = cal_normalized_histograms_ensemble(mhws_ensemble, Nbins)
    return mhws_ensemble, clim_ensemble, stats_ensemble, hist_ensemble


#==============================================================================
# Bias correction
#==============================================================================

def yearly_harmonic(t, a, phi):
    P = 365.25
    return a*np.cos(2*np.pi/P*t + phi)

def decompose_time_series(sst):
    """
    Decompose the time series into trend, seasonal and residual

    Parameters
    ----------
    sst : np.array
        time series

    Returns
    -------
    sst_trend : np.array
        trend
    sst_seasonal : np.array
        seasonal
    sst_residual : np.array
        residual

    """
    t = np.arange(0, len(sst))
    slope, intercept = np.polyfit(t, sst, 1)
    sst_trend = slope*t + intercept
    sst_untrended = sst - sst_trend
    popt, pcov = curve_fit(yearly_harmonic, t, sst_untrended, p0=[1, 0])

    sst_seasonal = yearly_harmonic(t, *popt)
    sst_residual = sst_untrended - sst_seasonal

    return sst_trend, sst_seasonal, sst_residual


def cal_bias_correction(sst_ref, sst_biased):
    """
    Bias correction of the sst_biased time series

    Parameters
    ----------
    sst_ref : np.array
        reference time series
    sst_biased : np.array
        biased time series

    both inputs are assumed to be daily time series covering the same time period

    Returns
    -------
    sst_corrected : np.array
        bias corrected time series

    """
    sst_trend, sst_seasonal, sst_residual = decompose_time_series(sst_ref)
    sst_biased_trend, sst_biased_seasonal, sst_biased_residual = decompose_time_series(sst_biased)

    # bias correction ratio
    r_bias = np.std(sst_residual)/np.std(sst_biased_residual)

    sst_corrected = sst_biased_trend + sst_biased_seasonal + r_bias*sst_biased_residual

    return sst_corrected, r_bias

def perform_bias_correction(sst, r_bias):
    """
    Bias correction of the sst time series

    Parameters
    ----------
    sst : np.array
        time series
    r_bias : float
        bias correction ratio

    Returns
    -------
    sst_corrected : np.array
        bias corrected time series

    """
    sst_trend, sst_seasonal, sst_residual = decompose_time_series(sst)
    sst_corrected = sst_trend + sst_seasonal + r_bias*sst_residual

    return sst_corrected

#==============================================================================
# marineheatwave.py helpers
#==============================================================================

def convert_orddates2datetime(ordinal_dates):
        """
        Converts a list of ordinal dates to a list of datetime objects 

        Parameters
        ----------
        ordinal_dates : list
            list of ordinal dates

        Returns
        -------
        dates : list
            list of datetime objects

        """
        return [datetime.datetime.fromordinal(t) for t in ordinal_dates]

def convert_npdatetime2ordinal(time):
        """
        Converts a numpy date64 array  to a list of ordinal dates

        Parameters
        ----------
        time : np.array
            numpy date64 array

        Returns
        -------
        ordinal_dates : list
            list of ordinal dates

        """
        dates = pd.to_datetime(time)
        return np.array([t.toordinal() for t in dates])


def convert_cftime_to_npdatetime64(time):
    """
    Converts a cftime date to a numpy date64 array

    Parameters
    ----------
    time : cftime date
        cftime date
    
    Returns
    -------
    time_new : np.array
        numpy date64 array

    """
    t_lst = [t.toordinal() for t in time]
    t_lst = convert_orddates2datetime(t_lst)
    time_new = np.array([np.datetime64(t) for t in t_lst])

    return time_new
