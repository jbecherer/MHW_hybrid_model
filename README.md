# MHW_hybrid_model

This repository supports the publication of the paper "Improving Marine Heatwave Statistics in Global Climate Models Using Machine Learning: A Case Study for the North-West European Shelf" by J. Becherer and T. Pohlmann, currently under review in *Climate Dynamics*.

It will contains the code necessary to generate all figures and tables featured in the paper, as well as the code for training the machine learning component of the hybrid model. Additionally, a selection of trained machine learning model weights is included.

---
## Setup

### Download the data

### dependencies
- scikit-learn
- xarray
- numpy
- pandas
- matplotlib
- cartopy
- torch

---
## main library
- `./code_proc/aisst.py`
-  `./code_proc/marineHeatWaves.py` slightly modified version git@github.com:ecjoliver/marineHeatWaves.git


---
# Data

use `./data/setup.sh` to download the data and set up the directory structure

- ERA5 variable collection : `./data/NWEuroShelf_era5_1982_2023_allVars_1deg_daily.nc`
- Bathymetry data on 1 deg grid `./data/bathymetry_1deg.nc`
- tidal amplitude on 1deg grid : `./data/tidal_current_amplitude_1deg.nc`

--- 
# ML module

## Generate the training data for the machine learning module
Based on the ERA5 data, bathymetry, and tidal current amplitude, the training data for the machine learning module is generated. The training data is split into training, validation, and test sets, where four random years are used for testing, and further four random years for validation, and the remaining years for training. The training data is saved in the `./data/ml_training/<region>/` folder.
Next to training there are also normalization parameters saved, which are calculated based on the entire time span of the training data. These parameters are used to normalize the input data before training and testing the machine learning models.


`./code_proc/create_training_data.py`
>   This function generates the training data for the machine learning module based on ERA5 data.
>
>   Input: 
>     - `./data/NWEuroShelf_era5_1982_2023_allVars_1deg_daily.nc`
>     - `./data/bathymetry_1deg.nc`
>     - `./data/tidal_current_amplitude_1deg.nc`
>    OUTPUT:
>    - Input data: ../data/ml_training/<region>/ml_input_data.csv
>    - Output data: ../data/ml_training/<region>/ml_output_data.csv
>    - Training input data: ../data/ml_training/<region>/ml_input_data_train.csv
>    - Training output data: ../data/ml_training/<region>/ml_output_data_train.csv
>    - Validation input data: ../data/ml_training/<region>/ml_input_data_val.csv
>    - Validation output data: ../data/ml_training/<region>/ml_output_data_val.csv
>    - Test input data: ../data/ml_training/<region>/ml_input_data_test.csv
>    - Test output data: ../data/ml_training/<region>/ml_output_data_test.csv
>    - Validation and test input data: ../data/ml_training/<region>/ml_input_data_valtest.csv
>    - Validation and test output data: ../data/ml_training/<region>/ml_output_data_valtest.csv

## Train the machine learning module
### general
in the folder `./models/NWEuroShelf/` you can find a csv files that contains the model specifications for the training.  In the meta script `./code_proc/meta_script_ml.sh` you can specify the region and which  csv file to use for training. Inside the csv file you can list an arbitrary number of models to train. The script will then train all models listed in the csv file.

For the paper we used the csv file `./models/NWEuroShelf/models_selction.csv` which contains the specifications for all neural network models used in the paper. 
For linear regression and random forest models we used the csv file `./models/NWEuroShelf/models_sklearn.csv`.

After selecting the preferred model in is written to the `./models/NWEuroShelf/best_model.csv` file. 
The file is later used for the hybrid model.

### Run the training scripts
`./code_proc/meta_script_ml.sh`
This the script that runs all the different training scripts.

### Train NN models
`./code_proc/pytorch_trainAllModels.py`
This scripts trains all NN models specified in a **csv file** that lies in the `./models/NWEuroShelf/`


### Train Linear Regression and Random Forrest models
`./code_proc/sklearn_training.py`
This scripts trains a linear regression and random forest model for a specified region.

### Evaluate the trained models
`./code_proc/cal_variance_correlation_4allmodels.py`
calculates all relevant statistics and writes them to a csv file.


## The Hybrid model
### Data
The data directory for MPI input data is  `./data/mpi/mon/` There are four base files that contain monthly averaged intput data from the MPI-ESM1.2 model. The files are: 
- `aisst_histssp585.nc` 
- `aisst_histssp126.nc`
- `aisst_ssp245.nc` 
-`aisst_ssp370.nc`
These cover the entire European Shelf including the Mediterranean sea. 


`generate_mpi_input_and_norm.py` :
In a further processsing step, we extract only data from the North-West European Shelf, calculate the SST monthly slope, and extract differenrt time periods 1982-2023, 1850-1882, and 2015-2100 for each scenario respectively. 
This generate the files:
- `./data/mpi/mon/aisst_histssp585_1982_2023_allvars.nc`
- `./data/mpi/mon/aisst_histssp585_1850_1982_allvars.nc`
- `./data/mpi/mon/aisst_ssp126_2015_2100_allvars.nc`
- `./data/mpi/mon/aisst_ssp246_2015_2100_allvars.nc`
- `./data/mpi/mon/aisst_ssp370_2015_2100_allvars.nc`
The script also generates the normalization parameters for the input data based on the Reference period (1982_2023), which are saved in `../models/NWEuroShelf/ml_norm_mpi.csv`. These parameters are used to normalize the input data for all time periods.

To genrate all these files, run the `run.sh` which calls the `generate_mpi_input_and_norm.py` script multiple times.


