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
`./code_proc/aisst.py`
`./code_proc/marineHeatWaves.py` slightly modified version git@github.com:ecjoliver/marineHeatWaves.git


---
# Data

- ERA5 variable collection : `./data/NWEuroShelf_era5_1982_2023_allVars_1deg_daily.nc`
- Bathymetry data on 1 deg grid `./data/bathymetry_1deg.nc`
- tidal amplitude on 1deg grid : `./data/tidal_current_amplitude_1deg.nc`

--- 
# ML module

## Generate the training data for the machine learning module
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

