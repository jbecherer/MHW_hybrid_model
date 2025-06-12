#!/bin/bash
#SBATCH --job-name=python_job   # Specify job name
#SBATCH --partition=compute # Specify partition name interactive, shared, compute
##SBATCH --mem=16G               # Specify amount of memory needed
#SBATCH --time=08:00:00         # Set a limit on the total run time
#SBATCH --output=log/log.python_job.o%j
#SBATCH --mail-type=FAIL
#SBATCH --account=uo0119

# this script is just a wrapper for python scripts that use to much compute to be run in my normal interactive ipython session

cd ./code_proc/
python create_training_data.py

#python mpi_dailyNWES.py
# python cal_mhw_maps.py
# python plot_rf_feature_importance.py

#python cal_variance_correlation_4allmodels.py NWEuroShelf model_selection_table.csv
#
# python plot_compare_model_variance.py NWEuroShelf models_diverse_new.csv
# python plot_compare_model_variance.py NWEuroShelf models_diverse_100.csv
# python plot_compare_model_variance.py NWEuroShelf models_epochs_new.csv
# python plot_compare_model_variance.py NWEuroShelf models_2plot.csv
#
# python generate_mpi_input_and_norm.py 
# python generate_mpi_input_and_norm.py 1850 1982 histssp585
# python generate_mpi_input_and_norm.py 1850 2100 histssp585
# python generate_mpi_input_and_norm.py 2010 2014 histssp585
# python generate_mpi_input_and_norm.py 2015 2100 ssp126
# python generate_mpi_input_and_norm.py 2015 2100 ssp245
# python generate_mpi_input_and_norm.py 2015 2100 ssp370
#
# python hybrid_model_applyNWES.py
#
# python -u cal_mean_sst.py 

# la=3
# python -u cal_mhw_maps_allScenarios.py $la
# python -u cal_mhw_returnperiods.py $la
