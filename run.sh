#!/bin/bash
# this script is just a wrapper for python scripts that use to much compute to be run in my normal interactive ipython session

cd ./code_proc/
# bash meta_script_ml.sh

#python mpi_dailyNWES.py
# python cal_mhw_maps.py
# python plot_rf_feature_importance.py

#python cal_variance_correlation_4allmodels.py NWEuroShelf model_selection_table.csv
#
# python plot_compare_model_variance.py NWEuroShelf models_2plot.csv
#
# python generate_mpi_input_and_norm.py era _ _
# python generate_mpi_input_and_norm.py 1850 1982 histssp585
# python generate_mpi_input_and_norm.py 2015 2100 histssp585
# python generate_mpi_input_and_norm.py 2015 2100 ssp126
# python generate_mpi_input_and_norm.py 2015 2100 ssp245
# python generate_mpi_input_and_norm.py 2015 2100 ssp370
#
# python hybrid_model_applyNWES.py
# bash merge_hybridmodel_years.sh
#
python -u cal_mean_sst.py 

# python -u cal_mhw_maps.py

# la=3
# python -u cal_mhw_maps_allScenarios.py $la
# python -u cal_mhw_returnperiods.py $la
#
# mkdir -p logs
# for la in {0..12}; do
#     python -u cal_mhw_maps_allScenarios.py $la > ./logs/cal_mhw_maps_allScenarios_${la}.log 2>&1 &
#     python -u cal_mhw_returnperiods.py $la > ./logs/cal_mhw_returnperiods_${la}.log 2>&1 &
# done
# wait
