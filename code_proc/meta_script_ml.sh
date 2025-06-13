#!/bin/bash


#regions='NorthSeaPoint NorthSea NWEuroShelf'

# create training data
# python ./create_training_data.py

regions="NWEuroShelf"
csv_files='best_model.csv'
csv_files='models_selection.csv'

for region in $regions; do
	echo "====================================="
	echo "Training models for region: $region"
	for csv_file in $csv_files; do
		echo $csv_file
		# train models
		python ./pytorch_trainAllModels.py $region $csv_file
		# python ./cal_variance_correlation_4allmodels.py $region $csv_file
		# python ../code_plot/plot_compare_model_variance.py $region $csv_file
	done

	# train sklearn models
	# echo "Training sklearn models" 
	# python ./sklearn_training.py $region 
	# python ./cal_variance_correlation_4allmodels.py $region models_sklearn.csv
	# python ../code_plot/plot_compare_model_variance.py $region models_sklearn.csv
done





