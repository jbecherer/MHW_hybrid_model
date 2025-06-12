import numpy as np
import pandas as pd

#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import joblib
import os, sys
from importlib import reload
import warnings
warnings.filterwarnings('ignore')

import aisst


#---------------helper function to show performance-----------------
def print_results(results):
    """Print GridSearchCV results

    Parameters
    ----------
    results : GridSearchCV object
        The GridSearchCV object to be printed

    Returns
    -------
    None

    """
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

def train_linear_regression(region, set='reduced'):
    """Train a linear regression model

    Parameters
    ----------
    region : str
        The region for which the model is to be trained
    set : str
        The feature set to be used for training the model. Default is 'reduced'
        other oprion is 'full' which uses all 10 features

    Returns
    -------
        
    The model is saved to a file:
        '../models/' + region + '/Linear_Regression_' + set + '.pkl'
    """

    #==============================================================================
    # load data
    #==============================================================================
    input_train = pd.read_csv('../data/ml_training/' + region + '/ml_input_data_train.csv')
    output_train = pd.read_csv('../data/ml_training/' + region + '/ml_output_data_train.csv')

    input_train = input_train.values
    input_train = aisst.reduce_feature_set(input_train, set)
    output_train = output_train.values

    #==============================================================================
    # Linear regression
    #==============================================================================
    lr = LinearRegression().fit(input_train, output_train)


    joblib.dump(lr, '../models/' + region + '/Linear_Regression_' + set + '.pkl')


def train_random_forest(region, set='reduced'):
    """Train a random forest model

    Parameters
    ----------
    region : str
        The region for which the model is to be trained
    set : str
        The feature set to be used for training the model. Default is 'reduced'
        other oprion is 'full' which uses all 10 features

    Returns
    -------

    The model is saved to a file:
        '../models/' + region + '/RF_model_' + set + '.pkl'
    """

    input_train = pd.read_csv('../data/ml_training/' + region + '/ml_input_data_train.csv')
    output_train = pd.read_csv('../data/ml_training/' + region + '/ml_output_data_train.csv')

    input_train = input_train.values
    input_train = aisst.reduce_feature_set(input_train, set)
    output_train = output_train.values

    #==============================================================================
    # Random forest
    #==============================================================================
    # Conduct search for best params while running cross-validation (GridSearchCV)
    rf = RandomForestRegressor()
    parameters = {
        'n_estimators': [100],
        #'n_estimators': [10, 50, 100, 200, 500],
        'max_depth': [2,4,8, None],
        # 'max_depth': [None],
        #'oob_score': [True] ,
        'max_features': [10],
        'verbose':[True]
    }

    if set == 'reduced':
        parameters['max_features'] = [6]

    cv_rf = GridSearchCV( rf, parameters, cv=5 )
    cv_rf.fit( input_train, output_train )

    print_results(cv_rf)

    joblib.dump(cv_rf.best_estimator_, '../models/' + region + '/RF_model_' + set + '.pkl')


if __name__ == '__main__':
    region = sys.argv[1]
    train_linear_regression(region)
    train_random_forest(region)
    print('Training completed!')

