# main
import sys
import numpy as np
import pandas as pd
sys.path.append("..")

# Models
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge

# missing data handling
from sklearn.preprocessing import StandardScaler, RobustScaler

# custom functions
from panel_imputation.panel_imputation import EmptySeriesImputer

# simulation of missing data
data = {
    'country': ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C"],
    'date': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01',
             '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01',
             '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
    'Variable1': [1, 2, 1.5, 3, 4, 100, 150, 130, 250, 400, 50, 110, 100, 150, 200],
    'Variable2': [30, 25, 23, 16, 9, 10, 7, 6.5, 4, 3, 100, 90, 80, 68, 56],
    'Variable3': [20, 10, 9.5, 7.5, 7, 12, 8, 7, 5, 4, 15, 13, 11, 8.5, 6],
    'Variable4': [0.5, 0.6, 0.4, 0.75, 1.25, 50, 40, 42, 35, 20, 0.5, 0.6, 0.5, 0.9, 1.5],
}
panel_df = pd.DataFrame(data).set_index(['country', 'date'])
panel_df.columns.name = 'variable'
# panel_df.loc['A', 'Variable1'] = np.NaN
panel_df.loc['A', 'Variable2'] = np.NaN
panel_df.loc['A', 'Variable3'] = np.NaN

# Defining param search space
search_params = {
    'fill_method': ['country', 'variable'],
    'scaler': [StandardScaler(), RobustScaler(quantile_range=(5.0, 95.0)),
               RobustScaler(quantile_range=(25.0, 75.0))],
    'model': [LinearRegression(), BayesianRidge(), Ridge()],
    'recursive_imputation': [False, True],
    'aggregate_by':['mean', 'pca'],
    'alternative_priority_bias': [False, True]
}
empty_imputer = EmptySeriesImputer('variable')
calibration_results = empty_imputer.run_calibration(panel_df, search_params, partial_replica=True, recursive_replication=True)
