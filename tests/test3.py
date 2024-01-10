# main
import sys
import pandas as pd
sys.path.append("..")

# # runtime optimization
# import cProfile

# missing data handling
from sklearn.preprocessing import StandardScaler, RobustScaler

# custom functions
from imputation.panel_imputation import EmptySeriesImputer

panel_incomplete_dataset = pd.read_csv('../data/data_test2.csv', index_col=['country', 'date'])
panel_incomplete_dataset.columns.name = 'variable'

# Defining param search space
search_params = {
    'fill_method': ['variable', 'country'],
    'scaler': [StandardScaler(), RobustScaler(quantile_range=(5.0, 95.0))],
    'recursive_imputation': [False, True],
    'aggregate_by':['mean', 'pca'],
    'alternative_priority_bias': [False]
}
empty_imputer = EmptySeriesImputer('variable')
calibration_results = empty_imputer.run_calibration(panel_incomplete_dataset, search_params, partial_replica=True, recursive_replication=True)
