# main
import sys
import pandas as pd
sys.path.append("..")

# runtime optimization
import cProfile

# custom functions
from imputation.panel_imputation import EmptySeriesImputer

panel_incomplete_dataset = pd.read_csv('../data/data_test2.csv', index_col=['country', 'date'])
panel_incomplete_dataset.columns.name = 'variable'
empty_imputer = EmptySeriesImputer('variable')
panel_dataset_imputed = empty_imputer.impute(panel_incomplete_dataset)
profiler = cProfile.Profile()
profiler.enable()
loss = empty_imputer.replica_loss(panel_dataset_imputed)
profiler.disable()
profiler.dump_stats('profile_results.prof')
