# main
import sys
import pandas as pd
sys.path.append("..")

# runtime optimization
import cProfile

# custom functions
from imputation.panel_imputation import EmptySeriesImputer

panel_incomplete_dataset = pd.read_csv('data/data_test2.csv')
empty_imputer = EmptySeriesImputer('variable')
profiler = cProfile.Profile()
profiler.enable()
panel_dataset_imputed = empty_imputer.impute(panel_incomplete_dataset)
profiler.disable()
profiler.dump_stats('profile_results.prof')
