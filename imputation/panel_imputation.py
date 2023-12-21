# main
import itertools
import numpy as np
import pandas as pd

# models
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# missing data handling
from sklearn.preprocessing import RobustScaler

# utils
import warnings
from tqdm.notebook import tqdm

# runtime optimization
from numba import jit

class EmptySeriesImputer():
    """
    A class for imputing missing values in a DataFrame using a combination of scaling, linear regression models, and
    a specified filling priority method.

    Parameters:
    - fill_method (str): The name of the index column to be used as the fill priority.
    - scaler (object, optional): An optional scaler object from scikit-learn for standardizing data. Default is RobustScaler.
    - model (object, optional): An optional scikit-learn model for regression. Default is LinearRegression(fit_intercept=False).
    - recursive_imputation (bool, optional): Perform recursive imputation if True. Default is True.
    - aggregate_by (str, optional): Method for aggregating predictions ('pca' or 'mean'). Default is 'pca'.
    - alternative_priority_bias (bool, optional): A bias parameter to change the order in which different group column pairs are filled. Default is False.
    - partial_replica (bool, optional): Allow partial replication during replica matrix creation. Default is False.
    - recursive_replication (bool, optional): Perform recursive replication during replica matrix creation. Default is False.

    Note:
    - The DataFrame used in the impute method must have exactly 2 different index columns, usually representing 'country' and 'date'.
    - The 'fill_method' should be the name of one of these indices (e.g., 'country').
    """
    def __init__(self, fill_method, scaler=None, model=None, recursive_imputation=True,
                 aggregate_by='pca', alternative_priority_bias=False, partial_replica=False,
                 recursive_replication=False):
        self.fill_method = fill_method
        self.scaler = RobustScaler(quantile_range=(5.0, 95.0)) if type(scaler)==type(None) else scaler
        self.model = LinearRegression(fit_intercept=False) if type(model)==type(None) else model
        self.alternative_priority_bias = alternative_priority_bias
        self.recursive_imputation = recursive_imputation
        self.aggregate_by = aggregate_by
        self.partial_replica = partial_replica
        self.recursive_replication = recursive_replication
        self.missing_priority = []

    def standardize_series(self, df):
        def ss(x):
            if x.isna().any():
                return np.array([np.NaN]*df.shape[0])
            else:
                x_scaled = self.scaler.fit_transform(x.values.reshape(-1, 1)).reshape(-1)
                return x_scaled
        return df.apply(ss)

    def fit_model(self, X, y):
        res = self.model.fit(X, y)
        return res.coef_

    def calc_imputation_priority(self, df):
        missing_map = df.groupby(self.fill_method, group_keys=True).apply(lambda x: x.isna().any())
        a_counts = missing_map.sum(axis=0)[missing_map.sum(axis=0)!=0]
        m_counts = missing_map.sum(axis=1)[missing_map.sum(axis=1)!=0]
        missing_priority_scores = pd.concat([a_counts, m_counts])
        if len(missing_priority_scores)==0:
            return []
        bias_key = a_counts[a_counts!=0].idxmin() if self.alternative_priority_bias else m_counts[m_counts!=0].idxmin()
        missing_priority_scores[bias_key] +=-1
        missing_priority = missing_map.stack()[missing_map.stack()].index.to_frame()
        missing_priority = missing_priority.replace(missing_priority_scores.to_dict())
        missing_priority = missing_priority.sum(axis=1).sort_values().index
        return missing_priority

    def setup_df(self, df):
        if self.fill_method not in df.index.names:
            Flipped = True
            df = df.T.stack(level=-1)
        else:
            Flipped = False
            df = df.copy()
        return df, Flipped

    def package_df(self, df, Flipped):
        if Flipped:
            df = df.T.stack(level=-1)
        else:
            df = df.copy()
        return df

    def aggregate_predictions(self, pred):
        agg_pred = None
        if self.aggregate_by == 'pca':
            pca = PCA(1)
            agg_pred = pca.fit_transform(pred).reshape(-1)
            flip_sign = np.sum(np.corrcoef(np.c_[pred, agg_pred].T)[-1, :-1]) < 0
            if flip_sign:
                agg_pred = -agg_pred
        elif self.aggregate_by == 'mean':
            agg_pred = pred.mean(axis=1)
        return agg_pred

    def aggregate_estimation(self, df_groups, group_names, group, columns, variable):
        # setup exog
        X = df_groups[np.where(group_names == group)[0][0]]
        finite_cols = columns[np.isfinite(X).all(axis=0)]
        finite_col_indices = np.where(np.isin(columns, finite_cols))[0]
        X = X[:, finite_col_indices]

        # estimate coefficients from all groups
        fitted_coefs = []
        for group_name, group_item in zip(group_names, df_groups):
            if (group_name!=group) and (group_item.size!=0):
                X_group = group_item[:, finite_col_indices]
                y_group = group_item[:, np.where(columns == variable)[0][0]]
                finite_cols_group = finite_cols[np.isfinite(X_group).all(axis=0)]
                finite_col_indices_group = np.where(np.isin(finite_cols, finite_cols_group))[0]
                if (finite_col_indices_group.size != 0) and not np.isnan(y_group.sum()):
                    X_group = X_group[:, finite_col_indices_group]
                    some_group_coefs = self.fit_model(X_group, y_group)
                    group_coefs = np.zeros_like(finite_cols, dtype=float)
                    available_coefs = np.where(np.isin(finite_cols, finite_cols_group))[0]
                    group_coefs[available_coefs] = some_group_coefs
                    fitted_coefs.append(group_coefs)
        fitted_coefs = np.array(fitted_coefs)
        fitted_preds = X @ fitted_coefs.T
        pred = self.aggregate_predictions(fitted_preds)
        return pred

    def recursive_impute(self, df_scaled, df_filled):
        # Perform estimation using numpy
        index_1 = df_scaled.index.get_level_values(self.fill_method).to_numpy()
        columns = df_scaled.columns.to_numpy()
        df_array = df_scaled.to_numpy()
        group_names, group_id = np.unique(index_1, return_index=True)
        group_names = group_names[np.argsort(group_id)]
        df_groups = np.split(df_array, len(group_names))
        #  perform (recursive) imputation
        for group, variable in self.missing_priority:
            pred = self.aggregate_estimation(df_groups, group_names, group, columns, variable)
            if self.recursive_imputation:
                df_scaled.loc[group, variable] = pred
            df_filled.loc[group, variable] = pred
        return df_filled

    def impute(self, df):
        """
        Impute missing values in the DataFrame using a combination of scaling and linear regression models.
        """
        df, Flipped = self.setup_df(df)
        df_filled = df.copy()
        a_miss = df.isna().all()
        m_miss = df.isna().groupby(self.fill_method).any().all(axis=1)
        df = df.drop(index=m_miss[m_miss].index, level=self.fill_method)
        df = df.drop(columns=a_miss[a_miss].index)

        # Calculating the priority for filling the series
        self.missing_priority = self.calc_imputation_priority(df)
        if len(self.missing_priority)==0:
            return self.package_df(df_filled, Flipped)

        # Standardize and add constant
        df_scaled = df.groupby(self.fill_method, group_keys=False).apply(self.standardize_series)
        df_scaled['const'] = 1

        #  perform (recursive) imputation
        df_filled = self.recursive_impute(df_scaled, df_filled)
        return self.package_df(df_filled, Flipped)

    def make_replica(self, df, original_missing_indices=None):
        """
        Create a replica matrix with imputed values by iteratively assuming that some series are empty and then filling them.
        """
        df, Flipped = self.setup_df(df)
        df_replica = df.copy()
        matrix_map = df.groupby(self.fill_method).sum().stack().index
        if self.partial_replica and type(original_missing_indices)==type(None):
            original_missing_indices = []
            warnings.warn("Please provide the original missing indices when partial replication is on.", UserWarning)
        elif not self.partial_replica:
            original_missing_indices = []
        for item in matrix_map:
            df.loc[item] = np.NaN
            replica_values = self.impute(df).loc[item].values
            if not (item in original_missing_indices):
                if self.recursive_replication:
                    df.loc[item] = replica_values
                else:
                    df.loc[item] = df_replica.loc[item].values
                df_replica.loc[item] = replica_values
            else:
                df.loc[item] = df_replica.loc[item].values
        return self.package_df(df_replica, Flipped)

    def fnorm(self, df, df_replica):
        """
        Compute the Frobenius norm between the provided matrices
        """
        return np.sqrt(np.sum(np.sum((df - df_replica)**2)))

    def flat_corr(self, df):
        """
         Compute the flattened correlation matrix of the panel dataframe
        """
        df, Flipped = self.setup_df(df)
        second_index = next((v for v in df.index.names if v != self.fill_method), None)
        return df.reset_index().pivot(index=second_index, columns=self.fill_method, values=df.columns).corr()

    def replica_loss(self, df, original_missing_indices=None):
        """
        Calculate the replica loss:  Frobenius norm between the flattened correlation matrices of the original dataframe and its replica
        """
        df_replica = self.make_replica(df, original_missing_indices)
        df_pivot_corr = self.flat_corr(df)
        df_replica_pivot_corr = self.flat_corr(df_replica)
        fnorm = self.fnorm(df_pivot_corr, df_replica_pivot_corr)
        return fnorm

    @staticmethod
    def run_calibration(df, search_params, partial_replica=False, recursive_replication=False):
        df = df.copy()
        param_combinations = [dict(zip(search_params.keys(), values)) for values in itertools.product(*search_params.values())]

        # Doing a grid search to find the best param set:
        calibration_results = []
        for param in tqdm(param_combinations):
            empty_imputer = EmptySeriesImputer(partial_replica=partial_replica,
                                            recursive_replication=recursive_replication,
                                            **param)
            df_calib = empty_imputer.impute(df)
            mp = empty_imputer.missing_priority
            param['loss'] = empty_imputer.replica_loss(df_calib, mp)
            calibration_results.append(param)
        calibration_results = pd.DataFrame(calibration_results).sort_values('loss')
        return calibration_results

