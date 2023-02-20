import numpy as np
import pandas as pd
from stateful_transformermixin import StatefulTransformerMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import shuffle


class StratifiedStatisticalImputer(BaseEstimator, StatefulTransformerMixin):
    """
    Impute missing values with statistical values (mean, median) to numerical columns based on 
    categories from another column.

    Parameter
    ----------
    strategy : (str), default='mean'
        The Imputation strategy.

        - If 'mean', then replace missing values using mean of the values from that category.
          Can only be used with numeric data.
        - If 'median', then replace missing values using median of the values from that category.
          Can only be used with numeric data.

    random_state: (int), default=None
        Seed number to for reproducibility.

    split: (bool), default=False
        If True, transform method returns X and y seperately.
        If False, transform method returns Concatenated dataframe with last column being y.

    Attributes
    ----------
    features_ : list
        Predictors in the data.

    n_features_: int
        Number of features in the data.

    categories_: list
        Categories in categorical data.

    cardinality_: int
        Number of categories in categorical data.

    means_: dict
        Mean value of imputation for each category.

    medians_: dict
        Median value of imputation for each category.

    Returns
    -------
    imputed_data: Pandas DataFrame, of shape (n_samples, m_features + 1)
        The numerical data after imputation.

    Examples
    --------
    With NumPy arrays

    >>> X = np.array([2.62, np.nan, 10.51, 3.05, 11.10, 2.90, np.nan, 10.89])
    >>> y = np.array(['Tiger', 'Smilodon', 'Smilodon', 'Tiger', 'Smilodon', 'Tiger', 'Tiger', 'Smilodon'])

    >>> imputer = StratifiedStatisticalImputer(strategy='mean', random_state=16)
    >>> imputer.fit_transform(X, y)
              F0 category col
    0  11.100000     Smilodon
    1   2.620000        Tiger
    2  10.833333     Smilodon
    3  10.890000     Smilodon
    4   3.050000        Tiger
    5   2.856667        Tiger
    6   2.900000        Tiger
    7  10.510000     Smilodon

    With Pandas Series

    >>> X = pd.Series([2.62, np.nan, 10.51, 3.05, 11.10, 2.90, np.nan, 10.89], name='canine length(in)')
    >>> y = pd.Series(['Tiger', 'Smilodon', 'Smilodon', 'Tiger', 'Smilodon', 'Tiger', 'Tiger', 'Smilodon'], name='species')

    >>> imputer = StratifiedStatisticalImputer(strategy='median', random_state=16)
    >>> imputer.fit_transform(X, y)
        canine length(in)   species
    0              11.10  Smilodon
    1               2.62     Tiger
    2              10.89  Smilodon
    3              10.89  Smilodon
    4               3.05     Tiger
    5               2.90     Tiger
    6               2.90     Tiger
    7              10.51  Smilodon

    Warning
    -------
    This transformer is experimental or in a prototype stage. The output from the `transform` or 
    `fit_transform` methods will result in a shuffled dataframe of only numerical variables. 
    Combining this transformed data with the original dataframe that may include both numerical 
    and categorical variables may produce incorrect combinations. Please use with caution. 
    Future versions of this transformer may address this issue.
    """
    allowed_strategies = ['mean', 'median']  # Class variable


    def __init__(self, strategy='mean', random_state=None, split=False):
        self.strategy = strategy
        self.random_state = random_state
        self.split = split
        strf_imp = StratifiedStatisticalImputer  # Alias for long class name
        # allowed_startegies = ['mean', 'median']
        if self.strategy not in strf_imp.allowed_strategies:
            raise ValueError(
                "Can only use these strategies: {0} got strategy={1}".format(
                    strf_imp.allowed_strategies,
                    self.strategy
                    )
            )


    def _check_input_data(self, X, y): 
        X = np.array(X)
        y = np.array(y)


        # ------------------ Raise error when ------------------ #
        

        # 1. When length of the input arrays are unequal
        if len(X) != len(y):
            raise ValueError("All arrays must be same length")

        # Reshape X to (n_samples, 1_feature) if X.shape is (n_samples,).
        if X.shape == (len(X),):
            X = X.reshape(len(X), 1)
                   

        # 2. When y is not of shape (n_samples,) or (n_samples, 1_feature)
        if y.shape != (len(y),):
            try:
                y.reshape(len(y),)
            except ValueError:
                raise ValueError(
                    "Expected input array of shape {0} or {1}, got {2}. Provide array of shape either {0} or {1}".format
                    ((len(y),), (len(y), 1), y.shape)
                )
            else:
                y = y.reshape(len(y),)

        # 3. When dtype of y is not 'object' or 'unicode'
        if y.dtype.kind not in ['O', 'U']:
            raise TypeError(
                "Expected dtype: {0}, got {1}".format(['O', 'U'], y.dtype.kind)
            )

        # 4. When dtype of X is not float, and a type casting attempt fails.
        if X.dtype.kind not in ('f'):
            try:
                X.astype('float64')
            except ValueError:
                raise TypeError(
                    "Expected dtype: {0}, got {1}".format('float', X.dtype.kind)
                )
            else:
                X = X.astype('float64')
        else:
            X = X.astype('float64')
        
        return X, y


    def fit(self, X, y):
        """
        Fit the imputer on data.

        Parameters
        ----------
        X : (int, float) array-like, of shape (n_samples, m_feature) 
            The data that has missing values and is subjected to imputation.
        
        y : (Object, Unicode) array-like, of shape (n_samples, ) or (n_samples, 1_feature)
            A data that has categories. To store categories and fetch its row indices.

        Returns
        -------
        self : 
            Returns the fitted object.

        """
        if type(X) == pd.DataFrame:
            self.features_ = [col for col in X.columns]
        elif type(X) == pd.Series:
            if X.name == None:
                self.features_ = ['F0']
            else:    
                self.features_ = [X.name]
        elif X.shape == (len(X),):
            self.features_ = ['F0']    
        else:
            self.features_ = [f'F{x}' for x in range(X.shape[1])]

        if type(y) == pd.DataFrame:
            labels = [col for col in y.columns]
            self.label_name_ = labels[0]
        elif type(y) == pd.Series:
            if y.name == None:
                self.label_name_ = 'category col'
            else:    
                self.label_name_ = y.name  
        else:
            self.label_name_ = 'category col'

        strf_imp = StratifiedStatisticalImputer # Alias for long class name
        X, y = self._check_input_data(X, y)

        # 1. Store number of features
        self.n_features_ = len(self.features_)

        # 2. Store unique values in y.
        self.categories_ = list(np.unique(y))

        # 3. Store cardinality of y.
        self.cardinality_ = len(np.unique(y))

        # Create a list of arrays that is grouped by y
        X_by_cat = []
        for cat in self.categories_:
            X_by_cat.append(X[y==cat])

        # 4. Store Mean value of imputation by category for each feature.
        self.means_ = pd.DataFrame(columns= ['Category'] + self.features_)
        for cat in range(len(self.categories_)):
            if self.n_features_ == 1:
                means = np.nanmean(a=X_by_cat[cat], axis=0)
                self.means_.loc[cat] = {'Category': self.categories_[cat], self.features_[0]: float(means)}  
            else:
                means = np.nanmean(a=X_by_cat[cat], axis=0).tolist()
                self.means_.loc[cat] = [self.categories_[cat]] + means
        self.means_ = self.means_.set_index('Category')

        # 5. Store Median value of imputation for each category.
        self.medians_ = pd.DataFrame(columns= ['Category'] + self.features_)
        for cat in range(len(self.categories_)):
            if self.n_features_ == 1:
                medians = np.nanmedian(a=X_by_cat[cat], axis=0) 
                self.medians_.loc[cat] = {'Category': self.categories_[cat], self.features_[0]: float(medians)} 
            else:
                medians = np.nanmedian(a=X_by_cat[cat], axis=0).tolist()
                self.medians_.loc[cat] = [self.categories_[cat]] + medians
        self.medians_ = self.medians_.set_index('Category')

        return self

    
    def transform(self, X, y):
        """
        Impute all missing values in `X`.

        Parameters
        ----------
        X : (int, float) array-like, of shape (n_samples, m_feature) 
            The data that has missing values and is subjected to imputation.
        
        y : (Object, Unicode) array-like, of shape (n_samples, ) or (n_samples, 1_feature)
            A data that has categories. To store categories and fetch its row indices.

        Returns
        -------
        imputed_data : pandas DataFrame of shape (n rows, m + 1 columns)
            `num_data` with imputed values

        """
        strf_imp = StratifiedStatisticalImputer # Alias for long class name
        check_is_fitted(self, ['features_', 'n_features_', 'categories_', 'cardinality_', 'means_', 'medians_'])

        X = np.array(X)
        y = np.array(y).flatten()

        if X.shape == (len(X),):
            X = X.reshape(-1, 1)  

        # Create a list of arrays that is grouped by y ✔
        X_by_cat = []
        for cat in self.categories_:
            X_by_cat.append(X[y==cat])

        # Impute missing values ✔
        cat_df = []
        for cat in range(len(self.categories_)):
            imputed_data = pd.DataFrame()
            for feature in range(len(self.features_)):
                if self.strategy == strf_imp.allowed_strategies[0]:
                    imputation_value = self.means_.iloc[cat, feature]
                elif self.strategy == strf_imp.allowed_strategies[1]:
                    imputation_value = self.medians_.iloc[cat, feature]
                
                arr = X_by_cat[cat][:, feature]
                arr[np.isnan(arr)] = imputation_value

                imputed_data[self.features_[feature]] = arr
            cat_df.append(imputed_data)

        imputed_df = pd.concat(cat_df)

        # Assign categorical variables accordingly ✔
        y_new = []
        for cat in range(len(self.categories_)):
            labels = [self.categories_[cat]] * len(X_by_cat[cat])
            y_new.append(labels)

        y_new = sum(y_new, [])
        imputed_df[self.label_name_] = y_new

        # Randomly shuffle the row indices ✔
        imputed_df = shuffle(imputed_df, random_state=self.random_state)
        imputed_df = imputed_df.reset_index(drop=True)

        # Split or not ✔
        if self.split:
            data = imputed_df.drop(self.label_name_, axis=1)
            labels = imputed_df[self.label_name_]
            return data, labels
        else:
            return imputed_df