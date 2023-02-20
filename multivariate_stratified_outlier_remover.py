import numpy as np
import pandas as pd
from stateful_transformermixin import StatefulTransformerMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import shuffle


class MultivariateStratifiedOutlierRemover(BaseEstimator, StatefulTransformerMixin):
    """
    Removes data points that fall outside of hyperrectangles defined by strata based on IQR(1.5) or IQR(3) 
    proximity rule.

    Parameters
    ----------
    strategy : (float), default=1.5
        The rule to compute outliers.

        - If 1.5, then compute outliers based on inter-quartile-range(1.5) proximity rule. 
        - If 3, then compute extreme values based on inter-quartile-range(3) proximity rule.

    random_state: (int), default=None
        Specify the seed for random number generation.

    split: (bool), default=False
        If True, transform method returns X and y seperately.
        If False, transform method returns concatenated dataframe with last column being y. 

    X : (int, float) array-like of shape (n_samples, m_features)
        Features or variables in the data.
        
    y : (object) array-like of shape (n_samples, 1_feature)
        Data that has categories.

    Attributes
    ----------
    features_: list
        if X is a Pandas DataFrame, display a list of features, else outputs 'None'.

    label_name_: list
        if y is a Pandas DataFrame, display the name of the column, else display 'category col'.

    n_samples_: int
        Number of samples

    n_features_: int
        Number of features

    categories_: list
        Categories in categorical data.

    cardinality_: int
        Number of categories in categorical data.

    IQR_: pd.DataFrame
        Interquartile range

    minima_: pd.DataFrame
        Minimums of variables.

    first_quartiles_: pd.DataFrame
        First quartiles or 25th Percentiles of the variables. 

    medians_: pd.DataFrame
        Median value of imputation for each category.

    third_quartiles_: pd.DataFrame
        Third quartiles or 75th Percentiles of the variables.
    
    maxima_: pd.DataFrame
        Maximums of variables.

    percent_: dict
        Percentage of outliers grouped by category in the variables.

    loss_: int
        Number of observations that will be removed.

    normalized_loss_: float
        Percentage of observations that will be removed.


    Returns
    -------
    Pandas DataFrame, of shape (n_samples_ - loss_, m_features_)
        The data after outlier removal.

    Warning!
    -------
    This transformer is experimental or in a prototype stage. The output from the `transform` 
    or `fit_transform` methods will result in a shuffled dataframe of only numerical variables. 
    Combining this transformed data with the original dataframe that may include both numerical and 
    categorical variables may produce incorrect combinations. Please use with caution. Future 
    versions of this transformer may address this issue.    
    """
    allowed_strategies = [1.5, 3]  # Class variable


    def __init__(self, strategy=1.5, random_state = None, split=False):
        self.strategy = strategy
        self.random_state = random_state
        self.split = split
        MOR = MultivariateStratifiedOutlierRemover  # Alias
        if self.strategy not in MOR.allowed_strategies:
            raise ValueError(
                "Can only use these strategies: {0} got strategy={1}".format(
                    MOR.allowed_strategies,
                    self.strategy
                    )
            )


    def _check_input_data(self, X, y):  
        X = np.array(X)
        y = np.array(y)

        # ------------------ Raise error when ------------------ #
        

        # 1. When length of the input arrays are unequal
        if len(y) != len(X):
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
        if X.dtype.kind not in ['i', 'f']:
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
        Fit the remover on data.

        Parameters
        ----------
        X : (numerical) array-like of shape (n_samples, m_features)
            Features or variables in the data.
        
        y : (categorical) array-like of shape (n_samples, 1_feature)
            Data that has categories.
 
        Returns
        -------
        self:
            Returns the fitted object
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

        X, y = self._check_input_data(X, y)

        # Store number of samples ✔
        self.n_samples_ = len(y)

        # Store number of features ✔
        self.n_features_ = X.shape[1]

        # Store categories ✔
        self.categories_ = list(np.unique(y))

        # Store number of categories ✔
        self.cardinality_ = len(np.unique(y))

        # Create a list of arrays that is grouped by y ✔
        X_by_cat = []
        for cat in self.categories_:
            X_by_cat.append(X[y==cat])   

        # Store interquartile range by category ✔
        self.IQR_ = pd.DataFrame(columns= ['Category'] + self.features_)
        for cat in range(len(self.categories_)):
            iqr = (np.quantile(a=X_by_cat[cat], q=0.75, axis=0) - np.quantile(a=X_by_cat[cat], q=0.25, axis=0)).tolist()
            self.IQR_.loc[cat] = [self.categories_[cat]] + iqr
        self.IQR_ = self.IQR_.set_index('Category')

        # Store minima of the variables by category ✔
        self.minima_ = pd.DataFrame(columns= ['Category'] + self.features_)
        for cat in range(len(self.categories_)):
            # To find minima, use formula -> minima = 25th quantile - IQR * 1.5 
            minima =  (np.quantile(a=X_by_cat[cat], q=0.25, axis=0) - ((np.quantile(a=X_by_cat[cat], q=0.75, axis=0) - np.quantile(a=X_by_cat[cat], q=0.25, axis=0)) * self.strategy)).tolist()
            self.minima_.loc[cat] = [self.categories_[cat]] + minima
        self.minima_ = self.minima_.set_index('Category')

        # Store first quartiles of the variables by category ✔
        self.first_quartiles_ = pd.DataFrame(columns= ['Category'] + self.features_)
        for cat in range(len(self.categories_)):
            first_quartiles = np.quantile(a=X_by_cat[cat], q=0.25, axis=0).tolist()
            self.first_quartiles_.loc[cat] = [self.categories_[cat]] + first_quartiles
        self.first_quartiles_ = self.first_quartiles_.set_index('Category')

        # Store medians of the variables by category ✔
        self.medians_ = pd.DataFrame(columns= ['Category'] + self.features_)
        for cat in range(len(self.categories_)):
            medians = np.median(a=X_by_cat[cat], axis=0).tolist()
            self.medians_.loc[cat] = [self.categories_[cat]] + medians
        self.medians_ = self.medians_.set_index('Category')

        # Store third quartiles of the variables by category ✔
        self.third_quartiles_ = pd.DataFrame(columns= ['Category'] + self.features_)
        for cat in range(len(self.categories_)):
            third_quartiles = np.quantile(a=X_by_cat[cat], q=0.75, axis=0).tolist()
            self.third_quartiles_.loc[cat] = [self.categories_[cat]] + third_quartiles
        self.third_quartiles_ = self.third_quartiles_.set_index('Category')

        # Store maxima of the variables by category ✔
        self.maxima_ = pd.DataFrame(columns= ['Category'] + self.features_)
        for cat in range(len(self.categories_)):
            # To find maxima, use formula -> maxima = 75th quantile + IQR * 1.5 
            maxima =  (np.quantile(a=X_by_cat[cat], q=0.75, axis=0) + ((np.quantile(a=X_by_cat[cat], q=0.75, axis=0) - np.quantile(a=X_by_cat[cat], q=0.25, axis=0)) * self.strategy)).tolist()
            self.maxima_.loc[cat] = [self.categories_[cat]] + maxima
        self.maxima_ = self.maxima_.set_index('Category')

        # Store indices of rows having outliers ✔
        self.outlier_indices_ = []
        for cat in range(len(self.categories_)):
            outlier_cells = []
            for col in range(len(self.features_)):
                cells = np.where(
                    np.where(
                        X_by_cat[cat].T[col] > self.maxima_.iloc[cat, col],
                        True,
                        np.where(X_by_cat[cat].T[col] < self.minima_.iloc[cat, col], True, False)
                    )
                )[0].tolist()
                outlier_cells.append(cells)
            cat_outlier_indices = list(set(sum(outlier_cells, [])))
            self.outlier_indices_.append(cat_outlier_indices)

        # Store percentage of rows having outliers ✔
        self.percent_ = {}
        for cat in range(len(self.categories_)):
            perc = round( len(self.outlier_indices_[cat]) / len(X_by_cat[cat]) * 100, 2 )
            self.percent_[self.categories_[cat]] = perc

        # Store number of rows having outliers ✔
        self.loss_ = len(sum(self.outlier_indices_, []))

        # Store percentage of rows having outliers ✔
        self.normalized_loss_ = round(self.loss_/len(y), 2)

        return self


    def transform(self, X, y):
        """
        Removes outliers based on the learned parameters.

        Parameters:
        -----------

        X: (numerical) array-like of shape (n_samples, m_features)
            Features or variables in the data to be Transformed.

        y : (object) array-like of shape (n_samples, 1_feature)
            Data that has categories.

        Returns:
        --------
        
        Pandas DataFrame, of shape (n_samples_ - loss_ , m_features_ + 1 column)
            The Data after outlier removal.
        """
        check_is_fitted(self, ['features_', 'label_name_', 'n_samples_', 'n_features_', 'categories_','cardinality_','IQR_','minima_','first_quartiles_','medians_','third_quartiles_','maxima_','outlier_indices_','percent_', 'loss_', 'normalized_loss_'])

        X, y = self._check_input_data(X, y)
        
        # Create a list of arrays that is grouped by y ✔
        X_by_cat = []
        for cat in self.categories_:
            X_by_cat.append(X[y==cat])

        # This step rewrites outlier indices while using only `transform` method ✔
        self.outlier_indices_ = []
        for cat in range(len(self.categories_)):
            outlier_cells = []
            for col in range(len(self.features_)):
                cells = np.where(
                    np.where(
                        X_by_cat[cat].T[col] > self.maxima_.iloc[cat, col],
                        True,
                        np.where(X_by_cat[cat].T[col] < self.minima_.iloc[cat, col], True, False)
                    )
                )[0].tolist()
                outlier_cells.append(cells)
            cat_outlier_indices = list(set(sum(outlier_cells, [])))
            self.outlier_indices_.append(cat_outlier_indices)

        # Store percentage of rows having outliers ✔
        self.percent_ = {}
        for cat in range(len(self.categories_)):
            perc = round( len(self.outlier_indices_[cat]) / len(X_by_cat[cat]) * 100, 2 )
            self.percent_[self.categories_[cat]] = perc

        # Store number of rows having outliers ✔
        self.loss_ = len(sum(self.outlier_indices_, []))

        # Store percentage of rows having outliers ✔
        self.normalized_loss_ = round(self.loss_/len(y), 2)

        # Based on the index postions in self.outlier_indices_, remove the marked observations from X_by_cat array ✔
        X_by_cat_remv = []
        for cat in range(len(self.categories_)):
            X_remv = np.delete(X_by_cat[cat], self.outlier_indices_[cat], axis=0)
            X_by_cat_remv.append(X_remv)

        # Assign the categorical variable accordingly ✔
        y_new = []
        for cat in range(len(self.categories_)):
            labels = [self.categories_[cat]] * len(X_by_cat_remv[cat])
            y_new.append(labels)

        X_by_cat_remv = np.vstack(X_by_cat_remv)
        y_new = sum(y_new, [])

        # Convert to a Dataframe ✔
        data_removed = pd.DataFrame(data=X_by_cat_remv, columns=self.features_)
        data_removed[self.label_name_] = np.array(y_new).reshape(-1, 1)

        # Randomly shuffle the row indices ✔
        data_removed = shuffle(data_removed, random_state=self.random_state)

        # Split or not ✔
        if self.split:
            data = data_removed.drop(self.label_name_, axis=1)
            labels = data_removed[self.label_name_]
            return data, labels
        else:
            return data_removed