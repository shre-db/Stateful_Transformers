# Stateful Transformers

Overview
--------
Currently this repository contains following classes based on scikit-Learn Library.
1. StatefulTransformerMixin
2. StratifiedStatisticalImputer
3. MultivariateStratifiedOutlierRemover
4. CustomPipeline
5. FullPipeline

These classes were built for specific needs of a project and may not be useful regularly in Machine Learning. The customizations made to these classes make it incompatible with other elements of scikit-learn implementation and should be used with caution. In most cases these classes are not recommended for use with other scikit-learn classes.
This README file provides a description of these classes, usage, contributing guidelines and other information related to the project.

Table of contents
-----------------
1. StatefulTransformerMixin
2. StratifiedStatisticalImputer
3. MultivariateStratifiedOutlierRemover
4. CustomPipeline
5. FullPipeline
6. Usage
7. Contributing
8. License
9. Contact

StatefulTransformerMixin
------------------------
StatefulTransformerMixin class is a custom mixin class that extends the functionality of the TransformerMixin class in scikit-learn. It overrides the `fit_transform` method of TransformerMixin to work with both X and y in a supervised learning setting. This is achieved by passing both X and y to the `transform` method of the transformer object.

The `fit_transform` method of StatefulTransformerMixin takes in X, y, and fit_params arguments. It fits the transformer object to the data using the `fit` method and then applies the transformer to the data using the `transform` method, passing in both X and y. The `**fit_params` argument allows for passing additional parameters to the `fit` method if needed.

The StatefulTransformerMixin class provides a convenient way to create custom transformers that can handle both input data X and target variable y in a supervised learning setting.

StratifiedStatisticalImputer
----------------------------

This is a Python class for imputing missing values with statistical values (mean, median) to numerical columns based on categories from another column. The class is named 'StratifiedStatisticalImputer', and it is implemented as a scikit-learn estimator, with BaseEstimator as the base class, and StatefulTransformerMixin as an additional mixin class.

The class takes in three parameters:

- `strategy` a string that specifies the imputation strategy, either 'mean' or 'median'.
- `random_state` an integer that sets the seed number for reproducibility.
- `split` a boolean that determines whether the transform method returns X and y separately or as a concatenated dataframe with the last column being y.

The class has the following attributes:

- `features_` a list of predictors in the data.
- `n_features_` an integer that specifies the number of features in the data.
- `categories_` a list of categories in the categorical data.
- `cardinality_` an integer that specifies the number of categories in the categorical data.
- `means_` a dictionary of mean values of imputation for each category.
- `medians_` a dictionary of median values of imputation for each category.

The class has a fit_transform method that takes in two arguments, X and y, and returns a pandas DataFrame of shape (n_samples, m_features + 1), which is the numerical data after imputation.

This class is experimental and in a prototype stage, and the output from the transform or fit_transform methods will result in a shuffled dataframe of only numerical variables. Combining this transformed data with the original dataframe that may include both numerical and categorical variables may produce incorrect combinations. This transformer should be used with caution, and future versions of this transformer may address this issue.

MultivariateStratifiedOutlierRemover
------------------------------------
This is a transformer class that removes data points that fall outside of hyper-rectangles defined by strata based on IQR(1.5) or IQR(3) proximity rule. It is built using scikit-learn's BaseEstimator class and a custom TransformerMixin class called StatefulTransformerMixin.

The transformer takes the following parameters:

- `strategy` A float, default value of 1.5. The rule to compute outliers:
    If 1.5, then compute outliers based on inter-quartile-range (1.5) proximity rule.
    If 3, then compute extreme values based on inter-quartile-range (3) proximity rule.
- `random_state` An int, default value of None. Specifies the seed for random number generation.
- `split` A bool, default value of False. If True, the transform method returns X and y separately. If False, the transform method returns a concatenated DataFrame with the last column being y.
- `X` An array-like of shape (n_samples, m_features). Features or variables in the data.
- `y` An array-like of shape (n_samples, 1_feature). Data that has categories.

The transformer has several attributes that store information about the data and computation:

- `features_` If X is a Pandas DataFrame, a list of features; otherwise, 'None'.
- `label_name_` If y is a Pandas DataFrame, the name of the column; otherwise, 'category col'.
- `n_samples_` An integer that stores the number of samples.
- `n_features_` An integer that stores the number of features.
- `categories_` A list that stores the categories in categorical data.
- `cardinality_` An integer that stores the number of categories in categorical data.
- `IQR_` A Pandas DataFrame that stores the interquartile range.
- `minima_` A Pandas DataFrame that stores the minimums of variables.
- `first_quartiles_` A Pandas DataFrame that stores the first quartiles or 25th percentiles of the variables.
- `medians_` A Pandas DataFrame that stores the median value of imputation for each category.
- `third_quartiles_` A Pandas DataFrame that stores the third quartiles or 75th percentiles of the variables.
- `maxima_` A Pandas DataFrame that stores the maximums of variables.
- `percent_` A dictionary that stores the percentage of outliers grouped by category in the variables.
- `loss_` An integer that stores the number of observations that will be removed.
- `normalized_loss_` A float that stores the percentage of observations that will be removed.

The transformer has three additional methods:

- `_check_input_data` A protected method that checks the input data and raises errors if the input is not valid.
- `fit` A method that fits the transformer by computing the various attributes listed above.
- `transform` A method that transforms the data by taking in both X and y as arguments. 

This transformer is experimental or in a prototype stage. The output from the `transform` or `fit_transform` methods will result in a shuffled DataFrame of only numerical variables. Combining this transformed data with the original DataFrame that may include both numerical and categorical variables may produce incorrect combinations. Please use with caution. Future versions of this transformer may address this issue.

CustomPipeline
--------------
This pipeline class is a modified version of the standard pipeline class provided by Scikit-Learn. When transformers like StratifiedStatisticalImputer and MultivariateStratifiedOutlierRemover are used as elements in a transformation pipeline, the standard `Pipeline` class provided by Scikit-Learn becomes incompatible with the requirements of data flow in the pipeline. This is because these transformers use a custom TransformerMixin class and not TransformerMixin directly. Therefore it is necessary to customize the `Pipeline` class as well. `CustomPipeline` inherits from Scikit-Learn's `Pipeline` class and modifies its `transform` and `fit_transform` methods using decorators. Overall, This pipeline class allows working with transformers that require both X and y for `transform` method, therefore while instantiating make sure the transformers under the hood allow `fit_transform` to be performed using both X and y.

FullPipeline
------------
This pipeline class is a modified version of the standard pipeline class provided by Scikit-Learn. Its purpose is to handle two levels of pipeline nesting that may include transformers derived from Base and Mixin classes like `TransformerMixin` or a custom Mixin class like `StatefulTransformerMixin`. The higher-level pipeline contain modules that are solely pipelines while the lower-level pipelines 
contain homogenous transformers (where transfomer's `transform` method takes only `X` or both `X` and `y`).
