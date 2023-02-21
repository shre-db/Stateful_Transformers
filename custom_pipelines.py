from sklearn.pipeline import Pipeline

class CustomPipeline(Pipeline):
    """
    This pipeline class allows working with transformers that require both X and y for `transform` 
    method. The `transform` and `fit_transform` methods are modified, therefore while instantiating 
    make sure the transformers under the hood allow `fit_transform` to be performed using 
    both X and y.
    """
    def modified_transform(transform):
        def wrapper(self, X, y=None):
            steps = super().get_params()['steps']
            for i in range(len(steps)):
                X, y = steps[i][1].transform(X, y)
            return X, y
        return wrapper
        

    def modified_fit_transform(fit_transform):
        def wrapper(self, X, y=None, **fit_params):
            steps = super().get_params()['steps']
            for i in range(len(steps)):
                X, y = steps[i][1].fit_transform(X, y)
            return X, y
        return wrapper


    @modified_transform
    def transform(self, X, y=None):
        return super().transform(X)
        

    @modified_fit_transform        
    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)


class FullPipeline(Pipeline):
    """
    This pipeline class is a modified version of the standard pipeline class provided by Scikit-Learn.
    Its purpose is to handle two levels of pipeline nesting that may include transformers derived
    from Base and Mixin classes like `TransformerMixin` or a custom Mixin class like `StatefulTransformerMixin`.
    The higher-level pipeline contain modules that are solely pipelines while the lower-level pipelines 
    contain homogenous transformers (where transfomer's `transform` method takes only `X` or both `X` and `y`).
    """
    def modified_transform(transform):
        def wrapper(self, X, y=None):
            steps = super().get_params()['steps']
            for i in range(len(steps)):
                if type(steps[i][1]) is CustomPipeline:
                    X, y = steps[i][1].transform(X, y)
                else:
                    X = steps[i][1].transform(X)
            return X, y
        return wrapper


    def modified_fit_transform(fit_transform):
        def wrapper(self, X, y=None, **fit_params):
            steps = super().get_params()['steps']
            for i in range(len(steps)):
                if type(steps[i][1]) is CustomPipeline:
                    X, y = steps[i][1].fit_transform(X, y)
                else:
                    X = steps[i][1].fit_transform(X, y)
            return X, y
        return wrapper

    
    @modified_transform
    def transform(self, X, y=None):
        return super().transform(X)
        

    @modified_fit_transform        
    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)
        