from sklearn.base import TransformerMixin

# Define a custom TransformerMixin class. This allows fit_transform method of concrete classes to
# work with both X and y.
class StatefulTransformerMixin(TransformerMixin):

    def fit_transform(self, X, y, **fit_params):
        # fit and transform methods of arity 2 (supervised transformation)
        return self.fit(X, y, **fit_params).transform(X, y)