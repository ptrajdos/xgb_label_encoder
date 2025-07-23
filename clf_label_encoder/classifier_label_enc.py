from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.calibration import LabelEncoder
from sklearn.multiclass import available_if
from xgboost import XGBClassifier
import numpy as np
from sklearn.utils.multiclass import type_of_target
import logging


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    Calling a prediction method will only be available if `refit=True`. In
    such case, we check first the fitted best estimator. If it is not
    fitted, we check the unfitted estimator.

    Checking the unfitted estimator allows to use `hasattr` on the `SearchCV`
    instance even before calling `fit`.
    """

    def check(self):
        if hasattr(self, "model_"):
            # raise an AttributeError if `attr` does not exist
            getattr(self.model_, attr)
            return True

    return check


class ClassifierWithLabelEncoder(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator_class=None, estimator_options=None):
        self.estimator_class = estimator_class
        self.estimator_options = estimator_options

    def _get_effective_model(self):
        effective_estim_class = (
            self.estimator_class if self.estimator_class else XGBClassifier
        )
        effective_estim_options = (
            self.estimator_options if self.estimator_options else {"random_state": 0}
        )
        return effective_estim_class(**effective_estim_options)

    def _create_encoder(self, X, y):

        target_type = type_of_target(y)

        if target_type in ["binary", "multiclass"]:
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)
            return

        self.label_encoder_ = None

    def _encode_y(self, y):
        if self.label_encoder_ is None:
            return y

        return self.label_encoder_.transform(y)

    def _decode_y(self, y):

        if self.label_encoder_ is None:
            return y

        return self.label_encoder_.inverse_transform(y)

    def fit(self, X, y, **kwargs):
        self._create_encoder(X, y)
        y_encoded = self._encode_y(y)
        self.model_ = self._get_effective_model()
        try:
            self.model_.fit(X, y_encoded, **kwargs)
        except Exception as e:
            logging.warning(f"Training base model has failed!", exc_info=True)
            raise e

        self.trained_ = True

        return self

    def _fit_check(self):
        check_is_fitted(
            self,
            ("trained_",),
        )

    def _more_tags(self):
        return self._get_effective_model()._more_tags()

    def __sklearn_tags__(self):
        return self._get_effective_model().__sklearn_tags__()

    @property
    def n_features_in_(self):
        return self.model_.n_features_in_

    @property
    def classes_(self):
        if self.label_encoder_ is not None:
            return self.label_encoder_.classes_

        return self.model_.classes_

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, weights=None):
        self._fit_check()
        return self.model_.predict_log_proba(X)

    def fit_predict(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.predict(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X, **kwargs):
        self._fit_check()
        return self.model_.decision_function(X, **kwargs)

    def _predict_dtype_changer(self, y_pred_decoded):

        if hasattr(self.classes_, "dtype"):
            pred_dtype = self.classes_.dtype
        else:
            pred_dtype = np.int_

        return np.astype(y_pred_decoded, pred_dtype)

    def predict(self, X, **kwargs):
        self._fit_check()
        y_pred_encoded = self.model_.predict(X, **kwargs)
        y_decoded = self._decode_y(y_pred_encoded)
        y_decoded_dty = self._predict_dtype_changer(y_decoded)
        return y_decoded_dty

    def predict_proba(self, X, **kwargs):
        self._fit_check()
        return self.model_.predict_proba(X, **kwargs)

    def score(self, X, y, **kwargs):
        self._fit_check()
        y_encoded = self._encode_y(y)
        return self.model_.score(X, y_encoded, **kwargs)
