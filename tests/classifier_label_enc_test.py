import unittest

from xgboost import XGBClassifier
from clf_label_encoder.classifier_label_enc import ClassifierWithLabelEncoder
from sklearn.utils.estimator_checks import check_estimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score





class ClassifierWithLabelEncoderTest(unittest.TestCase):

    def get_estimators(self):
        return {
            "DT": ClassifierWithLabelEncoder(
                estimator_class=DecisionTreeClassifier,
                estimator_options={"random_state": 0},
            ),
            "XGB": ClassifierWithLabelEncoder(
                estimator_class=XGBClassifier, estimator_options={"random_state": 0}
            ),
            "DEF": ClassifierWithLabelEncoder(),
            "RF": ClassifierWithLabelEncoder(
                estimator_class=RandomForestClassifier,
                estimator_options={"random_state": 0},
            ),
            "LR": ClassifierWithLabelEncoder(
                estimator_class=LogisticRegression,
                estimator_options={"random_state": 0},
            ),
        }

    def test_sklearn(self):

        checks_to_fail = {
            "check_estimator_tags_renamed": "XGB is failing with it!",
            "check_estimators_overwrite_params": "XGB",
            "check_dont_overwrite_parameters": "XGB",
            "check_classifiers_one_label": "XGB",
            "check_classifiers_one_label_sample_weights": "XBG",
            "check_fit2d_1sample": "XGB",
        }

        for clf_name, clf in self.get_estimators().items():
            with self.subTest(clf_name=clf_name):
                fail_checks = (
                    checks_to_fail if clf_name in ["XGB", "DEF", "XGBo"] else {}
                )
                check_estimator(clf, expected_failed_checks=fail_checks)

    def test_iris(self):
        X, y = load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


        dtypes = [np.str_, np.int_]
        for clf_name, clf in self.get_estimators().items():
            for dty in dtypes:
                with self.subTest(clf_name=clf_name, dty=dty):
                    y_t = np.astype(y_train, dty)
                    y_test_dty = np.astype(y_test, dty)
                    y_t_u = np.unique(y_t)

                    clf.fit(X_train, y_t)
                    y_pred = clf.predict(X_test)
                    self.assertTrue(np.isin(y_pred, y_t_u).all(), "Predicted class is not known")
                    kappa_score = cohen_kappa_score(y_pred, y_test_dty)
                    self.assertTrue(kappa_score>=0, "Kappa score is negative")



if __name__ == "__main__":
    unittest.main()
