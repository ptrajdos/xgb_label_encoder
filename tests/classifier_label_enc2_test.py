import unittest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.estimator_checks import check_estimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score


from clf_label_encoder.classifier_label_enc2 import ClassifierWithLabelEncoder2


class ClassifierWithLabelEncoder2Test(unittest.TestCase):

    def get_estimators(self):
        return {
            "DT": ClassifierWithLabelEncoder2(DecisionTreeClassifier(random_state=0)),
            "DEF": ClassifierWithLabelEncoder2(),
            "RF": ClassifierWithLabelEncoder2(RandomForestClassifier(random_state=0)),
            "LR": ClassifierWithLabelEncoder2(LogisticRegression(random_state=0)),
        }

    def test_sklearn(self):

        for clf_name, clf in self.get_estimators().items():
            with self.subTest(clf_name=clf_name):
                check_estimator(clf)

    def test_iris(self):
        X, y = load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        dtypes = [
            np.str_,
            np.int_,
            str,
            int,
        ]
        for clf_name, clf in self.get_estimators().items():
            for dty in dtypes:
                with self.subTest(clf_name=clf_name, dty=dty):
                    y_t = np.astype(y_train, dty)
                    y_test_dty = np.astype(y_test, dty)
                    y_t_u = np.unique(y_t)

                    clf.fit(X_train, y_t)
                    y_pred = clf.predict(X_test)
                    self.assertTrue(
                        np.isin(y_pred, y_t_u).all(), "Predicted class is not known"
                    )
                    kappa_score = cohen_kappa_score(y_pred, y_test_dty)
                    self.assertTrue(kappa_score >= 0, "Kappa score is negative")

    def test_pipeline(self):
        X, y = load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        dtypes = [np.str_, np.int_, str, int]
        for clf_name, clf in self.get_estimators().items():
            for dty in dtypes:
                with self.subTest(clf_name=clf_name, dty=dty):
                    y_t = np.astype(y_train, dty)
                    y_test_dty = np.astype(y_test, dty)
                    y_t_u = np.unique(y_t)

                    pipe = Pipeline(
                        [('trans', RobustScaler()), ('estimator', clf)]
                    )

                    pipe.fit(X_train, y_t)
                    y_pred = clf.predict(X_test)
                    self.assertTrue(
                        np.isin(y_pred, y_t_u).all(), "Predicted class is not known"
                    )
                    kappa_score = cohen_kappa_score(y_pred, y_test_dty)
                    self.assertTrue(kappa_score >= 0, "Kappa score is negative")

    def test_gridsearch(self):
        from sklearn.model_selection import GridSearchCV

        X, y = load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        dtypes = [np.str_, np.int_, str, int]
        estims = {
            "DT": ClassifierWithLabelEncoder2(DecisionTreeClassifier(random_state=0)),
            "RF": ClassifierWithLabelEncoder2(RandomForestClassifier(random_state=0)),
        }
        for clf_name, clf in estims.items():
            for dty in dtypes:
                with self.subTest(clf_name=clf_name, dty=dty):
                    y_t = np.astype(y_train, dty)
                    y_test_dty = np.astype(y_test, dty)
                    y_t_u = np.unique(y_t)

                    param_grid = {
                        'estimator__estimator__max_depth': [3, None],
                        'estimator__estimator__min_samples_split': [2, 5],
                    }

                    pipe = Pipeline(
                        [('trans', RobustScaler()), ('estimator', clf)]
                    )

                    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid)
                    grid_search.fit(X_train, y_t)
                    best_clf = grid_search.best_estimator_
                    y_pred = best_clf.predict(X_test)
                    self.assertTrue(
                        np.isin(y_pred, y_t_u).all(), "Predicted class is not known"
                    )
                    kappa_score = cohen_kappa_score(y_pred, y_test_dty)
                    self.assertTrue(kappa_score >= 0, "Kappa score is negative")



if __name__ == "__main__":
    unittest.main()
