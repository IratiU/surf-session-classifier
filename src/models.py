"""
models.py
==========
Script en el cual se construyen las funciones para cada método.
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


RANDOM_STATE = 42


def build_baseline_model(strategy="most_frequent"):
    """
    Construye el modelo base.

    Por defecto, predice siempre la clase mayoritaria.
    Sirve como referencia mínima para comparar modelos más complejos.
    """
    return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier",  DummyClassifier(strategy=strategy)),
            ])



def build_random_forest_model(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    criterion="gini",
    class_weight="balanced",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
):
    """
    Construye un Random Forest global.

    Se utilizan árboles relativamente poco profundos por defecto para reducir
    el riesgo de sobreajuste y mantener cierta interpretabilidad.
    """

    preprocessor = build_preprocessor()

    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", forest),
        ]
    )

    return model


def get_feature_importance(model):
    """
    Devuelve la importancia de variables de un modelo tipo Pipeline:
    preprocessor + classifier.
    """

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.named_steps[
        "column_transformer"
    ].get_feature_names_out()

    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return importance_df