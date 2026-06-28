"""
train.py
========
Script de entrenamiento principal para el modelo de predicción de sesiones de surf.

Flujo:
  1. Carga de datos y limpieza global (duplicados, ola=99)
  2. Split temporal train/test (sin shuffle)
  3. Entrenamiento de baseline y Random Forest
  4. Búsqueda de hiperparámetros con GridSearchCV + TimeSeriesSplit
  5. Evaluación comparativa en test
  6. Persistencia del mejor modelo con joblib

Uso:
  python src/train.py
  python src/train.py --data data/sesiones_surf.csv
  python src/train.py --no-tune          # salta el GridSearch (más rápido)
  python src/train.py --train-size 0.75
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla: seguro en HPC/servidor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_PATH = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_PATH.parent
sys.path.insert(0, str(SRC_PATH))

from preprocessing import clean_training_dataframe
from models import ( 
    build_baseline_model,
    build_random_forest_model,
    get_feature_importance,
)
from evaluation import evaluate_train_test 

# ── Constantes ────────────────────────────────────────────────────────────────
TARGET = "buena_sesion"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data"    / "sesiones_surf.csv"
MODELS_PATH       = PROJECT_ROOT / "models"
FIGURES_PATH      = PROJECT_ROOT / "figures"


# ── Helpers: datos ────────────────────────────────────────────────────────────

def load_and_clean(data_path: Path) -> pd.DataFrame:
    """
    Carga el CSV y aplica la limpieza global:
      - elimina duplicados exactos
      - elimina filas con altura_ola_m == 99 (valor erróneo conocido)
      - ordena por fecha (respeta naturaleza temporal)
    """
    print(f"[INFO] Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] Shape original: {df.shape}")

    df_clean = clean_training_dataframe(df)

    df_clean["fecha"] = pd.to_datetime(df_clean["fecha"], errors="coerce")
    df_clean = df_clean.sort_values("fecha").reset_index(drop=True)

    print(f"[INFO] Shape tras limpieza: {df_clean.shape}")
    return df_clean


def split_features_target(df: pd.DataFrame):
    """Separa predictores (X) y variable objetivo binaria (y: No->0, Yes->1)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError(
            "La variable objetivo contiene valores distintos de 'Yes' y 'No'. "
            "Revisa el CSV."
        )
    return X, y


def temporal_train_test_split(
    X: pd.DataFrame, y: pd.Series, train_size: float = 0.8
):
    """
    Divide los datos respetando el orden temporal.

    Los primeros `train_size` registros (los mas antiguos) van a train;
    el resto (los mas recientes) van a test.
    No se hace shuffle para evitar data leakage temporal.
    """
    split_idx = int(len(X) * train_size)

    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    print(
        f"[INFO] Train: {X_train.shape[0]} muestras  "
        f"({pd.to_datetime(X_train['fecha']).min().date()} -> "
        f"{pd.to_datetime(X_train['fecha']).max().date()})"
    )
    print(
        f"[INFO] Test:  {X_test.shape[0]} muestras  "
        f"({pd.to_datetime(X_test['fecha']).min().date()} -> "
        f"{pd.to_datetime(X_test['fecha']).max().date()})"
    )

    train_dist = y_train.value_counts(normalize=True).rename({0: "No", 1: "Yes"})
    test_dist  = y_test.value_counts(normalize=True).rename({0: "No", 1: "Yes"})
    dist_df = pd.DataFrame({"train": train_dist, "test": test_dist}).round(3)
    print(f"\n[INFO] Distribucion de clases:\n{dist_df.to_string()}\n")

    return X_train, X_test, y_train, y_test


# ── Helpers: modelos ──────────────────────────────────────────────────────────

def tune_random_forest(
    rf_model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    GridSearchCV con TimeSeriesSplit para seleccionar los mejores
    hiperparametros del Random Forest.

    Se usa F1 como metrica de optimizacion: equilibra bien precision y recall
    en un problema donde ambos tipos de error tienen coste real para el usuario
    (falso positivo -> sesion decepcionante; falso negativo -> oportunidad perdida).

    La validacion cruzada es temporal (TimeSeriesSplit) para no contaminar
    el futuro con informacion del pasado durante la busqueda.
    """
    param_grid = {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [None, 8, 12],
        "classifier__min_samples_leaf": [1, 5, 10],
        "classifier__class_weight": ["balanced"],
    }

    temporal_cv = TimeSeriesSplit(n_splits=3)
    f1_scorer   = make_scorer(f1_score, zero_division=0)

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=temporal_cv,
        verbose=1,
    )

    print("[INFO] Iniciando GridSearchCV (puede tardar varios minutos)...")
    grid_search.fit(X_train, y_train)

    print(f"[INFO] Mejores hiperparametros: {grid_search.best_params_}")
    print(
        f"[INFO] Mejor F1 medio en validacion temporal: "
        f"{round(grid_search.best_score_, 3)}"
    )

    cv_df = pd.DataFrame(grid_search.cv_results_)
    top5 = (
        cv_df[[
            "mean_test_score",
            "std_test_score",
            "param_classifier__n_estimators",
            "param_classifier__max_depth",
            "param_classifier__min_samples_leaf",
        ]]
        .sort_values("mean_test_score", ascending=False)
        .head(5)
    )
    print(f"\n[INFO] Top 5 combinaciones:\n{top5.to_string(index=False)}\n")

    return grid_search.best_estimator_


# ── Helpers: visualizaciones ──────────────────────────────────────────────────

def save_confusion_matrix(model, X_test, y_test, name: str) -> None:
    """
    Guarda la matriz de confusion como PNG en figures/.
    Usa backend Agg (sin pantalla) para compatibilidad con HPC/servidor.
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=["No", "Yes"],
        values_format="d",
    )
    disp.ax_.set_title(f"Matriz de confusion - {name}")
    plt.tight_layout()

    out_path = FIGURES_PATH / f"confusion_matrix_{name.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Matriz de confusion guardada en: {out_path}")

def save_model(model, name: str) -> None:
    """Persiste un modelo en models/<name>.pkl con joblib."""
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_PATH / f"{name}.pkl"
    joblib.dump(model, out_path)
    print(f"[INFO] Modelo guardado en: {out_path}")

# ── Separador visual ──────────────────────────────────────────────────────────

def _sep(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ── Pipeline principal ────────────────────────────────────────────────────────

def train(
    data_path: Path = DEFAULT_DATA_PATH,
    tune: bool = True,
    train_size: float = 0.8,
) -> Pipeline:
    """
    Ejecuta el pipeline completo de entrenamiento.

    Todos los modelos se guardan en models/ con su nombre correspondiente:
      - baseline.pkl
      - random_forest.pkl
      - random_forest_tuned.pkl  (solo si tune=True)

    Parametros
    ----------
    data_path : Path
        Ruta al CSV de datos crudos.
    tune : bool
        Si True, ejecuta GridSearchCV para buscar los mejores hiperparametros.
        Si False, usa los parametros iniciales del RF (mucho mas rapido).
    train_size : float
        Fraccion de datos (ordenados temporalmente) destinada a entrenamiento.

    Devuelve
    --------
    best_model : sklearn Pipeline
        Pipeline completo (preprocesador + clasificador) listo para predecir.
    """

    # ── 1. Datos ──────────────────────────────────────────────────────────────
    df = load_and_clean(data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X, y, train_size=train_size
    )

    results = {}

    # ── 2. Baseline ───────────────────────────────────────────────────────────
    baseline = build_baseline_model()
    baseline.fit(X_train, y_train)

    results["Baseline"] = evaluate_train_test(
        baseline, X_train, y_train, X_test, y_test, model_name="Baseline"
    )

    save_model(baseline, "baseline")

    # ── 3. Random Forest con parametros iniciales ─────────────────────────────
    _sep("RANDOM FOREST  (parametros iniciales)")
    rf = build_random_forest_model(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    results["Random Forest"] = evaluate_train_test(
        rf, X_train, y_train, X_test, y_test, model_name="Random Forest"
    )

    save_confusion_matrix(rf, X_test, y_test, name="Random Forest")
    save_model(rf, "random_forest") 
    best_model = rf

    # ── 4. Tuning (opcional) ──────────────────────────────────────────────────
    if tune:
        _sep("RANDOM FOREST TUNING  (GridSearchCV + TimeSeriesSplit)")
        rf_for_tuning = build_random_forest_model()
        best_rf = tune_random_forest(rf_for_tuning, X_train, y_train)

        results["RF tuned"] = evaluate_train_test(
            best_rf, X_train, y_train, X_test, y_test, model_name="RF tuned"
        )

        save_confusion_matrix(best_rf, X_test, y_test, name="RF_tuned")
        save_model(best_rf, "random_forest_tuned")

        best_model = best_rf
    else:
        print("[INFO] Tuning omitido (--no-tune). Se usara el RF con parametros iniciales.")

    # ── 5. Comparativa final en test ──────────────────────────────────────────
    _sep("COMPARATIVA EN TEST")
    comparison = pd.DataFrame(
        {name: res.loc["test"] for name, res in results.items()}
    ).T
    print(comparison.to_string())

    return best_model


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de prediccion de sesiones de surf.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        metavar="PATH",
        help="Ruta al CSV de datos crudos",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Omite el GridSearchCV (entrenamiento significativamente mas rapido)",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        metavar="FLOAT",
        help="Fraccion de datos (en orden temporal) para entrenamiento",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_path=args.data,
        tune=not args.no_tune,
        train_size=args.train_size,
    )