"""
predict.py
==========
Script de inferencia para predecir si una o varias sesiones nuevas serán buenas.

Uso:
  python src/predict.py --input data/nuevas_sesiones.csv
  python src/predict.py --input data/nuevas_sesiones.csv --model models/random_forest_tuned.pkl
  python src/predict.py --input data/nuevas_sesiones.csv --output data/predicciones.csv
"""



import argparse 
import sys
from pathlib import Path

import joblib
import pandas as pd

SRC_PATH = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_PATH.parent
sys.path.insert(0, str(SRC_PATH))


# Rutas del modelo y de los datos nuevos
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_tuned.pkl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predicciones.csv"
TARGET = "buena_sesion"


def load_model(model_path: Path):
    """
    Carga el modelo entrenado guardado con joblib.
    El modelo ya incluye el preprocesador dentro del Pipeline.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el modelo en: {model_path}. "
            "Comprueba la ruta o entrena primero el modelo con train.py."
        )

    return joblib.load(model_path)

def predict_sessions(model, input_path: Path) -> pd.DataFrame:
    """
    Lee un CSV con sesiones nuevas en bruto y devuelve las predicciones.
    """

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)

    X = df.copy()

    if TARGET in X.columns:
        X = X.drop(columns=[TARGET])

    # Predicción normal
    y_pred = model.predict(X)

    # Recuperar el cleaner dentro del pipeline
    cleaner = model.named_steps["preprocessor"].named_steps["cleaner"]

    # Aplicar solo la limpieza interpretable, no el one-hot ni el escalado
    X_clean = cleaner.transform(X)

    predictions = X_clean.copy()
    predictions["prediccion"] = y_pred
    predictions["prediccion_label"] = predictions["prediccion"].map({
        0: "No",
        1: "Yes",
    })

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        if 1 in model.classes_:
            class_1_index = list(model.classes_).index(1)
            predictions["probabilidad_buena_sesion"] = proba[:, class_1_index]

    return predictions




def save_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    """
    Guarda las predicciones en un CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"[INFO] Predicciones guardadas en: {output_path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Predice si nuevas sesiones de surf serán buenas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Ruta al CSV con sesiones nuevas en bruto",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta al modelo entrenado en formato .pkl",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Ruta donde guardar las predicciones",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    model = load_model(args.model)
    predictions = predict_sessions(model, args.input)

    print(predictions[["prediccion_label"]].head())

    save_predictions(predictions, args.output)