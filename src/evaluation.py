
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Calcula las métricas principales para clasificación binaria.

    La clase positiva es buena_sesion = Yes, codificada como 1.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_yes": precision_score(y_true, y_pred, zero_division=0),
        "recall_yes": recall_score(y_true, y_pred, zero_division=0),
        "f1_yes": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None and y_true.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics

def evaluate_model(model, X, y, model_name):
    """
    Evalúa un modelo entrenado sobre un conjunto de datos.

    Devuelve un diccionario con las métricas calculadas.
    """

    y_pred = model.predict(X)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]

    metrics = compute_metrics(y, y_pred, y_proba)

    print(f"===== {model_name} =====")
    print(f"Accuracy:          {metrics['accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Precision Yes:     {metrics['precision_yes']:.3f}")
    print(f"Recall Yes:        {metrics['recall_yes']:.3f}")
    print(f"F1 Yes:            {metrics['f1_yes']:.3f}")

    if "roc_auc" in metrics:
        print(f"ROC AUC:           {metrics['roc_auc']:.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=["No", "Yes"],
            zero_division=0,
        )
    )

    return metrics



def plot_roc_curve(model, X, y, model_name):
    """
    Representa la curva ROC del modelo.

    Solo se calcula si el modelo permite obtener probabilidades
    y si el conjunto evaluado contiene las dos clases.
    """

    if not hasattr(model, "predict_proba"):
        print(f"{model_name} no permite calcular probabilidades.")
        return

    if y.nunique() < 2:
        print("No se puede calcular la curva ROC porque solo hay una clase.")
        return

    RocCurveDisplay.from_estimator(model, X, y)
    plt.title(f"Curva ROC - {model_name}")
    plt.tight_layout()
    plt.show()


def evaluate_train_test(model, X_train, y_train, X_test, y_test, model_name):
    """
    Evalúa un modelo en train y test y devuelve una tabla comparativa.
    """

    train_metrics = evaluate_model(
        model,
        X_train,
        y_train,
        model_name=f"{model_name} - Train",
    )

    print("\n")

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        model_name=f"{model_name} - Test",
    )

    comparison = pd.DataFrame(
        {
            "train": train_metrics,
            "test": test_metrics,
        }
    ).T

    return comparison.round(3)



def plot_confusion_matrix(model, X, y, model_name):
    """
    Representa la matriz de confusión del modelo.
    """

    y_pred = model.predict(X)

    disp = ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred,
        labels=[0, 1],
        display_labels=["No", "Yes"],
        values_format="d",
    )

    plt.title(f"Matriz de confusión - {model_name}")
    plt.tight_layout()
    plt.show()

def compare_model_results(results_dict):
    """
    Construye una tabla comparativa a partir de varios resultados.

    Parámetros
    ----------
    results_dict : dict
        Diccionario con formato:
        {
            "Baseline": baseline_metrics,
            "Decision Tree": tree_metrics,
            ...
        }
    """

    return pd.DataFrame(results_dict).T.round(3)
