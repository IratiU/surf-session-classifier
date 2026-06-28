import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer





def clean_training_dataframe(df):
    """
    Limpieza inicial del dataframe completo antes de separar X e y.

    """

    df = df.copy()

    # Eliminar duplicados 
    n_duplicates = df.duplicated().sum()

    if n_duplicates > 0:
        print(f"Duplicados eliminados: {n_duplicates}")
        df = df.drop_duplicates().reset_index(drop=True)

    # El valor 99 en altura_ola_m representa un dato erróneo.
    # Como son pocos casos, eliminamos esas filas.
    n_bad_waves = (df["altura_ola_m"] == 99).sum()

    if n_bad_waves > 0:
        print(f"Filas eliminadas por altura_ola_m = 99: {n_bad_waves}")
        df = df[df["altura_ola_m"] != 99].reset_index(drop=True)

    return df


class SurfDataCleaner(BaseEstimator, TransformerMixin):
    """
    Limpieza básica y tratamiento de valores faltantes.

    Esta clase:
    - convierte fecha de string a datetime;
    - extrae el mes para poder imputar por spot y mes;
    - normaliza los nombres de los spots;
    - limpia variables categóricas;
    - convierte variables numéricas a formato numérico;
    - imputa temp_agua con la media por spot y mes;
    - imputa viento_kmh con la mediana por spot y mes;
    - trata marea y direccion_viento faltantes como categoría "unknown".
    """

    def __init__(self):
        self.temp_mean_by_spot_month_ = None
        self.wind_median_by_spot_month_ = None

        self.global_temp_mean_ = None
        self.global_wind_median_ = None

    def fit(self, X, y=None):
        """
        Aprende las estadísticas necesarias usando solo los datos de entrenamiento.

        Aquí no se imputan los datos.
        Solo se calculan las medias y medianas que luego se usarán en transform().
        """

        X = X.copy()

        # Convertir fecha y extraer mes
        X["fecha"] = pd.to_datetime(X["fecha"], errors="coerce")
        X["mes"] = X["fecha"].dt.month

        # Normalizar spot
        X["spot"] = (
            X["spot"]
            .replace(r"^\s*$", np.nan, regex=True)
            .astype("string")
            .str.strip()
            .str.lower()
        )

        # Convertir temp_agua a numérico
        X["temp_agua"] = (
            X["temp_agua"]
            .replace(r"^\s*$", np.nan, regex=True)
        )
        X["temp_agua"] = pd.to_numeric(X["temp_agua"], errors="coerce")

        # Convertir viento_kmh a numérico
        X["viento_kmh"] = (
            X["viento_kmh"]
            .replace(r"^\s*$", np.nan, regex=True)
        )
        X["viento_kmh"] = pd.to_numeric(X["viento_kmh"], errors="coerce")

        # Aprender media de temp_agua por spot y mes
        self.temp_mean_by_spot_month_ = (
            X.groupby(["spot", "mes"])["temp_agua"]
            .mean()
            .to_dict()
        )

        # Aprender mediana de viento_kmh por spot y mes
        self.wind_median_by_spot_month_ = (
            X.groupby(["spot", "mes"])["viento_kmh"]
            .median()
            .to_dict()
        )

        # Valores globales de respaldo
        self.global_temp_mean_ = X["temp_agua"].mean()
        self.global_wind_median_ = X["viento_kmh"].median()

        return self

    def transform(self, X):
        """
        Aplica la limpieza y las imputaciones aprendidas en fit().
        """

        X = X.copy()

        # Convertir fecha y extraer mes
        X["fecha"] = pd.to_datetime(X["fecha"], errors="coerce")
        X["mes"] = X["fecha"].dt.month

        # Normalizar spot
        X["spot"] = (
            X["spot"]
            .replace(r"^\s*$", np.nan, regex=True)
            .astype("string")
            .str.strip()
            .str.lower()
        )

        # Normalizar marea.
        # Si falta, no inventamos una marea concreta: usamos "unknown".
        X["marea"] = (
            X["marea"]
            .replace(r"^\s*$", np.nan, regex=True)
            .astype("string")
            .str.strip()
            .str.lower()
            .fillna("unknown")
        )

        # Normalizar direccion_viento.
        # Si falta, no inventamos una dirección concreta: usamos "unknown".
        X["direccion_viento"] = (
            X["direccion_viento"]
            .replace(r"^\s*$", np.nan, regex=True)
            .astype("string")
            .str.strip()
            .str.upper()
            .fillna("UNKNOWN")
        )

        numeric_cols = [
            "altura_ola_m",
            "periodo_s",
            "direccion_swell",
            "viento_kmh",
            "temp_agua",
            "valoracion",
        ]

        for col in numeric_cols:
            X[col] = X[col].replace(r"^\s*$", np.nan, regex=True)
            X[col] = pd.to_numeric(X[col], errors="coerce")


        # Imputar los valores faltantes de temp_agua
        missing_temp = X["temp_agua"].isna()

        for idx in X[missing_temp].index:
            key = (X.loc[idx, "spot"], X.loc[idx, "mes"])

            X.loc[idx, "temp_agua"] = self.temp_mean_by_spot_month_.get(
                key,
                self.global_temp_mean_
            )

        # Imputar los valores faltantes de viento_kmh
        missing_wind = X["viento_kmh"].isna()

        for idx in X[missing_wind].index:
            key = (X.loc[idx, "spot"], X.loc[idx, "mes"])

            X.loc[idx, "viento_kmh"] = self.wind_median_by_spot_month_.get(
                key,
                self.global_wind_median_
            )


        # Si aparece algún valor con altura de ola 99 en la predicción
        X.loc[X["altura_ola_m"] == 99, "altura_ola_m"] = np.nan
        

        # Eliminamos las columnas que no necesitamos para el modelo
        X = X.drop(columns=["fecha", "id_sesion", "num_surfistas"], errors="ignore")

        return X
    

def build_preprocessor():
    """
    Construye el preprocesador completo.

    Se puede usar dentro de un Pipeline junto con el modelo.
    """

    numeric_features = [
        "altura_ola_m",
        "periodo_s",
        "direccion_swell",
        "viento_kmh",
        "temp_agua",
        "valoracion",
        "mes",
    ]

    categorical_features = [
        "spot",
        "direccion_viento",
        "marea",
    ]


    # 2. Crear los pasos de preprocesado para cada tipo
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])



    # 3. Unir los transformadores con ColumnTransformer
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # 4. Crear el pipeline final con un modelo
    preprocessor = Pipeline(
        steps=[
            ("cleaner", SurfDataCleaner()),
            ("column_transformer", column_transformer),
        ]
    )
    return preprocessor

