#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# =========================================================
# Paso 1: Preprocesamiento
# =========================================================
def clean_data(train_df, test_df):

    # Crear Age
    train_df["Age"] = 2021 - train_df["Year"]
    test_df["Age"] = 2021 - test_df["Year"]

    # Eliminar columnas obligatorias
    train_df = train_df.drop(columns=["Year", "Car_Name"])
    test_df = test_df.drop(columns=["Year", "Car_Name"])

    # Separar X e Y
    y_train = train_df["Present_Price"]
    y_test  = test_df["Present_Price"]

    x_train = train_df.drop(columns=["Present_Price"])
    x_test  = test_df.drop(columns=["Present_Price"])


    print("Columns in x_train:", x_train.columns.tolist())
    print("Columns in x_test:", x_test.columns.tolist())

    return x_train, y_train, x_test, y_test


from sklearn.feature_selection import SelectKBest, f_regression
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# =========================================================
# Paso 3: Pipeline
# =========================================================
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def make_pipeline(x_train):

    CATEGORICAL = ["Fuel_Type", "Selling_type", "Transmission"]
    NUMERIC = ["Selling_Price", "Driven_kms", "Owner", "Age"]   # ← CORREGIDO

    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", MinMaxScaler(), NUMERIC),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("kbest", SelectKBest(score_func=f_regression)),
            ("reg", LinearRegression()),
        ]
    )

    return pipeline

# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# =========================================================
# Paso 4: GridSearchCV
# =========================================================
def make_grid_search(pipeline, x_train, y_train):

    # Fit temporal del transformer para calcular cuántas features produce
    transformer = pipeline.named_steps["transformer"]
    transformer.fit(x_train)

    n_features = transformer.transform(x_train).shape[1]

    param_grid = {"kbest__k": [n_features]}

    grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=10,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
        )

    grid.fit(x_train, y_train)
    return grid

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# =========================================================
# Paso 5: Guardar Modelo
# =========================================================
def save_estimator(model):

    os.makedirs("files/models", exist_ok=True)

    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
# =========================================================
# Paso 6: Calcular y Guardar Métricas
# =========================================================
def calculate_metrics(model, x_train, y_train, x_test, y_test):
    from sklearn.metrics import median_absolute_error

    metrics = []

    for x, y, label in [(x_train, y_train, "train"), (x_test, y_test, "test")]:
        y_pred = model.predict(x)

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mad = mean_absolute_error(y, y_pred)
        mad = median_absolute_error(y, y_pred)

        metrics.append(
            {
                "type": "metrics",
                "dataset": label,
                "r2": float(r2),
                "mse": float(mse),
                "mad": float(mad),
            }
        )

    return metrics


def save_metrics(metrics):

    os.makedirs("files/output", exist_ok=True)

    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for row in metrics:
            f.write(json.dumps(row))
            f.write("\n")

# =========================================================
# MAIN
# =========================================================
def main():
    train_df = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

    x_train, y_train, x_test, y_test = clean_data(train_df, test_df)

    pipeline = make_pipeline(x_train)
    model = make_grid_search(pipeline, x_train, y_train)

    save_estimator(model)

    metrics = calculate_metrics(model, x_train, y_train, x_test, y_test)
    save_metrics(metrics)


if __name__ == "__main__":
    main()

