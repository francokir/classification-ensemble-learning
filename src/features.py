import pandas as pd
import numpy as np


def obtener_categorias_fijas(df, columnas_categoricas):
    """Arma un diccionario donde para cada columna categórica guarda una lista ordenada de categorias posibles."""
    categorias_fijas = {}

    for col in columnas_categoricas:
        categorias_fijas[col] = sorted(df[col].dropna().astype(str).unique().tolist())

    return categorias_fijas


def ajustar_imputacion(df_train, columnas_numericas, columnas_categoricas):
    """Calcula los valores a usar para imputar los datos faltantes en las columnas numéricas y categóricas. Para las numéricas se usa la mediana,
    y para las categóricas se usa la moda."""
    valores_numericos = {}
    valores_categoricos = {}

    for col in columnas_numericas:
        valores_numericos[col] = df_train[col].median()

    for col in columnas_categoricas:
        valores_categoricos[col] = df_train[col].mode()[0]

    return valores_numericos, valores_categoricos


def aplicar_imputacion(df, valores_numericos, valores_categoricos):
    """Aplica la imputación de datos faltantes usando los valores calculados previamente."""
    df = df.copy()

    for col, valor in valores_numericos.items():
        df[col] = df[col].fillna(valor)

    for col, valor in valores_categoricos.items():
        df[col] = df[col].fillna(valor)

    return df


def ajustar_estandarizacion(df_train, columnas_numericas):
    """Calcula para cada variable numerica la media y el desvío estándar a usar para estandarizar los datos. 
    Si el desvío es 0 o NaN, se reemplaza por 1 para evitar divisiones por cero."""
    medias = {}
    desvios = {}

    for col in columnas_numericas:
        medias[col] = df_train[col].mean()
        desvio = df_train[col].std()

        if pd.isna(desvio) or desvio == 0:
            desvio = 1.0

        desvios[col] = desvio

    return medias, desvios


def aplicar_estandarizacion(df, columnas_numericas, medias, desvios):
    """Aplica la estandarización de las variables numéricas usando las medias y desvios calculados previamente."""
    df = df.copy()

    for col in columnas_numericas:
        df[col] = (df[col] - medias[col]) / desvios[col]

    return df


def crear_dummies(df, columnas_categoricas, categorias_fijas=None, drop_first=True):
    """Convierte variables categóricas en variables dummy (one-hot encoding). Si categorias_fijas es None, se calculan las categorías posibles 
    a partir del dataframe. Si drop_first es True, se omite la primera categoría para evitar multicolinealidad."""
    df = df.copy()
    columnas_dummies = []

    for col in columnas_categoricas:
        if categorias_fijas is None:
            categorias = sorted(df[col].dropna().astype(str).unique().tolist())
        else:
            categorias = categorias_fijas[col]

        if drop_first:
            categorias_a_usar = categorias[1:]
        else:
            categorias_a_usar = categorias

        for categoria in categorias_a_usar:
            nombre_columna = col + '_' + str(categoria)
            dummy = (df[col].astype(str) == str(categoria)).astype(int)
            columnas_dummies.append(dummy.rename(nombre_columna))

    if len(columnas_dummies) == 0:
        return pd.DataFrame(index=df.index)

    return pd.concat(columnas_dummies, axis=1)


def preparar_matrices_train_val(
    df_train,
    df_val,
    columnas_numericas,
    columnas_categoricas,
    target_col='rendimiento_binario',
    categorias_fijas=None,
    drop_first=True
):
    """Prepara las matrices de características y los vectores objetivo para los conjuntos de entrenamiento y 
    validación. Aplica imputación, estandarización y creación de dummies, usando los mismos parámetros calculados a 
    partir del conjunto de entrenamiento para ambos conjuntos ya que si no se hace así, se estaría filtrando 
    información del conjunto de validación al conjunto de entrenamiento."""
    
    df_train = df_train.copy()
    df_val = df_val.copy()

    valores_numericos, valores_categoricos = ajustar_imputacion(
        df_train,
        columnas_numericas,
        columnas_categoricas
    )

    df_train = aplicar_imputacion(df_train, valores_numericos, valores_categoricos)
    df_val = aplicar_imputacion(df_val, valores_numericos, valores_categoricos)

    medias, desvios = ajustar_estandarizacion(df_train, columnas_numericas)

    df_train = aplicar_estandarizacion(df_train, columnas_numericas, medias, desvios)
    df_val = aplicar_estandarizacion(df_val, columnas_numericas, medias, desvios)

    X_train_num = df_train[columnas_numericas].copy()
    X_val_num = df_val[columnas_numericas].copy()

    X_train_cat = crear_dummies(
        df_train,
        columnas_categoricas,
        categorias_fijas=categorias_fijas,
        drop_first=drop_first
    )

    X_val_cat = crear_dummies(
        df_val,
        columnas_categoricas,
        categorias_fijas=categorias_fijas,
        drop_first=drop_first
    )

    X_train_df = pd.concat([X_train_num, X_train_cat], axis=1)
    X_val_df = pd.concat([X_val_num, X_val_cat], axis=1)

    feature_names = X_train_df.columns.tolist()

    X_train = X_train_df.values.astype(float)
    X_val = X_val_df.values.astype(float)

    y_train = df_train[target_col].values.copy()
    y_val = df_val[target_col].values.copy()

    return X_train, y_train, X_val, y_val, feature_names