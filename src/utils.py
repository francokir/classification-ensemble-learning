import pandas as pd


def mostrar_fragmento_aleatorio(df, n=5, random_state=42):
    """Muestra un fragmento aleatorio del DataFrame para tener una idea de su contenido."""
    return df.sample(n=n, random_state=random_state)


def resumen_dataset(df):
    """Devuelve un resumen del dataset, muestrando el tipo de cada columna, la cantidad de valores nulos, 
    la cantidad de valores únicos, etc."""
    resumen = pd.DataFrame({
        'columna': df.columns,
        'tipo': df.dtypes.astype(str).values,
        'no_nulos': df.notnull().sum().values,
        'nulos': df.isnull().sum().values,
        'n_unicos': df.nunique().values
    })

    return resumen


def tabla_frecuencias(df, col):
    """Devuelve una tabla con las frecuencias y porcentajes de los valores en una columna."""
    conteo = df[col].value_counts(dropna=False)
    porcentaje = df[col].value_counts(normalize=True, dropna=False) * 100

    tabla = pd.DataFrame({
        'frecuencia': conteo,
        'porcentaje': porcentaje
    })

    return tabla


def resumen_numerico(df, columnas_numericas):
    """Devuelve un resumen estadístico de las columnas numéricas."""
    return df[columnas_numericas].describe().T


def medias_por_grupo(df, grupo, columnas_numericas):
    """Devuelve las medias de las columnas numéricas agrupadas por una columna específica."""
    return df.groupby(grupo)[columnas_numericas].mean()


def correlacion_con_target_binario_por_escuela(df, columnas_numericas):
    """Calcula la correlación de cada columna numérica con el target binario 'rendimiento_binario' para cada escuela."""
    resultados = []

    for escuela in sorted(df['escuela'].dropna().unique()):
        df_escuela = df[df['escuela'] == escuela]

        fila = {'escuela': escuela}

        for col in columnas_numericas:
            correlacion = df_escuela[col].corr(df_escuela['rendimiento_binario'])
            fila[col] = correlacion

        resultados.append(fila)

    return pd.DataFrame(resultados).set_index('escuela')
