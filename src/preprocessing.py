import numpy as np
import pandas as pd


def cargar_dataset(path):
    """Carga el dataset desde un archivo CSV."""
    return pd.read_csv(path)


def obtener_columnas():
    """Devuelve las listas de columnas numéricas, categóricas y el nombre del target."""
    columnas_numericas = [
        'horas_estudio',
        'asistencia',
        'nota_previa',
        'horas_sueno',
        'participacion',
        'horas_extracurricular',
        'acceso_internet',
        'distancia_escuela_km',
        'nivel_socioeconomico',
        'tamano_clase'
    ]

    columnas_categoricas = ['escuela', 'semestre']
    target = 'rendimiento'

    return columnas_numericas, columnas_categoricas, target


def crear_target_binario(df):
    """Crea una nueva columna 'rendimiento_binario' a partir de 'rendimiento'.0 = Insuficiente, 1 = Aprobado."""
    df = df.copy()

    df['rendimiento_binario'] = df['rendimiento'].apply(
        lambda x: 0 if x == 'Insuficiente' else 1
    )

    return df


def normalizar_escuela(df):
    """Limpia la columna 'escuela' convirtiendo a mayúsculas y eliminando espacios."""
    df = df.copy()

    df['escuela'] = df['escuela'].astype(str).str.upper().str.strip()

    return df


def corregir_nota_previa(df):
    """Corrige la columna 'nota_previa' dividiendo por 10 los valores mayores a 10, para que esten todos en la escala de 0 a 10."""
    df = df.copy()

    mascara = df['nota_previa'] > 10
    df.loc[mascara, 'nota_previa'] = df.loc[mascara, 'nota_previa'] / 10

    return df


def corregir_asistencia(df):
    """Corrige la columna 'asistencia' multiplicando por 100 los valores entre 0 y 1, para que esten todos en la escala de 0 a 100."""
    df = df.copy()

    mascara = (df['asistencia'] >= 0) & (df['asistencia'] < 1)
    df.loc[mascara, 'asistencia'] = df.loc[mascara, 'asistencia'] * 100

    return df


def resumen_faltantes(df):
    """Devuelve un resumen de las columnas con valores faltantes, indicando la cantidad y el porcentaje de faltantes."""
    faltantes = df.isnull().sum()
    porcentaje = (faltantes / len(df)) * 100

    resumen = pd.DataFrame({
        'faltantes': faltantes,
        'porcentaje': porcentaje
    })

    resumen = resumen[resumen['faltantes'] > 0].sort_values(by='faltantes', ascending=False)

    return resumen


def validar_rangos(df):
    """Chequea que las columnas numéricas estén dentro de rangos esperados y devuelve un resumen con la cantidad de valores fuera de rango."""
    resultados = []

    reglas = {
        'asistencia': (0, 100),
        'nota_previa': (0, 10),
        'horas_sueno': (0, 24),
        'participacion': (0, 10),
        'nivel_socioeconomico': (0, 1),
        'acceso_internet': (0, 1),
        'distancia_escuela_km': (0, np.inf),
        'tamano_clase': (0, np.inf),
        'horas_estudio': (0, np.inf),
        'horas_extracurricular': (0, np.inf)
    }

    for col, (minimo, maximo) in reglas.items():
        if col in df.columns:
            fuera_de_rango = ((df[col] < minimo) | (df[col] > maximo)).sum()
            resultados.append({
                'columna': col,
                'min_esperado': minimo,
                'max_esperado': maximo,
                'cantidad_fuera_de_rango': fuera_de_rango
            })

    return pd.DataFrame(resultados)


def resumen_outliers_iqr(df, columnas_numericas):
    """Para cada columna numérica, calcula el rango intercuartílico (IQR) y cuenta la cantidad de valores que se consideran outliers
    según la regla de Tukey (valores menores a Q1 - 1.5*IQR o mayores a Q3 + 1.5*IQR). Devuelve un resumen ordenado por cantidad de outliers."""
    resultados = []

    for col in columnas_numericas:
        serie = df[col].dropna()

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        cantidad = ((df[col] < limite_inferior) | (df[col] > limite_superior)).sum()

        resultados.append({
            'columna': col,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'limite_inferior': limite_inferior,
            'limite_superior': limite_superior,
            'cantidad_outliers': cantidad
        })

    return pd.DataFrame(resultados).sort_values(by='cantidad_outliers', ascending=False)


def limpieza_basica(df):
    """Realiza una limpieza básica del dataset aplicando las funciones de normalización y corrección definidas anteriormente."""
    df = df.copy()

    df = normalizar_escuela(df)
    df = corregir_nota_previa(df)
    df = corregir_asistencia(df)
    df = crear_target_binario(df)

    return df


def obtener_clase_minoritaria_y_mayoritaria(y):
    """Dada una serie de etiquetas binarias, devuelve cuál es la clase minoritaria y cuál es la mayoritaria."""
    clases, conteos = np.unique(y, return_counts=True)

    clase_minoritaria = clases[np.argmin(conteos)]
    clase_mayoritaria = clases[np.argmax(conteos)]

    return clase_minoritaria, clase_mayoritaria


def undersampling_random(X, y, random_state=42):
    """Realiza undersampling aleatorio de la clase mayoritaria para balancear el dataset."""
    rng = np.random.RandomState(random_state)

    clase_minoritaria, clase_mayoritaria = obtener_clase_minoritaria_y_mayoritaria(y)

    idx_min = np.where(y == clase_minoritaria)[0]
    idx_maj = np.where(y == clase_mayoritaria)[0]

    if len(idx_min) == len(idx_maj):
        return X.copy(), y.copy()

    idx_maj_sub = rng.choice(idx_maj, size=len(idx_min), replace=False)
    idx_final = np.concatenate([idx_min, idx_maj_sub])
    rng.shuffle(idx_final)

    return X[idx_final], y[idx_final]


def oversampling_duplicate(X, y, random_state=42):
    """Realiza oversampling de la clase minoritaria duplicando aleatoriamente muestras existentes para balancear el dataset."""
    rng = np.random.RandomState(random_state)

    clase_minoritaria, clase_mayoritaria = obtener_clase_minoritaria_y_mayoritaria(y)

    idx_min = np.where(y == clase_minoritaria)[0]
    idx_maj = np.where(y == clase_mayoritaria)[0]

    if len(idx_min) == len(idx_maj):
        return X.copy(), y.copy()

    n_faltan = len(idx_maj) - len(idx_min)
    idx_min_extra = rng.choice(idx_min, size=n_faltan, replace=True)

    X_extra = X[idx_min_extra]
    y_extra = y[idx_min_extra]

    X_bal = np.vstack([X, X_extra])
    y_bal = np.concatenate([y, y_extra])

    idx_final = np.arange(len(y_bal))
    rng.shuffle(idx_final)

    return X_bal[idx_final], y_bal[idx_final]


def smote_oversampling(X, y, k_neighbors=5, random_state=42):
    """Genera muestras sintéticas de la clase minoritaria utilizando el algoritmo SMOTE para balancear el dataset. Basicamente 
    crea nuevas muestras interpolando entre muestras minoritarias existentes y sus vecinos más cercanos."""
    rng = np.random.RandomState(random_state)

    clase_minoritaria, clase_mayoritaria = obtener_clase_minoritaria_y_mayoritaria(y)

    idx_min = np.where(y == clase_minoritaria)[0]
    idx_maj = np.where(y == clase_mayoritaria)[0]

    if len(idx_min) == len(idx_maj):
        return X.copy(), y.copy()

    X_min = X[idx_min]
    n_min = len(X_min)
    n_maj = len(idx_maj)

    if n_min < 2:
        return oversampling_duplicate(X, y, random_state=random_state)

    k = min(k_neighbors, n_min - 1)
    n_sinteticas = n_maj - n_min

    muestras_sinteticas = []

    for _ in range(n_sinteticas):
        i = rng.randint(0, n_min)
        xi = X_min[i]

        distancias = np.sqrt(np.sum((X_min - xi) ** 2, axis=1))
        vecinos = np.argsort(distancias)[1:k+1]

        if len(vecinos) == 0:
            xj = xi
        else:
            vecino_elegido = rng.choice(vecinos)
            xj = X_min[vecino_elegido]

        alpha = rng.rand()
        x_new = xi + alpha * (xj - xi)

        muestras_sinteticas.append(x_new)

    X_syn = np.array(muestras_sinteticas)
    y_syn = np.full(len(X_syn), clase_minoritaria)

    X_bal = np.vstack([X, X_syn])
    y_bal = np.concatenate([y, y_syn])

    idx_final = np.arange(len(y_bal))
    rng.shuffle(idx_final)

    return X_bal[idx_final], y_bal[idx_final]


def factor_cost_reweighting(y):
    """Calcula el factor C = P(clase mayoritaria) / P(clase minoritaria) para usar como peso en técnicas de reweighting de costo,
    donde las muestras de la clase minoritaria se ponderan más para compensar el desequilibrio."""
    clase_minoritaria, clase_mayoritaria = obtener_clase_minoritaria_y_mayoritaria(y)

    n_min = np.sum(y == clase_minoritaria)
    n_maj = np.sum(y == clase_mayoritaria)

    pi_1 = n_min / len(y)
    pi_2 = n_maj / len(y)

    C = pi_2 / pi_1

    return clase_minoritaria, clase_mayoritaria, C