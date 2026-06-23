import numpy as np


def hacer_split_desde_indices(df, train_idx, val_idx):
    """Agarra el dataframe original y los índices de entrenamiento y validación, y 
    devuelve los dataframes correspondientes. Sirve para pasar de indices a dataframes después de hacer un split."""
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    return df_train, df_val


def obtener_indices_random_split_estratificado(
    df,
    target_col='rendimiento_binario',
    test_size=0.2,
    random_state=42
):
    """Genera indices para un split train-validation aleatorio pero manteniendo la proporción de clases en el
    target_col. Es decir, hace un split estratificado."""
    rng = np.random.RandomState(random_state)

    train_indices = []
    val_indices = []

    clases = sorted(df[target_col].dropna().unique())

    for clase in clases:
        indices_clase = np.where(df[target_col].values == clase)[0]
        rng.shuffle(indices_clase)

        n_val = int(len(indices_clase) * test_size)

        if n_val == 0 and len(indices_clase) > 1:
            n_val = 1

        val_idx_clase = indices_clase[:n_val]
        train_idx_clase = indices_clase[n_val:]

        val_indices.extend(val_idx_clase.tolist())
        train_indices.extend(train_idx_clase.tolist())

    train_indices = np.array(sorted(train_indices))
    val_indices = np.array(sorted(val_indices))

    return train_indices, val_indices


def obtener_indices_group_split_por_escuela(df, escuelas_validacion):
    """Separa el dataset en entrenamiento y validación asegurando que las escuelas de validación no estén 
    presentes en el set de entrenamiento."""
    mascara_val = df['escuela'].isin(escuelas_validacion).values

    val_indices = np.where(mascara_val)[0]
    train_indices = np.where(~mascara_val)[0]

    return train_indices, val_indices


def ordenar_lista_semestres(lista_semestres):
    """Ordena semestres del tipo 2022-1, 2022-2, 2023-1, etc. Primero por año y luego por semestre."""
    lista_ordenada = sorted(
        lista_semestres,
        key=lambda x: (int(str(x).split('-')[0]), int(str(x).split('-')[1]))
    )

    return lista_ordenada


def obtener_indices_temporal_split(df, col_semestre='semestre'):
    """Genera un split temporal basado en la columna de semestre. El último semestre se usa para validación y
    el resto para entrenamiento."""
    semestres = ordenar_lista_semestres(df[col_semestre].dropna().unique().tolist())

    semestre_validacion = semestres[-1]
    semestres_entrenamiento = semestres[:-1]

    mascara_val = (df[col_semestre] == semestre_validacion).values
    mascara_train = df[col_semestre].isin(semestres_entrenamiento).values

    train_indices = np.where(mascara_train)[0]
    val_indices = np.where(mascara_val)[0]

    return train_indices, val_indices, semestres_entrenamiento, semestre_validacion


def crear_folds_kfold_aleatorio(df, n_splits=5, random_state=42):
    """Crea folds para cross validation aleatorio, mezclando los indices, dividiendo en n splits y 
    asegurando que cada fold tenga una parte de entrenamiento y validación. Se va a usar para buscar lambda y analizar
    coeficientes con folds aleatorios."""
    rng = np.random.RandomState(random_state)

    indices = np.arange(len(df))
    rng.shuffle(indices)

    folds = np.array_split(indices, n_splits)

    resultados = []

    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])

        resultados.append((train_idx, val_idx))

    return resultados


def crear_folds_group_kfold(df, group_col='escuela', n_splits=4, random_state=42):
    """Genera folds para cross validation pero por grupos, por defecto escuelas. Es decir, asegura que las escuelas
    de validación no estén presentes en el set de entrenamiento."""
    rng = np.random.RandomState(random_state)

    grupos = df[group_col].dropna().unique().tolist()
    grupos = np.array(grupos)

    rng.shuffle(grupos)

    grupos_por_fold = np.array_split(grupos, n_splits)

    resultados = []

    for grupos_val in grupos_por_fold:
        mascara_val = df[group_col].isin(grupos_val).values

        val_idx = np.where(mascara_val)[0]
        train_idx = np.where(~mascara_val)[0]

        resultados.append((train_idx, val_idx))

    return resultados