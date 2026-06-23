import numpy as np
import pandas as pd

from preprocessing import undersampling_random, oversampling_duplicate, smote_oversampling, factor_cost_reweighting
from features import preparar_matrices_train_val
from models import RegresionLogisticaBinaria, LogisticRegressionBinaryWeighted, RandomForestClassifierManual
from metrics import evaluar_modelo_binario, evaluar_modelo_multiclase
from data_splitting import (
    obtener_indices_temporal_split,
    crear_folds_kfold_aleatorio,
    crear_folds_group_kfold
)


def entrenar_y_evaluar_logistica(
    df_train,
    df_val,
    columnas_numericas,
    columnas_categoricas,
    lambda_reg=0.1,
    learning_rate=0.01,
    n_iter=6000,
    threshold=0.5,
    categorias_fijas=None
):
    """Prepara los datos, entrena un modelo de regresión logística binaria y evalúa su desempeño en el conjunto de validación. Esta funcion NO define el split, recibe
    los dataframes de train y validacion ya definidos."""
    X_train, y_train, X_val, y_val, feature_names = preparar_matrices_train_val(
        df_train,
        df_val,
        columnas_numericas,
        columnas_categoricas,
        target_col='rendimiento_binario',
        categorias_fijas=categorias_fijas,
        drop_first=True
    )

    modelo = RegresionLogisticaBinaria(
        learning_rate=learning_rate,
        n_iter=n_iter,
        lambda_reg=lambda_reg
    )

    modelo.fit(X_train, y_train)

    y_prob = modelo.predict_proba(X_val)

    resultado = evaluar_modelo_binario(y_val, y_prob, threshold=threshold)

    resultado['modelo'] = modelo
    resultado['feature_names'] = feature_names
    resultado['coeficientes'] = pd.Series(modelo.w, index=feature_names)
    resultado['intercepto'] = modelo.b

    return resultado


def evaluar_lambdas_en_folds(
    df,
    folds,
    columnas_numericas,
    columnas_categoricas,
    lambdas,
    learning_rate=0.01,
    n_iter=6000,
    threshold=0.5,
    categorias_fijas=None
):
    """Evalua una grilla de lambdas sobre varios folds y resume el F1 promedio de validacion para cada uno de los lambdas."""
    filas = []

    for lambda_reg in lambdas:
        f1_folds = []

        for train_idx, val_idx in folds:
            df_train = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy()

            resultado = entrenar_y_evaluar_logistica(
                df_train,
                df_val,
                columnas_numericas,
                columnas_categoricas,
                lambda_reg=lambda_reg,
                learning_rate=learning_rate,
                n_iter=n_iter,
                threshold=threshold,
                categorias_fijas=categorias_fijas
            )

            f1_folds.append(resultado['f1'])

        filas.append({
            'lambda': lambda_reg,
            'f1_promedio': float(np.nanmean(f1_folds)),
            'f1_std': float(np.nanstd(f1_folds))
        })

    tabla = pd.DataFrame(filas).sort_values(by='lambda').reset_index(drop=True)
    mejor_lambda = tabla.loc[tabla['f1_promedio'].idxmax(), 'lambda']

    return tabla, mejor_lambda


def buscar_mejor_lambda_random_cv(
    df,
    columnas_numericas,
    columnas_categoricas,
    lambdas,
    n_splits=5,
    random_state=42,
    learning_rate=0.01,
    n_iter=6000,
    threshold=0.5,
    categorias_fijas=None
):
    """Busca el mejor lambda para la regresion logistica binaria utilizando cross validation aleatoria (KFold).
    Crea los folds aleatorios con la funcion crear_folds_kfold_aleatorio y luego evalua cada lambda en cada fold utilizando la funcion evaluar_lambdas_en_folds."""
    folds = crear_folds_kfold_aleatorio(
        df,
        n_splits=n_splits,
        random_state=random_state
    )

    return evaluar_lambdas_en_folds(
        df,
        folds,
        columnas_numericas,
        columnas_categoricas,
        lambdas,
        learning_rate=learning_rate,
        n_iter=n_iter,
        threshold=threshold,
        categorias_fijas=categorias_fijas
    )


def buscar_mejor_lambda_group_cv(
    df,
    columnas_numericas,
    columnas_categoricas,
    lambdas,
    n_splits=4,
    random_state=42,
    learning_rate=0.01,
    n_iter=6000,
    threshold=0.5,
    categorias_fijas=None
):
    """Busca el mejor lambda para la regresion logistica binaria utilizando cross validation por grupos (GroupKFold).
    Crea los folds por grupos con la funcion crear_folds_group_kfold y luego evalua cada lambda en cada fold utilizando la funcion evaluar_lambdas_en_folds."""
    folds = crear_folds_group_kfold(
        df,
        group_col='escuela',
        n_splits=n_splits,
        random_state=random_state
    )

    return evaluar_lambdas_en_folds(
        df,
        folds,
        columnas_numericas,
        columnas_categoricas,
        lambdas,
        learning_rate=learning_rate,
        n_iter=n_iter,
        threshold=threshold,
        categorias_fijas=categorias_fijas
    )


def buscar_mejor_lambda_temporal(
    df,
    columnas_numericas,
    columnas_categoricas,
    lambdas,
    learning_rate=0.01,
    n_iter=6000,
    threshold=0.5,
    categorias_fijas=None
):
    """Busca el mejor lambda para la regresion logistica binaria utilizando un split temporal. El split se define con la funcion obtener_indices_temporal_split,
    que devuelve los indices de train y validacion (aca el fold es solo 1). Luego evalua cada lambda en ese split utilizando la funcion evaluar_lambdas_en_folds."""

    train_idx, val_idx, _, _ = obtener_indices_temporal_split(df)

    folds = [(train_idx, val_idx)]

    return evaluar_lambdas_en_folds(
        df,
        folds,
        columnas_numericas,
        columnas_categoricas,
        lambdas,
        learning_rate=learning_rate,
        n_iter=n_iter,
        threshold=threshold,
        categorias_fijas=categorias_fijas
    )


def obtener_coeficientes_en_folds(
    df,
    folds,
    columnas_numericas,
    columnas_categoricas,
    lambda_reg,
    learning_rate=0.01,
    n_iter=6000,
    threshold=0.5,
    categorias_fijas=None
):
    """Entrena la logistica en varios folds y guarda los coeficientes de cada fold en una tabla. 
    Esto sirve para analizar la estabilidad de los coeficientes, viendo como varian entre folds. Recibe los folds ya definidos y el lambda a usar."""
    coeficientes_folds = []

    for i, (train_idx, val_idx) in enumerate(folds, start=1):
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        resultado = entrenar_y_evaluar_logistica(
            df_train,
            df_val,
            columnas_numericas,
            columnas_categoricas,
            lambda_reg=lambda_reg,
            learning_rate=learning_rate,
            n_iter=n_iter,
            threshold=threshold,
            categorias_fijas=categorias_fijas
        )

        serie = resultado['coeficientes'].copy()
        serie.name = 'fold_' + str(i)

        coeficientes_folds.append(serie)

    tabla = pd.DataFrame(coeficientes_folds)

    return tabla


def obtener_coeficientes_random_cv(
    df,
    columnas_numericas,
    columnas_categoricas,
    lambda_reg,
    n_splits=5,
    random_state=42,
    learning_rate=0.01,
    n_iter=6000,
    threshold=0.5,
    categorias_fijas=None
):
    """Construye folds aleatorios con la funcion crear_folds_kfold_aleatorio y luego entrena la logistica en cada fold con el lambda dado, 
    guardando los coeficientes de cada fold en una tabla con obtener_coeficientes_en_folds. Basicamente obtiene los coeficientes de la logistica bajo folds aleatorios."""
    folds = crear_folds_kfold_aleatorio(
        df,
        n_splits=n_splits,
        random_state=random_state
    )

    return obtener_coeficientes_en_folds(
        df,
        folds,
        columnas_numericas,
        columnas_categoricas,
        lambda_reg,
        learning_rate=learning_rate,
        n_iter=n_iter,
        threshold=threshold,
        categorias_fijas=categorias_fijas
    )


def obtener_coeficientes_group_cv(
    df,
    columnas_numericas,
    columnas_categoricas,
    lambda_reg,
    n_splits=4,
    random_state=42,
    learning_rate=0.01,
    n_iter=6000,
    threshold=0.5,
    categorias_fijas=None
):
    """Construye folds por escuela con la funcion crear_folds_group_kfold y luego entrena la logistica en cada fold con el lambda dado,
    guardando los coeficientes de cada fold en una tabla con obtener_coeficientes_en_folds. Basicamente obtiene los coeficientes de la logistica bajo folds por grupos (escuelas).
    """
   
    folds = crear_folds_group_kfold(
        df,
        group_col='escuela',
        n_splits=n_splits,
        random_state=random_state
    )

    return obtener_coeficientes_en_folds(
        df,
        folds,
        columnas_numericas,
        columnas_categoricas,
        lambda_reg,
        learning_rate=learning_rate,
        n_iter=n_iter,
        threshold=threshold,
        categorias_fijas=categorias_fijas
    )


def comparar_estabilidad_coeficientes(tabla_random, tabla_group):
    """Compara el desvio estandar de cada coeficiente entre los folds aleatorios y los folds por grupos. 
    Esto sirve para analizar si los coeficientes son mas estables (menor desviacion) bajo un tipo de fold u otro."""

    std_random = tabla_random.std()
    std_group = tabla_group.std()

    comparacion = pd.DataFrame({
        'std_random': std_random,
        'std_group': std_group
    })

    comparacion['diferencia'] = comparacion['std_group'] - comparacion['std_random']
    comparacion = comparacion.sort_values(by='diferencia', ascending=False)

    return comparacion







# Funciones de Validacion para Clasificacion Multiclase


def codificar_labels(y, class_names):
    """Convierte etiquetas de clase a indices enteros basados en el orden de class_names. Por ejemplo Insuficiente -> 0, Regular -> 1. """
    mapa = {clase: i for i, clase in enumerate(class_names)}
    return np.array([mapa[valor] for valor in y], dtype=int)


def cross_validar_modelo_group_kfold(
    df,
    model_factory,
    columnas_numericas,
    columnas_categoricas,
    target_col='rendimiento',
    group_col='escuela',
    n_splits=4,
    random_state=42,
    categorias_fijas=None,
    class_names=None
):
    """Hace cross validation multiclase por grupos, evalua las predicciones out-of-fold y calcula metricas globales y por clase.
    Recibe una funcion model_factory que devuelve un modelo sin entrenar, los datos, las columnas a usar, el target, el grupo para los folds,
    el numero de splits, la semilla, las categorias fijas (para asegurar mismo orden en train y val) y los nombres de clase (para codificar labels). 
    Devuelve un diccionario con resultados y metricas. Cabe aclarar que es buena practica porque no promedia metricas fold por fold, reconstruye 
    las predicciones out-of-fold para todo el dataset y evalua una vez sobre ese conjunto."""

    if class_names is None:
        class_names = sorted(df[target_col].dropna().unique().tolist())

    folds = crear_folds_group_kfold(
        df,
        group_col=group_col,
        n_splits=n_splits,
        random_state=random_state
    )

    n_samples = len(df)
    n_classes = len(class_names)

    #Inicializa estructuras para guardar resultados out-of-fold
    y_true_full = codificar_labels(df[target_col].values, class_names)
    y_pred_oof = np.full(n_samples, -1, dtype=int)
    y_prob_oof = np.zeros((n_samples, n_classes))

    feature_names_ref = None
    importancias_folds = []

    for train_idx, val_idx in folds:
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        X_train, y_train_raw, X_val, y_val_raw, feature_names = preparar_matrices_train_val(
            df_train,
            df_val,
            columnas_numericas,
            columnas_categoricas,
            target_col=target_col,
            categorias_fijas=categorias_fijas,
            drop_first=True
        )

        y_train = codificar_labels(y_train_raw, class_names)

        # Entrena el modelo en el fold actual y guarda predicciones sobre validación.
        model = model_factory()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        y_pred_oof[val_idx] = y_pred
        y_prob_oof[val_idx] = y_prob

        feature_names_ref = feature_names

        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            importancias_folds.append(model.feature_importances_)

    # Evalúa el desempeño global usando todas las predicciones OOF.
    resultado = evaluar_modelo_multiclase(
        y_true_full,
        y_pred_oof,
        y_prob_oof,
        class_names
    )

    resultado['y_true'] = y_true_full
    resultado['y_pred'] = y_pred_oof
    resultado['y_prob'] = y_prob_oof
    resultado['feature_names'] = feature_names_ref
    resultado['class_names'] = class_names

    if len(importancias_folds) > 0:
        importancias_promedio = np.mean(np.vstack(importancias_folds), axis=0)

        if importancias_promedio.sum() > 0:
            importancias_promedio = importancias_promedio / importancias_promedio.sum()

        resultado['feature_importances'] = pd.Series(
            importancias_promedio,
            index=feature_names_ref
        ).sort_values(ascending=False)

    return resultado


def buscar_mejor_config_random_forest_group_kfold(
    df,
    configs,
    model_class,
    columnas_numericas,
    columnas_categoricas,
    target_col='rendimiento',
    group_col='escuela',
    n_splits=4,
    random_state=42,
    categorias_fijas=None,
    class_names=None
):
    """Evalua distintas configuraciones de RF y elige la mejor segun macro_F1. Recorre configuraciones creando un model factory para cada una,
    llama a cross_validar_modelo_group_kfold para obtener resultados OOF y metricas, guarda los resultados en una tabla y devuelve la tabla ordenada por macro_F1 y
    la mejor configuracion."""

    filas = []

    for config in configs:
        def model_factory(config=config):
            return model_class(**config)

        resultado = cross_validar_modelo_group_kfold(
            df,
            model_factory,
            columnas_numericas,
            columnas_categoricas,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits,
            random_state=random_state,
            categorias_fijas=categorias_fijas,
            class_names=class_names
        )

        resumen = resultado['tabla_resumen'].iloc[0].to_dict()

        fila = config.copy()
        fila['accuracy_global'] = resumen['accuracy_global']
        fila['macro_precision'] = resumen['macro_precision']
        fila['macro_recall'] = resumen['macro_recall']
        fila['macro_f1'] = resumen['macro_f1']
        fila['macro_auc_roc'] = resumen['macro_auc_roc']
        fila['macro_auc_pr'] = resumen['macro_auc_pr']

        filas.append(fila)

    tabla = pd.DataFrame(filas).sort_values(by='macro_f1', ascending=False).reset_index(drop=True)
    mejor_config = tabla.iloc[0].to_dict()

    return tabla, mejor_config


def armar_tabla_comparativa_modelos(resultados_modelos):
    """Arma una tabla comparativa con metricas globales de cada modelo. Recibe un diccionario con resultados de cada modelo (donde cada resultado tiene una tabla_resumen 
    con las metricas globales) y devuelve una tabla con las metricas globales de cada modelo para compararlos facilmente."""

    filas = []

    for nombre_modelo, resultado in resultados_modelos.items():
        resumen = resultado['tabla_resumen'].iloc[0].to_dict()
        resumen['modelo'] = nombre_modelo
        filas.append(resumen)

    tabla = pd.DataFrame(filas)

    columnas = [
        'modelo',
        'accuracy_global',
        'macro_precision',
        'macro_recall',
        'macro_f1',
        'macro_auc_roc',
        'macro_auc_pr'
    ]

    return tabla[columnas]


def armar_tabla_f1_por_clase_modelos(resultados_modelos):
    """Arma una tabla de F1 por clase para comparar modelos multiclase."""

    filas = []

    for nombre_modelo, resultado in resultados_modelos.items():
        tabla = resultado['tabla_por_clase'][['clase', 'f1']].copy()
        tabla['modelo'] = nombre_modelo
        filas.append(tabla)

    return pd.concat(filas, ignore_index=True)



def aplicar_rebalanceo(X_train, y_train, tecnica, random_state=42):
    """Aplica la tecnica de rebalanceo elegida a los datos de entrenamiento. Recibe las matrices de entrenamiento, la tecnica a aplicar 
    y la semilla para reproducibilidad. Siempre devuelve las matrices de entrenamiento rebalanceadas (o sin rebalancear si la tecnica es 'sin_rebalanceo') y 
    un vector de sample_weight que es None para las tecnicas que no lo requieren y tiene los pesos para cada muestra en el caso de cost re-weighting."""

    if tecnica == 'sin_rebalanceo':
        return X_train, y_train, None

    if tecnica == 'undersampling':
        X_bal, y_bal = undersampling_random(X_train, y_train, random_state=random_state)
        return X_bal, y_bal, None

    if tecnica == 'oversampling_duplicate':
        X_bal, y_bal = oversampling_duplicate(X_train, y_train, random_state=random_state)
        return X_bal, y_bal, None

    if tecnica == 'smote':
        X_bal, y_bal = smote_oversampling(X_train, y_train, k_neighbors=5, random_state=random_state)
        return X_bal, y_bal, None

    if tecnica == 'cost_reweighting':
        clase_minoritaria, clase_mayoritaria, C = factor_cost_reweighting(y_train)

        sample_weight = np.ones(len(y_train), dtype=float)
        sample_weight[y_train == clase_minoritaria] = C

        return X_train, y_train, sample_weight

    raise ValueError('Técnica de rebalanceo no reconocida')


def cross_validar_rebalanceo_group_kfold(
    df,
    tecnica,
    columnas_numericas,
    columnas_categoricas,
    target_col='rendimiento_binario',
    group_col='escuela',
    n_splits=4,
    random_state=42,
    categorias_fijas=None,
    lambda_reg=0.1,
    learning_rate=0.05,
    n_iter=5000,
    threshold=0.5,
    evaluar_minoritaria=False
):
    """Evalua una tecnica de rebalanceo utilizando cross validation por grupos (GroupKFold). Crea los folds por grupos con la funcion crear_folds_group_kfold, 
    luego en cada fold aplica la tecnica de rebalanceo. Se puede reorientar la evaluacion para enfocarse en la clase minoritaria invirtiendo las etiquetas y las probabilidades. 
    Finalmente reconstruye las predicciones out-of-fold para todo el dataset"""

    folds = crear_folds_group_kfold(
        df,
        group_col=group_col,
        n_splits=n_splits,
        random_state=random_state
    )

    y_true_full = df[target_col].values.astype(int)
    y_prob_oof = np.zeros(len(df))

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        X_train, y_train, X_val, y_val, feature_names = preparar_matrices_train_val(
            df_train,
            df_val,
            columnas_numericas,
            columnas_categoricas,
            target_col=target_col,
            categorias_fijas=categorias_fijas,
            drop_first=True
        )

        y_train = y_train.astype(int)
        y_val = y_val.astype(int)

        X_fit, y_fit, sample_weight = aplicar_rebalanceo(
            X_train,
            y_train,
            tecnica=tecnica,
            random_state=random_state + fold_id
        )

        modelo = LogisticRegressionBinaryWeighted(
            learning_rate=learning_rate,
            n_iter=n_iter,
            lambda_reg=lambda_reg,
            tol=1e-7,
            verbose=False
        )

        modelo.fit(X_fit, y_fit, sample_weight=sample_weight)

        y_prob = modelo.predict_proba(X_val)
        y_prob_oof[val_idx] = y_prob

    if evaluar_minoritaria:
        y_true_eval = 1 - y_true_full
        y_prob_eval = 1 - y_prob_oof
    else:
        y_true_eval = y_true_full
        y_prob_eval = y_prob_oof

    resultado = evaluar_modelo_binario(y_true_eval, y_prob_eval, threshold=threshold)
    resultado['tecnica'] = tecnica

    return resultado


def tabla_resultados_rebalanceo(resultados_dict):
    """Resume en una tabla comparativa las metricas globales de cada tecnica de rebalanceo. Recibe un diccionario con los resultados de cada tecnica 
    (donde cada resultado tiene una tabla_resumen con las metricas globales) y devuelve una tabla con las metricas globales de cada tecnica para compararlas facilmente."""
    filas = []

    nombres = {
        'sin_rebalanceo': 'Sin rebalanceo',
        'undersampling': 'Undersampling',
        'oversampling_duplicate': 'Oversampling duplicate',
        'smote': 'Oversampling SMOTE',
        'cost_reweighting': 'Cost re-weighting'
    }

    for tecnica, resultado in resultados_dict.items():
        filas.append({
            'Modelo': nombres.get(tecnica, tecnica),
            'Accuracy': resultado['accuracy'],
            'Precision': resultado['precision'],
            'Recall': resultado['recall'],
            'F1-Score': resultado['f1'],
            'AUC-ROC': resultado['auc_roc'],
            'AUC-PR': resultado['auc_pr']
        })

    return pd.DataFrame(filas)





def entrenar_final_y_evaluar_binario(
    df_dev,
    df_test,
    tecnica,
    columnas_numericas,
    columnas_categoricas,
    target_col='rendimiento_binario',
    categorias_fijas=None,
    lambda_reg=0.1,
    learning_rate=0.05,
    n_iter=5000,
    threshold=0.5,
    evaluar_minoritaria=False,
    random_state=42
):
    """Entrena el pipeline binario final con todo dev y evalua sobre test. Tambien si se quiere evaluar la clase minoritaria, invierte las etiquetas y probabilidades para 
    enfocarse en esa clase, tomando a la clase minoritaria como positiva. Devuelve un diccionario con resultados y metricas."""
    X_dev, y_dev, X_test, y_test, feature_names = preparar_matrices_train_val(
        df_dev,
        df_test,
        columnas_numericas,
        columnas_categoricas,
        target_col=target_col,
        categorias_fijas=categorias_fijas,
        drop_first=True
    )

    y_dev = y_dev.astype(int)
    y_test = y_test.astype(int)

    X_fit, y_fit, sample_weight = aplicar_rebalanceo(
        X_dev,
        y_dev,
        tecnica=tecnica,
        random_state=random_state
    )

    modelo = LogisticRegressionBinaryWeighted(
        learning_rate=learning_rate,
        n_iter=n_iter,
        lambda_reg=lambda_reg,
        tol=1e-7,
        verbose=False
    )

    modelo.fit(X_fit, y_fit, sample_weight=sample_weight)

    y_prob = modelo.predict_proba(X_test)

    if evaluar_minoritaria:
        y_true_eval = 1 - y_test
        y_prob_eval = 1 - y_prob
    else:
        y_true_eval = y_test
        y_prob_eval = y_prob

    resultado = evaluar_modelo_binario(y_true_eval, y_prob_eval, threshold=threshold)
    resultado['tecnica'] = tecnica
    resultado['feature_names'] = feature_names
    
    if evaluar_minoritaria:
        resultado['clase_positiva_reportada'] = 'Insuficiente'
    else:
        resultado['clase_positiva_reportada'] = 'No insuficiente'

        
    return resultado


def entrenar_final_y_evaluar_multiclase(
    df_dev,
    df_test,
    model_factory,
    columnas_numericas,
    columnas_categoricas,
    target_col='rendimiento',
    categorias_fijas=None,
    class_names=None
):
    """Entrena el mejor modelo multiclase final sobre todo dev y evalua sobre test. Prepara matrices de train/test, codifica labels a enteros, crea el modelo con
    model_factory, entrena, predice clases y probabilidades en test y evalua con evaluar_modelo_multiclase. Devuelve un diccionario con resultados y metricas."""

    if class_names is None:
        class_names = sorted(df_dev[target_col].dropna().unique().tolist())

    X_dev, y_dev_raw, X_test, y_test_raw, feature_names = preparar_matrices_train_val(
        df_dev,
        df_test,
        columnas_numericas,
        columnas_categoricas,
        target_col=target_col,
        categorias_fijas=categorias_fijas,
        drop_first=True
    )

    y_dev = codificar_labels(y_dev_raw, class_names)
    y_test = codificar_labels(y_test_raw, class_names)

    modelo = model_factory()
    modelo.fit(X_dev, y_dev)

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)

    resultado = evaluar_modelo_multiclase(y_test, y_pred, y_prob, class_names)
    resultado['feature_names'] = feature_names
    resultado['class_names'] = class_names

    if hasattr(modelo, 'feature_importances_') and modelo.feature_importances_ is not None:
        resultado['feature_importances'] = pd.Series(
            modelo.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)

    return resultado


def estimar_datos_nuevos_random_forest_multiclase(
    df,
    rf_params,
    columnas_numericas,
    columnas_categoricas,
    target_col='rendimiento',
    group_col='escuela',
    categorias_fijas=None,
    class_names=None,
    fracciones=None,
    n_splits=4,
    random_state=42,
    incremento_objetivo=0.01
):
    """Construye una learning curve aproximada para el random forest multiclase y extrapola cuantos datos harian falta para mejorar el accuracy en un incremento objetivo. 
    Para cada fraccion de datos, hace un muestreo estratificado por grupo, entrena un random forest con cross validation por grupos (GroupKFold) y guarda el accuracy global.
    Luego ajusta una curva de accuracy vs 1/sqrt(n) y extrapola cuantos datos harian falta para alcanzar el incremento objetivo de accuracy."""
    
    if fracciones is None:
        fracciones = [0.4, 0.55, 0.7, 0.85, 1.0]

    rng = np.random.RandomState(random_state)
    filas = []

    for fraccion in fracciones:
        idx_muestra = []

        for grupo, bloque in df.groupby(group_col):
            n_grupo = max(1, int(np.floor(len(bloque) * fraccion)))
            idx_grupo = rng.choice(bloque.index, size=n_grupo, replace=False)
            idx_muestra.extend(idx_grupo.tolist())

        idx_muestra = sorted(list(set(idx_muestra)))
        df_sub = df.loc[idx_muestra].reset_index(drop=True)

        def model_factory():
            return RandomForestClassifierManual(**rf_params)

        resultado = cross_validar_modelo_group_kfold(
            df_sub,
            model_factory,
            columnas_numericas,
            columnas_categoricas,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits,
            random_state=random_state,
            categorias_fijas=categorias_fijas,
            class_names=class_names
        )

        accuracy = resultado['tabla_resumen'].iloc[0]['accuracy_global']

        filas.append({
            'fraccion': fraccion,
            'n_muestras': len(df_sub),
            'accuracy': accuracy
        })

    tabla = pd.DataFrame(filas).sort_values('n_muestras').reset_index(drop=True)

    x = 1 / np.sqrt(tabla['n_muestras'].values)
    y = tabla['accuracy'].values

    pendiente, intercepto = np.polyfit(x, y, 1)

    accuracy_actual = tabla.iloc[-1]['accuracy']
    n_actual = tabla.iloc[-1]['n_muestras']
    accuracy_objetivo = accuracy_actual + incremento_objetivo

    n_estimado_total = np.nan
    datos_nuevos_estimados = np.nan

    if pendiente < 0 and accuracy_objetivo < intercepto:
        x_objetivo = (accuracy_objetivo - intercepto) / pendiente

        if x_objetivo > 0:
            n_estimado_total = 1 / (x_objetivo ** 2)
            datos_nuevos_estimados = max(0, n_estimado_total - n_actual)

    estimacion = {
        'pendiente': pendiente,
        'intercepto': intercepto,
        'accuracy_actual': accuracy_actual,
        'accuracy_objetivo': accuracy_objetivo,
        'n_actual': n_actual,
        'n_estimado_total': n_estimado_total,
        'datos_nuevos_estimados': datos_nuevos_estimados
    }

    return tabla, estimacion