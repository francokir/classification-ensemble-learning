import numpy as np
import pandas as pd


def matriz_confusion_binaria(y_true, y_pred):
    """Calcula la matriz de confusion para clasificación binaria  [[TN, FP],[FN, TP]] Sirve como base para otras 
    metricas."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return np.array([[tn, fp], [fn, tp]])


def accuracy_score_binario(y_true, y_pred):
    """Calcula la proporcion de predicciones correctas sobre el total de predicciones."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float((y_true == y_pred).mean())


def precision_score_binario(y_true, y_pred):
    """Calcula entre las predicciones positivas del modelo, cuales fueron correctas. (TP / (TP + FP))"""

    matriz = matriz_confusion_binaria(y_true, y_pred)
    tp = matriz[1, 1]
    fp = matriz[0, 1]

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score_binario(y_true, y_pred):
    """Calcula la proporcion de positivos reales . (TP / (TP + FN))"""

    matriz = matriz_confusion_binaria(y_true, y_pred)
    tp = matriz[1, 1]
    fn = matriz[1, 0]

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score_binario(y_true, y_pred):
    """Calcula la media armonica entre precision y recall, dando una medida balanceada entre ambos. 
    F1 = 2 * (precision * recall) / (precision + recall)"""
    
    precision = precision_score_binario(y_true, y_pred)
    recall = recall_score_binario(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def curva_roc(y_true, y_score):
    """Construye manualmente los puntos de la curva ROC ordenando las predicciones por su probabilidad
    y calculando TPR y FPR en cada umbral. Tambien devuelve los thresholds que son los valores de probabilidad 
    donde se producen cambios en la curva. Ordena de mayor a menor score para simular el barrido de umbrales"""


    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if len(np.unique(y_true)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    orden = np.argsort(-y_score)
    y_true = y_true[orden]
    y_score = y_score[orden]

    positivos = (y_true == 1).sum()
    negativos = (y_true == 0).sum()

    tpr = [0.0]
    fpr = [0.0]
    thresholds = [np.inf]

    tp = 0
    fp = 0
    score_anterior = None

    for i in range(len(y_true)):
        score_actual = y_score[i]

        if score_anterior is not None and score_actual != score_anterior:
            tpr.append(tp / positivos)
            fpr.append(fp / negativos)
            thresholds.append(score_anterior)

        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

        score_anterior = score_actual

    tpr.append(tp / positivos)
    fpr.append(fp / negativos)
    thresholds.append(score_anterior)

    return np.array(fpr), np.array(tpr), np.array(thresholds)


def curva_precision_recall(y_true, y_score):
    """Construye manualmente los puntos de la curva Precision-Recall. Devuelve recall, precision y thresholds. 
    Sirve para estudiar el trade-off entre precision y recall a medida que se varian los umbrales de decision."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if len(np.unique(y_true)) < 2:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([np.inf, -np.inf])

    orden = np.argsort(-y_score)
    y_true = y_true[orden]
    y_score = y_score[orden]

    positivos = (y_true == 1).sum()

    precision = [1.0]
    recall = [0.0]
    thresholds = [np.inf]

    tp = 0
    fp = 0
    score_anterior = None

    for i in range(len(y_true)):
        score_actual = y_score[i]

        if score_anterior is not None and score_actual != score_anterior:
            if tp + fp == 0:
                precision_actual = 1.0
            else:
                precision_actual = tp / (tp + fp)

            recall_actual = tp / positivos

            precision.append(precision_actual)
            recall.append(recall_actual)
            thresholds.append(score_anterior)

        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

        score_anterior = score_actual

    if tp + fp == 0:
        precision_actual = 1.0
    else:
        precision_actual = tp / (tp + fp)

    recall_actual = tp / positivos

    precision.append(precision_actual)
    recall.append(recall_actual)
    thresholds.append(score_anterior)

    return np.array(recall), np.array(precision), np.array(thresholds)

def area_bajo_curva(x, y):
    """Calcula el area bajo la curva utilizando la regla del trapecio. 
    Ordena los puntos por x y despues integra."""
    x = np.asarray(x)
    y = np.asarray(y)

    orden = np.argsort(x)
    x = x[orden]
    y = y[orden]

    return float(np.trapezoid(y, x))


def evaluar_modelo_binario(y_true, y_prob, threshold=0.5):
    """Se le pasa y_true, y_prob, umbraliza con threshold, genera y_pred, luego calcula todas las metricas:
    la matriz de confusion, accuracy, precision, recall, f1, curva ROC y curva Precision-Recall con sus respectivas areas bajo la curva.
    Devuelve un diccionario con todos estos resultados."""
    
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    matriz = matriz_confusion_binaria(y_true, y_pred)
    accuracy = accuracy_score_binario(y_true, y_pred)
    precision = precision_score_binario(y_true, y_pred)
    recall = recall_score_binario(y_true, y_pred)
    f1 = f1_score_binario(y_true, y_pred)

    if len(np.unique(y_true)) < 2:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        auc_roc = np.nan

        recall_curve = np.array([0.0, 1.0])
        precision_curve = np.array([1.0, 0.0])
        auc_pr = np.nan
    else:
        fpr, tpr, _ = curva_roc(y_true, y_prob)
        auc_roc = area_bajo_curva(fpr, tpr)

        recall_curve, precision_curve, _ = curva_precision_recall(y_true, y_prob)
        auc_pr = area_bajo_curva(recall_curve, precision_curve)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'matriz_confusion': matriz,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr,
        'auc_roc': auc_roc,
        'recall_curve': recall_curve,
        'precision_curve': precision_curve,
        'auc_pr': auc_pr
    }


def tabla_metricas_binarias(resultado, estrategia=''):
    """Convierte el diccionario de la funcion evaluar_modelo_binario en una tabla de pandas con una fila y 
    columnas para cada metrica. Agrega una columna 'estrategia' para identificar la estrategia de umbralizacion 
    utilizada."""
    return pd.DataFrame([{
        'estrategia': estrategia,
        'accuracy': resultado['accuracy'],
        'precision': resultado['precision'],
        'recall': resultado['recall'],
        'f1': resultado['f1'],
        'auc_roc': resultado['auc_roc'],
        'auc_pr': resultado['auc_pr']
    }])



#Metricas para Clasificacion Multiclase





def confusion_matrix_multiclass(y_true, y_pred, n_classes):
    """Construye la matriz de confusion para clasificacion multiclase. La matriz es de tamaño n_classes x n_classes, 
    donde la fila i representa las instancias reales de la clase i y la columna j representa las instancias 
    predichas como clase j. Cada elemento (i, j) contiene el conteo de instancias reales de la clase i que
    fueron predichas como clase j."""
    matriz = np.zeros((n_classes, n_classes), dtype=int)

    for real, pred in zip(y_true, y_pred):
        matriz[real, pred] += 1

    return matriz


def accuracy_multiclass(y_true, y_pred):
    """Calcula accuracy global multiclase."""
    return float((y_true == y_pred).mean())


def evaluar_modelo_multiclase(y_true, y_pred, y_prob, class_names):
    """Calcula todas las metricas para clasificacion multiclase. Para cada clase calcula precision, recall, f1,
    curva ROC y curva Precision-Recall con sus respectivas areas bajo la curva utilizando la estrategia de 
    one-vs-all. Devuelve un diccionario con todos estos resultados, incluyendo la matriz de confusion global,
    una tabla por clase y una tabla resumen con las metricas globales."""
    n_classes = len(class_names)

    matriz = confusion_matrix_multiclass(y_true, y_pred, n_classes)
    accuracy_global = accuracy_multiclass(y_true, y_pred)

    filas = []
    curvas_roc = {}
    curvas_pr = {}

    for idx, class_name in enumerate(class_names):
        # Para cada clase, se considera esa clase como positiva y las demás como negativas (one-vs-ALL)
        y_true_bin = (y_true == idx).astype(int)
        y_pred_bin = (y_pred == idx).astype(int)
        y_score_bin = y_prob[:, idx]

        accuracy = accuracy_score_binario(y_true_bin, y_pred_bin)
        precision = precision_score_binario(y_true_bin, y_pred_bin)
        recall = recall_score_binario(y_true_bin, y_pred_bin)
        f1 = f1_score_binario(y_true_bin, y_pred_bin)

        if len(np.unique(y_true_bin)) < 2:
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            auc_roc = np.nan

            recall_curve = np.array([0.0, 1.0])
            precision_curve = np.array([1.0, 0.0])
            auc_pr = np.nan
        else:
            fpr, tpr, _ = curva_roc(y_true_bin, y_score_bin)
            auc_roc = area_bajo_curva(fpr, tpr)

            recall_curve, precision_curve, _ = curva_precision_recall(y_true_bin, y_score_bin)
            auc_pr = area_bajo_curva(recall_curve, precision_curve)

        curvas_roc[class_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc_roc': auc_roc
        }

        curvas_pr[class_name] = {
            'recall': recall_curve,
            'precision': precision_curve,
            'auc_pr': auc_pr
        }

        filas.append({
            'clase': class_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr
        })

    tabla_por_clase = pd.DataFrame(filas)

    tabla_resumen = pd.DataFrame([{
        'accuracy_global': accuracy_global,
        'macro_precision': tabla_por_clase['precision'].mean(),
        'macro_recall': tabla_por_clase['recall'].mean(),
        'macro_f1': tabla_por_clase['f1'].mean(),
        'macro_auc_roc': tabla_por_clase['auc_roc'].mean(),
        'macro_auc_pr': tabla_por_clase['auc_pr'].mean()
    }])

    return {
        'matriz_confusion': matriz,
        'tabla_por_clase': tabla_por_clase,
        'tabla_resumen': tabla_resumen,
        'curvas_roc': curvas_roc,
        'curvas_pr': curvas_pr
    }