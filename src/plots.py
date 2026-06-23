import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def graficar_faltantes(df):
    faltantes = df.isnull().sum()
    faltantes = faltantes[faltantes > 0].sort_values(ascending=False)

    if len(faltantes) == 0:
        print('No hay valores faltantes')
        return

    plt.figure(figsize=(10, 5))
    plt.bar(faltantes.index, faltantes.values)
    plt.title('Cantidad de valores faltantes por columna')
    plt.xlabel('Columnas')
    plt.ylabel('Cantidad de faltantes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def graficar_histogramas(df, columnas_numericas):
    n = len(columnas_numericas)
    filas = math.ceil(n / 3)

    plt.figure(figsize=(15, 4 * filas))

    for i, col in enumerate(columnas_numericas, 1):
        plt.subplot(filas, 3, i)
        datos = df[col].dropna()
        plt.hist(datos, bins=30)
        plt.title(col)

    plt.tight_layout()
    plt.show()


def graficar_boxplots(df, columnas_numericas):
    n = len(columnas_numericas)
    filas = math.ceil(n / 3)

    plt.figure(figsize=(15, 4 * filas))

    for i, col in enumerate(columnas_numericas, 1):
        plt.subplot(filas, 3, i)
        datos = df[col].dropna()
        plt.boxplot(datos)
        plt.title(col)

    plt.tight_layout()
    plt.show()


def graficar_categoricas(df, columnas_categoricas):
    n = len(columnas_categoricas)

    plt.figure(figsize=(6 * n, 5))

    for i, col in enumerate(columnas_categoricas, 1):
        plt.subplot(1, n, i)
        conteo = df[col].value_counts(dropna=False).sort_index()
        plt.bar(conteo.index.astype(str), conteo.values)
        plt.title(col)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def graficar_target_multiclase(df):
    orden = ['Insuficiente', 'Regular', 'Bueno', 'Excelente']
    conteo = df['rendimiento'].value_counts().reindex(orden)

    plt.figure(figsize=(7, 5))
    plt.bar(conteo.index, conteo.values)
    plt.title('Distribución del target multiclase')
    plt.xlabel('Rendimiento')
    plt.ylabel('Cantidad')
    plt.tight_layout()
    plt.show()


def graficar_target_binario(df):
    conteo = df['rendimiento_binario'].value_counts().sort_index()

    plt.figure(figsize=(6, 5))
    plt.bar(['Desaprobado', 'Aprobado'], conteo.values)
    plt.title('Distribución del target binario')
    plt.xlabel('Rendimiento binario')
    plt.ylabel('Cantidad')
    plt.tight_layout()
    plt.show()


def graficar_target_por_grupo(df, grupo):
    tabla = pd.crosstab(df[grupo], df['rendimiento'], normalize='index') * 100

    columnas_ordenadas = [col for col in ['Insuficiente', 'Regular', 'Bueno', 'Excelente'] if col in tabla.columns]
    tabla = tabla[columnas_ordenadas]

    tabla.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'Distribución de rendimiento por {grupo}')
    plt.xlabel(grupo)
    plt.ylabel('Porcentaje')
    plt.legend(title='Rendimiento', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def graficar_boxplots_por_escuela(df, columnas_numericas):
    escuelas = sorted(df['escuela'].dropna().unique())

    for col in columnas_numericas:
        datos = []
        for escuela in escuelas:
            datos.append(df[df['escuela'] == escuela][col].dropna())

        plt.figure(figsize=(10, 5))
        plt.boxplot(datos, labels=escuelas)
        plt.title(f'{col} por escuela')
        plt.xlabel('Escuela')
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()


def graficar_heatmap_promedios_por_escuela(df, columnas_numericas):
    tabla = df.groupby('escuela')[columnas_numericas].mean()
    tabla_valores = tabla.values

    plt.figure(figsize=(12, 6))
    plt.imshow(tabla_valores, aspect='auto')
    plt.colorbar()

    plt.xticks(range(len(tabla.columns)), tabla.columns, rotation=45, ha='right')
    plt.yticks(range(len(tabla.index)), tabla.index)

    for i in range(len(tabla.index)):
        for j in range(len(tabla.columns)):
            plt.text(j, i, f'{tabla_valores[i, j]:.2f}', ha='center', va='center', fontsize=8)

    plt.title('Promedio de features numéricas por escuela')
    plt.tight_layout()
    plt.show()


def graficar_correlacion_target_por_escuela(df, columnas_numericas):
    resultados = []

    for escuela in sorted(df['escuela'].dropna().unique()):
        df_escuela = df[df['escuela'] == escuela]

        fila = {'escuela': escuela}

        for col in columnas_numericas:
            datos = df_escuela[[col, 'rendimiento_binario']].dropna()

            if len(datos) < 2:
                fila[col] = np.nan
            elif datos[col].nunique() < 2:
                fila[col] = np.nan
            elif datos['rendimiento_binario'].nunique() < 2:
                fila[col] = np.nan
            else:
                fila[col] = datos[col].corr(datos['rendimiento_binario'])

        resultados.append(fila)

    tabla = pd.DataFrame(resultados).set_index('escuela')

    plt.figure(figsize=(12, 5))
    sns.heatmap(tabla, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlación entre features y target binario por escuela')
    plt.tight_layout()
    plt.show()

def graficar_matriz_confusion(matriz, titulo='Matriz de confusión'):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matriz,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicho 0', 'Predicho 1'],
        yticklabels=['Real 0', 'Real 1']
    )
    plt.title(titulo)
    plt.xlabel('Predicción')
    plt.ylabel('Valor real')
    plt.tight_layout()
    plt.show()


def graficar_curva_roc(fpr, tpr, auc_roc, titulo='Curva ROC'):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label='AUC-ROC = {:.4f}'.format(auc_roc))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graficar_curva_pr(recall_curve, precision_curve, auc_pr, titulo='Curva Precision-Recall'):
    plt.figure(figsize=(6, 5))
    plt.plot(recall_curve, precision_curve, label='AUC-PR = {:.4f}'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graficar_f1_vs_lambda(tabla_resultados, titulo='F1 vs lambda'):
    plt.figure(figsize=(7, 5))
    plt.plot(tabla_resultados['lambda'], tabla_resultados['f1_promedio'], marker='o')

    if (tabla_resultados['lambda'] > 0).all():
        plt.xscale('log')

    plt.xlabel('lambda')
    plt.ylabel('F1 promedio de validación')
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def graficar_comparacion_estabilidad(tabla_std, top_n=15, titulo='Comparación de estabilidad'):
    top = tabla_std.head(top_n).copy()
    top = top.iloc[::-1]

    posiciones = np.arange(len(top))
    ancho = 0.35

    plt.figure(figsize=(10, 6))
    plt.barh(posiciones - ancho / 2, top['std_random'], height=ancho, label='KFold aleatorio')
    plt.barh(posiciones + ancho / 2, top['std_group'], height=ancho, label='GroupKFold')

    plt.yticks(posiciones, top.index)
    plt.xlabel('Desvío estándar del coeficiente')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graficar_boxplot_coeficientes(tabla_coeficientes, features, titulo='Distribución de coeficientes'):
    datos = []
    labels = []

    for feature in features:
        if feature in tabla_coeficientes.columns:
            serie = tabla_coeficientes[feature].dropna()

            if len(serie) > 0:
                datos.append(serie.values)
                labels.append(feature)

    if len(datos) == 0:
        print('No hay coeficientes para graficar')
        return

    plt.figure(figsize=(max(8, len(labels) * 0.8), 5))
    plt.boxplot(datos, labels=labels)
    plt.axhline(0, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def graficar_matriz_confusion_multiclase(matriz, class_names, titulo='Matriz de confusión multiclase'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matriz,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicción')
    plt.ylabel('Valor real')
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def graficar_roc_multiclase_ovr(curvas_roc, titulo='Curvas ROC one-vs-all'):
    plt.figure(figsize=(7, 5))

    for clase, datos in curvas_roc.items():
        plt.plot(datos['fpr'], datos['tpr'], label=f'{clase} (AUC={datos["auc_roc"]:.4f})')

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graficar_pr_multiclase_ovr(curvas_pr, titulo='Curvas Precision-Recall one-vs-all'):
    plt.figure(figsize=(7, 5))

    for clase, datos in curvas_pr.items():
        plt.plot(datos['recall'], datos['precision'], label=f'{clase} (AUC={datos["auc_pr"]:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graficar_comparacion_modelos(tabla_comparativa, metrica='macro_f1', titulo='Comparación de modelos'):
    plt.figure(figsize=(7, 5))
    plt.bar(tabla_comparativa['modelo'], tabla_comparativa[metrica])
    plt.ylabel(metrica)
    plt.title(titulo)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def graficar_f1_por_clase_modelos(tabla_f1, titulo='F1 por clase y modelo'):
    tabla_pivot = tabla_f1.pivot(index='clase', columns='modelo', values='f1')
    tabla_pivot.plot(kind='bar', figsize=(8, 5))
    plt.ylabel('F1')
    plt.title(titulo)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def graficar_importancia_features(importancias, top_n=15, titulo='Importancia de features'):
    top = importancias.head(top_n)
    top = top.iloc[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(top.index, top.values)
    plt.xlabel('Importancia')
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

def graficar_curvas_roc_rebalanceo(resultados_dict, titulo='Comparación de curvas ROC'):
    nombres = {
        'sin_rebalanceo': 'Sin rebalanceo',
        'undersampling': 'Undersampling',
        'oversampling_duplicate': 'Oversampling duplicate',
        'smote': 'Oversampling SMOTE',
        'cost_reweighting': 'Cost re-weighting'
    }

    plt.figure(figsize=(7, 5))

    for tecnica, resultado in resultados_dict.items():
        etiqueta = nombres.get(tecnica, tecnica)
        plt.plot(
            resultado['fpr'],
            resultado['tpr'],
            label=f'{etiqueta} (AUC={resultado["auc_roc"]:.4f})'
        )

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graficar_curvas_pr_rebalanceo(resultados_dict, titulo='Comparación de curvas Precision-Recall'):
    nombres = {
        'sin_rebalanceo': 'Sin rebalanceo',
        'undersampling': 'Undersampling',
        'oversampling_duplicate': 'Oversampling duplicate',
        'smote': 'Oversampling SMOTE',
        'cost_reweighting': 'Cost re-weighting'
    }

    plt.figure(figsize=(7, 5))

    for tecnica, resultado in resultados_dict.items():
        etiqueta = nombres.get(tecnica, tecnica)
        plt.plot(
            resultado['recall_curve'],
            resultado['precision_curve'],
            label=f'{etiqueta} (AUC={resultado["auc_pr"]:.4f})'
        )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graficar_comparacion_rebalanceo(tabla_metricas, metrica='F1-Score', titulo='Comparación de técnicas de rebalanceo'):
    plt.figure(figsize=(8, 5))
    plt.bar(tabla_metricas['Modelo'], tabla_metricas[metrica])
    plt.xticks(rotation=20)
    plt.ylabel(metrica)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def graficar_learning_curve_rf(tabla_learning, estimacion=None, titulo='Learning curve - Random Forest multiclase'):
    plt.figure(figsize=(7, 5))
    plt.plot(tabla_learning['n_muestras'], tabla_learning['accuracy'], marker='o')
    plt.xlabel('Cantidad de muestras')
    plt.ylabel('Accuracy')
    plt.title(titulo)

    if estimacion is not None and not pd.isna(estimacion['n_estimado_total']):
        plt.axhline(estimacion['accuracy_objetivo'], linestyle='--')
        plt.axvline(estimacion['n_estimado_total'], linestyle='--')

    plt.tight_layout()
    plt.show()