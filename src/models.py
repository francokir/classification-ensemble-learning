import numpy as np


class RegresionLogisticaBinaria:
    def __init__(
        self,
        learning_rate=0.01,
        n_iter=6000,
        lambda_reg=0.1,
        tol=1e-7,
        verbose=False
    ):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.verbose = verbose

        self.w = None
        self.b = 0.0
        self.loss_history = []

    def sigmoid(self, z):
        """Funcion sigmoide, transforma scores en probabilidades."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def calcular_loss(self, y, y_prob):
        """Calcula la función de pérdida logarítmica con regularización L2."""

        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)

        loss = -np.mean(y * np.log(y_prob) + (1 - y) * np.log(1 - y_prob))
        reg = self.lambda_reg * np.sum(self.w ** 2)

        return loss + reg

    def fit(self, X, y):
        """Entrena el modelo con gradiente descendente. Calcula probabilidades, errores, gradientes y actualiza
          pesos, en direccion opuesta al gradiente. Esto minimiza la log loss, no la accuracy, cabe aclarar esa diferencia."""
        
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []

        for i in range(self.n_iter):
            z = np.dot(X, self.w) + self.b
            y_prob = self.sigmoid(z)

            error = y_prob - y

            dw = (np.dot(X.T, error) / n_samples) + 2 * self.lambda_reg * self.w
            db = np.mean(error)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            loss = self.calcular_loss(y, y_prob)
            self.loss_history.append(loss)

            if self.verbose and i % 500 == 0:
                print('Iteración:', i, '- Loss:', loss)

            if i > 0:
                cambio = abs(self.loss_history[-1] - self.loss_history[-2])
                if cambio < self.tol:
                    break

        return self

    def predict_proba(self, X):
        """Devuelve P(y=1|X) para cada muestra. Esto defino lo que es una y_prob, es la probabilidad de que la clase 
        sea 1 dado X."""

        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Convierte probabilidades en clases usando threshold, umbral. Claramente para el caso
        binario el umbral comun es 0.5, pero se puede ajustar para mejorar precision o recall segun el caso."""

        y_prob = self.predict_proba(X)
        return (y_prob >= threshold).astype(int)
    


#Funciones de Clasificacion Multiclase



def softmax(z):
    """Convierte scores en probabilidades para cada clase, se necesita para la regresion logistica multiclase. 
    La funcion softmax toma un vector de scores y los transforma en probabilidades que suman 1."""
    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class LDAClassifierManual:
    def __init__(self, reg_epsilon=1e-6):
        self.reg_epsilon = reg_epsilon
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_ = None
        self.inv_covariance_ = None

    def fit(self, X, y):
        """Calcula las medias, la matriz de covarianza compartida y las priors para cada clase. 
        La matriz de covarianza se calcula como la media ponderada de las matrices de covarianza de cada clase, 
        ajustada por el número de muestras en cada clase. 
        Se añade un término de regularización a la matriz de covarianza para asegurar que sea invertible."""
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))

        covariance_sum = np.zeros((n_features, n_features))
        total_df = 0

        for idx, clase in enumerate(self.classes_):
            X_k = X[y == clase]
            n_k = len(X_k)

            self.priors_[idx] = n_k / len(X)
            self.means_[idx] = X_k.mean(axis=0)

            if n_k > 1:
                cov_k = np.cov(X_k, rowvar=False, bias=False)
                covariance_sum += (n_k - 1) * cov_k
                total_df += (n_k - 1)

        if total_df == 0:
            self.covariance_ = np.eye(n_features)
        else:
            self.covariance_ = covariance_sum / total_df

        self.covariance_ += self.reg_epsilon * np.eye(n_features)
        self.inv_covariance_ = np.linalg.inv(self.covariance_)

        return self

    def decision_function(self, X):
        """Calcula los scores para cada clase, estos se van a pasar a la softmax. Para calcular los scores hace
        uso de las medias, la matriz de covarianza y las priors. La formula del score es: score_k = X @ inv_covariance @ mean_k - 0.5 * mean_k @ inv_covariance @ mean_k + log(prior_k)"""
        scores = []

        for idx, clase in enumerate(self.classes_):
            mu = self.means_[idx]
            prior = self.priors_[idx]

            score = (
                X @ self.inv_covariance_ @ mu
                - 0.5 * mu @ self.inv_covariance_ @ mu
                + np.log(prior + 1e-15)
            )

            scores.append(score)

        return np.column_stack(scores)

    def predict_proba(self, X):
        """Calcula las probabilidades para cada clase usando la funcion softmax sobre los scores obtenidos de
        la decision_function."""
        scores = self.decision_function(X)
        return softmax(scores)

    def predict(self, X):
        """Predice la clase con mayor probabilidad para cada muestra."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class LogisticRegressionMulticlass:
    def __init__(self, learning_rate=0.05, n_iter=5000, lambda_reg=0.1, tol=1e-7, verbose=False):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.verbose = verbose

        self.W = None
        self.b = None
        self.loss_history = []

    def one_hot(self, y, n_classes):
        """Convierte etiquetas de clase en formato one-hot. Esto es necesario para calcular la función de pérdida 
        logarítmica en la regresión logística multiclase, ya que necesitamos comparar las probabilidades predichas 
        con las etiquetas reales en formato one-hot."""
        Y = np.zeros((len(y), n_classes))
        Y[np.arange(len(y)), y] = 1
        return Y

    def compute_loss(self, Y, P):
        """Calcula la función de pérdida logarítmica con regularización L2 para la regresión logística multiclase."""
        eps = 1e-15
        P = np.clip(P, eps, 1 - eps)

        loss = -np.mean(np.sum(Y * np.log(P), axis=1))
        reg = self.lambda_reg * np.sum(self.W ** 2)

        return loss + reg

    def fit(self, X, y):
        """Entrena el modelo usando gradiente descendente y entropía cruzada como función de pérdida."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.loss_history = []

        Y = self.one_hot(y, n_classes)

        for i in range(self.n_iter):
            scores = X @ self.W + self.b
            P = softmax(scores)

            error = P - Y

            dW = (X.T @ error) / n_samples + 2 * self.lambda_reg * self.W
            db = np.mean(error, axis=0)

            self.W = self.W - self.learning_rate * dW
            self.b = self.b - self.learning_rate * db

            loss = self.compute_loss(Y, P)
            self.loss_history.append(loss)

            if self.verbose and i % 500 == 0:
                print('Iteración:', i, '- Loss:', loss)

            if i > 0:
                cambio = abs(self.loss_history[-1] - self.loss_history[-2])
                if cambio < self.tol:
                    break

        return self

    def predict_proba(self, X):
        """Calcula las probabilidades para cada clase usando la función softmax sobre los scores obtenidos 
        de X @ W + b."""
        scores = X @ self.W + self.b
        return softmax(scores)

    def predict(self, X):
        """Predice la clase con mayor probabilidad para cada muestra."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class DecisionTreeClassifierManual:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=42,
        n_classes=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_classes = n_classes

        self.tree_ = None
        self.feature_importances_ = None
        self.rng = np.random.RandomState(self.random_state)

    def entropy(self, y):
        """Calcula la entropia de una distribucion de clases. La entropia es una medida de la impureza de un 
        conjunto de datos, y se utiliza para determinar la mejor división en un árbol de decisión. 
        Se calcula como -sum(p_i * log2(p_i)) donde p_i es la proporción de muestras de la clase i en el conjunto 
        de datos."""
        if len(y) == 0:
            return 0.0

        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts[counts > 0] / len(y)

        return -np.sum(probs * np.log2(probs + 1e-15))

    def class_counts(self, y):
        """Cuenta cuantos ejemplos hay de cada clase en un nodo."""
        return np.bincount(y, minlength=self.n_classes)

    def class_proba(self, y):
        """Calcula la probabilidad de cada clase en un nodo, esto se hace dividiendo el conteo de cada clase por
        el total de muestras en el nodo."""
        counts = self.class_counts(y)
        return counts / counts.sum()

    def majority_class(self, y):
        """Devuelve la clase mayoritaria en un nodo, esto se hace encontrando el índice de la clase con el conteo más alto."""
        return np.argmax(self.class_counts(y))

    def get_num_features_to_consider(self, n_features):
        """Determina cuantas features se van a considerar al buscar el split. Esto es util cuando el arbol se use dentro de RF."""
        if self.max_features is None:
            return n_features

        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))

        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)

        if isinstance(self.max_features, float):
            return max(1, int(n_features * self.max_features))

        return n_features

    def get_candidate_thresholds(self, values):
        """Genera los thresholds candidatos para dividir segun una feature numerica. Si hay pocos valores unicos, 
        se usan los puntos medios entre ellos. Si hay muchos, se usan percentiles."""
        unique_values = np.unique(values)

        if len(unique_values) <= 1:
            return np.array([])

        if len(unique_values) <= 20:
            return (unique_values[:-1] + unique_values[1:]) / 2

        quantiles = np.linspace(5, 95, 19)
        thresholds = np.unique(np.percentile(unique_values, quantiles))

        return thresholds

    def best_split(self, X, y):
        """Busca el mejor split posible para el nodo actual. Calcula entropia del nodo padre,
        inicializa el mejor split e itera sobre un subconjunto aleatorio de features y thresholds candidatos 
        para cada feature. Para cada split, calcula la entropia ponderada de los nodos hijos y el gain. Tambien descarta
        splits que no cumplen con el min_samples_leaf.
        Si el gain es mejor que el mejor gain encontrado hasta ahora, actualiza el mejor split.
        Es la funcion de decision del arbol."""


        n_samples, n_features = X.shape
        parent_entropy = self.entropy(y)

        best_feature = None
        best_threshold = None
        best_gain = -1

        m = self.get_num_features_to_consider(n_features)
        features = self.rng.choice(n_features, size=m, replace=False)

        for feature in features:
            thresholds = self.get_candidate_thresholds(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                entropy_left = self.entropy(y_left)
                entropy_right = self.entropy(y_right)

                weighted_entropy = (
                    (n_left / n_samples) * entropy_left
                    + (n_right / n_samples) * entropy_right
                )

                gain = parent_entropy - weighted_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def build_tree(self, X, y, depth):
        """Construye un arbol recursivamente, arma un nodo base, evalua condiciones de corte: nodo puro,
        max_depth, min_samples_split. Si se cumple alguna, el nodo se convierte en hoja.
        Si no, busca el mejor split y divide el dataset."""

        n_samples = len(y)

        node = {
            'pred_class': self.majority_class(y),
            'proba': self.class_proba(y)
        }

        if len(np.unique(y)) == 1:
            node['type'] = 'leaf'
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            node['type'] = 'leaf'
            return node

        if n_samples < self.min_samples_split:
            node['type'] = 'leaf'
            return node

        feature, threshold, gain = self.best_split(X, y)

        if feature is None or gain <= 0:
            node['type'] = 'leaf'
            return node

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node['type'] = 'leaf'
            return node

        self.feature_importances_[feature] += gain * n_samples

        node['type'] = 'node'
        node['feature'] = feature
        node['threshold'] = threshold
        node['left'] = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        """Entrena el arbol completo: si no sabe cuantas clases hay, las infiere, inicializa vector de importancias y
        construye el arbol. Al finalizar, normaliza las importancias."""
        if self.n_classes is None:
            self.n_classes = int(np.max(y)) + 1

        self.feature_importances_ = np.zeros(X.shape[1])
        self.tree_ = self.build_tree(X, y, depth=0)

        total_importance = self.feature_importances_.sum()
        if total_importance > 0:
            self.feature_importances_ = self.feature_importances_ / total_importance

        return self

    def predict_one(self, x, node):
        """Predice la clase de una sola observacion recorriendo el arbol desde la raiz hasta una hoja,
        tomando decisiones segun los splits."""
        if node['type'] == 'leaf':
            return node['pred_class']

        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])

        return self.predict_one(x, node['right'])

    def predict_proba_one(self, x, node):
        """Predice las probabilidades de clase para una sola observacion recorriendo el arbol desde la raiz hasta una hoja,
        tomando decisiones segun los splits."""

        if node['type'] == 'leaf':
            return node['proba']

        if x[node['feature']] <= node['threshold']:
            return self.predict_proba_one(x, node['left'])

        return self.predict_proba_one(x, node['right'])

    def predict(self, X):
        """Predice la clase de todas las observaciones, usando la funcion predict_one para cada una."""
        preds = [self.predict_one(x, self.tree_) for x in X]
        return np.array(preds)

    def predict_proba(self, X):
        """Predice las probabilidades de clase para todas las observaciones, usando la funcion predict_proba_one para cada una."""
        probs = [self.predict_proba_one(x, self.tree_) for x in X]
        return np.array(probs)


class RandomForestClassifierManual:
    def __init__(
        self,
        n_estimators=20,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_ = []
        self.feature_importances_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        """Entrena multiples arboles sobre muestras bootstrap, es decir muestras aleatorias con reemplazo del dataset original. 
        Para cada arbol, se calcula su importancia de features y al finalizar se promedian y normalizan. La idea es que reduce varianza del 
        arbol individual."""

        rng = np.random.RandomState(self.random_state)  
        n_samples = X.shape[0]

        self.n_classes_ = int(np.max(y)) + 1
        self.trees_ = []
        importances = []

        for i in range(self.n_estimators):
            if self.bootstrap:
                sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
            else:
                sample_idx = np.arange(n_samples)

            X_sample = X[sample_idx]
            y_sample = y[sample_idx]

            tree = DecisionTreeClassifierManual(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=rng.randint(0, 1000000),
                n_classes=self.n_classes_
            )

            tree.fit(X_sample, y_sample)

            self.trees_.append(tree)
            importances.append(tree.feature_importances_)

        self.feature_importances_ = np.mean(np.vstack(importances), axis=0)

        total_importance = self.feature_importances_.sum()
        if total_importance > 0:
            self.feature_importances_ = self.feature_importances_ / total_importance

        return self

    def predict_proba(self, X):
        """Promedia las probabilidades de todos los arboles para obtener la probabilidad final de cada clase. 
        Esto se hace obteniendo las probabilidades de cada arbol con predict_proba y luego promediando."""
        all_probs = [tree.predict_proba(X) for tree in self.trees_]
        return np.mean(np.stack(all_probs, axis=0), axis=0)

    def predict(self, X):
        """Predice la clase con mayor probabilidad para cada muestra, usando las probabilidades promedio de todos los arboles."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    



#Funciones de regresion logistica binaria con regularizacion L2 y ponderacion de muestras


class LogisticRegressionBinaryWeighted:
    def __init__(self, learning_rate=0.05, n_iter=5000, lambda_reg=0.1, tol=1e-7, verbose=False):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.verbose = verbose

        self.w = None
        self.b = 0.0
        self.loss_history = []

    def sigmoid(self, z):
        """Funcion sigmoide, transforma scores lineales en probabilidades."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def compute_loss(self, y, p, sample_weight):
        """Funcion de costo del modelo, es una log loss ponderada por muestra, con regularizacion L2. La parte de la log loss hace que
        si una observacion tiene peso grande, cuente mas en la loss si se predice mal, la regularizacion penaliza pesos grandes para evitar overfitting."""
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)

        loss = -np.sum(sample_weight * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(sample_weight)
        reg = self.lambda_reg * np.sum(self.w ** 2)

        return loss + reg

    def fit(self, X, y, sample_weight=None):
        """Entrena el modelo con gradiente descendente, inicializa los parametros antes de entrenar, si no se pasan pesos son todos iguales
        por lo que es flexible, sirve para cost-re weighting. Luego tambien normaliza los pesos para mantener la escala de train estable y calcula 
        probabilidades para la clase positiva, errores, gradientes ponderados por sample_weight y actualiza los pesos y sesgo en direccion opuesta al gradiente.
        Al finalizar cada iteracion calcula la loss ponderada y la guarda en el historial. El training para si la mejora de la loss es menor a tol."""
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []

        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        sample_weight = sample_weight.astype(float)
        sample_weight = sample_weight / sample_weight.mean()

        for i in range(self.n_iter):
            z = X @ self.w + self.b
            p = self.sigmoid(z)

            error = p - y

            dw = (X.T @ (sample_weight * error)) / np.sum(sample_weight) + 2 * self.lambda_reg * self.w
            db = np.sum(sample_weight * error) / np.sum(sample_weight)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            loss = self.compute_loss(y, p, sample_weight)
            self.loss_history.append(loss)

            if self.verbose and i % 500 == 0:
                print('Iteración:', i, '- Loss:', loss)

            if i > 0:
                cambio = abs(self.loss_history[-1] - self.loss_history[-2])
                if cambio < self.tol:
                    break

        return self

    def predict_proba(self, X):
        """ Devuelve la probabilidad estimada de la clase positiva."""
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Convierte probabilidades en clases usando un umbral, por defecto 0.5. Si la probabilidad de la clase positiva es mayor o igual al umbral, se predice 1, de lo contrario se predice 0."""
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)