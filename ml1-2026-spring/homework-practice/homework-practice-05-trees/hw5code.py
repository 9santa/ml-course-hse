import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector, min_samples_leaf=1):
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов нужно брать среднее двух соседних при сортировке значений признака
    * Поведение функции в случае константного признака может быть любым
    * При одинаковых приростах критерия Джини для нескольких порогов нужно выбирать сплит, у которого значение порога минимально
    * Достаточно поддерживать только бинарную классификацию.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на две различные подвыборки или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds, len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    x = np.asarray(feature_vector, dtype=float)
    y = np.asarray(target_vector)

    # Convert labels to 0 and 1
    y = (y == np.unique(y)[-1]).astype(int)

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    n = len(x_sorted)

    # Find positions of possible thresholds,
    # i.e. where the neighboring elements differ
    split_indices = np.where(
        x_sorted[:-1] != x_sorted[1:]
    )[0] + 1

    if len(split_indices) == 0:
        return np.array([]), np.array([]), None, None

    # Thresholds - mean of two different neighboring values
    thresholds = (
        x_sorted[split_indices - 1] + x_sorted[split_indices]
    ) / 2

    assert len(thresholds) == len(split_indices)

    # Кол-во объектов слева с справа для каждого порога
    n_left = split_indices
    n_right = n - split_indices

    cumsum = np.cumsum(y_sorted)

    # Кол-во объектов положительного класса слева и справа
    positive_left = cumsum[split_indices - 1]
    positive_right = cumsum[-1] - positive_left

    # Доли класса 1 (for all thresholds)
    p_left = positive_left / n_left
    p_right = positive_right / n_right
    p_total = cumsum[-1] / n

    # Gini impurity (for all thresholds)
    gini_left = 1 - (p_left ** 2 + (1 - p_left) ** 2)
    gini_right = 1 - (p_right ** 2 + (1 - p_right) ** 2)
    gini_total = 1 - (p_total ** 2 + (1 - p_total) ** 2)

    # Gini criterion (for all thresholds)
    ginis = (
        -1 * (n_left / n) * gini_left
        - (n_right / n) * gini_right
    )

    # Оставляем только пороги, удовлетворяющие min_samples_leaf
    valid = (
        (n_left >= min_samples_leaf)
        & (n_right >= min_samples_leaf)
    )

    thresholds = thresholds[valid]
    ginis = ginis[valid]

    if len(thresholds) == 0:
        return np.array([]), np.array([]), None, None

    # Лучший порог - макисмальный прирост
    best_idx = np.argmax(ginis)

    best_threshold = thresholds[best_idx]
    best_gini = ginis[best_idx]

    return thresholds, ginis, best_threshold, best_gini


class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже могут быть намеренно оставлены некоторые ошибки.
    Их нужно исправить в рамках задания.
    """
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        """
        sub_X: (n_samples, n_features) - матрица признаков рассматриваемого подмножества
        sub_y: (n_samples) - метки рассматриваемого подмножества
        node: dict - рассматриваемая вершина, содержит информацию о себе и ссылки на левого и правого сына
        """
        # fix 1: change != to ==
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if (
            (self._max_depth is not None and depth == self._max_depth)
            or
            (self._min_samples_split is not None and len(sub_y) < self._min_samples_split)
        ):
            node["type"] = "terminal"
            # [0][0] takes the class of the most common class
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, overall_best_gini, split = None, None, None, None
        # fix 2: change 1 to 0
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    # fix 4: correct ratio calculation
                    if key in clicks:
                        ratio[key] = clicks[key] / counts[key]
                    else:
                        ratio[key] = 0

                # fix 5: categories sorted by ratio (asc)
                sorted_categories = [cat for cat, _ in sorted(ratio.items(), key=lambda kv: kv[1])]

                # fix 6: category -> code
                categories_map = {cat: i for i, cat in enumerate(sorted_categories, start=0)}

                # fix 7: encode the feature column
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError("Supported feature types are: `real`, `categorical`")


            # _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            thresholds, ginis, best_threshold, best_gini = find_best_split(
                feature_vector,
                sub_y,
                min_samples_leaf=self._min_samples_leaf or 1,
            )

            if len(thresholds) == 0:
                # Этот признак нельзя использовать для разбиения, нет возможных порогов
                continue


            if overall_best_gini is None or best_gini > overall_best_gini:
                overall_best_gini = best_gini
                feature_best = feature
                split: np.ndarray[tuple[int]] = feature_vector < best_threshold

                if feature_type == "real":
                    threshold_best = best_threshold
                elif feature_type == "categorical":
                    threshold_best = [
                        cat for cat, code in categories_map.items()
                        if code < best_threshold
                    ]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            # [0][0] takes the class of the most common class
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):

        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            go_left = x[feature] < node["threshold"]

        elif feature_type == "categorical":
            go_left = x[feature] in node["categories_split"]

        else:
            raise ValueError("Unknown feature type")

        if go_left:
            return self._predict_node(x, node=node["left_child"])
        else:
            return self._predict_node(x, node=node["right_child"])


    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # fix 3: convert string target labels to 0s and 1s
        self._classes, y_encoded = np.unique(y, return_inverse=True)

        if len(self._classes) != 2:
            raise ValueError("The tree supports binary classification. `y` contains more than 2 classes.")

        self._fit_node(X, y_encoded, self._tree)

    def predict(self, X):
        predicted = []

        for x in X:
            predicted.append(self._predict_node(x, self._tree))

        return self._classes[np.array(predicted, dtype=int)]
