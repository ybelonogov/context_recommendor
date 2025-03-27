import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from numpy.linalg import norm
import random

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("UniversalContextualRecommender")


# Универсальный класс контекстных рекомендаций
class UniversalContextualRecommender:
    def __init__(self, dataset: pd.DataFrame, model_name: str,
                 user_col: str = 'user_id', item_col: str = 'item_id',
                 rating_col: str = 'rating', context_cols: list = None, **model_params):
        """
        Инициализация рекомендательной системы.

        :param dataset: DataFrame с данными об интеракциях и контекстными фичами.
        :param model_name: название модели для рекомендаций (например, 'camf', 'cslim').
        :param user_col: название столбца с идентификаторами пользователей.
        :param item_col: название столбца с идентификаторами объектов.
        :param rating_col: название столбца с оценками или степенью взаимодействия.
        :param context_cols: список столбцов с контекстной информацией (например, пол, возраст, жанр).
        :param model_params: дополнительные параметры для модели.
        """
        self.dataset = dataset.copy()
        self.model_name = model_name.lower()
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.context_cols = context_cols if context_cols is not None else []
        self.model_params = model_params
        self.model = None

        self._init_model()

    def _init_model(self):
        """
        Выбор и инициализация модели на основе переданного названия.
        """
        logger.info("Инициализация модели '%s'", self.model_name)
        if self.model_name == 'camf':
            self.model = CAMFModel(self.dataset, self.user_col, self.item_col,
                                   self.rating_col, self.context_cols, **self.model_params)
        elif self.model_name == 'cslim':
            self.model = CSLIMModel(self.dataset, self.user_col, self.item_col,
                                    self.rating_col, self.context_cols, **self.model_params)
        elif self.model_name in ['vbpr', 'vmf', 'amr', 'casualrec', 'dmrl', 'ctr', 'hft',
                                 'cdl', 'convmf', 'cdr', 'cvaecf', 'cvae', 'hrdr', 'lightgbm',
                                 'xgboost', 'deepfm']:
            logger.error("Модель %s пока не реализована.", self.model_name)
            raise NotImplementedError(
                f"Модель {self.model_name} пока не реализована. Реализуйте или подключите её код.")
        else:
            logger.error("Неизвестное название модели: %s", self.model_name)
            raise ValueError(f"Неизвестное название модели: {self.model_name}")

    def fit(self):
        """
        Обучение выбранной модели.
        Здесь можно добавить этап пред-фильтрации, если требуется.
        """
        if self.model is None:
            logger.error("Модель не инициализирована.")
            raise ValueError("Модель не инициализирована.")
        logger.info("Начало этапа обучения модели.")
        # Пример пред-фильтрации можно добавить здесь:
        # self.dataset = self.pre_filter(self.dataset)
        self.model.fit()
        logger.info("Обучение модели завершено.")

    def predict(self, default_context: dict = None):
        """
        Генерация ранжированных рекомендаций для каждого пользователя.
        В метод predict() для моделей, где используется контекст (например, CAMF),
        можно задать default_context – словарь с значениями контекстных признаков по умолчанию.

        :param default_context: словарь вида {context_col: значение}
        :return: словарь, где ключ — идентификатор пользователя, а значение — список отранжированных объектов.
        """
        if self.model is None:
            logger.error("Модель не инициализирована.")
            raise ValueError("Модель не инициализирована.")
        logger.info("Начало предсказания рекомендаций.")
        recs = self.model.predict(default_context)
        # Пример пост-фильтрации можно добавитья здесь:
        # recs = self.post_filter(recs)
        logger.info("Предсказание рекомендаций завершено.")
        return recs

    def pre_filter(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Пример пред-фильтрации: можно убрать объекты по сезонным или другим условиям.
        """
        # Здесь можно реализовать логику пред-фильтрации.
        return dataset

    def post_filter(self, recommendations: dict) -> dict:
        """
        Пример пост-фильтрации: можно до финального вывода дополнительно отфильтровать рекомендации.
        """
        # Здесь можно реализовать логику пост-фильтрации.
        return recommendations


# Реализация модели CAMF с использованием SGD
class CAMFModel:
    def __init__(self, dataset: pd.DataFrame, user_col: str, item_col: str,
                 rating_col: str, context_cols: list, **params):
        self.logger = logging.getLogger("CAMFModel")
        self.dataset = dataset
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.context_cols = context_cols
        # Гиперпараметры модели:
        self.n_factors = params.get('n_factors', 10)
        self.learning_rate = params.get('learning_rate', 0.005)
        self.reg = params.get('reg', 0.02)
        self.epochs = params.get('epochs', 10)

        # Инициализация параметров:
        self.users = self.dataset[self.user_col].unique()
        self.items = self.dataset[self.item_col].unique()

        # Латентные векторы пользователей и объектов
        self.U = {u: np.random.normal(scale=0.1, size=self.n_factors) for u in self.users}
        self.V = {i: np.random.normal(scale=0.1, size=self.n_factors) for i in self.items}
        # Смещения пользователей и объектов
        self.b_u = {u: 0.0 for u in self.users}
        self.b_i = {i: 0.0 for i in self.items}
        # Контекстные смещения: для каждого контекстного столбца и его уникального значения
        self.b_context = {}
        for col in self.context_cols:
            unique_vals = self.dataset[col].unique()
            for val in unique_vals:
                self.b_context[(col, val)] = 0.0
        # Глобальный средний рейтинг
        self.global_bias = self.dataset[self.rating_col].mean()
        self.logger.info("Инициализация CAMF модели завершена. Глобальный bias: %.4f", self.global_bias)

    def fit(self):
        """
        Обучение CAMF через SGD.
        Обходим все записи обучающей выборки и обновляем параметры модели.
        """
        self.logger.info("Начало обучения CAMF модели.")
        data = self.dataset.sample(frac=1).reset_index(drop=True)  # перемешиваем данные
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            data = data.sample(frac=1).reset_index(drop=True)
            for idx, row in data.iterrows():
                user = row[self.user_col]
                item = row[self.item_col]
                rating = row[self.rating_col]
                # Сумма контекстных смещений для данной записи
                context_bias = sum(self.b_context.get((col, row[col]), 0.0) for col in self.context_cols)
                # Предсказание
                pred = self.global_bias + self.b_u[user] + self.b_i[item] + np.dot(self.U[user],
                                                                                   self.V[item]) + context_bias
                err = rating - pred
                total_loss += err ** 2

                # Обновление латентных векторов
                U_old = self.U[user].copy()
                self.U[user] += self.learning_rate * (err * self.V[item] - self.reg * self.U[user])
                self.V[item] += self.learning_rate * (err * U_old - self.reg * self.V[item])

                # Обновление смещений
                self.b_u[user] += self.learning_rate * (err - self.reg * self.b_u[user])
                self.b_i[item] += self.learning_rate * (err - self.reg * self.b_i[item])
                # Обновление контекстных смещений
                for col in self.context_cols:
                    key = (col, row[col])
                    self.b_context[key] += self.learning_rate * (err - self.reg * self.b_context[key])
            rmse = np.sqrt(total_loss / len(data))
            self.logger.info("Эпоха %d/%d: RMSE = %.4f", epoch, self.epochs, rmse)
        self.logger.info("Обучение CAMF модели завершено.")

    def predict_score(self, user, item, context_dict: dict = None):
        """
        Расчёт предсказанного рейтинга для пользователя и объекта.
        :param user: идентификатор пользователя.
        :param item: идентификатор объекта.
        :param context_dict: словарь с контекстными признаками по умолчанию (например, {'gender': 'M', 'age': 25})
        :return: предсказанный рейтинг.
        """
        context_bias = 0.0
        if context_dict is not None:
            for col in self.context_cols:
                val = context_dict.get(col, None)
                if val is not None:
                    context_bias += self.b_context.get((col, val), 0.0)
        return self.global_bias + self.b_u[user] + self.b_i[item] + np.dot(self.U[user], self.V[item]) + context_bias

    def predict(self, default_context: dict = None):
        """
        Для каждого пользователя рассчитываем предсказанные рейтинги для объектов,
        с которыми он ещё не взаимодействовал, и возвращаем ранжированный список.
        :param default_context: словарь с контекстом, который используется для предсказания.
        :return: словарь {user: [item1, item2, ...]}.
        """
        self.logger.info("Начало генерации рекомендаций (CAMF).")
        recommendations = {}
        user_items = self.dataset.groupby(self.user_col)[self.item_col].apply(set).to_dict()
        for user in self.users:
            scores = {}
            for item in self.items:
                if item in user_items.get(user, set()):
                    continue
                score = self.predict_score(user, item, default_context)
                scores[item] = score
            ranked_items = sorted(scores, key=scores.get, reverse=True)
            recommendations[user] = ranked_items
        self.logger.info("Генерация рекомендаций (CAMF) завершена.")
        return recommendations


# Реализация модели CSLIM (Item-based Collaborative Filtering с косинусной схожестью)
class CSLIMModel:
    def __init__(self, dataset: pd.DataFrame, user_col: str, item_col: str,
                 rating_col: str, context_cols: list, **params):
        self.logger = logging.getLogger("CSLIMModel")
        self.dataset = dataset
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.context_cols = context_cols  # в данной реализации не используются, но могут быть задействованы для дообучения
        self.use_ratings = params.get('use_ratings', True)
        self.item_similarity = None
        self.user_item_matrix = None

    def fit(self):
        """
        Формируем матрицу взаимодействий и вычисляем косинусную схожесть между объектами.
        """
        self.logger.info("Формирование матрицы 'пользователь-объект' для CSLIM")
        self.user_item_matrix = self.dataset.pivot_table(index=self.user_col,
                                                         columns=self.item_col,
                                                         values=self.rating_col,
                                                         fill_value=0)
        R = self.user_item_matrix.values.astype(np.float32)
        norms = norm(R, axis=0)
        norms[norms == 0] = 1e-10
        similarity_matrix = np.dot(R.T, R) / (np.outer(norms, norms))
        items = self.user_item_matrix.columns
        self.item_similarity = {}
        for i, item_i in enumerate(items):
            sim_dict = {}
            for j, item_j in enumerate(items):
                if item_i == item_j:
                    continue
                sim_dict[item_j] = similarity_matrix[i, j]
            self.item_similarity[item_i] = sim_dict
        self.logger.info("Вычисление схожести объектов завершено.")

    def predict(self, default_context: dict = None):
        """
        Для каждого пользователя рассчитываем сумму взвешенных оценок по схожести для объектов,
        с которыми он ещё не взаимодействовал, и возвращаем ранжированный список.
        :param default_context: в данной реализации не используется.
        :return: словарь {user: [item1, item2, ...]}.
        """
        self.logger.info("Начало генерации рекомендаций (CSLIM).")
        recommendations = {}
        user_groups = self.dataset.groupby(self.user_col)
        for user, group in user_groups:
            interacted = group.set_index(self.item_col)[self.rating_col].to_dict()
            scores = defaultdict(float)
            for item_i, rating in interacted.items():
                for item_j, sim in self.item_similarity.get(item_i, {}).items():
                    if item_j in interacted:
                        continue
                    scores[item_j] += sim * rating
            ranked_items = sorted(scores, key=scores.get, reverse=True)
            recommendations[user] = ranked_items
        self.logger.info("Генерация рекомендаций (CSLIM) завершена.")
        return recommendations


# Пример использования:
if __name__ == "__main__":
    # Пример датасета с контекстными признаками
    data = {
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u3', 'u4'],
        'item_id': ['i1', 'i2', 'i2', 'i3', 'i1', 'i3', 'i2'],
        'rating': [5, 3, 4, 2, 1, 5, 4],
        'gender': ['M', 'M', 'F', 'F', 'M', 'M', 'F'],  # контекст: пол
        'age': [25, 25, 30, 30, 22, 22, 28]  # контекст: возраст
    }
    df = pd.DataFrame(data)

    logger.info("=== Запуск CAMF модели ===")
    camf_recommender = UniversalContextualRecommender(df, model_name='camf',
                                                      user_col='user_id', item_col='item_id',
                                                      rating_col='rating',
                                                      context_cols=['gender', 'age'],
                                                      n_factors=5, learning_rate=0.01, reg=0.02, epochs=15)
    camf_recommender.fit()
    default_context = {'gender': 'M', 'age': 25}
    camf_recs = camf_recommender.predict(default_context=default_context)
    logger.info("Рекомендации (CAMF) для пользователей:")
    for user, items in camf_recs.items():
        logger.info("Пользователь %s: %s", user, items)

    logger.info("=== Запуск CSLIM модели ===")
    cslim_recommender = UniversalContextualRecommender(df, model_name='cslim',
                                                       user_col='user_id', item_col='item_id',
                                                       rating_col='rating', context_cols=['gender', 'age'],
                                                       use_ratings=True)
    cslim_recommender.fit()
    cslim_recs = cslim_recommender.predict()
    logger.info("Рекомендации (CSLIM) для пользователей:")
    for user, items in cslim_recs.items():
        logger.info("Пользователь %s: %s", user, items)
