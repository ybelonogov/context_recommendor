import os
import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from numpy.linalg import norm
import random

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("UniversalContextualRecommender")

#############################################
# Wrapper для модели LightFM из библиотеки LightFM
from lightfm import LightFM
from lightfm.data import Dataset as LightFMDataset

class LightFMModel:
    def __init__(self, dataset: pd.DataFrame, user_col: str, item_col: str,
                 rating_col: str, context_cols: list, **params):
        self.logger = logging.getLogger("LightFMModel")
        self.dataset = dataset
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.context_cols = context_cols  # контекст не используется стандартно
        self.epochs = params.get('epochs', 10)
        self.learning_rate = params.get('learning_rate', 0.05)
        self.model = LightFM(loss='warp', learning_rate=self.learning_rate)
        self.lightfm_dataset = LightFMDataset()
        self.interactions = None

    def fit(self):
        self.logger.info("Начало обучения LightFM модели.")
        users = self.dataset[self.user_col].unique().tolist()
        items = self.dataset[self.item_col].unique().tolist()
        self.lightfm_dataset.fit(users, items)
        interactions_iter = (
            (row[self.user_col], row[self.item_col], row[self.rating_col])
            for _, row in self.dataset.iterrows()
        )
        self.interactions, _ = self.lightfm_dataset.build_interactions(interactions_iter)
        self.model.fit(self.interactions, epochs=self.epochs, num_threads=4)
        self.logger.info("Обучение LightFM модели завершено.")

    def predict(self, default_context: dict = None):
        self.logger.info("Генерация рекомендаций с помощью LightFM.")
        user_id_map, _, item_id_map, _ = self.lightfm_dataset.mapping()
        user_reverse = {uid: user for user, uid in user_id_map.items()}
        item_reverse = {iid: item for item, iid in item_id_map.items()}
        num_users, num_items = self.interactions.shape
        recommendations = {}
        for user in range(num_users):
            scores = self.model.predict(user, np.arange(num_items))
            top_items = np.argsort(-scores)[:10]
            recommendations[user_reverse[user]] = [item_reverse[i] for i in top_items]
        self.logger.info("Рекомендации с LightFM сгенерированы.")
        return recommendations

#############################################
# Wrapper для модели SVD из библиотеки Surprise
try:
    from surprise import SVD, Dataset as SurpriseDataset, Reader
except ImportError:
    SVD = None

class SurpriseSVDModel:
    def __init__(self, dataset: pd.DataFrame, user_col: str, item_col: str,
                 rating_col: str, context_cols: list, **params):
        self.logger = logging.getLogger("SurpriseSVDModel")
        self.dataset = dataset
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.context_cols = context_cols
        self.epochs = params.get('epochs', 20)
        self.lr_all = params.get('lr_all', 0.005)
        self.reg_all = params.get('reg_all', 0.02)
        if SVD is None:
            raise ImportError("Библиотека Surprise не установлена.")
        self.model = SVD(n_epochs=self.epochs, lr_all=self.lr_all, reg_all=self.reg_all)
        self.trainset = None

    def fit(self):
        self.logger.info("Начало обучения Surprise SVD модели.")
        from surprise import Reader
        reader = Reader(rating_scale=(self.dataset[self.rating_col].min(), self.dataset[self.rating_col].max()))
        data = SurpriseDataset.load_from_df(self.dataset[[self.user_col, self.item_col, self.rating_col]], reader)
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)
        self.logger.info("Обучение Surprise SVD модели завершено.")

    def predict(self, default_context: dict = None):
        self.logger.info("Генерация рекомендаций с помощью Surprise SVD.")
        recommendations = {}
        users = self.trainset.all_users()
        items = self.trainset.all_items()
        user_reverse = {uid: self.trainset.to_raw_uid(uid) for uid in users}
        item_reverse = {iid: self.trainset.to_raw_iid(iid) for iid in items}
        for u in users:
            scores = {}
            for i in items:
                pred = self.model.predict(self.trainset.to_raw_uid(u), self.trainset.to_raw_iid(i)).est
                scores[i] = pred
            top_items = sorted(scores, key=scores.get, reverse=True)[:10]
            recommendations[user_reverse[u]] = [item_reverse[i] for i in top_items]
        self.logger.info("Рекомендации Surprise SVD сгенерированы.")
        return recommendations

#############################################
# Универсальный класс, позволяющий выбрать библиотечную модель
class UniversalContextualRecommender:
    def __init__(self, dataset: pd.DataFrame, model_name: str,
                 user_col: str = 'user_id', item_col: str = 'item_id',
                 rating_col: str = 'rating', context_cols: list = None, **model_params):
        """
        :param dataset: DataFrame с данными.
        :param model_name: название модели ('lightfm', 'implicit_als', 'surprise_svd').
        :param user_col: название столбца с идентификаторами пользователей.
        :param item_col: название столбца с идентификаторами объектов.
        :param rating_col: название столбца с оценками.
        :param context_cols: список столбцов с контекстными признаками.
        :param model_params: дополнительные параметры для выбранной модели.
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
        logger.info("Инициализация модели '%s'", self.model_name)
        if self.model_name == 'lightfm':
            self.model = LightFMModel(self.dataset, self.user_col, self.item_col,
                                      self.rating_col, self.context_cols, **self.model_params)
        elif self.model_name == 'surprise_svd':
            self.model = SurpriseSVDModel(self.dataset, self.user_col, self.item_col,
                                          self.rating_col, self.context_cols, **self.model_params)
        else:
            logger.error("Неизвестная или неподдерживаемая модель: %s", self.model_name)
            raise ValueError(f"Неизвестная или неподдерживаемая модель: {self.model_name}")

    def fit(self):
        if self.model is None:
            logger.error("Модель не инициализирована.")
            raise ValueError("Модель не инициализирована.")
        logger.info("Начало обучения модели.")
        self.model.fit()
        logger.info("Обучение модели завершено.")

    def predict(self, default_context: dict = None):
        if self.model is None:
            logger.error("Модель не инициализирована.")
            raise ValueError("Модель не инициализирована.")
        logger.info("Начало генерации рекомендаций.")
        recs = self.model.predict(default_context)
        logger.info("Генерация рекомендаций завершена.")
        return recs

    def pre_filter(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset

    def post_filter(self, recommendations: dict) -> dict:
        return recommendations
