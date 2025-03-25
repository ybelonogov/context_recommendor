import os
import pandas as pd
import numpy as np

# Импортируем необходимые компоненты из Cornac
import cornac
from cornac.data import Dataset
from cornac.models import MF, BPR, PMF, SVD
from cornac.models.recommender import Recommender
from cornac.eval_methods import BaseMethod


class ContextualRecommender(Recommender):
    """
    Класс, который:
    - получает датасет (с информацией о пользователях, ресторанах, рейтингах и контексте),
    - инициализирует выбранную модель Cornac,
    - обучает модель,
    - выдаёт top-N рекомендаций для каждого пользователя.
    """

    def __init__(
            self,
            model_name: str,
            train_data: pd.DataFrame,
            user_col: str = "user_id",
            item_col: str = "item_id",
            rating_col: str = "rating",
            context_col: str = "last_click",
            rating_scale=(1, 5),
            name="ContextualRecommender",
            **model_params
    ):
        super().__init__(name=name)
        self.model_name = model_name
        self.train_data = train_data
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.context_col = context_col
        self.rating_scale = rating_scale
        self.model_params = model_params

        # Проверка наличия нужных колонок
        for col in [user_col, item_col, rating_col, context_col]:
            if col not in train_data.columns:
                raise ValueError(f"Столбец '{col}' не найден в train_data.")

        self.user_mapping = {}
        self.item_mapping = {}
        self.context_mapping = {}

        self.model = self._init_model()

    def _init_model(self):
        model_dict = {
            "MF": MF,
            "BPR": BPR,
            "PMF": PMF,
            "SVD": SVD,
        }
        if self.model_name not in model_dict:
            raise ValueError(
                f"Модель '{self.model_name}' не поддерживается. Доступные варианты: {list(model_dict.keys())}"
            )
        model_cls = model_dict[self.model_name]
        model = model_cls(**self.model_params)
        return model

    def build(self):
        """
        Подготавливает данные:
        - создаёт маппинги для пользователей, ресторанов и контекста,
        - формирует список кортежей (user, item, rating) для Cornac.Dataset.
        """
        self.user_mapping = {u: idx for idx, u in enumerate(self.train_data[self.user_col].unique())}
        self.item_mapping = {i: idx for idx, i in enumerate(self.train_data[self.item_col].unique())}
        self.context_mapping = {c: idx for idx, c in enumerate(self.train_data[self.context_col].unique())}

        data_uir = []
        for _, row in self.train_data.iterrows():
            user_idx = self.user_mapping[row[self.user_col]]
            item_idx = self.item_mapping[row[self.item_col]]
            rating_val = float(row[self.rating_col])
            data_uir.append((user_idx, item_idx, rating_val))

        # Сохраняем список взаимодействий для дальнейшего использования
        self.data_uir = data_uir
        self.dataset = Dataset.from_uir(data_uir)

    def fit(self):
        if not hasattr(self, "dataset"):
            raise ValueError("Сначала вызовите метод build() для подготовки данных.")
        eval_method = BaseMethod.from_splits(
            train_data=self.data_uir,
            val_data=None,
            test_data=self.data_uir,  # передаем непустой тестовый набор
            rating_threshold=None
        )
        self.model.fit(eval_method.train_set)

    def score(self, user_idx, item_idx, context_idx=None):
        # Стандартные модели Cornac не учитывают контекст «из коробки».
        # Здесь context_idx пока не используется – можно расширить функционал при необходимости.
        return self.model.score(user_idx, item_idx)

    def recommend_for_user(self, user_raw_id, top_n=5, given_context=None):
        """
        Выдаёт top-N рекомендаций для указанного пользователя.
        Если задан контекст (например, последний клик), он может быть использован в дальнейшем расширении логики.
        """
        if user_raw_id not in self.user_mapping:
            return []

        user_idx = self.user_mapping[user_raw_id]
        if given_context is not None:
            context_idx = self.context_mapping.get(given_context, 0)
        else:
            context_idx = 0

        all_items = list(self.item_mapping.keys())
        scores = []
        for it_raw in all_items:
            it_idx = self.item_mapping[it_raw]
            pred = self.score(user_idx, it_idx, context_idx=context_idx)
            scores.append((it_raw, pred))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


if __name__ == "__main__":
    # Путь к CSV-файлу внутри папки "restaurant+consumer+data".
    data_file = os.path.join("restaurant+consumer+data", "Restaurant_Consumer_Data.csv")

    # Если файл не найден, создадим демонстрационный DataFrame (можно убрать этот блок, если файл точно присутствует)
    if not os.path.exists(data_file):
        print(f"Файл {data_file} не найден. Создаём демонстрационные данные.")
        # Демонстрационные данные с колонками, аналогичными оригинальному датасету
        demo_data = {
            "ConsumerID": [101, 102, 103, 101, 102],
            "RestaurantID": [201, 202, 203, 204, 205],
            "Rating": [5, 3, 4, 2, 5],
            # Допустим, информации по последнему клику нет, добавляем фиктивную колонку
            "last_click": ["default", "default", "default", "default", "default"]
        }
        df = pd.DataFrame(demo_data)
    else:
        # Загружаем датасет из CSV
        df = pd.read_csv(data_file)

    # Выводим первые строки датасета для ознакомления
    print("Первые строки датасета:")
    print(df.head())

    # Если в датасете отсутствует колонка для контекста (например, 'last_click'),
    # добавляем фиктивный столбец со значением 'default'.
    if "last_click" not in df.columns:
        df["last_click"] = "default"

    # Если имена столбцов не совпадают с требуемыми, выполняем переименование.
    # Например, если в датасете имена "ConsumerID" и "RestaurantID", переименовываем их.
    if "ConsumerID" in df.columns and "RestaurantID" in df.columns:
        df = df.rename(columns={"ConsumerID": "user_id", "RestaurantID": "item_id"})
    elif "user_id" not in df.columns or "item_id" not in df.columns:
        raise ValueError("Невозможно обнаружить нужные колонки для пользователей и ресторанов.")

    # Если в датасете нет колонки с рейтингом, создаём или генерируем её.
    if "rating" not in df.columns:
        np.random.seed(42)
        df["rating"] = np.random.randint(1, 6, size=len(df))

    # Инициализируем рекомендатель.
    recommender = ContextualRecommender(
        model_name="SVD",  # Можно выбирать: MF, BPR, PMF, SVD
        train_data=df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        context_col="last_click",
        rating_scale=(1, 5),
        k=10,
        max_iter=50
    )

    # Подготавливаем данные и обучаем модель
    recommender.build()
    recommender.fit()

    # Для демонстрации выбираем первого пользователя из датасета
    sample_user = df["user_id"].iloc[0]
    print(f"\nTop-5 рекомендаций для пользователя {sample_user} (контекст 'default'):")
    recommendations = recommender.recommend_for_user(user_raw_id=sample_user, top_n=5, given_context="default")
    for rank, (item, score) in enumerate(recommendations, start=1):
        print(f"{rank}. Ресторан: {item}, предсказанный рейтинг: {score:.4f}")
