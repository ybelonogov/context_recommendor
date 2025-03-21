import logging
import numpy as np
from collections import defaultdict

# Подключаем нужные модели
from cornac.data import GraphModality
from cornac.models import (
    FM,      # Factorization Machine
    SoRec,   # Social Recommendation
    PMF,     # Probabilistic Matrix Factorization
    BPR,     # Bayesian Personalized Ranking
    SVD,     # SVD-based collaborative filtering
    NMF      # Non-negative Matrix Factorization
)


class Recommender:
    def __init__(
        self,
        data,
        model='FM',
        user_graph=None,
        user_subset_ratio=1.0,
        min_interactions=2,
        use_full_dataset=False,
        seed=42,
        **model_kwargs
    ):
        """
        Параметры:
          data: List[(user, item, rating, возможно timestamp)],
                все взаимодействия (explicit), например из MovieLens/FilmTrust.
          model: Строка с названием модели, одно из:
                 ['fm', 'sorec', 'pmf', 'bpr', 'svd', 'nmf'] (регистр не важен).
          user_graph: Если используем SoRec, можно передать список ребер [(u,v,1), ...].
          user_subset_ratio: какую долю пользователей брать [0..1].
          min_interactions: минимальное число взаимодействий, чтобы пользователь попал в выборку.
          use_full_dataset: если True — всё в train, нет test.
          seed: для воспроизводимости random.
          **model_kwargs: доп. параметры, передаваемые в выбранную модель.
        """

        self.model_name = model
        self.user_graph = None
        np.random.seed(seed)

        # 1) Группируем взаимодействия по пользователям
        ratings_by_user = defaultdict(list)
        for row in data:
            user = row[0]
            item = row[1]
            rating = row[2]
            # Если есть 4-й элемент (timestamp), захватим его
            timestamp = row[3] if len(row) > 3 else None
            ratings_by_user[user].append((item, rating, timestamp))

        # 2) Фильтруем пользователей с достаточным кол-вом интеракций
        filtered_users = []
        for u, interactions in ratings_by_user.items():
            if len(interactions) >= min_interactions:
                filtered_users.append(u)

        # 3) Даунсэмплинг пользователей (user_subset_ratio)
        if user_subset_ratio < 1.0:
            subset_size = int(len(filtered_users) * user_subset_ratio)
            selected_users = set(np.random.choice(filtered_users, size=subset_size, replace=False))
        else:
            selected_users = set(filtered_users)

        self.train_data = []
        self.test_data = []

        # 4) Для каждого пользователя:
        #    - Сортируем взаимодействия по timestamp (если есть),
        #    - Если use_full_dataset=False, последнее уходит в test, остальные в train,
        #      иначе всё в train
        for u in selected_users:
            items = ratings_by_user[u]
            # Если есть timestamp, сортируем
            if items[0][2] is not None:
                items = sorted(items, key=lambda x: x[2])  # сортируем по времени

            if use_full_dataset:
                for (itm, r, ts) in items:
                    self.train_data.append((u, itm, r))
            else:
                # Последняя запись в test, остальные в train
                if len(items) == 1:
                    self.train_data.append((u, items[0][0], items[0][1]))
                else:
                    *train_part, last_inter = items
                    for (itm, r, ts) in train_part:
                        self.train_data.append((u, itm, r))
                    self.test_data.append((u, last_inter[0], last_inter[1]))

        logging.info(
            "Recommender: выбрано %d пользователей, train=%d, test=%d",
            len(selected_users), len(self.train_data), len(self.test_data)
        )

        # 5) Если SoRec — подключаем social graph
        #    (для других моделей user_graph игнорируется)
        # if user_graph is not None and model.lower() == 'sorec':
        #     self.user_graph = GraphModality(data=user_graph, symmetric=True)

        # 6) Создаём нужную модель
        model_lower = model.lower()
        if model_lower == 'fm':
            self.model = FM(seed=seed, **model_kwargs)
        elif model_lower == 'sorec':
            self.model = SoRec(seed=seed, **model_kwargs)
        elif model_lower == 'pmf':
            self.model = PMF(seed=seed, **model_kwargs)
        elif model_lower == 'bpr':
            self.model = BPR(seed=seed, **model_kwargs)
        elif model_lower == 'svd':
            self.model = SVD(seed=seed, **model_kwargs)
        elif model_lower == 'nmf':
            self.model = NMF(seed=seed, **model_kwargs)
        else:
            raise ValueError(f"Неизвестная модель: {model}")

        # 7) Определяем ID-шники юзеров и айтемов (только из train, чтобы сошлось с моделью Cornac)
        self.user_ids = sorted({u for u, _, _ in self.train_data})
        self.item_ids = sorted({i for _, i, _ in self.train_data})
        self.user_to_index = {u: idx for idx, u in enumerate(self.user_ids)}
        self.item_to_index = {i: idx for idx, i in enumerate(self.item_ids)}

        # 8) Запомним, какие items юзер видел в train (чтобы не рекомендовать их повторно)
        self.train_items_by_user = defaultdict(set)
        for u, i, r in self.train_data:
            self.train_items_by_user[u].add(i)


    def fit(self):
        """Тренируем выбранную модель. Если есть test, считаем метрики."""
        from cornac.eval_methods import RatioSplit

        # Создаём RatioSplit с test_size=0.0, т.к. Cornac мы передаём только train
        ratio_split = RatioSplit(
            data=self.train_data,
            test_size=0.0,
            user_graph=self.user_graph,
            exclude_unknowns=False,
            seed=0
        )
        train_set = ratio_split.train_set
        self.model.fit(train_set)

        # Если есть тест, логируем метрики
        if len(self.test_data) > 0:
            p, r, ndcg = self.evaluate(self.test_data, k=10)
            logging.info(
                "%s Test => P@10=%.4f, R@10=%.4f, NDCG@10=%.4f",
                self.model_name.upper(), p, r, ndcg
            )


    def recommend(self, user_id, top_n=10):
        """Формируем top-N для данного user_id."""
        if user_id not in self.user_to_index:
            return []
        import numpy as np
        u_idx = self.user_to_index[user_id]
        scores = self.model.score(u_idx)  # вектор скоров для каждого item
        scores = np.array(scores)

        # Убираем items, которые юзер видел
        seen = self.train_items_by_user[user_id]
        for it in seen:
            if it in self.item_to_index:
                idx = self.item_to_index[it]
                scores[idx] = -np.inf

        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        top_items = [self.item_ids[i] for i in top_indices]
        return top_items


    def evaluate(self, test_data, k=10, rating_threshold=2.5):
        """
        Считаем Precision@K, Recall@K, NDCG@K на test_data.
        Предполагается, что test_data = [(user, item, rating), ...].
        """
        import numpy as np
        true_items_by_user = defaultdict(set)
        for u, i, r in test_data:
            if r >= rating_threshold:
                true_items_by_user[u].add(i)

        prec_list, rec_list, ndcg_list = [], [], []

        for user, relevant_items in true_items_by_user.items():
            if user not in self.user_to_index:
                continue
            top_k = self.recommend(user, top_n=k)
            hits = sum(1 for it in top_k if it in relevant_items)

            prec = hits / k
            rec = hits / len(relevant_items)
            prec_list.append(prec)
            rec_list.append(rec)

            # NDCG
            dcg, idcg = 0.0, 0.0
            for rank, it in enumerate(top_k, start=1):
                rel = 1.0 if it in relevant_items else 0.0
                dcg += rel / np.log2(rank + 1)
            # Ideal DCG
            sorted_relevances = [1.0]*len(relevant_items) + [0.0]*(k - len(relevant_items))
            for rank, rel in enumerate(sorted_relevances[:k], start=1):
                idcg += rel / np.log2(rank + 1)
            ndcg = dcg/idcg if idcg > 0 else 0.0
            ndcg_list.append(ndcg)

        p = np.mean(prec_list) if prec_list else 0.0
        r = np.mean(rec_list) if rec_list else 0.0
        n = np.mean(ndcg_list) if ndcg_list else 0.0
        return p, r, n
