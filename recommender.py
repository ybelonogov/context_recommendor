from cornac.data import GraphModality, TextModality
from cornac.data.text import BaseTokenizer



from cornac.models import FM, SoRec, CTR
from cornac.metrics import Precision, Recall, NDCG
import numpy as np

class Recommender:
    def __init__(self, train_data, model=None, user_graph=None, item_text=None, **model_kwargs):
        """
        train_data: list of (user, item, rating) tuples for training.
        model: either a string name of a Cornac model or an instantiated model object.
        user_graph: list of (user, user) trust pairs or Cornac GraphModality for social context.
        item_text: dict or list of (item, text) for item descriptions, or Cornac TextModality.
        model_kwargs: additional keyword args to pass when instantiating the model (if model is a name).
        """
        self.train_data = train_data

        # Set up user and item ID mappings for consistent indexing
        self.user_ids = sorted({u for u, _, _ in train_data})
        self.item_ids = sorted({i for _, i, _ in train_data})
        self.user_to_index = {u: idx for idx, u in enumerate(self.user_ids)}
        self.item_to_index = {i: idx for idx, i in enumerate(self.item_ids)}

        # Prepare Cornac modalities for user graph and item text if provided
        self.user_graph = None
        if user_graph is not None:
            # If raw list of edges provided, create GraphModality
            if isinstance(user_graph, GraphModality):
                self.user_graph = user_graph
            else:
                self.user_graph = GraphModality(data=user_graph, symmetric=True)

        self.item_text = None
        if item_text is not None:
            if isinstance(item_text, TextModality):
                self.item_text = item_text
            else:
                # If item_text is a dict or list of (id, text), convert to TextModality
                if isinstance(item_text, dict):
                    ids, docs = zip(*item_text.items())
                else:  # list of tuples
                    ids, docs = zip(*item_text)
                # Ensure item IDs used in text are in training items
                ids = list(ids)
                docs = list(docs)
                # Initialize TextModality (use basic tokenizer and limit vocabulary size for speed)

                # Создание токенизатора
                tokenizer = BaseTokenizer()
                self.item_text = TextModality(corpus=docs, ids=ids, tokenizer=tokenizer, max_doc_freq=1.0)

        # Instantiate the model if a name is given
        if isinstance(model, str):
            model_name = model.lower()
            if model_name == 'fm':
                # Factorization Machine model (we use default or provided k2 factors)
                self.model = FM(**model_kwargs)
            elif model_name == 'sorec':
                self.model = SoRec(**model_kwargs)
            elif model_name == 'ctr':
                self.model = CTR(**model_kwargs)
            else:
                raise ValueError(f"Unknown model name: {model}")
        else:
            # If an instance of a Cornac model is provided
            self.model = model

        # Store training user->items mapping for filtering in recommendations
        self.train_items_by_user = {}
        for u, i, r in train_data:
            self.train_items_by_user.setdefault(u, set()).add(i)

    def fit(self):
        """Train the model on the training data (leveraging any provided side information)."""
        # Cornac models expect data as an iterable of (user, item, rating)
        # We can train directly if modalities were set at model init; otherwise, we use an evaluation method.

        # If we have side info, use Cornac's RatioSplit to prepare data and then fit through Cornac's Experiment or directly.
        from cornac.eval_methods import RatioSplit
        ratio_split = RatioSplit(data=self.train_data, test_size=0.0, exclude_unknowns=False,
                                 user_graph=self.user_graph, item_text=self.item_text, seed=0)
        train_set = ratio_split.train_set

        # Fit the model on the training set
        self.model.fit(train_set)
        return self

    def recommend(self, user_id, top_n=10):
        """Generate top-N recommendations for a given user."""
        if user_id not in self.user_to_index:
            return []  # Unknown user
        uidx = self.user_to_index[user_id]
        # Get score predictions for all items for this user
        scores = self.model.score(uidx)  # returns scores for all items
        scores = np.array(scores)
        # Exclude items the user has already interacted with in training
        if user_id in self.train_items_by_user:
            seen_items = self.train_items_by_user[user_id]
            seen_indices = [self.item_to_index[i] for i in seen_items if i in self.item_to_index]
            scores[seen_indices] = -np.inf  # lower the score to exclude seen items
        # Get indices of top N scores
        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        # Map back to item IDs
        top_items = [self.item_ids[idx] for idx in top_indices]
        return top_items

    def evaluate(self, test_data, k=10, rating_threshold=2.5):
        """
        Compute Precision@K, Recall@K, and NDCG@K on the provided test data.
        Only items with rating >= rating_threshold are considered "relevant" for evaluation.
        """
        # Build ground-truth relevance sets from test data
        true_items_by_user = {}
        for u, i, r in test_data:
            if r >= rating_threshold:
                true_items_by_user.setdefault(u, set()).add(i)

        # Compute metrics
        prec_list = []
        rec_list = []
        ndcg_list = []
        for user, true_items in true_items_by_user.items():
            if not true_items:
                continue  # skip users with no relevant items
            # Get top-K recommendations for the user
            top_k = self.recommend(user, top_n=k)
            if not top_k:
                continue
            # Precision@K: fraction of recommended items that are relevant
            hits = sum(1 for item in top_k if item in true_items)
            prec = hits / k
            prec_list.append(prec)
            # Recall@K: fraction of relevant items that are recommended
            rec = hits / len(true_items)
            rec_list.append(rec)
            # NDCG@K: discounted gain of recommended hits
            dcg = 0.0
            idcg = 0.0
            for rank, item in enumerate(top_k, start=1):
                rel = 1.0 if item in true_items else 0.0
                dcg += rel / np.log2(rank + 1)
            # Calculate ideal DCG (IDCG) for this user (i.e., all top K are relevant if possible)
            sorted_relevances = sorted([1.0] * len(true_items) + [0.0] * (k - len(true_items)), reverse=True)
            for rank, rel in enumerate(sorted_relevances, start=1):
                idcg += rel / np.log2(rank + 1)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_list.append(ndcg)

        # Average metrics over all users
        precision_at_k = np.mean(prec_list) if prec_list else 0.0
        recall_at_k = np.mean(rec_list) if rec_list else 0.0
        ndcg_at_k = np.mean(ndcg_list) if ndcg_list else 0.0
        return precision_at_k, recall_at_k, ndcg_at_k


def generate_recommendations(train_data, model_name, top_n=5, **model_kwargs):
    """
    Функция принимает:
      - train_data: список кортежей (user, item, rating)
      - model_name: строка с названием модели (например, 'FM', 'SoRec', 'CTR')
      - top_n: число рекомендуемых айтемов для каждого пользователя (по умолчанию 5)
      - model_kwargs: дополнительные параметры для модели Cornac
    Функция обучает модель и возвращает словарь, где ключ – идентификатор пользователя,
    а значение – список рекомендованных айтемов.
    """
    # Инициализируем и обучаем модель с использованием CornacRecommender
    recommender = Recommender(train_data, model=model_name, **model_kwargs)
    recommender.fit()

    recommendations = {}
    # Проходим по всем пользователям, имеющим данные в тренировочном наборе
    for user in recommender.user_ids:
        recommendations[user] = recommender.recommend(user, top_n=top_n)

    return recommendations