import os
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

# Импортируем наш класс для рекомендаций
from recommender import Recommender

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='experiment.log',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def run_recommendation_pipeline(dataset_type, model_name, output_csv, top_n=10, **model_kwargs):
    """
    Запускает модель рекомендаций на датасете FilmTrust или MovieLens и сохраняет результаты в CSV.

    Аргументы:
      dataset_type (str): 'filmtrust' или 'movielens'
      model_name (str): название модели ('FM', 'SoRec', 'CTR')
      output_csv (str): путь к CSV-файлу для сохранения результатов
      top_n (int): количество рекомендаций для каждого пользователя (по умолчанию 10)
      model_kwargs: дополнительные параметры для модели Cornac

    Возвращает:
      dict: Рекомендации в формате {user: [item1, item2, ...]}
    """

    if dataset_type.lower() == 'filmtrust':
        ratings_path = os.path.join('filmtrust', 'ratings.txt')
        trust_path = os.path.join('filmtrust', 'trust.txt')

        ratings_df = pd.read_csv(ratings_path, sep=' ', header=None, names=['user', 'item', 'rating'])
        ratings = ratings_df.values.tolist()

        trust_df = pd.read_csv(trust_path, sep=' ', header=None, names=['user', 'friend'])

        user_trust_edges = [(u, v, 1) for u, v in trust_df.values]

        unique_items = ratings_df['item'].unique()
        item_desc = {item: f"Описание фильма {item}" for item in unique_items}

    elif dataset_type.lower() == 'movielens':
        data_dir = 'ml-20m/'
        ratings_df = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
        movies_df = pd.read_csv(os.path.join(data_dir, 'movies.csv'))

        ratings = list(ratings_df[['userId', 'movieId', 'rating']].itertuples(index=False, name=None))

        user_trust_edges = None

        item_desc = {row['movieId']: row['title'] for _, row in movies_df.iterrows()}
    else:
        raise ValueError("Неверный тип датасета. Используйте 'filmtrust' или 'movielens'.")

    np.random.seed(42)
    train_data = []
    test_data = []
    ratings_by_user = defaultdict(list)
    for uid, iid, rating in ratings:
        ratings_by_user[uid].append((iid, rating))
    for uid, items in ratings_by_user.items():
        if len(items) < 2:
            train_data.append((uid, items[0][0], items[0][1]))
        else:
            np.random.shuffle(items)
            cutoff = int(len(items) * 0.8)
            for iid, rating in items[:cutoff]:
                train_data.append((uid, iid, rating))
            for iid, rating in items[cutoff:]:
                test_data.append((uid, iid, rating))

    logging.info("Dataset %s: Всего взаимодействий: %d, тренировка: %d, тест: %d",
                 dataset_type, len(ratings), len(train_data), len(test_data))

    if model_name.lower() == 'sorec':
        recommender = Recommender(train_data, model=model_name, user_graph=user_trust_edges, **model_kwargs)
    elif model_name.lower() == 'ctr':
        recommender = Recommender(train_data, model=model_name, item_text=item_desc, **model_kwargs)
    else:
        recommender = Recommender(train_data, model=model_name, **model_kwargs)

    recommender.fit()

    recommendations = {}
    for user in recommender.user_ids:
        recommendations[user] = recommender.recommend(user, top_n=top_n)

    recs_list = []
    for user, recs in recommendations.items():
        recs_str = ",".join(map(str, recs))
        recs_list.append({'user': user, 'recommendations': recs_str})

    recs_df = pd.DataFrame(recs_list)
    recs_df.to_csv(output_csv, index=False)
    logging.info("Рекомендации для %s модели на датасете %s сохранены в %s",
                 model_name, dataset_type, output_csv)

    return recommendations


if __name__ == "__main__":

    experiments = [
        # Эксперименты для FilmTrust
        # {
        #     'dataset_type': 'filmtrust',
        #     'model_name': 'FM',
        #     'output_csv': 'filmtrust_FM_recommendations.csv',
        #     'top_n': 10,
        #     'model_kwargs': {'k2': 3, 'max_iter': 10, 'seed': 42, 'verbose': True}
        # },
        {
            'dataset_type': 'filmtrust',
            'model_name': 'SoRec',
            'output_csv': 'filmtrust_SoRec_recommendations.csv',
            'top_n': 10,
            'model_kwargs': {'k': 3, 'max_iter': 10, 'seed': 42, 'verbose': True}
        },
        {
            'dataset_type': 'filmtrust',
            'model_name': 'CTR',
            'output_csv': 'filmtrust_CTR_recommendations.csv',
            'top_n': 10,
            'model_kwargs': {'k': 3, 'max_iter': 10, 'seed': 42, 'verbose': True}
        },
        # Эксперименты для MovieLens
        # {
        #     'dataset_type': 'movielens',
        #     'model_name': 'FM',
        #     'output_csv': 'movielens_FM_recommendations.csv',
        #     'top_n': 10,
        #     'model_kwargs': {'k2': 3, 'max_iter': 2, 'seed': 42, 'verbose': True}
        # },
        {
            'dataset_type': 'movielens',
            'model_name': 'CTR',
            'output_csv': 'movielens_CTR_recommendations.csv',
            'top_n': 10,
            'model_kwargs': {'k': 3, 'max_iter': 2, 'seed': 42, 'verbose': True}
        }
    ]

    for exp in experiments:
        try:
            logging.info("Запуск эксперимента: Dataset: %s, Model: %s",
                         exp['dataset_type'], exp['model_name'])
            run_recommendation_pipeline(
                dataset_type=exp['dataset_type'],
                model_name=exp['model_name'],
                output_csv=exp['output_csv'],
                top_n=exp['top_n'],
                **exp['model_kwargs']
            )
            logging.info("Эксперимент успешно завершён: Dataset: %s, Model: %s",
                         exp['dataset_type'], exp['model_name'])
        except Exception as e:
            logging.error("Ошибка в эксперименте: Dataset: %s, Model: %s. Ошибка: %s",
                          exp['dataset_type'], exp['model_name'], str(e))
