import pandas as pd
import logging
import os

from recommender import Recommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='experiment.log',
    filemode='w'
)

def run_pipeline(dataset_type, model_name, output_csv, user_subset_ratio=1.0,
                 use_full_dataset=False, min_interactions=2, **model_kwargs):
    """
    dataset_type: 'filmtrust' или 'movielens'
    model_name: 'FM', 'SoRec', 'PMF', 'BPR', 'SVD', 'NMF'
    output_csv: куда сохранить рекомендации
    user_subset_ratio: доля пользователей
    use_full_dataset: если True, всё в train
    min_interactions: минимум взаимодействий у юзера
    """
    # 1) Читаем датасет
    if dataset_type.lower() == 'filmtrust':
        ratings_path = 'filmtrust/ratings.txt'
        trust_path   = 'filmtrust/trust.txt'
        ratings_df = pd.read_csv(ratings_path, sep=' ', header=None, names=['user','item','rating'])
        trust_df   = pd.read_csv(trust_path,  sep=' ', header=None, names=['user','friend'])

        # data = [(user, item, rating)]
        data = ratings_df.values.tolist()
        # user_graph = [(u,v,1), ...]
        user_graph = [(u,v,1) for (u,v) in trust_df.values]

    elif dataset_type.lower() == 'movielens':
        data_dir = 'ml-20m'
        ratings_df = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
        # ratings.csv: userId, movieId, rating, timestamp
        data = []
        for row in ratings_df.itertuples(index=False):
            data.append((row.userId, row.movieId, row.rating, row.timestamp))
        user_graph = None
    else:
        raise ValueError("Unknown dataset type")


    #    Если модель SoRec -> user_graph передаём,
    #    Иначе можно None (или игнорировать).
    # if model_name.lower() == 'sorec':
    #     rec = Recommender(
    #         data=data,
    #         model=model_name,
    #         user_graph=user_graph,
    #         user_subset_ratio=user_subset_ratio,
    #         min_interactions=min_interactions,
    #         use_full_dataset=use_full_dataset,
    #         **model_kwargs
    #     )
    # else:
    rec = Recommender(
        data=data,
        model=model_name,
        user_graph=None,  # для PMF, BPR, SVD, NMF, FM соц. граф не нужен
        user_subset_ratio=user_subset_ratio,
        min_interactions=min_interactions,
        use_full_dataset=use_full_dataset,
        **model_kwargs
    )

    rec.fit()

    recs_list = []
    for user in rec.user_ids:
        items = rec.recommend(user, top_n=10)
        recs_str = ",".join(map(str, items))
        recs_list.append({'user': user, 'recommendations': recs_str})

    recs_df = pd.DataFrame(recs_list)
    recs_df.to_csv(output_csv, index=False)
    logging.info("Saved recommendations => %s", output_csv)


if __name__ == "__main__":
    experiments = [
        # {
        #     'dataset_type': 'filmtrust',
        #     'model_name': 'FM',
        #     'output_csv': 'filmtrust_fm.csv',
        #     'user_subset_ratio': 0.001,
        #     'use_full_dataset': False,
        #     'min_interactions': 2,
        #     'seed': 42,
        #     'k2': 3,
        #     'max_iter': 3
        # },
        {
            'dataset_type': 'movielens',
            'model_name': 'PMF',
            'output_csv': 'ml_pmf.csv',
            'user_subset_ratio': 0.2,
            'use_full_dataset': False,
            'min_interactions': 5,
            'seed': 42,
            'k': 10,
            'max_iter': 20
        },
        {
            'dataset_type': 'movielens',
            'model_name': 'BPR',
            'output_csv': 'ml_bpr_full.csv',
            'user_subset_ratio': 0.2,
            'use_full_dataset': False,
            'min_interactions': 5,
            'seed': 42,
            'k': 10,
            'max_iter': 30
        },
        {
            'dataset_type': 'filmtrust',
            'model_name': 'SoRec',
            'output_csv': 'filmtrust_sorec.csv',
            'user_subset_ratio': 0.2,
            'use_full_dataset': False,
            'min_interactions': 2,
            'seed': 42,
            'k': 5,
            'max_iter': 20
        },
        {
            'dataset_type': 'filmtrust',
            'model_name': 'SVD',
            'output_csv': 'filmtrust_svd.csv',
            'user_subset_ratio': 0.2,
            'use_full_dataset': False,
            'min_interactions': 2,
            'seed': 42,
            'k': 5,
            'max_iter': 20
        },
        {
            'dataset_type': 'movielens',
            'model_name': 'NMF',
            'output_csv': 'ml_nmf.csv',
            'user_subset_ratio': 0.2,
            'use_full_dataset': False,
            'min_interactions': 5,
            'seed': 42,
            'k': 10,
            'max_iter': 50
        },
        {
            'dataset_type': 'movielens',
            'model_name': 'PMF',
            'output_csv': 'ml_pmf.csv',
            'user_subset_ratio': 0.2,
            'use_full_dataset': True,
            'min_interactions': 5,
            'seed': 42,
            'k': 10,
            'max_iter': 20
        },
        {
            'dataset_type': 'movielens',
            'model_name': 'BPR',
            'output_csv': 'ml_bpr_full.csv',
            'user_subset_ratio': 0.2,
            'use_full_dataset': True,
            'min_interactions': 5,
            'seed': 42,
            'k': 10,
            'max_iter': 30
        },
        {
            'dataset_type': 'filmtrust',
            'model_name': 'SoRec',
            'output_csv': 'filmtrust_sorec.csv',
            'user_subset_ratio': 0.2,
            'use_full_dataset': True,
            'min_interactions': 2,
            'seed': 42,
            'k': 5,
            'max_iter': 20
        },
        {
            'dataset_type': 'filmtrust',
            'model_name': 'SVD',
            'output_csv': 'filmtrust_svd.csv',
            'user_subset_ratio': 0.2,
            'use_full_dataset': True,
            'min_interactions': 2,
            'seed': 42,
            'k': 5,
            'max_iter': 20
        },
        {
            'dataset_type': 'movielens',
            'model_name': 'NMF',
            'output_csv': 'ml_nmf.csv',
            'user_subset_ratio': 0.2,
            'use_full_dataset': True,
            'min_interactions': 5,
            'seed': 42,
            'k': 10,
            'max_iter': 50
        }
    ]

    for exp in experiments:
        logging.info("=== Запуск эксперимента: %s + %s ===",
                     exp['dataset_type'], exp['model_name'])
        run_pipeline(**exp)
        logging.info("=== Эксперимент завершён ===")
