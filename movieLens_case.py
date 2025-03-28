import os
import pandas as pd
import numpy as np
import random
import logging
from sklearn.model_selection import train_test_split

from recommender import UniversalContextualRecommender

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MovieLensDataIntegration")

def load_data(directory: str):
    files = {
        "ratings": os.path.join(directory, "u.data"),
        "movies": os.path.join(directory, "u.item"),
        "users": os.path.join(directory, "u.user")
    }

    logger.info("Загрузка файлов из директории: %s", directory)

    # Файл u.data: userID, movieID, rating, timestamp (разделитель - табуляция)
    df_ratings = pd.read_csv(files["ratings"], sep="\t", names=["userID", "movieID", "rating", "timestamp"])

    # Файл u.item: movieID, title, release_date, video_release_date, IMDb_URL и 19 бинарных признаков жанров
    movie_columns = ["movieID", "title", "release_date", "video_release_date", "IMDb_URL"] + [f"genre_{i}" for i in range(19)]
    df_movies = pd.read_csv(files["movies"], sep="|", names=movie_columns, encoding="latin1", header=None)

    # Файл u.user: userID, age, gender, occupation, zip_code (разделитель - |)
    df_users = pd.read_csv(files["users"], sep="|", names=["userID", "age", "gender", "occupation", "zip_code"])

    logger.info("Файлы загружены: ratings: %s, movies: %s, users: %s", df_ratings.shape, df_movies.shape, df_users.shape)
    return df_ratings, df_movies, df_users

def merge_data(directory: str) -> pd.DataFrame:
    df_ratings, df_movies, df_users = load_data(directory)

    # Объединяем рейтинги с информацией о фильмах по movieID
    df = pd.merge(df_ratings, df_movies, on="movieID", how="left")
    # Объединяем с данными пользователей по userID
    df = pd.merge(df, df_users, on="userID", how="left")

    logger.info("Объединённые данные имеют размер: %s", df.shape)
    return df

def save_recommendations_to_csv(recommendations: dict, output_file: str):
    """
    Сохраняет словарь рекомендаций в CSV‑файл.
    Каждая строка содержит идентификатор пользователя и строковое представление списка рекомендованных фильмов.
    """
    rows = []
    for user, recs in recommendations.items():
        rows.append({
            'user': user,
            'recommended_items': ",".join(map(str, recs))
        })
    recs_df = pd.DataFrame(rows)
    recs_df.to_csv(output_file, index=False)
    logger.info("Результаты сохранены в %s", output_file)

def create_and_save_ground_truth(df: pd.DataFrame, test_ratio: float = 0.3, random_state: int = 42):
    """
    Разбивает объединённый датасет на тренировочную и тестовую выборки,
    затем для тестовой выборки агрегирует true movieID для каждого пользователя
    и сохраняет результат в CSV (test_ground_truth_MovieLens.csv).
    Если в данных отсутствует столбец 'placeID', используется 'movieID'.
    """
    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_state)
    # Определяем столбец для идентификатора фильма
    id_col = "placeID" if "placeID" in df.columns else "movieID"
    ground_truth_df = test_df.groupby("userID")[id_col].apply(lambda x: ",".join(map(str, x))).reset_index()
    ground_truth_df.rename(columns={id_col: "true_items"}, inplace=True)
    ground_truth_csv = "test_ground_truth_MovieLens.csv"
    ground_truth_df.to_csv(ground_truth_csv, index=False)
    logger.info("Ground truth (тестовая выборка) сохранена в %s", ground_truth_csv)
    return train_df, test_df

def main(tested: bool):
    data_directory = "ml-100k"
    df = merge_data(data_directory)

    exclude_cols = [
        'userID', 'movieID', 'rating', 'timestamp',
        'title', 'release_date', 'video_release_date', 'IMDb_URL'
    ]

    context_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info("Контекстные признаки: %s", context_cols)
    if tested:
        logger.info("Включён режим тестирования. Будет выполнено разбиение на train/test и сохранение ground truth.")
        train_df, test_df = create_and_save_ground_truth(df, test_ratio=0.3, random_state=42)
        df = train_df
    else:
        logger.info("Режим тестирования выключен. Модели обучаются на всём датасете.")

    # ------------------- CAMF модель -------------------
    logger.info("Запуск CAMF модели")
    recommender_camf = UniversalContextualRecommender(
        dataset=df,
        model_name='camf',
        user_col='userID',
        item_col='movieID',
        rating_col='rating',
        context_cols=context_cols,
        n_factors=10,
        learning_rate=0.01,
        reg=0.02,
        epochs=10
    )
    recommender_camf.fit()
    camf_recs = recommender_camf.predict(default_context=None)
    save_recommendations_to_csv(camf_recs, "camf_movielens_recommendations.csv")

    # ------------------- CSLIM модель -------------------
    logger.info("Запуск CSLIM модели")
    recommender_cslim = UniversalContextualRecommender(
        dataset=df,
        model_name='cslim',
        user_col='userID',
        item_col='movieID',
        rating_col='rating',
        context_cols=context_cols,
        use_ratings=True
    )
    recommender_cslim.fit()
    cslim_recs = recommender_cslim.predict(default_context=None)
    save_recommendations_to_csv(cslim_recs, "cslim_movielens_recommendations.csv")

    # ------------------- LightFM модель -------------------
    logger.info("Запуск модели LightFM")
    recommender_lightfm = UniversalContextualRecommender(
        dataset=df,
        model_name='lightfm',
        user_col='userID',
        item_col='movieID',
        rating_col='rating',
        context_cols=context_cols,
        epochs=10,
        learning_rate=0.05
    )
    recommender_lightfm.fit()
    lightfm_recs = recommender_lightfm.predict(default_context=None)
    save_recommendations_to_csv(lightfm_recs, "lightfm_movielens_recommendations.csv")

    # ------------------- Surprise SVD модель -------------------
    logger.info("Запуск модели Surprise SVD")
    recommender_surprise = UniversalContextualRecommender(
        dataset=df,
        model_name='surprise_svd',
        user_col='userID',
        item_col='movieID',
        rating_col='rating',
        context_cols=context_cols,
        epochs=20,
        lr_all=0.005,
        reg_all=0.02
    )
    recommender_surprise.fit()
    surprise_recs = recommender_surprise.predict(default_context=None)
    save_recommendations_to_csv(surprise_recs, "surprise_svd_movielens_recommendations.csv")

if __name__ == '__main__':
    # Передайте tested=True, если хотите выполнить разбиение на train/test и обучение на тренировочных данных,
    # иначе tested=False – обучение на всём датасете.
    main(tested=True)
