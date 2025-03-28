import os
import pandas as pd
import logging

from recommender import UniversalContextualRecommender

# Предполагается, что класс UniversalContextualRecommender уже реализован.
# Например, если он находится в модуле my_recommender, импортируйте его:


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RestaurantConsumerDataIntegration")


def load_data(directory: str):
    files = {
        "chefmozaccepts": os.path.join(directory, "chefmozaccepts.csv"),
        "chefmozcuisine": os.path.join(directory, "chefmozcuisine.csv"),
        "chefmozhours4": os.path.join(directory, "chefmozhours4.csv"),
        "chefmozparking": os.path.join(directory, "chefmozparking.csv"),
        "geoplaces2": os.path.join(directory, "geoplaces2.csv"),
        "rating_final": os.path.join(directory, "rating_final.csv"),
        "usercuisine": os.path.join(directory, "usercuisine.csv"),
        "userpayment": os.path.join(directory, "userpayment.csv"),
        "userprofile": os.path.join(directory, "userprofile.csv")
    }

    logger.info("Загрузка файлов из директории: %s", directory)
    df_accepts = pd.read_csv(files["chefmozaccepts"])
    df_cuisine = pd.read_csv(files["chefmozcuisine"])
    df_hours = pd.read_csv(files["chefmozhours4"])
    df_parking = pd.read_csv(files["chefmozparking"])

    # Для geoplaces2 задаем явную кодировку
    try:
        df_geoplaces = pd.read_csv(files["geoplaces2"], encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Не удалось прочитать %s с кодировкой utf-8, пробуем latin1", files["geoplaces2"])
        df_geoplaces = pd.read_csv(files["geoplaces2"], encoding="latin1")

    df_rating = pd.read_csv(files["rating_final"])
    df_usercuisine = pd.read_csv(files["usercuisine"])
    df_userpayment = pd.read_csv(files["userpayment"])
    df_userprofile = pd.read_csv(files["userprofile"])

    logger.info("Файлы загружены:")
    logger.info(" chefmozaccepts: %s", df_accepts.shape)
    logger.info(" chefmozcuisine: %s", df_cuisine.shape)
    logger.info(" chefmozhours4: %s", df_hours.shape)
    logger.info(" chefmozparking: %s", df_parking.shape)
    logger.info(" geoplaces2: %s", df_geoplaces.shape)
    logger.info(" rating_final: %s", df_rating.shape)
    logger.info(" usercuisine: %s", df_usercuisine.shape)
    logger.info(" userpayment: %s", df_userpayment.shape)
    logger.info(" userprofile: %s", df_userprofile.shape)

    return (df_accepts, df_cuisine, df_hours, df_parking,
            df_geoplaces, df_rating, df_usercuisine, df_userpayment, df_userprofile)


def merge_data(directory: str) -> pd.DataFrame:
    """
    Объединяет данные:
      - Используется rating_final.csv как базовая таблица (User-Item-Rating).
      - По ключу placeID объединяются файлы с данными по ресторанам.
      - По ключу userID объединяются файлы с данными по потребителям.
    """
    (df_accepts, df_cuisine, df_hours, df_parking,
     df_geoplaces, df_rating, df_usercuisine, df_userpayment, df_userprofile) = load_data(directory)

    # Объединяем ресторанную информацию
    df = df_rating.copy()
    df = df.merge(df_geoplaces, on="placeID", how="left")
    df = df.merge(df_cuisine, on="placeID", how="left")
    df = df.merge(df_hours, on="placeID", how="left")
    df = df.merge(df_parking, on="placeID", how="left")
    df = df.merge(df_accepts, on="placeID", how="left")

    # Объединяем информацию о потребителях
    df = df.merge(df_userprofile, on="userID", how="left")
    df = df.merge(df_usercuisine, on="userID", how="left")
    df = df.merge(df_userpayment, on="userID", how="left")

    logger.info("Объединённые данные имеют размер: %s", df.shape)
    return df


def save_recommendations_to_csv(recommendations: dict, output_file: str):
    """
    Сохраняет словарь рекомендаций в CSV‑файл.
    Каждая строка содержит идентификатор пользователя и строковое представление списка рекомендованных объектов.
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


def main():
    # Задайте путь к директории с данными
    data_directory = "restaurant+consumer+data"
    df = merge_data(data_directory)

    # Определяем контекстные признаки: исключаем ключевые колонки, отвечающие за идентификаторы и основную оценку.
    # Здесь используем userID, placeID и rating (а также food_rating, service_rating если есть) как основные.
    exclude_cols = ['userID', 'placeID', 'rating', 'food_rating', 'service_rating']
    context_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info("Контекстные признаки: %s", context_cols)

    # ------------------- CAMF модель -------------------
    logger.info("Запуск CAMF модели")
    recommender_camf = UniversalContextualRecommender(
        dataset=df,
        model_name='camf',
        user_col='userID',  # идентификатор пользователя
        item_col='placeID',  # идентификатор ресторана
        rating_col='rating',  # основная оценка
        context_cols=context_cols,
        n_factors=10,
        learning_rate=0.01,
        reg=0.02,
        epochs=10
    )
    recommender_camf.fit()
    camf_recs = recommender_camf.predict(default_context=None)
    save_recommendations_to_csv(camf_recs, "camf_restaurant_recommendations.csv")

    # ------------------- CSLIM модель -------------------
    logger.info("Запуск CSLIM модели")
    recommender_cslim = UniversalContextualRecommender(
        dataset=df,
        model_name='cslim',
        user_col='userID',
        item_col='placeID',
        rating_col='rating',
        context_cols=context_cols,  # для CSLIM контекстные признаки не используются, но передаются для согласованности
        use_ratings=True
    )
    recommender_cslim.fit()
    cslim_recs = recommender_cslim.predict(default_context=None)
    save_recommendations_to_csv(cslim_recs, "cslim_restaurant_recommendations.csv")
 # ------------------- Модель LightFM -------------------
    logger.info("Запуск модели LightFM")
    recommender_lightfm = UniversalContextualRecommender(
        dataset=df,
        model_name='lightfm',
        user_col='userID',
        item_col='placeID',
        rating_col='rating',
        context_cols=context_cols,
        epochs=10,
        learning_rate=0.05
    )
    recommender_lightfm.fit()
    lightfm_recs = recommender_lightfm.predict()
    save_recommendations_to_csv(lightfm_recs, "lightfm_restaurant_recommendations.csv")



    # ------------------- Модель Surprise SVD -------------------
    logger.info("Запуск модели Surprise SVD")
    recommender_surprise = UniversalContextualRecommender(
        dataset=df,
        model_name='surprise_svd',
        user_col='userID',
        item_col='placeID',
        rating_col='rating',
        context_cols=context_cols,
        epochs=20,
        lr_all=0.005,
        reg_all=0.02
    )
    recommender_surprise.fit()
    surprise_recs = recommender_surprise.predict()
    save_recommendations_to_csv(surprise_recs, "surprise_svd_restaurant_recommendations.csv")

if __name__ == '__main__':
    main()
