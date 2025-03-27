import os
import pandas as pd
import logging

from recommender import UniversalContextualRecommender

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

    # Для geoplaces2 задаём явную кодировку
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
      - rating_final.csv используется как базовая таблица (User-Item-Rating).
      - По ключу placeID объединяются файлы с информацией о ресторанах.
      - По ключу userID объединяются файлы с информацией о пользователях.
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

    # Объединяем информацию о пользователях
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
    data_directory = "restaurant+consumer+data"
    df = merge_data(data_directory)

    # Определяем контекстные признаки: исключаем основные колонки
    exclude_cols = ['userID', 'placeID', 'rating', 'food_rating', 'service_rating']
    context_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info("Контекстные признаки: %s", context_cols)

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

    # ------------------- Модель Implicit ALS -------------------
    # logger.info("Запуск модели Implicit ALS")
    # recommender_implicit = UniversalContextualRecommender(
    #     dataset=df,
    #     model_name='implicit_als',
    #     user_col='userID',
    #     item_col='placeID',
    #     rating_col='rating',
    #     context_cols=context_cols,
    #     factors=50,
    #     iterations=10
    # )
    # recommender_implicit.fit()
    # implicit_recs = recommender_implicit.predict()
    # save_recommendations_to_csv(implicit_recs, "implicit_als_restaurant_recommendations.csv")

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
