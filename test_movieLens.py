import os
import pandas as pd
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MovieLensEvaluateMetrics")


def load_ground_truth_csv(file_path: str) -> dict:
    """
    Загружает CSV с ground truth.
    Ожидается, что CSV имеет колонки:
      - user или userID: идентификатор пользователя,
      - true_items: строка с разделёнными запятыми идентификаторами релевантных фильмов.
    Возвращает словарь {user: set(этих объектов)}.
    """
    df = pd.read_csv(file_path)
    # Если используется 'userID' вместо 'user', берем его
    user_col = "user" if "user" in df.columns else "userID"
    gt = {}
    for _, row in df.iterrows():
        user = row[user_col]
        items = [item.strip() for item in str(row["true_items"]).split(",") if item.strip()]
        gt[user] = set(items)
    logger.info("Загружен ground truth для %d пользователей из %s", len(gt), file_path)
    return gt


def load_recommendations(file_path: str) -> dict:
    """
    Загружает CSV с рекомендациями.
    Ожидается, что CSV имеет колонки:
      - user или userID: идентификатор пользователя,
      - recommended_items: строка с разделёнными запятыми идентификаторами фильмов.
    Возвращает словарь {user: [фильмы]}.
    """
    df = pd.read_csv(file_path)
    recs = {}
    user_col = "user" if "user" in df.columns else "userID"
    for _, row in df.iterrows():
        user = row[user_col]
        items = [item.strip() for item in str(row["recommended_items"]).split(",") if item.strip()]
        recs[user] = items
    logger.info("Загружено рекомендаций для %d пользователей из %s", len(recs), file_path)
    return recs


def calculate_map_k(true_labels, predicted_labels, k=50) -> float:
    """
    Вычисляет AP@k для одного пользователя.

    :param true_labels: Список истинных (релевантных) объектов.
    :param predicted_labels: Список рекомендованных объектов.
    :param k: Число рекомендаций для оценки.
    :return: AP@k для данного пользователя.
    """
    predicted_labels_k = predicted_labels[:k]
    true_set = set(true_labels)
    hit_count, precision_sum = 0, 0.0
    for i, item in enumerate(predicted_labels_k, start=1):
        if item in true_set:
            hit_count += 1
            precision_sum += hit_count / i
    return precision_sum / min(len(true_labels), k) if hit_count > 0 else 0.0


def map_at_k_metric(recommendations: dict, ground_truth: dict, k=50) -> float:
    """
    Вычисляет MAP@k по всем пользователям.

    :param recommendations: Словарь {user: [рекомендуемые объекты]}.
    :param ground_truth: Словарь {user: set(релевантных объектов)}.
    :param k: Число рекомендаций для оценки.
    :return: Среднее AP@k (MAP@k).
    """
    ap_list = []
    for user, recs in recommendations.items():
        if user not in ground_truth or not ground_truth[user]:
            continue
        ap = calculate_map_k(list(ground_truth[user]), recs, k)
        ap_list.append(ap)
    return np.mean(ap_list) if ap_list else 0.0


def main():
    # Файл ground truth, который должен быть сохранён заранее, например: "test_ground_truth_MovieLens.csv"
    gt_file = "test_ground_truth_MovieLens.csv"
    if not os.path.exists(gt_file):
        logger.error("Файл ground truth '%s' не найден!", gt_file)
        return
    ground_truth = load_ground_truth_csv(gt_file)

    # Файлы с рекомендациями для моделей MovieLens
    rec_files = {
        "CAMF": "camf_movielens_recommendations.csv",
        "CSLIM": "cslim_movielens_recommendations.csv",
        "LightFM": "lightfm_movielens_recommendations.csv",
        "SurpriseSVD": "surprise_svd_movielens_recommendations.csv"
    }

    results = {}
    for model, file in rec_files.items():
        if not os.path.exists(file):
            logger.warning("Файл с рекомендациями для модели %s ('%s') не найден. Пропускаем.", model, file)
            continue
        recommendations = load_recommendations(file)
        map_value = map_at_k_metric(recommendations, ground_truth, k=50)
        results[model] = map_value
        logger.info("MAP@50 для модели %s: %.4f", model, map_value)

    print("\nРезультаты MAP@50 по моделям (MovieLens, используя test_ground_truth_MovieLens.csv):")
    for model, score in results.items():
        print(f"{model}: {score:.4f}")


if __name__ == '__main__':
    main()
