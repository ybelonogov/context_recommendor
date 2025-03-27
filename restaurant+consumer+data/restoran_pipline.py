#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, csr_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

def main():
    # === ПУТИ К ФАЙЛАМ (замените, если нужно) ===
    path_rating    = 'rating_final.csv'
    path_user      = 'userprofile.csv'
    path_userpay   = 'userpayment.csv'
    path_usercuis  = 'usercuisine.csv'
    path_places    = 'geoplaces2.csv'
    path_placecuis = 'chefmozcuisine.csv'
    path_placepark = 'chefmozparking.csv'
    path_placepay  = 'chefmozaccepts.csv'

    # === ЧТЕНИЕ CSV ===
    # Используем encoding='latin-1'; если не работает, попробуйте 'cp1252', 'ISO-8859-1', или без параметра.
    df_rating    = pd.read_csv(path_rating, encoding='latin-1')
    df_user      = pd.read_csv(path_user, encoding='latin-1')
    df_userpay   = pd.read_csv(path_userpay, encoding='latin-1')
    df_usercuis  = pd.read_csv(path_usercuis, encoding='latin-1')
    df_places    = pd.read_csv(path_places, encoding='latin-1')
    df_placecuis = pd.read_csv(path_placecuis, encoding='latin-1')
    df_placepark = pd.read_csv(path_placepark, encoding='latin-1')
    df_placepay  = pd.read_csv(path_placepay, encoding='latin-1')

    # === БИНАРИЗАЦИЯ РЕЙТИНГА ===
    # rating_final.csv => userID, placeID, rating, food_rating, service_rating
    # Предположим, rating >= 3 => 1, иначе 0
    df_rating['bin_rating'] = (df_rating['rating'] >= 3).astype(int)

    # Убедимся, что userID, placeID - строки
    df_rating['userID']  = df_rating['userID'].astype(str)
    df_rating['placeID'] = df_rating['placeID'].astype(str)

    # === ОБРАБОТКА ПОЛЬЗОВАТЕЛЕЙ (userprofile) ===
    df_user['userID'] = df_user['userID'].astype(str)

    # == userpayment.csv: userID, Upayment (возможно несколько способов)
    df_userpay['userID']   = df_userpay['userID'].astype(str)
    df_userpay['Upayment'] = df_userpay['Upayment'].fillna('Unknown')
    df_userpay_oh = pd.get_dummies(df_userpay, columns=['Upayment'])
    df_userpay_agg = df_userpay_oh.groupby('userID').sum().reset_index()

    # == usercuisine.csv: userID, Rcuisine
    df_usercuis['userID']   = df_usercuis['userID'].astype(str)
    df_usercuis['Rcuisine'] = df_usercuis['Rcuisine'].fillna('Unknown')
    df_usercuis_oh = pd.get_dummies(df_usercuis, columns=['Rcuisine'])
    df_usercuis_agg = df_usercuis_oh.groupby('userID').sum().reset_index()

    # Объединяем всё в df_user_all
    df_user_all = df_user.merge(df_userpay_agg, on='userID', how='left')
    df_user_all = df_user_all.merge(df_usercuis_agg, on='userID', how='left')
    df_user_all = df_user_all.fillna(0)

    # Поля для OHE из userprofile
    cat_cols = ['smoker','drink_level','dress_preference','ambience','transport','marital_status']
    # Преобразуем в строку (потому что могли быть NaN)
    for c in cat_cols:
        if c in df_user_all.columns:
            df_user_all[c] = df_user_all[c].astype(str)

    df_user_all_oh = pd.get_dummies(df_user_all, columns=[c for c in cat_cols if c in df_user_all.columns])

    # === ОБРАБОТКА РЕСТОРАНОВ (places) ===
    df_places['placeID'] = df_places['placeID'].astype(str)

    # == chefmozcuisine: placeID, Rcuisine
    df_placecuis['placeID'] = df_placecuis['placeID'].astype(str)
    df_placecuis['Rcuisine'] = df_placecuis['Rcuisine'].fillna('Unknown')
    df_placecuis_oh = pd.get_dummies(df_placecuis, columns=['Rcuisine'])
    df_placecuis_agg = df_placecuis_oh.groupby('placeID').sum().reset_index()

    # == chefmozparking: placeID, parking_lot
    df_placepark['placeID'] = df_placepark['placeID'].astype(str)
    df_placepark['parking_lot'] = df_placepark['parking_lot'].fillna('Unknown')
    df_placepark_oh = pd.get_dummies(df_placepark, columns=['parking_lot'])
    df_placepark_agg = df_placepark_oh.groupby('placeID').sum().reset_index()

    # == chefmozaccepts: placeID, Rpayment
    df_placepay['placeID']  = df_placepay['placeID'].astype(str)
    df_placepay['Rpayment'] = df_placepay['Rpayment'].fillna('Unknown')
    df_placepay_oh = pd.get_dummies(df_placepay, columns=['Rpayment'])
    df_placepay_agg = df_placepay_oh.groupby('placeID').sum().reset_index()

    # Сливаем всё про place
    df_place_all = df_places.merge(df_placecuis_agg, on='placeID', how='left')
    df_place_all = df_place_all.merge(df_placepark_agg, on='placeID', how='left')
    df_place_all = df_place_all.merge(df_placepay_agg, on='placeID', how='left')
    df_place_all = df_place_all.fillna(0)

    # Поля для OHE из geoplaces2
    cat_cols_2 = ['city','state','alcohol','smoking_area','dress_code','accessibility','price','Rambience','franchise','area']
    for c in cat_cols_2:
        if c in df_place_all.columns:
            df_place_all[c] = df_place_all[c].astype(str)

    df_place_all_oh = pd.get_dummies(df_place_all, columns=[c for c in cat_cols_2 if c in df_place_all.columns])

    # === ПРЕОБРАЗОВАНИЕ userID/placeID => user_idx, item_idx ===
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df_user_all_oh['userID'] = df_user_all_oh['userID'].astype(str)
    df_user_all_oh['user_idx'] = user_encoder.fit_transform(df_user_all_oh['userID'])

    df_place_all_oh['placeID'] = df_place_all_oh['placeID'].astype(str)
    df_place_all_oh['item_idx'] = item_encoder.fit_transform(df_place_all_oh['placeID'])

    # Сортируем, чтобы user_idx / item_idx шли по порядку
    df_user_all_oh = df_user_all_oh.sort_values('user_idx').reset_index(drop=True)
    df_place_all_oh = df_place_all_oh.sort_values('item_idx').reset_index(drop=True)

    num_users = df_user_all_oh['user_idx'].nunique()
    num_items = df_place_all_oh['item_idx'].nunique()

    # === РАЗБИВКА RATING НА TRAIN/TEST ===
    df_rating['user_idx'] = user_encoder.transform(df_rating['userID'])
    df_rating['item_idx'] = item_encoder.transform(df_rating['placeID'])

    # Убираем строки, где user_idx или item_idx = -1 (если LabelEncoder не нашёл соответствий)
    df_rating = df_rating[(df_rating['user_idx'] >= 0) & (df_rating['item_idx'] >= 0)]

    df_train, df_test = train_test_split(df_rating, test_size=0.2, random_state=42)

    train_rows = df_train['user_idx'].values
    train_cols = df_train['item_idx'].values
    train_vals = df_train['bin_rating'].values

    test_rows = df_test['user_idx'].values
    test_cols = df_test['item_idx'].values
    test_vals = df_test['bin_rating'].values

    train_interactions = coo_matrix((train_vals, (train_rows,train_cols)), shape=(num_users, num_items))
    test_interactions  = coo_matrix((test_vals,  (test_rows, test_cols)),  shape=(num_users, num_items))

    # === ФОРМИРУЕМ user_features, item_features ===

    # Убираем ID
    user_feats_cols = [c for c in df_user_all_oh.columns if c not in ['userID', 'user_idx']]
    item_feats_cols = [c for c in df_place_all_oh.columns if c not in ['placeID', 'item_idx']]

    df_user_feats = df_user_all_oh[user_feats_cols]
    df_item_feats = df_place_all_oh[item_feats_cols]

    # Удаляем поля, которые точно текстовые / ненужные:
    # (вставьте те, что показаны в логе: 'the_geom_meter','name','address','interest','hijos' и т.д.)
    cols_to_drop_user = ['personality', 'religion', 'hijos', 'interest', 'the_geom_meter', 'name', 'address']
    df_user_feats = df_user_feats.drop(columns=cols_to_drop_user, errors='ignore')

    # Пробуем жёстко конвертировать всё в float:
    df_user_feats_num = (df_user_feats
                         .apply(lambda col: pd.to_numeric(col, errors='coerce'))
                         .fillna(0)
                         .astype(float)
                         )
    X_user = df_user_feats_num.values
    print("DEBUG user feats dtype =", X_user.dtype)  # должно быть float64
    user_feature_matrix = csr_matrix(X_user)

    # Аналогично для item
    cols_to_drop_item = ['the_geom_meter', 'name', 'address', '...']
    df_item_feats = df_item_feats.drop(columns=cols_to_drop_item, errors='ignore')

    df_item_feats_num = (df_item_feats
                         .apply(lambda col: pd.to_numeric(col, errors='coerce'))
                         .fillna(0)
                         .astype(float)
                         )
    X_item = df_item_feats_num.values
    print("DEBUG item feats dtype =", X_item.dtype)
    item_feature_matrix = csr_matrix(X_item)

    print('=== SHAPES ===')
    print('user_feature_matrix:', user_feature_matrix.shape)
    print('item_feature_matrix:', item_feature_matrix.shape)
    print('train_interactions:', train_interactions.shape, 'NNZ=', train_interactions.nnz)
    print('test_interactions: ', test_interactions.shape,  'NNZ=', test_interactions.nnz)

    # === ОБУЧАЕМ LIGHTFM ===
    model = LightFM(loss='warp', random_state=42)
    model.fit(
        interactions=train_interactions,
        user_features=user_feature_matrix,
        item_features=item_feature_matrix,
        epochs=10,
        num_threads=2
    )

    # === ОЦЕНКА ===
    train_precision = precision_at_k(
        model,
        train_interactions,
        user_features=user_feature_matrix,
        item_features=item_feature_matrix,
        k=10
    ).mean()
    test_precision = precision_at_k(
        model,
        test_interactions,
        user_features=user_feature_matrix,
        item_features=item_feature_matrix,
        k=10
    ).mean()

    train_auc = auc_score(
        model,
        train_interactions,
        user_features=user_feature_matrix,
        item_features=item_feature_matrix
    ).mean()
    test_auc = auc_score(
        model,
        test_interactions,
        user_features=user_feature_matrix,
        item_features=item_feature_matrix
    ).mean()

    print(f'Train P@10 = {train_precision:.4f}, AUC={train_auc:.4f}')
    print(f'Test  P@10  = {test_precision:.4f}, AUC={test_auc:.4f}')

    # === ПРИМЕР РЕКОМЕНДАЦИИ ДЛЯ user_idx=0 ===
    user_id_example = 0
    scores = model.predict(
        user_ids=user_id_example,
        item_ids=np.arange(num_items),
        user_features=user_feature_matrix,
        item_features=item_feature_matrix
    )
    top_items = np.argsort(-scores)[:10]
    print(f'Top-10 recommendations for user_idx={user_id_example} => item_idx={top_items}')

if __name__ == '__main__':
    main()
