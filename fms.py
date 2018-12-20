from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from lightfm.cross_validation import random_train_test_split
import numpy as np
import pandas as pd


def get_dataset_performance(
    df_ratings,
    df_train,
    df_test,
    user_features_train=None,
    movie_features_train=None,
    user_features_test=None,
    movie_features_test=None
):
    np.random.seed(2018)
    train_interactions, train_weights, train_user_features, train_movie_features = LightFM_interactions(
        df_train,
        df_ratings,
        user_features=user_features_train,
        movie_features=movie_features_train
    )
    test_interactions, test_weights, test_user_features, test_movie_features = LightFM_interactions(
        df_test,
        df_ratings,
        user_features=user_features_test,
        movie_features=movie_features_test
    )

    if (user_features_train is None):
        train_user_features = None
    if (movie_features_train is None):
        train_movie_features = None
    if (user_features_test is None):
        test_user_features = None
    if (movie_features_test is None):
        test_movie_features = None

    model = fit_LightFM_model(train_interactions, train_weights, train_user_features, train_movie_features)
    return test_LightFM_model(model, test_interactions, train_interactions, test_user_features, test_movie_features, k=5)

def LightFM_interactions(df_frac, df_ratings, user_features=None, movie_features=None):
    umr = umr_arrays(df_frac)
    unique_elements = unique_users_movies(df_ratings)
    dataset = init_lightfm_dataset(unique_elements, user_features, movie_features)
    interactions, weights = build_interactions(dataset, umr)
    if user_features is not None:
        user_feature_matrix = build_user_features(dataset, user_features)
    else:
        user_feature_matrix = []
    if movie_features is not None:
        movie_feature_matrix = build_movie_features(dataset, movie_features)
    else:
        movie_feature_matrix = []
    return([interactions, weights, user_feature_matrix, movie_feature_matrix])

def umr_arrays(df_ratings):
    users = df_ratings['userId'].values
    users = np.reshape(users, (len(users),1))
    movies = df_ratings['movieId'].values
    movies = np.reshape(movies, (len(movies),1))
    ratings = df_ratings['rating'].values
    ratings = np.reshape(ratings, (len(ratings),1))
    umr = np.concatenate((users, movies, ratings), axis=1)
    return(umr)

def unique_users_movies(df_ratings):
    unique_users = np.unique(df_ratings['userId'])
    unique_movies = np.unique(df_ratings['movieId'])
    unique_elements = [unique_users, unique_movies]
    return(unique_elements)

def init_lightfm_dataset(unique_elements, user_features=None, movie_features=None):
    unique_users = unique_elements[0]
    unique_movies = unique_elements[1]
    if (user_features is not None):
        user_features = [*user_features[0][1]]
    if (movie_features is not None):
        movie_features = [*movie_features[0][1]]
    dataset = Dataset()
    dataset.fit(users=unique_users, items=unique_movies, user_features=user_features, item_features=movie_features)
    return(dataset)

def build_interactions(dataset, umr):
    interactions, weights = dataset.build_interactions((i[0], i[1], i[2]) for i in umr)
    return([interactions, weights])

def build_user_features(dataset, user_features):
    user_feature_matrix = dataset.build_user_features(user_features)
    return(user_feature_matrix)

def build_movie_features(dataset, movie_features):
    movie_feature_matrix = dataset.build_item_features(movie_features)
    return(movie_feature_matrix)

def fit_LightFM_model(interactions, weights, user_feature_matrix, movie_feature_matrix, no_components=40, loss='warp'):
    model = LightFM(no_components=no_components, loss=loss, random_state=2019)
    model.fit(
        interactions=interactions,
        sample_weight=weights,
        user_features=user_feature_matrix,
        item_features=movie_feature_matrix
    )
    return(model)

def test_LightFM_model(model, test_interactions, train_interactions, user_features, movie_features, k=5):
    test_precision = precision_at_k(
        model,
        test_interactions,
        train_interactions,
        k=k,
        user_features=user_features,
        item_features=movie_features,
        num_threads=2
    ).mean()
    test_recall = recall_at_k(
        model,
        test_interactions,
        train_interactions,
        k=k,
        user_features=user_features,
        item_features=movie_features,
        num_threads=2
    ).mean()
    test_auc = auc_score(
        model,
        test_interactions,
        train_interactions,
        user_features=user_features,
        item_features=movie_features,
        num_threads=2
    ).mean()
    print('Model')
    print('Precision at k=', str(k), ': ', round(test_precision, 3), sep='')
    print('Recall at k=', str(k) + ': ', round(test_recall, 3), sep='')
    print('AUC: ', round(test_auc, 3), sep='')
    return({'precision': round(test_precision, 3),
            'recall': round(test_recall, 3),
            'auc': round(test_auc, 3)})
