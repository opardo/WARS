from fms import *
from features import *
import time

def tune_basic(movie_size, user_size, df_ratings, wap_movies):
    np.random.seed(2018)
    df_reviews = df_ratings.groupby(['movieId']).userId.count().reset_index()
    reviewed_movies = np.setdiff1d(df_reviews[df_reviews['userId'] >= 1500].index, wap_movies)
    sampled_movies = np.random.choice(reviewed_movies, size=movie_size-6, replace=False)
    selected_movies = np.concatenate((wap_movies, sampled_movies))
    
    df_ratings_sample1 = df_ratings[df_ratings.movieId.isin(selected_movies)]
    users = np.unique(df_ratings_sample1.userId)
    sampled_users = np.random.choice(users, size=user_size, replace=False)
    df_ratings_sample = df_ratings_sample1[df_ratings_sample1.userId.isin(sampled_users)]
    
    df_wap_ratings = df_ratings_sample[df_ratings_sample.movieId.isin(wap_movies)]
    df_test_sample = df_wap_ratings.sample(frac = 0.50)
    df_train_sample = pd.concat([df_ratings_sample, df_test_sample]).drop_duplicates(keep=False)
    
    t0 = time.time()
    res = get_dataset_performance(df_ratings_sample, df_train_sample, df_test_sample)
    t1 = time.time()
    res['time'] = t1-t0
    return res

def tune_features(feature_count, df_ratings, wap_movies, df_movies, df_tags, df_genome, pop_tags_dict, rel_tags_dict, feats='ugt'):
    np.random.seed(2018)
    df_reviews = df_ratings.groupby(['movieId']).userId.count().reset_index()
    reviewed_movies = np.setdiff1d(df_reviews[df_reviews['userId'] >= 1500].index, wap_movies)
    sampled_movies = np.random.choice(reviewed_movies, size=244, replace=False)
    selected_movies = np.concatenate((wap_movies, sampled_movies))

    df_ratings_sample1 = df_ratings[df_ratings.movieId.isin(selected_movies)]
    users = np.unique(df_ratings_sample1.userId)
    sampled_users = np.random.choice(users, size=2000, replace=False)
    df_ratings_sample = df_ratings_sample1[df_ratings_sample1.userId.isin(sampled_users)]

    df_wap_ratings = df_ratings_sample[df_ratings_sample.movieId.isin(wap_movies)]
    df_test_sample = df_wap_ratings.sample(frac = 0.50)
    df_train_sample = pd.concat([df_ratings_sample, df_test_sample]).drop_duplicates(keep=False)
    
    genres_dict = get_genres_dict(df_movies)
    number = feature_count
    pop_tags_dict = get_popular_tags_dict(df_tags, number)
    rel_tags_dict = get_relevant_tags_dict(df_genome, number)

    user_features_all_train = get_user_all_features(df_train_sample, df_movies, df_tags, genres_dict, pop_tags_dict)
    user_features_all_test = get_user_all_features(df_test_sample, df_movies, df_tags, genres_dict, pop_tags_dict)
    movie_features_all_train = get_movie_all_features(df_train_sample, df_movies, df_genome, genres_dict, rel_tags_dict)
    movie_features_all_test = get_movie_all_features(df_test_sample, df_movies, df_genome, genres_dict, rel_tags_dict)
    
    if feats == 'ugt':
        user_features_train = user_features_all_train['features_mgt']
        user_features_test = user_features_all_test['features_mgt']
        movie_features_train = movie_features_all_train['features_ugt']
        movie_features_test = movie_features_all_test['features_ugt']
    else:
        user_features_train = user_features_all_train['features_mg']
        user_features_test = user_features_all_test['features_mg']
        movie_features_train = movie_features_all_train['features_ug']
        movie_features_test = movie_features_all_test['features_ug']

    t0 = time.time()
    res = get_dataset_performance(df_ratings_sample, df_train_sample, df_test_sample, user_features_train, 
                                  movie_features_train, user_features_test, movie_features_test)
    t1 = time.time()
    res['time'] = t1-t0
    return res