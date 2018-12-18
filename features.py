import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import collections

# DICTIONARIES

def get_genres_dict(df_movies):
    genres_values = df_movies['genres'].map(lambda x: x.split('|')).values
    genres = list(set(
        [genre for movie_genres in genres_values for genre in movie_genres if genre != "(no genres listed)"]
    ))
    genres_dict = {genres[i]: i for i in range(0,len(genres))}
    return(genres_dict)

def get_popular_tags_dict(df_tags, number):
    df_popular_tags = df_tags.groupby('tag').userId.count().reset_index()\
        .sort_values(by=['userId'], ascending=False)\
        .head(number)
    popular_tags = df_popular_tags['tag'].values
    pop_tags_dict = {popular_tags[i]: i for i in range(0,len(popular_tags))}
    return(pop_tags_dict)

def get_relevant_tags_dict(df_genome, number):
    df_relevant_tags = df_genome.groupby('tag').relevance.mean().reset_index()\
        .sort_values(by=['relevance'], ascending=False)\
        .head(number)
    relevant_tags = df_relevant_tags['tag'].values
    rel_tags_dict = {relevant_tags[i]: i for i in range(0,len(relevant_tags))}
    return(rel_tags_dict)

# USER FEATURES

## Movies/Genres features

def user_movie_genres(df_ratings, df_movies, genres_dict):
    df_user_genres_list = genres_list_per_user(df_ratings, df_movies)
    df_user_total_movies = total_movies_per_user(df_ratings)
    df_user_movies_genres = genres_features_per_user(df_user_genres_list, df_user_total_movies, genres_dict)
    total_movies = len(pd.unique(df_ratings.movieId))
    df_user_movies_genres['movies_perc'] = df_user_movies_genres.total_movies / total_movies
    df_user_movies_genres = df_user_movies_genres[['userId', 'movies_perc', 'genres_features']]
    return(df_user_movies_genres)

def genres_list_per_user(df_ratings, df_movies):
    df_ratings_movies = pd.merge(df_ratings, df_movies, on='movieId', how='left')
    df_ratings_movies['genres_list'] = df_ratings_movies['genres'].apply(lambda x:x.split('|'))
    df_user_genres_list = df_ratings_movies\
        .groupby('userId')\
        .genres_list\
        .apply(lambda x:[genre for genres in x for genre in genres])\
        .reset_index()
    return(df_user_genres_list)

def total_movies_per_user(df_ratings):
    df_user_total_movies = df_ratings.groupby('userId').movieId.count().reset_index()
    df_user_total_movies.rename(columns={'movieId':'total_movies'}, inplace=True)
    return(df_user_total_movies)

def genres_features_per_user(df_user_genres_list, df_user_total_movies, genres_dict):
    df_user_movies_genres = pd.merge(df_user_genres_list, df_user_total_movies, on='userId', how='inner')
    df_user_movies_genres['genres_features'] = df_user_movies_genres\
        .apply(lambda x: get_user_genres_features(x['genres_list'], x['total_movies'], genres_dict), axis=1)
    return(df_user_movies_genres)

def get_user_genres_features(genres_list, total_movies, genres_dict):
    genres_num = len(genres_dict)
    genres_features = [0.0] * genres_num
    genres_list = [genre for genre in genres_list if genre != "(no genres listed)"]
    genres_freq = collections.Counter(genres_list)
    for genre, freq in genres_freq.items():
        genres_features[genres_dict[genre]] = freq / total_movies
    return(genres_features)

## Tags features

def user_tags(df_ratings, df_tags, pop_tags_dict):
    df_user_tags_list = tags_list_per_user(df_ratings, df_tags)
    df_user_tags = tags_features_per_user(df_user_tags_list, pop_tags_dict)
    return(df_user_tags)

def tags_list_per_user(df_ratings, df_tags):
    df_tags.rename(columns={'timestamp':'tag_timestamp'}, inplace=True)
    df_ratings_tags = pd.merge(df_ratings, df_tags, on=['userId','movieId'], how='left')
    df_user_tags_list = df_ratings_tags\
        .groupby('userId')\
        .tag\
        .apply(lambda x: [tag for tag in x if tag is not np.nan])\
        .reset_index()
    df_user_tags_list.rename(columns={'tag':'tags_list'}, inplace=True)
    return(df_user_tags_list)

def tags_features_per_user(df_user_tags_list, pop_tags_dict):
    df_user_tags_list['tags_features'] = df_user_tags_list\
        .apply(lambda x: get_user_tags_features(x['tags_list'], pop_tags_dict), axis=1)
    df_user_tags = df_user_tags_list[['userId', 'tags_features']]
    return(df_user_tags)

def get_user_tags_features(tags_list, pop_tags_dict):
    tags_num = len(pop_tags_dict)
    tags_features = [0.0] * tags_num
    pop_tags_list = [tag for tag in tags_list if tag in pop_tags_dict.keys()]
    tags_freq = collections.Counter(pop_tags_list)
    for tag, freq in tags_freq.items():
        tags_features[pop_tags_dict[tag]] = freq / len(tags_list)
    return(tags_features)

## Merge features

def get_user_all_features(df_ratings, df_movies, df_tags, genres_dict, pop_tags_dict):
    df_user_movies_genres = user_movie_genres(df_ratings, df_movies, genres_dict)
    df_user_tags = user_tags(df_ratings, df_tags, pop_tags_dict)
    df_user_mgt = pd.merge(df_user_movies_genres, df_user_tags, on='userId', how='inner')
    df_user_mgt['features_m'] = df_user_mgt.apply(lambda x: [x['movies_perc']], axis=1)
    df_user_mgt['features_mg'] = df_user_mgt.\
            apply(lambda x: [x['movies_perc']] + x['genres_features'], axis=1)
    df_user_mgt['features_mgt'] = df_user_mgt.\
            apply(lambda x: [x['movies_perc']] + x['genres_features'] + x['tags_features'], axis=1)
    df_user_features = df_user_mgt[['userId','features_m','features_mg','features_mgt']]
    
    user_all_features = {}
    user_all_features['features_m'] = LightFM_features(df_user_features, 'userId', 'features_m')
    user_all_features['features_mg'] = LightFM_features(df_user_features, 'userId', 'features_mg')
    user_all_features['features_mgt'] = LightFM_features(df_user_features, 'userId', 'features_mgt')
    
    return(user_all_features)
    
    
    return(df_user_features)

# MOVIES FEATURES

## Users features

def movie_users(df_ratings):
    total_users = len(pd.unique(df_ratings.userId))
    df_movie_users = df_ratings.groupby('movieId').userId.count().reset_index()
    df_movie_users.rename(columns={'userId':'total_users'}, inplace=True)
    df_movie_users['users_perc'] = df_movie_users.total_users / total_users
    df_movie_users = df_movie_users[['movieId', 'users_perc']]
    return(df_movie_users)

## Genres features

def movie_genres(df_ratings, df_movies, genres_dict):
    df_rating_movies = df_ratings[['movieId']].drop_duplicates(keep='first')
    df_movie_genres = pd.merge(df_rating_movies, df_movies, on='movieId', how='inner')
    df_movie_genres['genres_features'] = df_movie_genres\
        .apply(lambda x: get_movie_genres_features(x['genres'], genres_dict), axis=1)
    df_movie_genres = df_movie_genres[['movieId', 'genres_features']]
    return(df_movie_genres)

def get_movie_genres_features(genres_str, genres_dict):
    genres_num = len(genres_dict)
    genres_features = [0] * genres_num
    genres_list = [genre for genre in genres_str.split('|') if genre != "(no genres listed)"]
    for genre in genres_list:
        genres_features[genres_dict[genre]] = 1 / len(genres_list)
    return(genres_features)

## Tags features

def movie_tags(df_ratings, df_genome, rel_tags_dict):
    df_movie_genome_list = genome_list_per_movie(df_ratings, df_genome)
    df_movie_tags = tags_features_per_movie(df_movie_genome_list, rel_tags_dict)
    return(df_movie_tags)

def genome_list_per_movie(df_ratings, df_genome):
    df_rating_movies = df_ratings[['movieId']].drop_duplicates(keep='first')
    df_movie_genome = pd.merge(df_rating_movies, df_genome, on='movieId', how='inner')
    df_movie_genome_list = df_movie_genome\
        .groupby('movieId')[['tag', 'relevance']]\
        .apply(lambda x:pd.Series({'genome': x.values.tolist()}))\
        .reset_index()
    df_movie_genome_list.rename(columns={'genome':'genome_list'}, inplace=True)
    return(df_movie_genome_list)

def tags_features_per_movie(df_movie_genome_list, rel_tags_dict):
    df_movie_genome_list['tags_features'] = df_movie_genome_list\
        .apply(lambda x: get_rel_tags_features(x['genome_list'], rel_tags_dict), axis=1)
    df_movie_tags = df_movie_genome_list[['movieId', 'tags_features']]
    return(df_movie_tags)

def get_rel_tags_features(genome_list, rel_tags_dict):
    tags_num = len(rel_tags_dict)
    rel_tags_features = [0] * tags_num
    try:
        rel_genome = [genome for genome in genome_list if genome[0] in rel_tags_dict.keys()]
    except:
        rel_genome = []
    if (len(rel_genome) > 0):
        for tag, relevance in rel_genome:
            rel_tags_features[rel_tags_dict[tag]] = relevance
    return(rel_tags_features)

## Merge features

def get_movie_all_features(df_ratings, df_movies, df_genome, genres_dict, rel_tags_dict):
    df_movie_users = movie_users(df_ratings)
    df_movie_genres = movie_genres(df_ratings, df_movies, genres_dict)
    df_movie_tags = movie_tags(df_ratings, df_genome, rel_tags_dict)
    df_movie_ug = pd.merge(df_movie_users, df_movie_genres, on='movieId', how='inner')
    df_movie_ugt = pd.merge(df_movie_ug, df_movie_tags, on='movieId', how='inner')

    df_movie_ugt['features_u'] = df_movie_ugt.apply(lambda x: [x['users_perc']], axis=1)
    df_movie_ugt['features_ug'] = df_movie_ugt.\
        apply(lambda x: [x['users_perc']] + x['genres_features'], axis=1)
    df_movie_ugt['features_ugt'] = df_movie_ugt.\
        apply(lambda x: [x['users_perc']] + x['genres_features'] + x['tags_features'], axis=1)
    df_movie_features = df_movie_ugt[['movieId', 'features_u', 'features_ug', 'features_ugt']]
    
    movie_all_features = {}
    movie_all_features['features_u'] = LightFM_features(df_movie_features, 'movieId', 'features_u')
    movie_all_features['features_ug'] = LightFM_features(df_movie_features, 'movieId', 'features_ug')
    movie_all_features['features_ugt'] = LightFM_features(df_movie_features, 'movieId', 'features_ugt')
    
    return(movie_all_features)

# LIGHT FM UTILS
def LightFM_features(df_features, user_movie, features):
    df_features_tupples = df_features.apply(lambda x: (x[user_movie],x[features]), axis=1).values
    features = LightFM_array(df_features_tupples)
    return(features)
    
def LightFM_array(df_features_tupples):
    features = []
    for row in range(len(df_features_tupples)):
        elem = df_features_tupples[row][0]
        feature_map = df_features_tupples[row][1]
        feature_dict = {}
        for feat in range(len(feature_map)):
            feature_dict[feat] = feature_map[feat]
        features.append((elem,feature_dict))
    return(features)