import os
import pprint
import re
from typing import Dict, Text

import numpy as np
import tensorflow as tf
from pathlib import Path
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds

from app.models import Movie, WatchedMovie

candidates = 10

batch_size = 1

DATA_PATH = Path(__file__).resolve().parents[3] / "data"

# ratings = tf.data.experimental.make_csv_dataset(os.path.join(DATA_PATH, "ratings.csv"),
#                                                 batch_size=batch_size, num_epochs=5,
#                                                 column_names=['userId', 'movieId', 'rating', 'timestamp'])
# # test = tf.data.experimental.make_csv_dataset(os.path.join(DATA_PATH,"1605562828.0/ratings_test.csv"),
# #                                              batch_size=batch_size, num_epochs=5,
# #                                              column_names=['userid', 'movieid',
# #                                                            'rating'])
# movies = tf.data.experimental.make_csv_dataset(os.path.join(DATA_PATH,"movies.csv"), batch_size=batch_size,
#                                                num_epochs=5, shuffle=False,
#                                                column_names=['movieId','title','genres'])
# # Ratings data.
# ratings = tfds.load("movielens/20m-ratings", split="train")
# # Features of all the available movies.
# movies = tfds.load("movielens/20m-movies", split="train")
#
# ratings = ratings.map(lambda x: {
#     "movie_title": x["movie_title"],
#     "user_id": x["user_id"],
# })
#
# movies = movies.map(lambda x: x['movie_title'])
# tf.random.set_seed(42)
# shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
#
# train = shuffled.take(80_000)
# test = shuffled.skip(80_000).take(20_000)
#
# movie_titles = movies.batch(1_000)
# user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
#
# unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
# unique_user_ids = np.unique(np.concatenate(list(user_ids)))
#
# embedding_dimension = 32
# user_model = tf.keras.Sequential([
#   tf.keras.layers.StringLookup(
#       vocabulary=unique_user_ids, mask_token=None),
#   # We add an additional embedding to account for unknown tokens.
#   tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
# ])
#
# movie_model = tf.keras.Sequential([
#   tf.keras.layers.StringLookup(
#       vocabulary=unique_movie_titles, mask_token=None),
#   tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
# ])
#
# metrics = tfrs.metrics.FactorizedTopK(
#   candidates=movies.batch(128).map(movie_model)
# )
#
# task = tfrs.tasks.Retrieval(
#   metrics=metrics
# )
#

class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model, task):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_movie_embeddings)
#
#
# model = MovielensModel(user_model, movie_model, task)
# model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
#
# cached_train = train.shuffle(100_000).batch(8192).cache()
# cached_test = test.batch(4096).cache()
#
# model.fit(cached_train, epochs=100)
#
# model.evaluate(cached_test, return_dict=True)
#
# model.save_weights(Path(__file__).resolve().parents[1]/'weights/retrieval')
# # Create a model that takes in raw query features, and
# index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k=candidates)
# # recommends movies out of the entire movies dataset.
# index.index_from_dataset(
#   tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
# )
#
# # Get recommendations.
# _, titles = index(tf.constant(["0"]))
# print(f"Recommendations for user 0: {titles[0, :3]}")


# cached_train = train.shuffle(100_000).batch(8192).cache()
# cached_test = test.batch(4096).cache()
# dict_user = {
#  'movie_title':[b"Frozen (2013)"],
#  'user_id': [b'0']}
# new_user = tf.data.Dataset.from_tensor_slices(dict_user)
# ratings = ratings.concatenate(new_user)

# dict_user = {
#  'movie_title':[b"Toy Story (1995)"],
#  'user_id': [b'0']}
# new_user = tf.data.Dataset.from_tensor_slices(dict_user)
# ratings = ratings.concatenate(new_user)
#
# dict_user = {
#  'movie_title':[b"Ratatouille (2007)"],
#  'user_id': [b'0']}
# new_user = tf.data.Dataset.from_tensor_slices(dict_user)
# ratings = ratings.concatenate(new_user)
#
# dict_user = {
#  'movie_title':[b"Incredibles 2 (2018)"],
#  'user_id': [b'0']}
# new_user = tf.data.Dataset.from_tensor_slices(dict_user)
# ratings = ratings.concatenate(new_user)
#
#
# ratings = ratings.map(lambda x: {
#     "movie_title": x["movie_title"],
#     "user_id": x["user_id"],
# })

def add_new_user(watched_movies):
    ratings = tfds.load("movielens/100k-ratings", split="train")
    # Features of all the available movies.

    movies = tfds.load("movielens/100k-movies", split="train")

    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        # 'year':  print(x['movie_title'])
    })

    # movies = movies.map(lambda x: x['movie_title'])
    # for watched_movie in watched_movies:
    #     movie = Movie.query.filter(watched_movie.movie_id == Movie.movielens_id).first()
    #     dict_user = {
    #         'movie_title': [bytes(movie.name +' ('+ str(movie.release_year) + ')', 'utf-8')],
    #         'user_id': [b'0']}
    #     new_user = tf.data.Dataset.from_tensor_slices(dict_user)
    #     ratings = ratings.concatenate(new_user)
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    embedding_dimension = 32
    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    movie_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
        tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).map(movie_model)
    )

    task = tfrs.tasks.Retrieval(
        metrics=metrics
    )



    # movies = movies.map(lambda x: x['movie_title'])
    # tf.random.set_seed(42)
    # shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
    #
    # train = shuffled.take(80_000)
    # test = shuffled.skip(80_000).take(20_000)
    #
    # # movie_titles = movies.batch(1_000)
    # user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
    #
    # # unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    # unique_user_ids = np.unique(np.concatenate(list(user_ids)))
    #
    # embedding_dimension = 32
    # user_model = tf.keras.Sequential([
    #   tf.keras.layers.StringLookup(
    #       vocabulary=unique_user_ids, mask_token=None),
    #   # We add an additional embedding to account for unknown tokens.
    #   tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    # ])

    # movie_model = tf.keras.Sequential([
    #   tf.keras.layers.StringLookup(
    #       vocabulary=unique_movie_titles, mask_token=None),
    #   tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    # ])

    # metrics = tfrs.metrics.FactorizedTopK(
    #   candidates=movies.batch(128).map(movie_model)
    # )
    #
    # task = tfrs.tasks.Retrieval(
    #   metrics=metrics
    # )

    model = MovielensModel(user_model, movie_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    model.fit(cached_train, epochs=1)

    model.evaluate(cached_test, return_dict=True)

    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model) #, k=candidates)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
      tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
    )

    # Get recommendations.
    _, titles = index(tf.constant(["0"]))
    print(f"Recommendations for user 0: {titles[0, :3]}")
    return titles


# ratings = ratings.map(lambda x: {
#     "movie_id": x["movieId"],
#     "user_id": x["userId"],
#     "user_rating": x["rating"]
# })
#
# tf.random.set_seed(42)
# shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
#
# train = shuffled.take(80_000)
# test = shuffled.skip(80_000).take(20_000)
#
# movie_ids = ratings.batch(1_000_000).map(lambda x: x["movie_id"])
# user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
#
# unique_movie_titles = np.unique(np.concatenate(list(movie_ids)))
# unique_user_ids = np.unique(np.concatenate(list(user_ids)))
#
#
# class RankingModel(tf.keras.Model):
#
#     def __init__(self):
#         super().__init__()
#         embedding_dimension = 32
#
#         # Compute embeddings for users.
#         self.user_embeddings = tf.keras.Sequential([
#             tf.keras.layers.IntegerLookup(
#                 vocabulary=unique_user_ids, mask_token=None),
#             tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
#         ])
#
#         # Compute embeddings for movies.
#         self.movie_embeddings = tf.keras.Sequential([
#             tf.keras.layers.IntegerLookup(
#                 vocabulary=unique_movie_titles, mask_token=None),
#             tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
#         ])
#
#         # Compute predictions.
#         self.ratings = tf.keras.Sequential([
#             # Learn multiple dense layers.
#             tf.keras.layers.Dense(256, activation="relu"),
#             tf.keras.layers.Dense(64, activation="relu"),
#             # Make rating predictions in the final layer.
#             tf.keras.layers.Dense(1)
#         ])
#
#     def call(self, inputs):
#         user_id, movie_title = inputs
#
#         user_embedding = self.user_embeddings(user_id)
#         movie_embedding = self.movie_embeddings(movie_title)
#
#         return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))
#
#
# task = tfrs.tasks.Ranking(
#     loss=tf.keras.losses.MeanSquaredError(),
#     metrics=[tf.keras.metrics.RootMeanSquaredError()]
# )
#
#
# class MovielensModel(tfrs.models.Model):
#
#     def __init__(self):
#         super().__init__()
#         self.ranking_model: tf.keras.Model = RankingModel()
#         self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
#             loss=tf.keras.losses.MeanSquaredError(),
#             metrics=[tf.keras.metrics.RootMeanSquaredError()]
#         )
#
#     def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
#         return self.ranking_model((features["user_id"], features["movie_id"]))
#
#     def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
#         labels = features.pop("user_rating")
#
#         rating_predictions = self(features)
#
#         # The task computes the loss and the metrics.
#         return self.task(labels=labels, predictions=rating_predictions)
#
#
# model = MovielensModel()
# model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
#
# cached_train = train.cache().shuffle(100_000).batch(8192)
# # train.shuffle(100_000).batch(8192).cache()
# cached_test = test.cache().batch(4096)
#
# model.fit(cached_train, epochs=3)
# model.evaluate(cached_test, return_dict=True)
#
# new_rate = [1,1,4.0,964982703]
# ratings.concatenate(new_rate)
#
#
#
# ratings = ratings.map(lambda x: {
#     "movie_id": x["movieId"],
#     "user_id": x["userId"],
#     "user_rating": x["rating"]
# })
#
# tf.random.set_seed(42)
# shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
#
# train = shuffled.take(80_000)
# test = shuffled.skip(80_000).take(20_000)
#
# movie_ids = ratings.batch(1_000_000).map(lambda x: x["movie_id"])
# user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
#
# unique_movie_titles = np.unique(np.concatenate(list(movie_ids)))
# unique_user_ids = np.unique(np.concatenate(list(user_ids)))
#
#
# class RankingModel(tf.keras.Model):
#
#     def __init__(self):
#         super().__init__()
#         embedding_dimension = 32
#
#         # Compute embeddings for users.
#         self.user_embeddings = tf.keras.Sequential([
#             tf.keras.layers.IntegerLookup(
#                 vocabulary=unique_user_ids, mask_token=None),
#             tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
#         ])
#
#         # Compute embeddings for movies.
#         self.movie_embeddings = tf.keras.Sequential([
#             tf.keras.layers.IntegerLookup(
#                 vocabulary=unique_movie_titles, mask_token=None),
#             tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
#         ])
#
#         # Compute predictions.
#         self.ratings = tf.keras.Sequential([
#             # Learn multiple dense layers.
#             tf.keras.layers.Dense(256, activation="relu"),
#             tf.keras.layers.Dense(64, activation="relu"),
#             # Make rating predictions in the final layer.
#             tf.keras.layers.Dense(1)
#         ])
#
#     def call(self, inputs):
#         user_id, movie_title = inputs
#
#         user_embedding = self.user_embeddings(user_id)
#         movie_embedding = self.movie_embeddings(movie_title)
#
#         return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))
#
#
# task = tfrs.tasks.Ranking(
#     loss=tf.keras.losses.MeanSquaredError(),
#     metrics=[tf.keras.metrics.RootMeanSquaredError()]
# )
#
#
# class MovielensModel(tfrs.models.Model):
#
#     def __init__(self):
#         super().__init__()
#         self.ranking_model: tf.keras.Model = RankingModel()
#         self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
#             loss=tf.keras.losses.MeanSquaredError(),
#             metrics=[tf.keras.metrics.RootMeanSquaredError()]
#         )
#
#     def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
#         return self.ranking_model((features["user_id"], features["movie_id"]))
#
#     def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
#         labels = features.pop("user_rating")
#
#         rating_predictions = self(features)
#
#         # The task computes the loss and the metrics.
#         return self.task(labels=labels, predictions=rating_predictions)
#
#
# model = MovielensModel()
# model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
#
# cached_train = train.cache().shuffle(100_000).batch(8192)
# # train.shuffle(100_000).batch(8192).cache()
# cached_test = test.cache().batch(4096)
#
# model.fit(cached_train, epochs=3)
# model.evaluate(cached_test, return_dict=True)

def load_model(user_id):
    ratings, train, test, unique_user_ids, unique_movie_titles, movies = load_data()
    user_model, movie_model, task = build_model(unique_user_ids, unique_movie_titles, movies)
    model = MovielensModel(user_model, movie_model, task)
    model.load_weights(Path(__file__).resolve().parents[1] / 'weights/retrieval_new')
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k=candidates)
    index.index_from_dataset(
      tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
    )

    # Get recommendations.
    _, titles = index(tf.constant([str(100000 + int(user_id))]))
    print(f"Recommendations for user 0: {titles[0, :3]}")
    return titles



def load_data(ratings=None):
    if not ratings:
        ratings = tfds.load("movielens/latest-small-ratings", split="train")
        # Features of all the available movies.

    movies = tfds.load("movielens/latest-small-movies", split="train")

    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        # 'year':  print(x['movie_title'])
    })
    watched_movies = WatchedMovie.query.filter().all()

    for watched_movie in watched_movies:
        movie = Movie.query.filter(watched_movie.movie_id == Movie.movielens_id).first()
        dict_user = {
            'movie_title': [bytes(movie.name + ' (' + str(movie.release_year) + ')', 'utf-8')],
            'user_id': [bytes(str(100000+ watched_movie.user_id), 'utf-8')]}
        new_user = tf.data.Dataset.from_tensor_slices(dict_user)
        ratings = ratings.concatenate(new_user)


    movies = movies.map(lambda x: x['movie_title'])

    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)
    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))
    return ratings, train, test, unique_user_ids, unique_movie_titles, movies




def build_model(unique_user_ids, unique_movie_titles, movies):

    embedding_dimension = 32
    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension),
        # tf.keras.layers.Flatten()
    ])

    movie_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
        tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension),
        # tf.keras.layers.Flatten()
    ])

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).map(movie_model)
    )

    task = tfrs.tasks.Retrieval(
        metrics=metrics
    )
    return user_model, movie_model, task
