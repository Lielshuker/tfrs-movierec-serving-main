import tensorflow as tf
from pathlib import Path

# def run(load_model=False):
# from recommenders.models.new_retreival import load_data, build_model, MovielensModel


def job_predict():
    from recommenders.models.new_retreival import load_data, build_model, MovielensModel

    # ratings , train, test, unique_user_ids, unique_movie_titles, movies = load_data()
    # user_model, movie_model, task = build_model(unique_user_ids, unique_movie_titles, movies)
    # model = MovielensModel(user_model, movie_model, task)
    #
    # model.load_weights(Path(__file__).resolve().parents[1] / 'weights/retrieval')
    # model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    # cached_train = train.shuffle(100_000).batch(8192).cache()
    # cached_test = test.batch(4096).cache()
    # model.fit(cached_train, epochs=1)
    # model.evaluate(cached_test, return_dict=True)

    # for watched_movie in watched_movies:
    #     movie = Movie.query.filter(watched_movie.movie_id == Movie.movielens_id).first()
    #     dict_user = {
    #         'movie_title': [bytes(movie.name + ' (' + str(movie.release_year) + ')', 'utf-8')],
    #         'user_id': [bytes(str(100000+ movie.id), 'utf-8')]}
    #     new_user = tf.data.Dataset.from_tensor_slices(dict_user)
    #     ratings = ratings.concatenate(new_user)

    ratings , train, test, unique_user_ids, unique_movie_titles, movies = load_data()
    user_model, movie_model, task = build_model(unique_user_ids, unique_movie_titles, movies)
    model = MovielensModel(user_model, movie_model, task)

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()
    model.fit(cached_train, epochs=200)
    model.evaluate(cached_test, return_dict=True)
    model.save_weights(Path(__file__).resolve().parents[1] / 'weights/retrieval_new')
    # model.load_weights(Path(__file__).resolve().parents[1] / 'weights/retrieval')
    # model.fit(cached_train, epochs=1)

# job_predict()
