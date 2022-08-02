from flask import Flask, request, jsonify
from pathlib import Path
from flask_migrate import Migrate

# from app import app
# from app import app
from config import Config
from flask_sqlalchemy import SQLAlchemy

import numpy as np
# from recommenders.models.try_retrieval_model import  load
from recommenders.models.new_retreival import load_model
from recommenders.models.scheduler_predict import job_predict
# from train.run_experiment import _get_optimizer
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

# DB_PATH = Path(__file__).resolve().parents[2] / "app.db"
# WANDB_PATH = Path(__file__).resolve().parents[1] / "wandb"
# BEST_RETR_RUN = "run-20201113_034446-boy4apok/files"
# BEST_RANK_RUN = "run-20201113_102942-9xmrl22y/files"
from run import app

MIN_CANDID = 100
NUM_RECOMMEND = 10 



# retrieval_model = retrieve.RetrievalModel(WANDB_PATH / BEST_RETR_RUN)
# ranking_model = rank.RankingModel(WANDB_PATH / BEST_RANK_RUN)

# @api.route("/")
# def index():
#     return "recommenders api"
#
#
# def new_user():
#     experiment_config = {
#         "dataset": "Dataset",
#         "dataset_args": {
#             "batch_size": 512,
#             "test_fraction": 0.3
#         },
#         "model": "RetrievalModel",
#         "network": "retrieval_basic_factorization",
#         "network_args": {
#             "embedding_dimension": 32,
#         },
#         "train_args": {
#             "epochs": 1,
#             "learning_rate": 0.01,
#             "optimizer": "SGD"
#
#         }
#     }
#     models_module = importlib.import_module("recommenders.models")
#     model_class_ = getattr(models_module, experiment_config["model"])
#
#     networks_module = importlib.import_module("recommenders.networks")
#     network_fn_ = getattr(networks_module, experiment_config["network"])
#     network_args = experiment_config.get("network_args", {})
#
#     datasets_module = importlib.import_module("recommenders.datasets")
#     dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
#     dataset_args = experiment_config.get("dataset_args", {})
#     dataset = dataset_class_(**dataset_args)
#     dataset.load_or_generate_data()
#     # for watch in watched_movie:
#     #     dataset.test.concatenate([(0, watch.movie_id, watch.rate, 0)])
#
#     model = model_class_(
#         dataset=dataset, network_fn=network_fn_, network_args=network_args
#     )
#
#     # if use_wandb:
#     wandb.init(config=experiment_config)
#
#     callbacks = list()
#     callbacks.append(WandbCallback())
#     optimizer = _get_optimizer(experiment_config["train_args"]["optimizer"])
#     model.compile(optimizer=optimizer(learning_rate=experiment_config["train_args"]["learning_rate"]))
#     model.fit(dataset.train,
#             epochs=experiment_config["train_args"]["epochs"],
#             validation_data=dataset.test,
#             validation_freq=20,
#             callbacks=callbacks)
#     return model, dataset




@app.route("/v1/predict")
def predict():
    user_id = request.args.get("user_id")
    # watched_movie = WatchedMovie.query.filter(WatchedMovie.user_id == user_id).all()
    titles = load_model(user_id)
    # history = [m.movie_id for m in watched_movie]
    # retrieval_model_new, dataset = new_user()
    # u_feature = {'userid':tf.constant([str(user_id)])}
    # query_embedding = retrieval_model_new.query_model.predict(u_feature).squeeze()
    # candids = retrieval_model.index.get_nns_by_vector(query_embedding, 10 + len(history))
    # candids = list(set(candids) - set(history))
    # item_pred = sorted(list(zip(candids)),key=lambda tup: -tup[0])[:NUM_RECOMMEND]
    titles = np.array(titles[0,])
    rec = []
    for t in titles:
        rec.append(t.decode("utf-8") )
    out = { 'recs': rec }
    
    return jsonify(out)
    

