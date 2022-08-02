from pathlib import Path
import importlib
import tensorflow as tf
import wandb
from annoy import AnnoyIndex
from wandb.integration.keras import WandbCallback

from train.run_experiment import _get_optimizer

DEFAULT_CONFIG =  {
        "dataset": "Dataset",
        "model": "RetreivalModel",
        "network": "retrieval_basic_factorization",
        "network_args": {
            "embedding_dimension": 32,
        }
}

class RetrievalModel:
    def __init__(self, weight_dir,config=DEFAULT_CONFIG):

        networks_module = importlib.import_module("recommenders.networks")
        network_fn_ = getattr(networks_module, config["network"])
        network_args = config.get("network_args", {})
        
        datasets_module = importlib.import_module("recommenders.datasets")
        dataset_class_ = getattr(datasets_module, config["dataset"])
        dataset_args = config.get("dataset_args", {})
        dataset = dataset_class_(**dataset_args)
        dataset.load_or_generate_data(update_to_latest_db=False)
        self.dataset = dataset

        self.query_model, _ = network_fn_(dataset.unique_user_ids, dataset.unique_movie_ids, **network_args)
        self.query_model.load_weights(weight_dir / 'query')
        self.index = AnnoyIndex(self.query_model.embedding_dimension, 'dot')
        self.index.load(str(weight_dir /'candid.annoy'))


    def predict(self, features, num_candids = 100):

        query_embedding =  self.query_model.predict(features).squeeze()
        candids = self.index.get_nns_by_vector(query_embedding, num_candids)
        return candids

    def dataset(self):
        return self.dataset()

    # def fit(self):
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
    #     wandb.init(config=experiment_config)
    #
    #     callbacks = list()
    #     callbacks.append(WandbCallback())
    #
    #     optimizer = _get_optimizer(experiment_config["train_args"]["optimizer"])
    #     self.query_model.compile(optimizer=optimizer(learning_rate=experiment_config["train_args"]["learning_rate"]))
    #     self.query_model.fit(self.dataset.train,
    #               epochs=experiment_config["train_args"]["epochs"],
    #               validation_data=self.dataset.test,
    #               validation_freq=20,
    #               callbacks=callbacks)



