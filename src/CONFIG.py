from pathlib import Path
import torch
import os

path_dict = {
    "ROOT_DIR": Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
}

path_dict["TRAIN_PATH"] = path_dict["ROOT_DIR"] / "input" / "train_with_GKF.csv"
path_dict["TEST_PATH"] = path_dict["ROOT_DIR"] / "input" / "test.csv"
path_dict["BERT_PATH"] = "bert-base-uncased"
path_dict["TOKENIZER_PATH"] = "bert-base-uncased"
path_dict["MODEL_FILENAME"] = "models/some_model325"
path_dict["MODEL_PATH"] = path_dict["ROOT_DIR"] / path_dict["MODEL_FILENAME"]

config_dict = {
    "base_model": "bert-base-uncased",
    "loss_criterion": "nn.BCEWithLogitsLoss",
    "wandbLog": True,
    "wandbProjectName": "google-quest-challenge-kaggle",
    "wandb_log_logits": False,
    "gpus": "1",
    "auto_select_gpus": False,
    "freeze_bert": False,
    "maxlen": 512,
    "lr": 3e-5,
    "bert_dropout": 0.3,
    "bert_output_used": "maxpooled",
    "batch_size": 4,
    "max_epochs": 10,
    "save_model_bin": True,
    "save_model_onnx": False,
    "wandb_save_model": False,
    "accumulate_grad_batches": 4,
}
config_dict["effective_batch_size"] = (
    config_dict["batch_size"] * config_dict["accumulate_grad_batches"]
)
config_dict["log_every_n_steps"] = config_dict["accumulate_grad_batches"] * 10
config_dict["MODEL_FILENAME"] = path_dict["MODEL_FILENAME"]
