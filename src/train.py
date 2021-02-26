import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

import wandb
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from quest_dataset import QuestData
from model import QuestModel


class ToggleBertBaseTraining(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print("-" * 100)
        print("ToggleBertBaseTraining Callback working.............")
        if trainer.current_epoch == 0:
            print(
                f"current_epoch is: {trainer.current_epoch} and freezing BERT layer's parameters"
            )
            for p in pl_module.model.bert.parameters():
                p.requires_grad = False
        else:
            print(
                f"current_epoch is: {trainer.current_epoch} and unfreezing BERT layer's parameters for training"
            )
            for p in pl_module.model.bert.parameters():
                p.requires_grad = True
        print("-" * 100)


class SaveModelWeights(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        os.makedirs("../models/", exist_ok=True)
        print("-" * 100)
        print("SaveModelWeight Callback working.............")
        print(f"trainer.current_epoch: {trainer.current_epoch}")
        if trainer.current_epoch >= 2:
            print(
                f"val_spearman: {trainer.logger_connector.logged_metrics['val_spearman']}"
            )
            m_filepath = f"../models/{pl_module.hparams.model_name}-epoch-{trainer.current_epoch}-val_spearman{trainer.logger_connector.logged_metrics['val_spearman'].item():.5f}-fold-{pl_module.hparams.fold}.pt"
            torch.save(pl_module.model.state_dict(), m_filepath)
            print(f"saved current model weights in file: {m_filepath}")
        print("-" * 100)


if __name__ == "__main__":
    pl.seed_everything(420)

    parser = ArgumentParser()

    # trainer related arguments
    parser.add_argument(
        "--gpus",
        default=1,
        help="if value is 0 cpu will be used, if string then that gpu device will be used",
    )
    parser.add_argument("--checkpoint_callback", action="store_true")
    parser.add_argument("--logger", action="store_true")
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)
    parser.add_argument("--accumulate_grad_batches", default=2, type=int)
    parser.add_argument("--model_name", default="quest", type=str)

    # data related arguments
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--fold", default=0, type=int, choices=[0, 1, 2, 3, 4])

    # model related arguments
    parser.add_argument("--maxlen", default=512, type=int)
    parser.add_argument("--bert_lr", default=1e-5, type=int)
    parser.add_argument("--linear_lr", default=5e-3, type=int)
    parser.add_argument("--bert_dropout", default=0.3, type=float)
    parser.add_argument(
        "--bert_output_used",
        default="maxpooled",
        type=str,
        choices=["maxpooled", "weighted_sum"],
    )
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.effective_batch_size = args.batch_size * args.accumulate_grad_batches
    args.log_every_n_steps = args.accumulate_grad_batches * 5

    pl_model = QuestModel(args)
    data = QuestData(args)

    if args.logger:
        args.logger = WandbLogger(
            project="google-quest-challenge-kaggle", group=str(args.fold)
        )

    early_stop_callback = EarlyStopping(
        monitor="val_spearman", min_delta=0.0000, patience=2, verbose=False, mode="max"
    )

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="models",
    #     filename=args.model_name + "-{epoch:02d}-{val_spearman:.5f}" + f"fold={args.fold}",
    #     verbose=True,
    #     save_top_k=2,
    #     monitor='val_spearman',
    # )

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            ToggleBertBaseTraining(),
            SaveModelWeights(),
            # checkpoint_callback,
            early_stop_callback,
        ],
    )

    print(
        f"Training model_name={args.model_name} on fold={args.fold} for max_apochs={args.max_epochs} with and effective batch_size of effective_batch_size={args.effective_batch_size}"
    )
    trainer.fit(pl_model, data)
