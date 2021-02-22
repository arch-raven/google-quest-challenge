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
        if trainer.current_epoch == 0:
            print(f"current_epoch is: {trainer.current_epoch} and freezing BERT layer's parameters")
            for p in pl_module.bert.parameters():
                p.requires_grad = False
        else:
            print(f"current_epoch is: {trainer.current_epoch} and unfreezing BERT layer's parameters for training")
            for p in pl_module.bert.parameters():
                p.requires_grad = True

def main():
    pl.seed_everything(420)
    
    parser = ArgumentParser()
    parser.add_argument("--fold", type=int, choices=[0,1,2,3,4])
    parser.add_argument("--gpus", default=1, help="if value is 0 cpu will be used, if string then that gpu device will be used")
    parser.add_argument("--maxlen", default=512, type=int)
    parser.add_argument("--bert_lr", default=1e-5, type=int)
    parser.add_argument("--linear_lr", default=5e-3, type=int)
    parser.add_argument("--bert_dropout", default=0.3, type=float)
    parser.add_argument("--bert_output_used", default="maxpooled", type=str, choices=["maxpooled", "weighted_sum"])
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--accumulate_grad_batches", default=4, type=int)
    
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    args.effective_batch_size = args.batch_size*args.accumulate_grad_batches
    args.log_every_n_steps = args.accumulate_grad_batches * 5
    
    model = QuestModel(args)
    data = QuestData(args)
    
    args.logger = WandbLogger(project="google-quest-challenge-kaggle", group=str(args.fold))
    
    early_stop_callback = EarlyStopping(
        monitor="val_spearman", min_delta=0.0000, patience=2, verbose=False, mode="max"
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="quest-{epoch:02d}-{FOLD}",
    )
    
    trainer = pl.Trainer.from_argparse_args(
        args, 
        callbacks=[
            ToggleBertBaseTraining(), 
            checkpoint_callback, 
            early_stop_callback,
        ]
    )
    