import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import transformers
from transformers import AdamW, get_cosine_schedule_with_warmup
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from quest_dataset import get_dataloaders
from CONFIG import path_dict, config_dict
from scipy.stats import spearmanr
from argparse import ArgumentParser

global path_dict, FOLD

class Model(pl.LightningModule):
    def __init__(self, path_dict=path_dict, config_dict=config_dict, **kwargs):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(path_dict["BERT_PATH"])
        self.bert_dropout = nn.Dropout(config_dict["bert_dropout"])
        self.linear = nn.Linear(768, 30)

        self.save_hyperparameters(config_dict)

    @staticmethod
    def loss(logits, targets):
        return nn.BCEWithLogitsLoss()(logits, targets)

    def forward(self, ids_seq, attn_masks, token_type_ids):

        bert_out = self.bert(
            ids_seq, attention_mask=attn_masks, token_type_ids=token_type_ids
        )
        # using maxpooled output
        max_out = self.bert_dropout(bert_out[1])
        return self.linear(max_out)

    def shared_step(self, batch):
        ids_seq, attn_masks, token_type_ids, target = (
            batch["ids_seq"],
            batch["attn_masks"],
            batch["token_type_ids"],
            batch["target"],
        )
        logits = self(ids_seq, attn_masks, token_type_ids)
        loss = self.loss(logits, target)
        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self.shared_step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss = self.shared_step(batch)
        self.log(
            "valid_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        return {"valid_loss": loss, "logits": logits, "true_preds": batch["target"]}

    def configure_optimizers(self):
        grouped_parameters = [
            {"params":self.bert.parameters(), "lr":self.hparams.bert_lr},
            {"params":self.linear.parameters(), "lr":self.hparams.linear_lr}
        ]
        optim = AdamW(grouped_parameters, lr=self.hparams.bert_lr)
        num_training_steps = 304*self.hparams.max_epochs                                                                                      ##change this, hardcoded rn
        sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=num_training_steps)
        
        return [optim] , [sched]

    def get_dummy_inputs(self):
        return (
            torch.zeros((8, 512), dtype=torch.long, device=self.device),
            torch.zeros((8, 512), dtype=torch.long, device=self.device),
            torch.zeros((8, 512), dtype=torch.long, device=self.device),
        )

    def validation_epoch_end(
        self, validation_step_outputs
    ):  # args are defined as part of pl API
        # if self.hparams["save_model_bin"]:
        #     torch.save(self.state_dict(), self.hparams["MODEL_FILENAME"] + ".pt")

        if self.hparams["save_model_onnx"]:
            dummy_inputs = self.get_dummy_inputs()
            torch.onnx.export(
                self,
                dummy_inputs,
                f"{path_dict['MODEL_PATH']}.onnx",
                input_names=["seq", "attn_masks", "token_type_ids"],
                dynamic_axes={
                    "seq": {0: "sequence"},
                    "attn_masks": {0: "sequence"},
                    "token_type_ids": {0: "sequence"},
                },
                opset_version=11,
            )
            if self.hparams["wandb_save_model"]:
                wandb.save(path_dict["MODEL_FILENAME"] + ".onnx")

        if self.hparams["wandb_log_logits"]:
            flattened_logits = torch.sigmoid(
                torch.cat([out['logits'] for out in validation_step_outputs])
            )
            self.logger.experiment.log(
                {
                    "valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
                    "global_step": self.global_step,
                },
                commit=False,
            )
        y_pred = (
            torch.sigmoid(torch.cat([out['logits'] for out in validation_step_outputs]))
            .to("cpu")
            .detach()
            .numpy()
        )
        y_true = (
            torch.cat([out['true_preds'] for out in validation_step_outputs]).to("cpu").detach().numpy()
        )

        spearman_corr = self.spearman_metric(y_true, y_pred)
        self.log("val_spearman", spearman_corr, logger=True)

    @staticmethod
    def spearman_metric(y_true, y_pred, return_scores=False):
        corr = [
            spearmanr(pred_col, target_col).correlation
            for pred_col, target_col in zip(y_pred.T, y_true.T)
        ]
        if return_scores:
            return corr
        else:
            return np.nanmean(corr)


def main():
    pl.seed_everything(42)
    
    parser = ArgumentParser()
    parser.add_argument("--fold", type=int, choices=[0,1,2,3,4])
    args = parser.parse_args()
    FOLD = args.fold
    
    path_dict["MODEL_FILENAME"] = f"models/free-log-fire-onFold{args.fold}"
    path_dict["MODEL_PATH"] = path_dict["ROOT_DIR"] / path_dict["MODEL_FILENAME"]
    config_dict["MODEL_FILENAME"] = path_dict["MODEL_FILENAME"]
    
    model = Model(config_dict=config_dict, fold=args.fold)
    train_dl, valid_dl = get_dataloaders(fold=args.fold, config_dict=model.hparams)

    wandb_logger = WandbLogger(project="google-quest-challenge-kaggle", group=str(args.fold))

    early_stop_callback = EarlyStopping(
        monitor="val_spearman", min_delta=0.0000, patience=3, verbose=False, mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="quest-{epoch:02d}-{FOLD}",
    )

    class ToggleBertTraining(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch == 0:
                print(f"current_epoch is: {trainer.current_epoch} and freezing BERT layer's parameters")
                for p in pl_module.bert.parameters():
                    p.requires_grad = False
            else:
                print(f"current_epoch is: {trainer.current_epoch} and unfreezing BERT layer's parameters for training")
                for p in pl_module.bert.parameters():
                    p.requires_grad = True
                
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=model.hparams["gpus"],
        log_gpu_memory="min_max",
        auto_select_gpus=model.hparams["auto_select_gpus"],
        accumulate_grad_batches=model.hparams["accumulate_grad_batches"],
        log_every_n_steps=model.hparams["log_every_n_steps"],
        max_epochs=model.hparams["max_epochs"],
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            ToggleBertTraining(),
        ],
    )

    trainer.fit(model, train_dl, valid_dl)

if __name__ == "__main__":
    main()

    