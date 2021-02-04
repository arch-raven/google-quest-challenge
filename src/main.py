import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

from quest_dataset import get_dataloaders
from CONFIG import path_dict, config_dict
from scipy.stats import spearmanr

global path_dict


class Model(pl.LightningModule):
    def __init__(self, path_dict=path_dict, config_dict=config_dict, **kwargs):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(path_dict["BERT_PATH"])
        self.bert_dropout = nn.Dropout(config_dict["bert_dropout"])
        self.linear = nn.Linear(768, 30)

        self.save_hyperparameters(config_dict, kwargs)
        self.toggle_freeze_bert_to(self.hparams["freeze_bert"])
        self.lr = self.hparams["lr"]

    def toggle_freeze_bert_to(self, freeze_bert):
        for p in self.bert.parameters():
            p.requires_grad = freeze_bert

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
        return AdamW(self.parameters(), self.lr)

    def get_dummy_inputs(self):
        return (
            torch.zeros((8, 512), dtype=torch.long, device=self.device),
            torch.zeros((8, 512), dtype=torch.long, device=self.device),
            torch.zeros((8, 512), dtype=torch.long, device=self.device),
        )

    def validation_epoch_end(
        self, validation_step_outputs
    ):  # args are defined as part of pl API
        if self.hparams["save_model_bin"]:
            torch.save(model.state_dict(), self.hparams["MODEL_FILENAME"] + ".bin")

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
                torch.cat(validation_step_outputs["logits"])
            )
            self.logger.experiment.log(
                {
                    "valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
                    "global_step": self.global_step,
                },
                commit=False,
            )
        y_pred = (
            torch.sigmoid(torch.cat(validation_step_outputs["logits"]))
            .to("cpu")
            .detach()
            .numpy()
        )
        y_true = (
            torch.cat(validation_step_outputs["true_preds"]).to("cpu").detach().numpy()
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


if __name__ == "__main__":
    FOLD = 0

    model = Model(config_dict=config_dict, fold=FOLD)
    train_dl, valid_dl = get_dataloaders(fold=FOLD, config_dict=model.hparams)

    wandb_logger = WandbLogger(project=model.hparams["wandbProjectName"])

    early_stop_callback = EarlyStopping(
        monitor="valid_loss", min_delta=0.00, patience=3, verbose=False, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="quest-{epoch:02d}-{FOLD}",
    )

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
        ],
    )

    trainer.fit(model, train_dl, valid_dl)
