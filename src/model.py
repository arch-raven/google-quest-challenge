import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

import transformers
from transformers import AdamW, get_cosine_schedule_with_warmup

from scipy.stats import spearmanr


class QuestModel(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.bert_dropout = nn.Dropout(args.bert_dropout)
        self.linear = nn.Linear(768, 30)

        self.save_hyperparameters(args)

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
            "valid_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return {"valid_loss": loss, "logits": logits, "true_preds": batch["target"]}

    def configure_optimizers(self):
        grouped_parameters = [
            {"params": self.bert.parameters(), "lr": self.hparams.bert_lr},
            {"params": self.linear.parameters(), "lr": self.hparams.linear_lr},
        ]
        optim = AdamW(grouped_parameters, lr=self.hparams.bert_lr)
        # 4863 is total number of samples in train split
        num_training_steps = (
            4863 // (self.hparams.batch_size * self.hparams.accumulate_grad_batches)
        ) * self.hparams.max_epochs
        sched = get_cosine_schedule_with_warmup(
            optim, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        return [optim], [sched]

    def validation_epoch_end(self, validation_step_outputs):
        y_pred = (
            torch.sigmoid(torch.cat([out["logits"] for out in validation_step_outputs]))
            .to("cpu")
            .detach()
            .numpy()
        )
        y_true = (
            torch.cat([out["true_preds"] for out in validation_step_outputs])
            .to("cpu")
            .detach()
            .numpy()
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
    pass
