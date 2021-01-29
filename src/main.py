import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import transformers
from transformers import AdamW , get_linear_schedule_with_warmup
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from quest_dataset import get_dataloaders
from CONFIG import path_dict, config_dict

global path_dict

class Model(pl.LightningModule):
    def __init__(self, path_dict=path_dict, config_dict=config_dict):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(path_dict["BERT_PATH"])
        self.bert_dropout = nn.Dropout(config_dict['bert_dropout'])
        self.linear = nn.Linear(768,30)

        self.save_hyperparameters(config_dict)
        if self.hparams['freeze_bert']:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.lr = self.hparams['lr']

    @staticmethod
    def loss(logits, targets):
        return nn.BCEWithLogitsLoss()(logits, targets)

    def forward(self, ids_seq, attn_masks,token_type_ids):

        bert_out = self.bert(ids_seq, attention_mask=attn_masks, token_type_ids=token_type_ids)
        #using maxpooled output
        max_out = self.bert_dropout(bert_out[1])
        return self.linear(max_out)

    def shared_step(self, batch):
        ids_seq, attn_masks, token_type_ids, target = batch['ids_seq'], batch['attn_masks'], batch['token_type_ids'], batch['target']
        logits = self(ids_seq, attn_masks, token_type_ids)
        loss = self.loss(logits, target)
        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self.shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss = self.shared_step(batch)
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {"valid_loss":loss ,"logits":logits}

    def configure_optimizers(self):
        return AdamW(self.parameters(), self.lr)

    def get_dummy_inputs(self):
        return (torch.zeros((8,512), dtype=torch.long, device=self.device),
               torch.zeros((8,512), dtype=torch.long, device=self.device),
               torch.zeros((8,512), dtype=torch.long, device=self.device))

    def validation_epoch_end(self, validation_step_outputs):  # args are defined as part of pl API
        if self.hparams['save_model']:
            dummy_inputs = self.get_dummy_inputs()
            torch.onnx.export(self, dummy_inputs, path_dict['MODEL_PATH'], input_names=['seq','attn_masks','token_type_ids'],dynamic_axes={'seq':{0:'sequence'},'attn_masks':{0:'sequence'},'token_type_ids':{0:'sequence'}}, opset_version=11)
            if self.hparams["wandb_save_model"]:
                wandb.save(path_dict['MODEL_FILENAME'])

        if self.hparams["wandb_log_logits"]:
            flattened_logits = torch.sigmoid(torch.cat(validation_step_outputs['logits']))
            self.logger.experiment.log(
                {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": self.global_step},
                commit=False
                )

if __name__ == "__main__":

    model = Model(config_dict=config_dict)
    train_dl, valid_dl = get_dataloaders(config_dict=model.hparams)

    wandb_logger = WandbLogger(project=model.hparams["wandbProjectName"])
    
    early_stop_callback = EarlyStopping(
            monitor='valid_loss',
            min_delta=0.00,
            patience=5,
            verbose=False,
            mode='min'
        )
    
    trainer = pl.Trainer(
            logger=wandb_logger,
            gpus=model.hparams["gpus"],
            log_gpu_memory = 'min_max',
            auto_select_gpus=model.hparams["auto_select_gpus"], 
            accumulate_grad_batches=model.hparams['accumulate_grad_batches'],
            log_every_n_steps = model.hparams["log_every_n_steps"],
            max_epochs=model.hparams["max_epochs"],
            callbacks=[early_stop_callback]
        )

    trainer.fit(model, train_dl, valid_dl)

