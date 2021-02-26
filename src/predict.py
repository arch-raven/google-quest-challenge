from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np

import torch
import pytorch_lightning as pl

from quest_dataset import QuestData
from model import QuestModel


if __name__ == "__main__":
    pl.seed_everything(420)

    parser = ArgumentParser()

    parser.add_argument("path_to_ckpt", type=str, help="path to checkpoint file")
    parser.add_argument("--ckpt_folder", default="models", type=str)
    parser.add_argument(
        "--output_filename", default="none", type=str, help="regex pattern or filename"
    )
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

    args = parser.parse_args()

    data = QuestData(args)
    pl_model = QuestModel(args)

    pl_model.model.load_state_dict(torch.load(args.path_to_ckpt))

    trainer.test(pl_model, data.test_dataloader())
