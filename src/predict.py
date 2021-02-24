from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np

import torch
import pytorch_lightning as pl

from quest_dataset import QuestData
from model import QuestModel


def predict_and_save(args, model, dataloader):
    outputs = []
    for batch in dataloader:
        ids_seq, attn_masks, token_type_ids, target = (
            batch["ids_seq"],
            batch["attn_masks"],
            batch["token_type_ids"],
            batch["target"],
        )

        logits = model.forward(ids_seq, attn_masks, token_type_ids)
        outputs.append(logits)

    y_pred = torch.sigmoid(torch.cat(outputs)).to("cpu").detach().numpy()

    submission_df = pd.read_csv("../output/sample_submission.csv")
    submission_df.loc[:, data.target_cols] = y_pred

    if not args.output_filename.endswith(".csv"):
        args.output_filename = f"{args.output_filename}.csv"

    submission_df.to_csv(
        f"output/{args.output_filename}",
        index=False,
    )
    print(f"predictions saved in file {args.output_filename}")


if __name__ == "__main__":
    pl.seed_everything(420)

    parser = ArgumentParser()
    parser.add_argument("--ckpt_folder", default="models", type=str)
    parser.add_argument(
        "--output_filename", default="none", type=str, help="regex pattern or filename"
    )
    parser.add_argument("--fold", default=0, type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--gpus",
        default=1,
        help="if value is 0 cpu will be used, if string then that gpu device will be used",
    )
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
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--accumulate_grad_batches", default=2, type=int)
    parser.add_argument("--model_name", default="quest", type=str)
    args = parser.parse_args()

    data = QuestData(args)
    model = QuestModel(args)

    test_dataloader = data.test_dataloader()

    for checkpoint in [f for f in os.listdir(args.ckpt_folder) if f.endswith(".ckpt")]:
        args.output_filename = f"{checkpoint[:-5]}.csv"
        if not os.path.exists(f"outputs/{args.output_filename}"):
            print(f"Loading model from checkpoint: {args.output_filename}")
            model.load_from_checkpoint(f"models/{checkpoint}")
            predict_and_save(args, model, test_dataloader)
        else:
            print(f"prediction csv file for {checkpoint} already exists")
