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
    submission_df.loc[:,data.target_cols] = y_pred 
    
    if not args.output_filename.endswith(".csv"):
        args.output_filename = f"{args.output_filename}.csv"
    
    submission_df.to_csv(
        f"output/{args.output_filename}",
        index=False, 
    )
    print(f"predictions saved in file {args.output_filename}")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ckpt_folder", default="models",type=str)
    parser.add_argument("output_filename", default="none",type=str, help="regex pattern or filename")
    args = parser.parse_args()
    
    data = QuestData(args)
    model = QuestModel(args)
    
    test_dataloader = data.test_dataloader()
    
    for checkpoint in [f for f in os.listdir(args.ckpt_folder) if f.endswith(".ckpt")]:
        args.output_filename = f"{checkpoint[:-5]}.csv"
        if not os.path.exists(args.output_filename): 
            model.load_from_checkpoint(checkpoint)
            predict_and_save(args, model, test_dataloader)
        else:
            print(f"prediction csv file for {checkpoint} already exists")