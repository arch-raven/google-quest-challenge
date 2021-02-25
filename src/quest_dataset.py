import pandas as pd
import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import transformers
import pytorch_lightning as pl


ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class qDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        qtitle,
        qbody,
        answer,
        target=None,
    ):
        self.hparams = args
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        if target is None:
            self.target = [0] * len(self.qtitle)
        else:
            self.target = target
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        self.maxlen = self.hparams.maxlen

    def __len__(self):
        return len(self.qtitle)

    def __getitem__(self, idx):
        # print("idx: ", idx)
        qtitle = self.qtitle[idx]
        qbody = self.qbody[idx]
        answer = self.answer[idx]

        inputs = self.tokenizer(
            " ".join(qtitle.split()) + " " + " ".join(qbody.split()),
            " ".join(answer.split()),
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.maxlen,
        )
        return {
            "ids_seq": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attn_masks": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "target": torch.tensor(self.target[idx], dtype=torch.float),
        }


TARGET_COLUMNS = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]


class QuestData(pl.LightningDataModule):
    def __init__(self, args, target_cols=TARGET_COLUMNS):
        super().__init__()
        self.hparams = args
        self.target_cols = target_cols

    def train_dataloader(self):
        df = pd.read_csv(ROOT_DIR / "input" / "train_with_GKF.csv").fillna("none")
        df = df.loc[df["fold"] != self.hparams.fold]

        qtitle = df.loc[:, "question_title"].values
        qbody = df.loc[:, "question_body"].values
        answer = df.loc[:, "answer"].values
        target = df.loc[:, self.target_cols].values

        ds = qDataset(self.hparams, qtitle, qbody, answer, target)

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )

    def val_dataloader(self):
        df = pd.read_csv(ROOT_DIR / "input" / "train_with_GKF.csv").fillna("none")
        df = df.loc[df["fold"] == self.hparams.fold]

        qtitle = df.loc[:, "question_title"].values
        qbody = df.loc[:, "question_body"].values
        answer = df.loc[:, "answer"].values
        target = df.loc[:, self.target_cols].values

        ds = qDataset(self.hparams, qtitle, qbody, answer, target)

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=8,
            drop_last=True,
        )

    def test_dataloader(self):
        df = pd.read_csv(ROOT_DIR / "input" / "test.csv").fillna("none")

        qtitle = df.loc[:, "question_title"].values
        qbody = df.loc[:, "question_body"].values
        answer = df.loc[:, "answer"].values

        ds = qDataset(self.hparams, qtitle, qbody, answer)

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--fold", default=0, type=int, choices=[0, 1, 2, 3, 4])
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
    parser.add_argument("--batch_size", default=4, type=int)
    # parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--gpus",
        default=1,
        help="if value is 0 cpu will be used, if string then that gpu device will be used",
    )
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--accumulate_grad_batches", default=4, type=int)

    args = parser.parse_args()

    args.effective_batch_size = args.batch_size * args.accumulate_grad_batches
    args.log_every_n_steps = args.accumulate_grad_batches * 5

    dm = QuestData(args)
