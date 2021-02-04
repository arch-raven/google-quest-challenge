import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split
from CONFIG import path_dict, config_dict


class qDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        qtitle,
        qbody,
        answer,
        target=None,
        path_dict=path_dict,
        config_dict=config_dict,
    ):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        if target is None:
            self.target = [0] * len(self.qtitle)
        else:
            self.target = target
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            path_dict["TOKENIZER_PATH"]
        )
        self.maxlen = config_dict["maxlen"]

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


def return_dl(df, split, target_cols, config_dict=config_dict):
    qtitle = df.loc[:, "question_title"].values
    qbody = df.loc[:, "question_body"].values
    answer = df.loc[:, "answer"].values
    target = df.loc[:, target_cols].values

    ds = qDataset(qtitle, qbody, answer, target)

    return torch.utils.data.DataLoader(
        ds,
        batch_size=config_dict["batch_size"],
        shuffle=split == "train",
        num_workers=8,
        drop_last=True,
    )


def get_dataloaders(
    fold, target_cols=target_columns, path_dict=path_dict, config_dict=config_dict
):
    data = pd.read_csv(path_dict["TRAIN_PATH"]).fillna("none")

    df_train = data.loc[data["fold"] != fold]
    df_valid = data.loc[data["fold"] == fold]

    return (
        return_dl(df_train, "train", target_cols=target_cols, config_dict=config_dict),
        return_dl(df_valid, "valid", target_cols=target_cols, config_dict=config_dict),
    )


target_columns = [
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

if __name__ == "__main__":
    train_dl, valid_dl = get_dataloaders(0, target_columns)
