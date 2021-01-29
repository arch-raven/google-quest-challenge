import pandas as pd 
import torch
import transformers
from sklearn.model_selection import train_test_split
from CONFIG import path_dict, config_dict

class qDataset(torch.utils.data.Dataset):
    def __init__(self, qtitle, qbody, answer, target=None, path_dict=path_dict, config_dict=config_dict):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        if target is None:
            self.target = [0]*len(self.qtitle)
        else:
            self.target = target
        self.tokenizer = transformers.BertTokenizer.from_pretrained(path_dict["TOKENIZER_PATH"])
        self.maxlen = config_dict['maxlen']
        
    def __len__(self):
        return len(self.qtitle)
    
    def __getitem__(self, idx):
        # print("idx: ", idx)
        qtitle = self.qtitle[idx]
        qbody = self.qbody[idx]
        answer = self.answer[idx]    

        inputs = self.tokenizer(
            " ".join(qtitle.split()) + ' ' + " ".join(qbody.split()),
            " ".join(answer.split()),
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.maxlen,
        )
        return {
            'ids_seq' : torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attn_masks' : torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids' :torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'target' : torch.tensor(self.target[idx], dtype=torch.float)
        }
    
def return_dl(df, split, config_dict=config_dict):
    qtitle = df.loc[:,'question_title'].values
    qbody = df.loc[:,'question_body'].values
    answer = df.loc[:,'answer'].values
    target = df.iloc[:,10:].values

    ds = qDataset(qtitle, qbody, answer, target)

    return torch.utils.data.DataLoader(
        ds,
        batch_size=config_dict["batch_size"],
        shuffle=split=="train",
        num_workers=8,
        drop_last=True,
    )

def get_dataloaders(path_dict=path_dict, config_dict=config_dict):
    data = pd.read_csv(path_dict["TRAIN_PATH"], index_col='qa_id')
    df_train, df_valid = train_test_split(data, random_state=42, test_size=0.1)
    return return_dl(df_train, "train", config_dict=config_dict), return_dl(df_valid, "valid",config_dict=config_dict)
        
if __name__ == "__main__":
    train_dl ,valid_dl = get_dataloaders()    