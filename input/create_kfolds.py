import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

SEED=42
train = pd.read_csv('train.csv').fillna("none")

gkf = GroupKFold(n_splits=5).split(X=train.question_body, groups=train.question_body)

train.loc[:,'fold'] = -1

for i, (tr_idx, val_idx) in enumerate(gkf):
    train.loc[val_idx,'fold'] = i

train.to_csv("train_with_GKF.csv", index=False)