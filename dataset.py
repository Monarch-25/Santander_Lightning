import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import TensorDataset,random_split,DataLoader
import torch
from math import ceil
import pytorch_lightning as pl

train = pd.read_csv('train.csv')
test = pd.read_csv('train.csv')

#creating a train and validation set
X = torch.tensor(train.drop(['ID_code','target'],axis = 1).values).to(torch.float32)
y = torch.tensor(train.target.values).to(torch.float32)

train_ds,val_ds = random_split(TensorDataset(X,y),[int(0.8*len(y)),ceil(0.2*len(y))])

#creating a test dataset
test_inputs = torch.tensor(test.drop(['ID_code'],axis = 1).values)   
test_ids = test['ID_code']


class santanderDM(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def train_dataloader(self)->TRAIN_DATALOADERS:
        return DataLoader(dataset=train_ds,shuffle=True,num_workers=12,batch_size=512)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=val_ds,num_workers=12,batch_size=512)