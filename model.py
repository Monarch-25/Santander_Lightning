from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from torch import optim,nn
from torchmetrics import AUROC
import torch
import torch.nn.functional as F
rand_mat = torch.randn(1,200)

class NN_baseline(pl.LightningModule):
    def __init__(self,input_size) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size,50),
            nn.ReLU(inplace=True),
            nn.Linear(50,1)
        )

        self.loss_fn = nn.BCELoss()
        self.acc = AUROC('binary')

    def forward(self,x):
        x = x.to(torch.float32)
        return F.sigmoid(self.net(x))
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        scores = self.forward(x).squeeze()
        loss = self.loss_fn(scores,y)
        acc = self.acc(scores,y)
        self.log_dict({'train_loss':loss,'auc':acc},prog_bar=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x,y = batch
        scores = self.forward(x).squeeze()
        loss = self.loss_fn(scores,y)
        acc = self.acc(scores,y)
        self.log_dict({'val_step':loss,'auc':acc},prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(),lr = 3e-4)
    
class NN_baseline_improved(pl.LightningModule):
    def __init__(self,input_size,hidden_dim) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(1,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim*input_size*1,1)
        self.loss_fn = nn.BCELoss()
        self.acc = AUROC('binary')

    def forward(self,x):
        x = x.to(torch.float32)
        bacth_size = x.shape[0]
        #curr x.shape -> (batch_size,input_size)
        x = x.view(-1,1) #will basically turn each feature as its own example
        #curr x.shape -> (batch_size*input_size,1)
        x = F.relu(self.fc1(x)).reshape(bacth_size,-1) #(batch_size,hidden_dim*input_size)
        return F.sigmoid(self.fc2(x)).view(-1)
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        scores = self.forward(x).squeeze()
        loss = self.loss_fn(scores,y)
        acc = self.acc(scores,y)
        self.log_dict({'train_loss':loss,'auc':acc},prog_bar=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x,y = batch
        scores = self.forward(x).squeeze()
        loss = self.loss_fn(scores,y)
        acc = self.acc(scores,y)
        self.log_dict({'val_step':loss,'auc':acc},prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(),lr = 3e-4)