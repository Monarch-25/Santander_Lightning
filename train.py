from dataset import santanderDM
from model import  NN_baseline,NN_baseline_improved
import pytorch_lightning as pl

if __name__=='__main__':
    model = NN_baseline_improved(input_size=200,hidden_dim=32)
    dm = santanderDM()

    trainer = pl.Trainer(max_epochs=5,accelerator='gpu',devices=1)
    trainer.fit(model=model,datamodule=dm)