from pathlib import Path
import json
import numpy as np
import pandas as pd
def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
import torch
from torch import nn

class DNN(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),
                                  nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x):return self.layers(x).squeeze(1)

class CNN(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool1d(2),
                               nn.Conv1d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool1d(2))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(32*(n//4),16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x):return self.head(self.conv(x[:,None,:])).squeeze(1)

def predict(model,X,mean,sd):
    model.eval()
    z=torch.from_numpy(((X-mean)/sd).astype(np.float32))
    with torch.no_grad():return torch.sigmoid(model(z)).numpy().astype(np.float64)

def restore(folder):
    cfg=json.loads((folder/'配置.json').read_text(encoding='utf-8'))
    model=(DNN if cfg['family']=='DNN' else CNN)(len(cfg['columns']))
    model.load_state_dict(torch.load(folder/'模型.pt',map_location='cpu',weights_only=True))
    standard=np.load(folder/'标准化.npz')
    return model,standard['mean'],standard['sd'],cfg

def fit(X,y,family,seed,decay,epochs,folder):
    folder.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);model=(DNN if family=='DNN' else CNN)(X.shape[1]);mean=X.mean(0);sd=X.std(0);sd[sd==0]=1
    values=torch.from_numpy(((X-mean)/sd).astype(np.float32));labels=torch.from_numpy(y.astype(np.float32));optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=decay);loss=nn.BCEWithLogitsLoss();g=torch.Generator().manual_seed(seed+1700);trace=[]
    for epoch in range(1,max(epochs)+1):
        model.train()
        for batch in torch.randperm(len(y),generator=g).split(32):
            optimizer.zero_grad(set_to_none=True);v=loss(model(values[batch]),labels[batch]);v.backward();optimizer.step()
        model.eval()
        with torch.no_grad():p=torch.sigmoid(model(values)).numpy().astype(float)
        trace.append(dict(epoch=epoch,training_mse=float(np.mean((p-y)**2))))
        if epoch in epochs:
            d=folder/f'epoch{epoch}';d.mkdir();torch.save(model.state_dict(),d/'模型.pt');np.savez(d/'标准化.npz',mean=mean,sd=sd);write_json(d/'配置.json',dict(family=family,columns=list(range(X.shape[1])),seed=seed,epochs=epoch,weight_decay=decay))
    pd.DataFrame(trace).to_csv(folder/'训练轨迹.csv',index=False)