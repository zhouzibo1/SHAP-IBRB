"""Manuscript-scale DNN and1D-CNN, fixed100epochs, paired training-only standardization."""
from pathlib import Path
import sys,json,time
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
for path in [ROOT,HERE,HERE/'依赖',HERE/'基线依赖']:sys.path.insert(0,str(path))
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata
from NLST训练内可行性评估 import score,write_json

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

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    prior=Path((HERE/'最新原数据1000交叉验证目录.txt').read_text(encoding='utf-8').strip())
    baseline=Path((HERE/'最新原数据1000四项基线目录.txt').read_text(encoding='utf-8').strip())
    out=HERE/('原数据1000深度基线_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    out.mkdir();(HERE/'最新原数据1000深度基线目录.txt').write_text(str(out),encoding='utf-8')
    source=pd.read_csv(ROOT/'Lung Cancer Dataset.csv');X,y=source.iloc[:,:-1].to_numpy(float),source.iloc[:,-1].to_numpy(int)
    refs=[e for e in json.loads((prior/'模型清单.json').read_text(encoding='utf-8')) if e['family']=='soft']
    write_json(out/'运行前固定方案.json',dict(reference=str(prior),existing_baselines=str(baseline),training_n=1000,
        inputs=['same7 in original column order','all17 in original column order'],outer_folds=10,
        manuscript_DNN='hidden64-32-16 ReLU,Adam,100epochs',manuscript_CNN='two1D convolution layers with maxpool',
        explicit_CNN='channels16/32,kernel3,padding1,ReLU,maxpool2 after eachconv,dense16 output1',
        optimizer='Adam lr.001,beta.9/.999,eps1e-8,weight_decay0,batch32,100epochs',
        loss='binary crossentropy with logits; raw sigmoid scores; threshold.5',
        standardization='mean/std only on each1000 trainingpool',
        selection='fixed architectures/settings,epoch100; no hyperparametersearch or external earlystop',
        versions=dict(torch=torch.__version__),cpu_threads=2,deterministic=True,
        scope='development baseline extension; manuscript unspecified CNN details now explicitly supplied',
        lock='save all40 models before any outer scores',automatic_adoption=False,NLST_test_scored_n=0))
    entries=[]
    for ref in refs:
        tr=np.array(ref['training_rows'])-1;te=np.array(ref['test_rows'])-1
        assert len(tr)==1000 and len(te)==1000 and not np.intersect1d(tr,te).size
        for suffix,cols in [('same7',np.array(ref['columns'])),('all17',np.arange(X.shape[1]))]:
            mean=X[tr][:,cols].mean(0);sd=X[tr][:,cols].std(0);sd[sd==0]=1
            values=torch.from_numpy(((X[tr][:,cols]-mean)/sd).astype(np.float32));labels=torch.from_numpy(y[tr].astype(np.float32))
            for family,constructor in [('DNN',DNN),('CNN',CNN)]:
                start=time.perf_counter();torch.manual_seed(ref['seed'])
                model=constructor(len(cols));opt=torch.optim.Adam(model.parameters(),lr=.001)
                loss_fn=nn.BCEWithLogitsLoss();generator=torch.Generator().manual_seed(ref['seed']+1700)
                trace=[]
                print(f'START repeat{ref["repeat"]} fold{ref["fold"]} {family}_{suffix}',flush=True)
                for epoch in range(1,101):
                    model.train();permutation=torch.randperm(len(tr),generator=generator)
                    for batch in permutation.split(32):
                        opt.zero_grad(set_to_none=True);loss=loss_fn(model(values[batch]),labels[batch]);loss.backward();opt.step()
                    model.eval()
                    with torch.no_grad():
                        logits=model(values);p=torch.sigmoid(logits).numpy().astype(np.float64)
                        trace.append(dict(epoch=epoch,training_BCE=float(loss_fn(logits,labels)),training_mse=float(np.mean((p-y[tr])**2))))
                folder=out/f'repeat{ref["repeat"]}_fold{ref["fold"]}'/f'{family}_{suffix}';folder.mkdir(parents=True)
                torch.save(model.state_dict(),folder/'模型.pt');np.savez(folder/'标准化.npz',mean=mean,sd=sd)
                cfg=dict(family=family,columns=cols.tolist(),seed=ref['seed'],shuffle_seed=ref['seed']+1700,
                    epochs=100,batch_size=32,learning_rate=.001,weight_decay=0,checkpoint='epoch100',training_rows=ref['training_rows'],test_rows=ref['test_rows'])
                write_json(folder/'配置.json',cfg);pd.DataFrame(trace).to_csv(folder/'训练轨迹.csv',index=False)
                entries.append(dict(repeat=ref['repeat'],fold=ref['fold'],model=f'{family}1000_{suffix}',
                    family=family,columns=cols.tolist(),model_path=str(folder),fit_seconds=time.perf_counter()-start,
                    training_rows=ref['training_rows'],test_rows=ref['test_rows']))
                write_json(out/'模型清单.json',entries)
                print(f'SAVED repeat{ref["repeat"]} fold{ref["fold"]} {family}_{suffix} seconds{time.perf_counter()-start:.1f}',flush=True)
    write_json(out/'全部评价前锁定.json',dict(all_models_ready=True,model_count=40))
    rows=[]
    for e in entries:
        folder=Path(e['model_path']);model,mean,sd,cfg=restore(folder);cols=np.array(e['columns'])
        tr=np.array(e['training_rows'])-1
        np.testing.assert_allclose(mean,X[tr][:,cols].mean(0),atol=1e-12)
        np.testing.assert_allclose(sd,np.where(X[tr][:,cols].std(0)==0,1,X[tr][:,cols].std(0)),atol=1e-12)
        for role,field in [('train','training_rows'),('holdout','test_rows')]:
            idx=np.array(e[field])-1;p=predict(model,X[idx][:,cols],mean,sd)
            path=folder/f'{role}_预测.csv';pd.DataFrame(dict(source_row_1based=idx+1,y_true=y[idx],score=p)).to_csv(path,index=False)
            read=pd.read_csv(path,float_precision='round_trip');np.testing.assert_allclose(read.score,p,atol=1e-12)
            metric=score(y[idx],p);pos=y[idx].sum();neg=len(idx)-pos
            auc=(rankdata(read.score)[y[idx]==1].sum()-pos*(pos+1)/2)/(pos*neg)
            np.testing.assert_allclose([auc,np.mean((read.score-y[idx])**2)],[metric['auc'],metric['mse']],atol=1e-12)
            if role=='train':
                trace=pd.read_csv(folder/'训练轨迹.csv');assert len(trace)==100
                np.testing.assert_allclose(trace.training_mse.iloc[-1],metric['mse'],atol=1e-12)
            rows.append(dict(repeat=e['repeat'],fold=e['fold'],model=e['model'],subset=role,**metric))
    result=pd.DataFrame(rows);result.to_csv(out/'全部指标.csv',index=False)
    old=pd.read_csv(baseline/'含既有基线全部指标.csv')
    combined=pd.concat([old,result],ignore_index=True);combined.to_csv(out/'全部基线指标.csv',index=False)
    hold=combined[combined.subset.eq('holdout')]
    hold.groupby('model')[['accuracy','auc','mse','recall']].agg(['mean','std']).to_csv(out/'全部基线十折汇总.csv')
    write_json(out/'独立交付核验.json',dict(passed=True,new_models=40,prediction_files=80,train_only_standardization=True,
        epoch100_trace_and_predictions_verified=True,raw_mse_and_rank_auc_verified=True,NLST_test_scored_n=0))
    write_json(out/'完成记录.json',dict(completed=True,new_models=40,combined_models_per_outer_fold=18,automatic_adoption=False))
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
