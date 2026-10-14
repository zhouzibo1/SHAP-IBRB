"""Bounded train-only epoch/weight-decay selection for same7 neural baselines."""
from pathlib import Path
import sys,json,time
from datetime import datetime,timezone,timedelta
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(HERE),str(ROOT),str(HERE/'依赖'),str(HERE/'基线依赖')]
import numpy as np
import pandas as pd
import torch
from torch import nn
from 原数据1000深度基线 import DNN,CNN,predict,restore
from NLST训练内可行性评估 import score,write_json
EPOCHS=[50,100];DECAY=[0.,.001]

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

def run_case(ref,out,source):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    data=pd.read_csv(ROOT/'Lung Cancer Dataset.csv');X=data.iloc[:,:-1].to_numpy(float);y=data.iloc[:,-1].to_numpy(int);name=f'repeat{ref["repeat"]}_fold{ref["fold"]}';folder=out/name;folder.mkdir();records=[];entries=[];choices=[]
    for f in range(1,4):
        md=source/name/f'inner{f}';info=json.loads((md/'成员.json').read_text(encoding='utf-8'));tr=np.array(info['training_rows'])-1;va=np.array(info['validation_rows'])-1;cols=np.array(info['columns']);write_json(folder/f'inner{f}_成员.json',info)
        for family in ['DNN','CNN']:
            for decay in DECAY:
                d=folder/f'inner{f}'/f'{family}_decay{decay}';fit(X[tr][:,cols],y[tr],family,ref['seed']+900+f-1,decay,EPOCHS,d)
                for epoch in EPOCHS:
                    model,mean,sd,_=restore(d/f'epoch{epoch}');p=predict(model,X[va][:,cols],mean,sd);pd.DataFrame(dict(source_row_1based=va+1,y_true=y[va],score=p)).to_csv(d/f'epoch{epoch}'/'内层预测.csv',index=False);records.append(dict(repeat=ref['repeat'],fold=ref['fold'],inner_fold=f,family=family,decay=decay,epoch=epoch,mse=float(np.mean((p-y[va])**2)),model_path=str(d/f'epoch{epoch}'),columns=json.dumps(cols.tolist())))
    table=pd.DataFrame(records);tr=np.array(ref['training_rows'])-1;cols=np.array(ref['columns'])
    for family in ['DNN','CNN']:
        means=table[table.family.eq(family)].groupby(['decay','epoch']).mse.mean();grid=[(d,e) for d in DECAY for e in EPOCHS];decay,epoch=min(grid,key=lambda c:means.loc[c]);d=folder/f'{family}_final';fit(X[tr][:,cols],y[tr],family,ref['seed'],decay,[epoch],d)
        entries.append(dict(repeat=ref['repeat'],fold=ref['fold'],family=family,model=f'{family}1000_same7_inner',model_path=str(d/f'epoch{epoch}'),columns=cols.tolist(),training_rows=ref['training_rows'],test_rows=ref['test_rows'],epoch=epoch,decay=decay));choices.append(dict(repeat=ref['repeat'],fold=ref['fold'],family=family,selected_decay=decay,selected_epoch=epoch,means=[dict(decay=c[0],epoch=c[1],mse=float(means.loc[c])) for c in grid]))
    table.to_csv(folder/'内层搜索.csv',index=False);write_json(folder/'训练内选择.json',choices);print('FITTED '+name,flush=True);return entries,records,choices

def main():
    if len(sys.argv)>1:
        out=Path(sys.argv[1]);evaluate(out);return
    source=Path((HERE/'最新原数据1000基线公平选参目录.txt').read_text(encoding='utf-8').strip());plan=json.loads((source/'运行前固定方案.json').read_text(encoding='utf-8'));reference=Path(plan['reference']);refs=sorted(json.loads((reference/'模型清单.json').read_text(encoding='utf-8')),key=lambda e:(e['repeat'],e['fold']))
    out=HERE/('原数据1000深度基线内层验证_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'));out.mkdir();(HERE/'最新深度基线内层验证目录.txt').write_text(str(out),encoding='utf-8')
    write_json(out/'运行前固定方案.json',dict(source=str(source),reference=str(reference),architecture='unchanged DNN64/32/16 and original two-convolution CNN',inputs='same training-only SHAP7',optimizer='Adam lr.001 batch32',epochs=EPOCHS,weight_decay=DECAY,selection='same inner3 folds,unweighted mean raw Brier;grid order breaks ties',SHAP='reuse30 independently checked inner refits and their members',outer_training_n=1000,outer_evaluation_n=1000,expected_trajectories=120,expected_inner_checkpoints=240,expected_final_models=20,lock='all models and choices before outer scoring',scope='development CV;old fixed protocols remain sensitivity conditions',device='CPU,one torch thread per worker'))
    entries=[];records=[];choices=[]
    with ProcessPoolExecutor(max_workers=3,max_tasks_per_child=1) as pool:
        futures=[pool.submit(run_case,ref,out,source) for ref in refs]
        for future in as_completed(futures):
            e,r,c=future.result();entries+=e;records+=r;choices+=c;write_json(out/'模型清单.json',entries);write_json(out/'全部训练内选择.json',choices);pd.DataFrame(records).to_csv(out/'全部内层搜索.csv',index=False)
    write_json(out/'全部评价前锁定.json',dict(all_ready=True,models=20,inner_checkpoint_conditions=240));evaluate(out)

def evaluate(out):
    torch.set_num_threads(1);assert json.loads((out/'全部评价前锁定.json').read_text(encoding='utf-8'))['all_ready'];data=pd.read_csv(ROOT/'Lung Cancer Dataset.csv');X=data.iloc[:,:-1].to_numpy(float);y=data.iloc[:,-1].to_numpy(int);rows=[]
    for e in json.loads((out/'模型清单.json').read_text(encoding='utf-8')):
        folder=Path(e['model_path']);model,mean,sd,_=restore(folder)
        for subset,field in [('train','training_rows'),('holdout','test_rows')]:
            idx=np.array(e[field])-1;p=predict(model,X[idx][:,e['columns']],mean,sd);pd.DataFrame(dict(source_row_1based=idx+1,y_true=y[idx],score=p)).to_csv(folder/f'{subset}_预测.csv',index=False);rows.append(dict(repeat=e['repeat'],fold=e['fold'],model=e['model'],subset=subset,**score(y[idx],p)))
    pd.DataFrame(rows).to_csv(out/'全部指标.csv',index=False);write_json(out/'完成记录.json',dict(completed=True,final_models=20,inner_trajectories=120,inner_checkpoint_conditions=240));print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
