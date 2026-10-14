from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import lightgbm as lgb
SEEDS=[20261003,20261004,20261005]
CONTINUOUS=['age','baseline_nodule_count','index_long_diameter_mm','index_perpendicular_diameter_mm']
BINARY=['male','cigsmok']
NOMINAL=['race','index_lobe','index_margin','index_attenuation']
def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def standardize_fit(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd==0] = 1.
    return mu, sd, (X-mu)/sd

def fold_ids(y, n_folds, seed):
    rng = np.random.default_rng(seed)
    ids = np.empty(len(y), dtype=int)
    # Offset the larger class so total fold sizes stay balanced.
    offset = 0
    for c in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == c))
        ids[indices] = (np.arange(len(indices)) + offset) % n_folds
        offset = (offset + len(indices)) % n_folds
    return ids

def shap_fit(X, y, seed, directory):
    directory.mkdir(parents=True, exist_ok=True)
    mu, sd, z = standardize_fit(X)
    ym, ys = float(y.mean()), float(y.std())
    base = lgb.train({'objective': 'quantile', 'alpha': .5, 'seed': seed,
                     'num_threads': 2, 'verbosity': -1, 'deterministic': True,
                     'force_col_wise': True}, lgb.Dataset(z, label=(y-ym)/ys),
                     num_boost_round=200)
    values = base.predict(z, pred_contrib=True)
    np.testing.assert_allclose(values.sum(1), base.predict(z), atol=1e-10, rtol=1e-10)
    importance = np.abs(values[:, :-1]).mean(0)
    columns = np.sort(np.argsort(-importance, kind='stable')[:7])
    (directory/'SHAP模型.txt').write_text(base.model_to_string(), encoding='utf-8')
    np.savez(directory/'标准化.npz', mean=mu, sd=sd, target_mean=ym, target_sd=ys)
    write_json(directory/'排序.json', {'importance': importance.tolist(), 'columns': columns.tolist()})
    return columns

def preprocess_fit(frame):
    impute = {}
    categories = {}
    for column in CONTINUOUS:
        impute[column] = float(frame[column].median())
    for column in BINARY + NOMINAL:
        impute[column] = float(frame[column].mode().iloc[0])
    for column in NOMINAL:
        categories[column] = sorted(frame[column].fillna(impute[column]).unique().tolist())
    return {'impute': impute, 'categories': categories, 'fit_n': len(frame)}

def transform(frame, prep):
    values, names = [], []
    for column in CONTINUOUS + BINARY:
        values.append(frame[column].fillna(prep['impute'][column]).to_numpy(float))
        names.append(column)
    for column in NOMINAL:
        raw = frame[column].fillna(prep['impute'][column]).to_numpy(float)
        for level in prep['categories'][column]:
            values.append((raw == level).astype(float))
            names.append(f'{column}_level_{int(level)}')
    return np.column_stack(values), names

run=SimpleNamespace(preprocess_fit=preprocess_fit,transform=transform,write_json=write_json)

def folds(y,k,seed):
    rng=np.random.default_rng(seed);a=np.empty(len(y),int)
    for c in [0,1]:
        for f,ids in enumerate(np.array_split(rng.permutation(np.flatnonzero(y==c)),k)):a[ids]=f
    return a

def upstream(frame,folder):
    folder.mkdir(parents=True,exist_ok=True);prep=run.preprocess_fit(frame);X,names=run.transform(frame,prep);y=frame.label_365d.to_numpy(int);mu,sd=X.mean(0),X.std(0);sd[sd==0]=1
    params=dict(objective='binary',seed=SEEDS[0],num_threads=2,verbosity=-1,deterministic=True,force_col_wise=True)
    model=lgb.train(params,lgb.Dataset((X-mu)/sd,label=y,feature_name=names),num_boost_round=200)
    contributions=model.predict((X-mu)/sd,pred_contrib=True);np.testing.assert_allclose(contributions.sum(1),model.predict((X-mu)/sd,raw_score=True),atol=1e-9)
    imp=np.abs(contributions[:,:-1]).mean(0);eligible=np.flatnonzero(X.std(0)>0);selected=eligible[np.argsort(-imp[eligible],kind='stable')[:7]];assert len(selected)==7
    np.savez(folder/'SHAP标准化.npz',mean=mu,sd=sd,selected=selected);(folder/'SHAP模型.txt').write_text(model.model_to_string(),encoding='utf-8');pd.DataFrame(dict(encoded_feature=names,mean_absolute_SHAP=imp,selected=np.isin(np.arange(len(names)),selected))).to_csv(folder/'SHAP排序.csv',index=False)
    run.write_json(folder/'上游拟合.json',dict(training_pids=frame.index.tolist(),preprocessing=prep,all_encoded_names=names,selected_columns=selected.tolist(),selected_features=[names[j] for j in selected],params=params))
    return X,y,names,selected,prep
