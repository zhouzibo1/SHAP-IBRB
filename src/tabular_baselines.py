from pathlib import Path
import json
import numpy as np
import pandas as pd
def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
import lightgbm as lgb
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from upstream_core import standardize_fit

def lr_fit(X, y, penalty):
    mu, sd, z = standardize_fit(X)
    def loss(w):
        logits = w[0] + z@w[1:]
        value = np.logaddexp(0, logits).sum()-y@logits + .5*penalty*(w[1:]@w[1:])
        residual = expit(logits)-y
        return value, np.r_[residual.sum(), z.T@residual+penalty*w[1:]]
    result = minimize(loss, np.zeros(z.shape[1]+1), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    assert result.success, result.message
    return result.x, mu, sd

def lr_predict(model, X):
    w, mu, sd = model
    return expit(w[0]+((X-mu)/sd)@w[1:])

def gb_fit(X, y, seed, leaves, reg):
    mu, sd, z = standardize_fit(X)
    base = lgb.train({'objective': 'binary', 'learning_rate': .05, 'max_depth': 6,
                     'num_leaves': leaves, 'min_data_in_leaf': 20, 'lambda_l2': reg,
                     'seed': seed, 'num_threads': 2, 'verbosity': -1,
                     'deterministic': True, 'force_col_wise': True},
                     lgb.Dataset(z, label=y), num_boost_round=200)
    return base, mu, sd

def gb_predict(model, X):
    base, mu, sd = model
    return base.predict((X-mu)/sd)

def make_model(family,candidate,seed):
    if family=='DT':return DecisionTreeClassifier(max_depth=candidate[0],criterion=candidate[1],random_state=seed)
    if family=='SVM':
        base=make_pipeline(StandardScaler(),SVC(kernel='linear',C=candidate,random_state=seed))
        return CalibratedClassifierCV(base,method='sigmoid',
            cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed),ensemble=False,n_jobs=1)
    if family=='XGB':return XGBClassifier(n_estimators=candidate,max_depth=6,learning_rate=.05,
        objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=2,random_state=seed)
    return ExplainableBoostingClassifier(n_jobs=1,random_state=seed)

def fit_logistic(X, y, ridge=1.):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.
    z = (X - mu) / sd
    def objective(w):
        logits = w[0] + z @ w[1:]
        residual = expit(logits) - y
        value = np.logaddexp(0., logits).sum() - y @ logits + .5 * ridge * (w[1:] @ w[1:])
        return value, np.r_[residual.sum(), z.T @ residual + ridge * w[1:]]
    fitted = minimize(objective, np.zeros(X.shape[1] + 1), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    return fitted.x, mu, sd

def tree_fit(X,y,candidate):
    params=dict(objective='binary',learning_rate=.05,min_data_in_leaf=20,lambda_l2=1.,seed=SEEDS[0],num_threads=2,verbosity=-1,deterministic=True,force_col_wise=True);params.update(TREE_GRID[candidate])
    return lgb.train(params,lgb.Dataset(X,label=y),num_boost_round=200)
SEEDS=[20261003,20261004,20261005]
TREE_GRID=[dict(max_depth=3,num_leaves=7),dict(max_depth=6,num_leaves=31)]
