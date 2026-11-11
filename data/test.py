#%%
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask.distributed import Client,wait, LocalCluster
import xgboost 
#%%
import xgboost.dask as dxgb
#%%
import pandas as pd
#%%
from dask_ml.preprocessing import DummyEncoder, LabelEncoder
#%%
#from dask.distributed import Client
#import joblib

ro.r['load']('5v_cleandf.rdata')
df = ro.r['df']

with localconverter(ro.default_converter + pandas2ri.converter):
    pd_df = ro.conversion.rpy2py(df)

pd_df.to_parquet('test_data.parquet', engine="pyarrow", index=False)
dask_df = dd.read_parquet("test_data.parquet")
#pd_df = ro.conversion.rpy2py(df)
#print(pd_df.describe())
#dask_df = dd.from_pandas(pd_df, npartitions=8)
# ############################
#     The pred col should be the disposition
# ######
#%%
cluster = LocalCluster(
    n_workers=3, threads_per_worker=2,
    memory_limit ='2GB'
)
client = Client(cluster)
#%%
m = pd.read_csv("cc_symptom_mapping.csv")
feat = m[m.treatment == "feature"]

group_cols = feat.groupby("group")["raw_column"].agg(list).to_dict()

group_series = {g: dask_df[cols].max(axis=1) for g, cols in group_cols.items()}

X = dask_df[["age", "gender"]].assign(**group_series)
y = dask_df["disposition"]

X,y = client.persist([X,y])
#%%
encoder = DummyEncoder(columns = ["gender"])
X["gender"] = X["gender"].astype(
    pd.CategoricalDtype(
        categories=["Female", "Male"]
    ))
X.categorize(columns=["gender"])
X = encoder.fit_transform(X)
labels = LabelEncoder()
y = labels.fit_transform(y)

X,y = client.persist([X,y])
wait([X,y])
#%%
x_train, test_x, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=42)
#%%
d_train = dxgb.DaskDMatrix(client,x_train,y_train)
#%%
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.1
}

output = dxgb.train(
    client,
    params,
    dtrain=d_train,
    num_boost_round=100,
    #evals=[(x_train, "train"), (test_x, "test")]
)

model = output["booster"]
#%%
y_pred_proba = dxgb.predict( client, model, test_x)
#%%
y_pred_proba = y_pred_proba.compute()
#%%
y_pred_proba.shape
#%%
import numpy as np

y_pred = np.asarray(y_pred_proba).ravel()
y_pred_class = (y_pred >= 0.5).astype(int)
y_true = y_test.compute().to_numpy()

#%%
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

#from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_true, y_pred_class))

print(classification_report(y_true, y_pred_class))
#%%
from sklearn.ensemble import RandomForestClassifier
from dask.distributed import wait

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    max_features="sqrt",
    random_state=42
)

# Train
rf.fit(x_train, y_train)
#%%
y_preds_rf = rf.predict(test_x)
#y_preds_rf = y_preds_rf.compute()
print("Accuracy:", accuracy_score(y_true, y_preds_rf))

print(classification_report(y_true, y_preds_rf))
#%%
import json, joblib, dask, sklearn, pandas, rpy2
from pathlib import Path
#%%
out = Path("risk_model"); out.mkdir(exist_ok=True)
joblib.dump(model, out / "model.joblib" )

(out / "schema.json").write_text(json.dumps({
    "features": list(X.columns),
    "age_range": [18, 110],
    "missing_policy": "median_impute_then_flag",
}, indent=2))

(out / "thresholds.json").write_text(json.dumps({
    "self_care_max": 0.10,        # placeholders: choose from validation data
    "see_doctor_max": 0.35,       # by targeting a sensitivity level, not accuracy
}, indent=2))

(out / "metadata.json").write_text(json.dumps({
    "trained_on": "yale ed admission dataset",
    "target": "admission",
    "sklearn": sklearn.__version__,
    "dask": dask.__version__,
    "lightgbm": xgboost.__version__,
    "auroc": 0.0, "brier": 0.0,   # fill from your held-out site
}, indent=2))
#%%
