#import pandas as pd
#import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import dask.dataframe as dd
import re
from dask.distributed import Client
import joblib

from sklearn.ensemble import RandomForestClassifier

ro.r['load']('5v_cleandf.rdata')
df = ro.r['df']

with localconverter(ro.default_converter + pandas2ri.converter):
    pd_df = ro.conversion.rpy2py(df)

#pd_df = ro.conversion.rpy2py(df)
#%%
#print(pd_df.describe())
dask_df = dd.from_pandas(pd_df, npartitions = 8)
# ############################
#     The pred col should be the disposition
# ######

suffixes = ("_max", "_min", "_mean", "_count", "_last", "_median", "_npos")
prefixes = ("meds_","ecodes", "cc_", "triage_", "dep_")

cols_to_drop = [
    col for col in dask_df.columns
    if col.endswith(suffixes) or col.startswith(prefixes)
]
#print(cols_to_drop)
dask_df = dask_df.drop(cols_to_drop,axis = 1)
# print(dask_df.describe())

print(list(dask_df.columns))
print(len(dask_df.columns))

print(dask_df['disposition'].unique())