# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%



# %%
# !pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html --user
# !pip install tqdm pandas sklearn spacy nltk transformers emojis --ignore-installed --user
# # !pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# !pip install tensorflow==2.3.0 --user
# !pip install pytorch-lightning wandb --user
# !pip install --upgrade --user gensim spacy
# !python -m spacy download en_core_web_sm --user
# !pip install Cython
# !pip install pytorch-lightning wandb --user


# %%
# from google.colab import drive
# drive.mount('/content/drive')

# # import os
# # os.chdir('/content/drive')


# %%
# !pip install -qqq pytorch_lightning wandb torchmetrics --upgrade
# !pip install -qqq kaggle --upgrade


# %%
# !pip install pandas --upgrade


# %%
# ! mkdir ~/.kaggle
# ! cp '/content/drive/My Drive/kaggle/kaggle.json' ~/.kaggle/
# ! chmod 600 ~/.kaggle/kaggle.json


# %%
# !kaggle competitions download -p /mnt/maths/mt601/kaggle/ventilator/ ventilator-pressure-prediction


# %%
# !unzip /mnt/maths/mt601/kaggle/ventilator/ventilator-pressure-prediction.zip -d /mnt/maths/mt601/kaggle/ventilator/data/


# %%
import numpy as np
import pandas as pd
import gc
import os

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW, lr_scheduler, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, _LRScheduler
import math
import torch.nn.functional as F
from copy import deepcopy
import pickle
import torchmetrics
from scipy.stats import iqr

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize, LabelEncoder
from sklearn.model_selection import train_test_split, GroupKFold, KFold
import wandb
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
wandb.login(key='6fa032c9de89fb7104cc7828743a6ad1ece62906')

from IPython.display import display
pl.seed_everything(2021)


# %%
import os
os.chdir('/mnt/maths/mt601/kaggle/ventilator/')


# %%
DATA_PATH = "/mnt/maths/mt601/kaggle/ventilator/data/"

sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')
# df_train = pd.read_csv(DATA_PATH + 'train.csv')
# df_test = pd.read_csv(DATA_PATH + 'test.csv')

# df_train.sort_values(['breath_id','time_step'],inplace=True)
# df_test.sort_values(['breath_id','time_step'],inplace=True)
# df = df_train[df_train['breath_id'] < 5].reset_index(drop=True)
kf = KFold(n_splits=10,random_state=2021, shuffle=True)

df_train=pd.read_feather(DATA_PATH + 'df_train.ftr')
df_test=pd.read_feather(DATA_PATH + 'df_test.ftr')

# %% [markdown]
# # Dataset & Dataloader

# %%

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        if col!='open_channels':
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def add_features(df):
    df['cross']= df['u_in'] * df['u_out']
    df['cross2']= df['time_step'] * df['u_out']
    df['area'] = df['time_step'] * df['u_in']
    df['area1'] = df.groupby('breath_id')['area'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    print("Step-1...Completed")
    
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df['u_in_lag5'] = df.groupby('breath_id')['u_in'].shift(5)
    df['u_out_lag5'] = df.groupby('breath_id')['u_out'].shift(5)
    df['u_in_lag_back5'] = df.groupby('breath_id')['u_in'].shift(-5)
    df['u_out_lag_back5'] = df.groupby('breath_id')['u_out'].shift(-5)
    df['u_in_lag6'] = df.groupby('breath_id')['u_in'].shift(6)
    df['u_out_lag6'] = df.groupby('breath_id')['u_out'].shift(6)
    df['u_in_lag_back6'] = df.groupby('breath_id')['u_in'].shift(-6)
    df['u_out_lag_back6'] = df.groupby('breath_id')['u_out'].shift(-6)
    df['u_in_lag7'] = df.groupby('breath_id')['u_in'].shift(7)
    df['u_out_lag7'] = df.groupby('breath_id')['u_out'].shift(7)
    df['u_in_lag_back7'] = df.groupby('breath_id')['u_in'].shift(-7)
    df['u_out_lag_back7'] = df.groupby('breath_id')['u_out'].shift(-7)
    df['u_in_lag8'] = df.groupby('breath_id')['u_in'].shift(8)
    df['u_out_lag8'] = df.groupby('breath_id')['u_out'].shift(8)
    df['u_in_lag_back8'] = df.groupby('breath_id')['u_in'].shift(-8)
    df['u_out_lag_back8'] = df.groupby('breath_id')['u_out'].shift(-8)
    df['u_in_lag9'] = df.groupby('breath_id')['u_in'].shift(9)
    df['u_out_lag9'] = df.groupby('breath_id')['u_out'].shift(9)
    df['u_in_lag_back9'] = df.groupby('breath_id')['u_in'].shift(-9)
    df['u_out_lag_back9'] = df.groupby('breath_id')['u_out'].shift(-9)
    df['u_in_lag10'] = df.groupby('breath_id')['u_in'].shift(10)
    df['u_out_lag10'] = df.groupby('breath_id')['u_out'].shift(10)
    df['u_in_lag_back10'] = df.groupby('breath_id')['u_in'].shift(-10)
    df['u_out_lag_back10'] = df.groupby('breath_id')['u_out'].shift(-10)
    df = df.fillna(0)
    print("Step-2...Completed")
    
    df['breath_id_u_in_max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id_u_in_min'] = df.groupby(['breath_id'])['u_in'].transform('min')
    df['breath_id_u_in_median'] = df.groupby(['breath_id'])['u_in'].transform('median')
    df['breath_id_u_out_median'] = df.groupby(['breath_id'])['u_out'].transform('median')
    df['breath_id_u_in_mean'] = df.groupby(['breath_id'])['u_in'].transform('mean')
    df['breath_id_u_out_mean'] = df.groupby(['breath_id'])['u_out'].transform('mean')
    df['breath_id_u_in_IQR'] = df.groupby(['breath_id'])['u_in'].transform(iqr)
    df['breath_id_u_out_IQR'] = df.groupby(['breath_id'])['u_out'].transform(iqr)
    df['breath_id_u_in_diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id_u_in_diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    df['breath_id_u_in_diffmin'] = df.groupby(['breath_id'])['u_in'].transform('min') - df['u_in']
    df['breath_id_u_in_diffmedian'] = df.groupby(['breath_id'])['u_in'].transform('median') - df['u_in']

    df['ewm_u_in_mean_3'] = df.groupby('breath_id')['u_in'].ewm(halflife=3).mean().reset_index(level=0,drop=True)
    df['ewm_u_in_var_3'] = df.groupby('breath_id')['u_in'].ewm(halflife=3).var().reset_index(level=0,drop=True)
    df['ewm_u_in_std_3'] = df.groupby('breath_id')['u_in'].ewm(halflife=3).std().reset_index(level=0,drop=True)
    df['ewm_u_in_corr_3'] = df.groupby('breath_id')['u_in'].ewm(halflife=3).corr().reset_index(level=0,drop=True)
    df['ewm_u_in_cov_3'] = df.groupby('breath_id')['u_in'].ewm(halflife=3).cov().reset_index(level=0,drop=True)
    df['ewm_u_in_mean_5'] = df.groupby('breath_id')['u_in'].ewm(halflife=5).mean().reset_index(level=0,drop=True)
    df['ewm_u_in_var_5'] = df.groupby('breath_id')['u_in'].ewm(halflife=5).var().reset_index(level=0,drop=True)
    df['ewm_u_in_std_5'] = df.groupby('breath_id')['u_in'].ewm(halflife=5).std().reset_index(level=0,drop=True)
    df['ewm_u_in_corr_5'] = df.groupby('breath_id')['u_in'].ewm(halflife=5).corr().reset_index(level=0,drop=True)
    df['ewm_u_in_cov_5'] = df.groupby('breath_id')['u_in'].ewm(halflife=5).cov().reset_index(level=0,drop=True)
    df['ewm_u_in_mean_7'] = df.groupby('breath_id')['u_in'].ewm(halflife=7).mean().reset_index(level=0,drop=True)
    df['ewm_u_in_var_7'] = df.groupby('breath_id')['u_in'].ewm(halflife=7).var().reset_index(level=0,drop=True)
    df['ewm_u_in_std_7'] = df.groupby('breath_id')['u_in'].ewm(halflife=7).std().reset_index(level=0,drop=True)
    df['ewm_u_in_corr_7'] = df.groupby('breath_id')['u_in'].ewm(halflife=7).corr().reset_index(level=0,drop=True)
    df['ewm_u_in_cov_7'] = df.groupby('breath_id')['u_in'].ewm(halflife=7).cov().reset_index(level=0,drop=True)
    df['ewm_u_in_mean_9'] = df.groupby('breath_id')['u_in'].ewm(halflife=9).mean().reset_index(level=0,drop=True)
    df['ewm_u_in_var_9'] = df.groupby('breath_id')['u_in'].ewm(halflife=9).var().reset_index(level=0,drop=True)
    df['ewm_u_in_std_9'] = df.groupby('breath_id')['u_in'].ewm(halflife=9).std().reset_index(level=0,drop=True)
    df['ewm_u_in_corr_9'] = df.groupby('breath_id')['u_in'].ewm(halflife=9).corr().reset_index(level=0,drop=True)
    df['ewm_u_in_cov_9'] = df.groupby('breath_id')['u_in'].ewm(halflife=9).cov().reset_index(level=0,drop=True)
    
    df['rolling_3_mean'] = df.groupby('breath_id')['u_in'].rolling(window=3, min_periods=1).mean().reset_index(level=0,drop=True)
    df['rolling_3_max'] = df.groupby('breath_id')['u_in'].rolling(window=3, min_periods=1).max().reset_index(level=0,drop=True)
    df['rolling_3_min'] = df.groupby('breath_id')['u_in'].rolling(window=3, min_periods=1).min().reset_index(level=0,drop=True)
    df['rolling_3_std'] = df.groupby('breath_id')['u_in'].rolling(window=3, min_periods=1).std().reset_index(level=0,drop=True)
    df['rolling_3_median'] = df.groupby('breath_id')['u_in'].rolling(window=3, min_periods=1).median().reset_index(level=0,drop=True)
    df['rolling_5_mean'] = df.groupby('breath_id')['u_in'].rolling(window=5, min_periods=1).mean().reset_index(level=0,drop=True)
    df['rolling_5_max'] = df.groupby('breath_id')['u_in'].rolling(window=5, min_periods=1).max().reset_index(level=0,drop=True)
    df['rolling_5_min'] = df.groupby('breath_id')['u_in'].rolling(window=5, min_periods=1).min().reset_index(level=0,drop=True)
    df['rolling_5_std'] = df.groupby('breath_id')['u_in'].rolling(window=5, min_periods=1).std().reset_index(level=0,drop=True)
    df['rolling_5_median'] = df.groupby('breath_id')['u_in'].rolling(window=5, min_periods=1).median().reset_index(level=0,drop=True)
    df['rolling_7_mean'] = df.groupby('breath_id')['u_in'].rolling(window=7, min_periods=1).mean().reset_index(level=0,drop=True)
    df['rolling_7_max'] = df.groupby('breath_id')['u_in'].rolling(window=7, min_periods=1).max().reset_index(level=0,drop=True)
    df['rolling_7_min'] = df.groupby('breath_id')['u_in'].rolling(window=7, min_periods=1).min().reset_index(level=0,drop=True)
    df['rolling_7_std'] = df.groupby('breath_id')['u_in'].rolling(window=7, min_periods=1).std().reset_index(level=0,drop=True)
    df['rolling_7_median'] = df.groupby('breath_id')['u_in'].rolling(window=7, min_periods=1).median().reset_index(level=0,drop=True)
    df['rolling_9_mean'] = df.groupby('breath_id')['u_in'].rolling(window=9, min_periods=1).mean().reset_index(level=0,drop=True)
    df['rolling_9_max'] = df.groupby('breath_id')['u_in'].rolling(window=9, min_periods=1).max().reset_index(level=0,drop=True)
    df['rolling_9_min'] = df.groupby('breath_id')['u_in'].rolling(window=9, min_periods=1).min().reset_index(level=0,drop=True)
    df['rolling_9_std'] = df.groupby('breath_id')['u_in'].rolling(window=9, min_periods=1).std().reset_index(level=0,drop=True)
    df['rolling_9_median'] = df.groupby('breath_id')['u_in'].rolling(window=9, min_periods=1).median().reset_index(level=0,drop=True)

    df['expand_mean_3'] = df.groupby('breath_id')['u_in'].expanding(3).mean().reset_index(level=0,drop=True)
    df['expand_max_3'] = df.groupby('breath_id')['u_in'].expanding(3).max().reset_index(level=0,drop=True)
    df['expand_min_3'] = df.groupby('breath_id')['u_in'].expanding(3).min().reset_index(level=0,drop=True)
    df['expand_std_3'] = df.groupby('breath_id')['u_in'].expanding(3).std().reset_index(level=0,drop=True)
    df['expand_median_3'] = df.groupby('breath_id')['u_in'].expanding(3).median().reset_index(level=0,drop=True)
    df['expand_mean_5'] = df.groupby('breath_id')['u_in'].expanding(5).mean().reset_index(level=0,drop=True)
    df['expand_max_5'] = df.groupby('breath_id')['u_in'].expanding(5).max().reset_index(level=0,drop=True)
    df['expand_min_5'] = df.groupby('breath_id')['u_in'].expanding(5).min().reset_index(level=0,drop=True)
    df['expand_std_5'] = df.groupby('breath_id')['u_in'].expanding(5).std().reset_index(level=0,drop=True)
    df['expand_median_5'] = df.groupby('breath_id')['u_in'].expanding(5).median().reset_index(level=0,drop=True)
    df['expand_mean_7'] = df.groupby('breath_id')['u_in'].expanding(7).mean().reset_index(level=0,drop=True)
    df['expand_max_7'] = df.groupby('breath_id')['u_in'].expanding(7).max().reset_index(level=0,drop=True)
    df['expand_min_7'] = df.groupby('breath_id')['u_in'].expanding(7).min().reset_index(level=0,drop=True)
    df['expand_std_7'] = df.groupby('breath_id')['u_in'].expanding(7).std().reset_index(level=0,drop=True)
    df['expand_median_7'] = df.groupby('breath_id')['u_in'].expanding(7).median().reset_index(level=0,drop=True)
    df['expand_mean_1'] = df.groupby('breath_id')['u_in'].expanding(9).mean().reset_index(level=0,drop=True)
    df['expand_max_1'] = df.groupby('breath_id')['u_in'].expanding(9).max().reset_index(level=0,drop=True)
    df['expand_min_1'] = df.groupby('breath_id')['u_in'].expanding(9).min().reset_index(level=0,drop=True)
    df['expand_std_1'] = df.groupby('breath_id')['u_in'].expanding(9).std().reset_index(level=0,drop=True)
    df['expand_median_1'] = df.groupby('breath_id')['u_in'].expanding(9).median().reset_index(level=0,drop=True)
    df = df.fillna(0)
    print("Step-3...Completed")
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    df['u_in_diff5'] = df['u_in'] - df['u_in_lag5']
    df['u_out_diff5'] = df['u_out'] - df['u_out_lag5']
    df['u_in_diff6'] = df['u_in'] - df['u_in_lag6']
    df['u_out_diff6'] = df['u_out'] - df['u_out_lag6']
    df['u_in_diff7'] = df['u_in'] - df['u_in_lag7']
    df['u_out_diff7'] = df['u_out'] - df['u_out_lag7']
    df['u_in_diff8'] = df['u_in'] - df['u_in_lag8']
    df['u_out_diff8'] = df['u_out'] - df['u_out_lag8']
    df['u_in_diff9'] = df['u_in'] - df['u_in_lag9']
    df['u_out_diff9'] = df['u_out'] - df['u_out_lag9']
    df['u_in_diff10'] = df['u_in'] - df['u_in_lag10']
    df['u_out_diff10'] = df['u_out'] - df['u_out_lag10']
    print("Step-4...Completed")
    
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] =df['u_in_cumsum'] /df['count']
    
    df['u_in_1st_derivative'] = (df['u_in'].diff().fillna(0) / df['time_step'].diff().fillna(0)).fillna(0)
    
    df['time_diff'] = df['time_step']-df['time_step'].shift(1).fillna(0)
    df['power'] = df['time_diff']*df['u_in']
    df['power_cumsum'] = df.groupby(['breath_id'])['power'].cumsum()
    
    df['u_in_partition_out_sum'] = df.groupby(['breath_id',"u_out"])['u_in'].transform("sum")
    
    df = reduce_mem_usage(df)
    
    # u_in_half is time:0 - time point of u_out:1 rise (almost 1.0s)
    df['tmp'] = df['u_out']*(-1)+1 # inversion of u_out
    df['u_in_half'] = df['tmp'] * df['u_in']
    
    u_in_half_mean_dict = df.groupby('breath_id')['u_in_half'].mean().to_dict()
    df['u_in_half_mean'] = df['breath_id'].map(u_in_half_mean_dict)
    
    u_in_half_std_dict = df.groupby('breath_id')['u_in_half'].std().to_dict()
    df['u_in_half_std'] = df['breath_id'].map(u_in_half_std_dict)
    del u_in_half_std_dict
    
    del u_in_half_mean_dict, df['tmp']    
    del df['power']
    
    # All entries are first point of each breath_id
    first_df = df.loc[0::80,:]
    # All entries are first point of each breath_id
    last_df = df.loc[79::80,:]
    
    # u_in: first point, last point
    u_in_first_dict = dict(zip(first_df['breath_id'], first_df['u_in']))
    df['u_in_first'] = df['breath_id'].map(u_in_first_dict)
    del u_in_first_dict
    #u_in_last_dict = dict(zip(first_df['breath_id'], last_df['u_in']))
    #df['u_in_last'] = df['breath_id'].map(u_in_last_dict)
    #del u_in_last_dict
    
    # time(sec) of end point
    time_end_dict = dict(zip(last_df['breath_id'], last_df['time_step']))     
    df['time_end'] = df['breath_id'].map(time_end_dict)
    del time_end_dict
    del last_df
    
    df['u_out_diff'] = df['u_out'].diff()
    df['u_out_diff'].fillna(0, inplace=True)
    df['u_out_diff'].replace(-1, 0, inplace=True)
    uout1_df = df[df['u_out_diff']==1]
    
    # Register Area when u_out becomes 1
    #uout1_area_dict = dict(zip(first_df['breath_id'], first_df['u_in']))
    #df['area_uout1'] = df['breath_id'].map(uout1_area_dict)
    #del uout1_area_dict
    
    # time(sec) when u_out becomes 1
    uout1_dict = dict(zip(uout1_df['breath_id'], uout1_df['time_step']))
    df['time_uout1'] = df['breath_id'].map(uout1_dict)
    del uout1_dict, df['u_out_diff']
    
    # Dict that puts 0 at the beginning of the 80row cycle
    first_0_dict = dict(zip(first_df['id'], [0]*len(uout1_df)))

    del first_df
    del uout1_df   
    
    gc.collect()
    
    # Faster version u_in_diff creation, faster than groupby
    df['u_in_diff'] = df['u_in'].diff()
    df['tmp'] = df['id'].map(first_0_dict) # put 0, the 80row cycle
    df.iloc[0::80, df.columns.get_loc('u_in_diff')] = df.iloc[0::80, df.columns.get_loc('tmp')]

    # Create u_in vibration
    df['diff_sign'] = np.sign(df['u_in_diff'])
    df['sign_diff'] = df['diff_sign'].diff()
    df['tmp'] = df['id'].map(first_0_dict) # put 0, the 80row cycle
    df.iloc[0::80, df.columns.get_loc('sign_diff')] = df.iloc[0::80, df.columns.get_loc('tmp')]
    del first_0_dict
    
    # Count the number of inversions, so take the absolute value and sum
    df['sign_diff'] = abs(df['sign_diff']) 
    sign_diff_dict = df.groupby('breath_id')['sign_diff'].sum().to_dict()
    df['diff_vib'] = df['breath_id'].map(sign_diff_dict)
    
    del sign_diff_dict
    
    #if 'diff_sign' in df.columns:
    #df.drop(['diff_sign', 'sign_diff'], axis=1, inplace=True)
    #if 'tmp' in df.columns:
    #    df.drop(['tmp'], axis=1, inplace=True)
    
    del df['sign_diff'], df['diff_sign'], df['tmp']
    
    gc.collect()
    
    df = reduce_mem_usage(df)
    
    
    df['1/C_']=(1/df['C']).astype(str)
    df['R/C_']=(df['R']/df['C']).astype(str)
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R_C'] = df["R"].astype(str) + '_' + df["C"].astype(str)
    df = pd.get_dummies(df)
    print("Step-5...Completed")
    
    return df

def prepare_data(df):
    df=df.drop('id',axis=1)
    unique_breath=df['breath_id'].nunique()
    pressures = torch.tensor(df['pressure'].values.reshape(unique_breath,-1,1),dtype=torch.float32)
    u_outs = torch.tensor(df['u_out'].values.reshape(unique_breath,-1,1),dtype=torch.int64)
    features = torch.tensor(df.drop(['breath_id','pressure'],axis=1).values.reshape(unique_breath,-1,len(df.columns)-2),dtype=torch.float32)

    return {
        'pressure':pressures,
        'u_out':u_outs,
        'feature':features
    }


# %%
# df_train=add_features(df_train)
# df_test=add_features(df_test)
# from tqdm.notebook import tqdm
# feature_cols=[i for i in df_train.columns if i not in ['id','breath_id','u_out','pressure','fold'] and not ('R_' in i or 'R_' in i)]
# for col in tqdm(feature_cols):
#     scaler=RobustScaler()
#     df_train[[col]]=scaler.fit_transform(df_train[[col]])
#     df_test[[col]]=scaler.transform(df_test[[col]])
# df_train.to_feather(DATA_PATH+'df_train.ftr')
# df_test.to_feather(DATA_PATH+'df_test.ftr')


# %%
drop_cols1=['u_out_diff1','u_out_diff2','u_out_diff3','u_out_diff4','u_out_diff5','u_out_diff6','u_out_diff7','u_out_diff8','u_out_diff9','u_out_diff10','one']
drop_cols2=['breath_id_u_out_median','breath_id_u_out_IQR']
# drop_cols3=['ewm_u_in_corr_3','ewm_u_in_corr_5','ewm_u_in_corr_7','ewm_u_in_corr_9']
df_train.drop(drop_cols1+drop_cols2,axis=1,inplace=True)
df_test.drop(drop_cols1+drop_cols2,axis=1,inplace=True)
input_dim=df_train.shape[1]-3

min_pressure=df_train.pressure.min()
max_pressure=df_train.pressure.max()

# %%
class VentilatorDataset(Dataset):
    def __init__(self, df):
        if "pressure" not in df.columns:
            df['pressure'] = 0
        self.inputs=prepare_data(df)
                
    def __len__(self):
        return self.inputs['pressure'].shape[0]

    def __getitem__(self, idx):
        return {k:v[idx] for k,v in self.inputs.items()}

# %% [markdown]
# # Modelling

# %%
class VentilatorLoss(torchmetrics.Metric):
    """
    Directly optimizes the competition metric
    """
    def __init__(self):
        super().__init__()
        self.add_state("sum_absolute_error", torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state("n_observation", torch.tensor(0), dist_reduce_fx='sum')
    def update(self, preds, y, u_out):
        w = 1 - u_out
        ae = w * torch.abs(y - preds)
        self.sum_absolute_error += torch.sum(ae)
        self.n_observation += torch.sum(w)
        print(torch.sum(ae),torch.sum(w),self.sum_absolute_error,self.n_observation)
    def compute(self):
        return self.sum_absolute_error / self.n_observation

class NNLoss(nn.Module):
    """
    Loss for the neural network
    """
    def __init__(self):
        super().__init__()
    def __call__(self, preds, y, u_out):
        w = 1 - u_out
        # mae = w * (y - preds).pow(2)
        mae = w * (y - preds).abs()
        mae = mae.sum() / w.sum()

        return mae

class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.min_pressure=min_pressure
        self.max_pressure=max_pressure
    def __call__(self,pred):
        return pred * (self.max_pressure - self.min_pressure) + self.min_pressure



# %%


class VentilatorLSTM(pl.LightningModule):
    def __init__(
        self,
        CONFIG
    ):
        super().__init__()
        for k,v in CONFIG.items():
            globals()[k]=v

        self.CONFIG=CONFIG
        self.tr_inp_feature=tr_inp_feature
        self.model_type = 'LSTM'
        
        self.tr_inp_feature=tr_inp_feature
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, tr_inp_feature),
            nn.SELU(),
            nn.LayerNorm(tr_inp_feature)
        )

        self.lstm1 = nn.LSTM(tr_inp_feature, dense_dim,
                            dropout=0.1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(dense_dim*2, dense_dim//2,
                            dropout=0.1, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(dense_dim//2*2, dense_dim//4,
                            dropout=0.1, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(dense_dim//4*2, dense_dim//8,
                            dropout=0.1, batch_first=True, bidirectional=True)

        if scaled_head:
            self.reg_head = nn.Sequential(
                nn.LayerNorm(dense_dim//4),
                nn.SELU(),
                nn.Linear(dense_dim//4, 1),
                nn.Sigmoid(),
                ScalingLayer()
            )
        else:
            self.reg_head = nn.Sequential(
                nn.LayerNorm(dense_dim//4),
                nn.SELU(),
                nn.Linear(dense_dim//4, 1)
            )

        # self.lstm1 = nn.LSTM(tr_inp_feature, dense_dim, num_layers=4,
        #                     dropout=0.0, batch_first=True, bidirectional=True)
        # self.reg_head = nn.Sequential(
        #     nn.LayerNorm(dense_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(dense_dim*2, 1)
        # )

        for n, m in self.named_modules():
            if isinstance(m, nn.LSTM):
                print(f'init {m}')
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.GRU):
                print(f"init {m}")
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
        
        self.loss_fn=NNLoss()
        self.lr_scheduler=lr_scheduler

    def forward(self, train_batch): 
        features=train_batch['feature']
        
        features = self.mlp(features)

        # LSTM
        features,_ = self.lstm1(features)
        features, _ = self.lstm2(features)
        features,_ = self.lstm3(features)
        features, _ = self.lstm4(features)

        pred = self.reg_head(features)

        return pred
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.CONFIG['lr'], weight_decay=self.CONFIG['weight_decay'])
        scheduler = CosineAnnealingWarmRestarts(optimizer, self.CONFIG['T_MAX'], eta_min=self.CONFIG['min_lr'],verbose=0, T_mult=self.CONFIG['T_MULT'])
        # scheduler = CosineAnnealingWarmRestarts(optimizer, self.CONFIG['T_MAX'], eta_min=self.CONFIG['min_lr'],verbose=0)
        return {
            "optimizer":optimizer,
            "lr_scheduler" : scheduler
        }

    def compute_loss(self,preds, y, u_out):
        return self.loss_fn(preds, y, u_out)

    def training_step(self,train_batch,batch_idx):
        y=train_batch['pressure']
        u_out=train_batch['u_out']
        
        preds=self.forward(train_batch)
        loss=self.compute_loss(preds, y, u_out)
        self.log('train_loss',loss,on_epoch=True,on_step=True)
        return loss

    def validation_step(self,val_batch,batch_idx):
        y=val_batch['pressure']
        u_out=val_batch['u_out']
        
        preds=self.forward(val_batch)
        loss=self.compute_loss(preds, y, u_out)
        self.log('val_loss',loss,on_epoch=True,on_step=False)


# %%
# CONFIG_transformer={
#     'input_dim':50,
#     'tr_inp_feature':1024,
#     'tr_n_head':1,
#     'tr_n_hidden':512, 
#     'tr_n_layers':4, 
#     'tr_dropout':0.1,
#     'lstm_dim':512,
#     'lstm_layers':2,
#     'lstm_dropout':0.1,
#     'logit_dim':512,
#     'lr_scheduler':torch.optim.lr_scheduler.StepLR,
#     'max_epochs':1000,
#     'batch_size':64,
#     'lr':1e-5,
#     'weight_decay':0
# }

CONFIG_LSTM={
    'input_dim':input_dim,
    'tr_inp_feature':512,
    'dense_dim':512,
    'lr_scheduler':'CosineAnnealingWarmRestart',
    'max_epochs':1000,
    'batch_size':8192,
    'lr':1e-3,
    'weight_decay':1e-4,
    'min_lr':0,
    'T_MAX':1000,
    'T_MULT':1,
    'scaled_head':True
}


# %%
preds=[]
CONFIG=CONFIG_LSTM
group_name='LSTM'
job_type='3LSTM 512 512 scale head SELU'
tags=[
    'cosine_lr_1000',
    'SELU'
]

result=[]
test_ds=VentilatorDataset(df_test)
train_ds=VentilatorDataset(df_train)
test_data_loader = DataLoader(test_ds,batch_size=CONFIG['batch_size'],shuffle=False,num_workers=4)

for k, (train_index, val_index) in enumerate(kf.split(train_ds)):
    train_data_loader = DataLoader(Subset(train_ds, train_index.tolist()),batch_size=CONFIG['batch_size'],shuffle=True,num_workers=4)
    val_data_loader = DataLoader(Subset(train_ds, val_index.tolist()),batch_size=CONFIG['batch_size'],shuffle=False,num_workers=4)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss",mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss',mode='min',patience=50)
    # model=VentilatorTransformer(CONFIG_transformer)
    model=VentilatorLSTM(CONFIG)
    wandb_logger = WandbLogger(project='ventilator',config=CONFIG,group=group_name,job_type=job_type,name=f'fold_{k+1}_'+'_'.join(tags),log_model=True,tags=tags)
    wandb_logger.watch(model,'all',5)
    trainer = pl.Trainer(accelerator='dp',precision=16,max_epochs=CONFIG['max_epochs'],gpus=-1, auto_select_gpus=True,callbacks=[checkpoint_callback,early_stop_callback],logger=wandb_logger,log_every_n_steps=5)
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
    
    pred=trainer.predict(dataloaders=test_data_loader)
    preds.append(torch.cat(pred,axis=0).flatten().detach().cpu().unsqueeze(1))
    
    wandb.finish()


# %%
# artifacts=[
#            'minhhieutran2112/ventilator/model-3w59z7m8:v0',
#            'minhhieutran2112/ventilator/model-184k1ewx:v0',
#            'minhhieutran2112/ventilator/model-ag1ipaee:v0',
#            'minhhieutran2112/ventilator/model-14nbs7pk:v0',
#            'minhhieutran2112/ventilator/model-1z4v7gdv:v0'
# ]
# run = wandb.init()
# preds1=[]
# CONFIG=CONFIG_LSTM

# loss=[]
# for k in range(5):
#     if k in [1,3]:
#         continue
#     artifact = run.use_artifact(artifacts[k], type='model')
#     artifact_dir = artifact.download()
#     model=VentilatorLSTM(CONFIG)
#     model=model.load_from_checkpoint('artifacts/'+artifacts[k].split('/')[-1]+'/model.ckpt',CONFIG=CONFIG)

#     tmp_test_ds=deepcopy(test_ds)
#     # tmp_test_ds=Subset(train_ds, val_index.tolist()).dataset
#     scaler=pickle.load(open(f'/content/drive/MyDrive/kaggle/ventilator/scaler/{job_type}/fold_{k+1}_scaler.sav', 'rb'))
#     tmp_test_ds.inputs['feature']=torch.tensor(scaler.transform(tmp_test_ds.inputs['feature'].reshape(-1,64)).reshape(-1,80,64),dtype=torch.float32)

#     # model=VentilatorTransformer(CONFIG_transformer)
#     pred=[]
#     trainer=pl.Trainer(gpus=-1)

#     # for batch in DataLoader(tmp_test_ds,batch_size=1024,shuffle=False):
#     #     batch={k:v.to('cuda') for k,v in batch.items()}
#     #     y=batch['pressure']
#     #     feature=batch['feature']
#     #     u_out=batch['u_out']
#     #     preds=model.forward(batch)
#     #     tmp_loss=model.compute_loss(preds, y, u_out)
#     #     loss.append(tmp_loss.detach().cpu())

#     pred=trainer.predict(model,DataLoader(tmp_test_ds,batch_size=1024,shuffle=False))
#     preds1.append(torch.cat(pred,axis=0).flatten().cpu().tolist())
#     del tmp_test_ds, model
#     gc.collect()


# %%
# preds_agg=np.nanmean(np.vstack(preds1),axis=0).tolist()
# df_test['pressure']=preds_agg
# sub=df_test[sub.columns]
# sub.to_csv('/content/drive/MyDrive/kaggle/ventilator/submission_mean.csv', index=False)

# preds_agg=np.nanmedian(np.vstack(preds1),axis=0).tolist()
# df_test['pressure']=preds_agg
# sub=df_test[sub.columns]
# sub.to_csv('/content/drive/MyDrive/kaggle/ventilator/submission_median.csv', index=False)


# %%
preds_agg=np.nanmean(torch.cat(preds,axis=1),axis=1).tolist()
df_test['pressure']=preds_agg
sub=df_test[sub.columns]
sub.to_csv(DATA_PATH + 'submission_mean.csv', index=False)


# %%
preds_agg=np.nanmedian(torch.cat(preds,axis=1),axis=1).tolist()
df_test['pressure']=preds_agg
sub=df_test[sub.columns]
sub.to_csv(DATA_PATH + 'submission_median.csv', index=False)


# %%
# !kaggle competitions submit ventilator-pressure-prediction -f '/content/drive/MyDrive/kaggle/ventilator/submission_mean.csv' -m '4LSTM_1mlp_1reghead_cosine_lr_max20_mult1 mean'


