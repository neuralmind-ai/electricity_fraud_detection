import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import PowerTransformer
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold

def download_data():
    # Download dos dados à partir do repositório de origem
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z01')
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z02')
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.zip')

    # Unzip dos dados ordenados em múltiplos pacotes diferentes
    os.system('cat data.z01 data.z02 data.zip > data_compress.zip')
    os.system('unzip -n -q data_compress')

def get_dataset(filepath):
    """## Salvar a coluna flags para ser concatenada após o processamento"""

    df_raw = pd.read_csv(filepath,index_col=0)
    flags = df_raw.FLAG.copy()

    # Remover a coluna flags 
    df_raw.drop(['FLAG'], axis=1, inplace=True)

    """## Ordenar"""

    #reordenar por data
    df_raw = df_raw.T.copy()
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw.sort_index(inplace=True, axis=0)
    df_raw = df_raw.T.copy()
    df_raw['FLAG'] = flags
    return df_raw


"""# Processamento dataset"""
def get_processed_dataset(filepath):

    df_raw = get_dataset(filepath)
    flags = df_raw['FLAG']
    df_raw.drop(['FLAG'], axis=1, inplace=True)

    # df Original com NAN por 0
    df_zero = df_raw.copy()
    df_zero = df_zero.apply(lambda row: row.fillna(0))

    """## Transformar Yeo - Johson"""

    # transformar com Yeo - Johson
    pt = PowerTransformer(method='yeo-johnson', standardize=False) 
    skl_yeojohnson = pt.fit(df_zero.values)
    lambdas_found  = skl_yeojohnson.lambdas_
    skl_yeojohnson = pt.transform(df_zero.values)
    df_yj = pd.DataFrame(data=skl_yeojohnson, columns=df_zero.columns, index=df_zero.index)

    """## Aplicar Z-score"""

    # aplizar Z-score
    df_zscore = pd.DataFrame(data=zscore(df_yj), columns=df_zero.columns, index=df_zero.index)
    df_zscore['flag'] = flags
    return df_zscore.iloc[:, 5:]

class FraudDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()    
        self.target = y.copy()
        self.x = X.copy()
        
        self.x = torch.from_numpy(self.x).type(torch.FloatTensor)
        self.target = torch.from_numpy(self.target).type(torch.LongTensor)
        self.x = self.x[:,0:1029]
        
    def __len__(self):
        return self.target.shape[0] 
    
    def __getitem__(self, index):
        target = self.target[index]
        x = self.x[index]
        x = x.view(1,-1,7)
        
        return x,target