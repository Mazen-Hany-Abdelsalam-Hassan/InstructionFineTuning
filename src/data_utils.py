import os

import pandas as pd
from typing import List

from config import (DATA_PATH ,train_df_dir , val_df_dir ,test_df_dir,
                    COLUMNS_NAME , SEED )

def split_data(df:pd.DataFrame,  train_split: float = .8
               , val_split: float = .10, seed = SEED , col_name:List[str] = COLUMNS_NAME):

    assert set(col_name).issubset(set(df.columns)) , "KeyError Instruction ,Input , Response not in table"
    df = df[col_name]
    train_split_value = int(len(df) * train_split)
    val_split_value = int(len(df) *  (train_split + val_split)  )
    train = df.iloc[:train_split_value]
    validation = df.iloc[train_split_value: val_split_value]
    test = df.iloc[val_split_value]

    os.makedirs(DATA_PATH , exist_ok=True)
    train.to_csv(  train_df_dir ,index= False )
    validation.to_csv(train_df_dir, index=False)
    test.to_csv(train_df_dir, index=False)