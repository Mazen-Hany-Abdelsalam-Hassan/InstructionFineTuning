import os
import pandas as pd
from typing import List
import  torch
import copy
from torch.utils.data import Dataset , DataLoader
from config import (DATA_PATH ,train_df_dir , val_df_dir ,test_df_dir,
                    COLUMNS_NAME , SEED , PADDINGTOKEN ,SYS_prompt , TOKENIZER ,MAX_LENGTH)

def split_data(df:pd.DataFrame,  train_split: float = .8
               , val_split: float = .10, seed = SEED , col_name:List[str] = COLUMNS_NAME):

    assert set(col_name).issubset(set(df.columns)) , "KeyError Instruction ,Input , Response not in table"
    df = df[col_name]
    df = df.sample(frac=1,ignore_index=True , random_state=seed)
    train_split_value = int(len(df) * train_split)
    val_split_value = int(len(df) *  (train_split + val_split)  )
    train = df.iloc[:train_split_value]
    validation = df.iloc[train_split_value: val_split_value]
    test = df.iloc[val_split_value:]
    os.makedirs(DATA_PATH , exist_ok=True)
    train.to_csv(  train_df_dir ,index= False )
    validation.to_csv(val_df_dir, index=False)
    test.to_csv(test_df_dir, index=False)

def Apply_Prompt_template(x , template , Max_length):
    Instruction =x['instruction']
    Input = x['input']
    Response =x['response']
    Full_prompt =  template.format(Instruction = Instruction , Input = Input , Response = Response)
    Full_prompt_tokenized =  TOKENIZER.encode(Full_prompt)[:Max_length]
    Full_prompt_tokenized.append(PADDINGTOKEN)
    return Full_prompt_tokenized


class InstructionFineTuning(Dataset):
    def __init__(self, df_dir: pd.DataFrame, Max_lenght: int = 550):
        super().__init__()
        df = pd.read_csv(df_dir)
        self.X = df.apply(lambda x: Apply_Prompt_template(x, template=SYS_prompt, Max_length=Max_lenght), axis=1)
        self.X = list(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        return x

    def __len__(self):
        return len(self.X)


def collate_function(data_sample):
    batch = copy.deepcopy(data_sample)
    max_seq =  max(len(x) for x in batch)
    X , Y =[] ,[]
    for single_input in batch:
        seq2pad = max_seq - len(single_input)
        x_padding = [PADDINGTOKEN ]* (seq2pad)
        y_padding =   (seq2pad+1) * [-100]
        x = single_input [0:]
        y = single_input[1:]
        x.extend(x_padding)
        y.extend(y_padding)
        X.append(x)
        Y.append(y)
    return torch.tensor(X) , torch.tensor(Y)


def create_loaders(df_dir :str,
                  Max_length =MAX_LENGTH,
                  train_split: float = .8
               , val_split: float = .10,
                  seed = SEED ,
                  col_name:List[str] = COLUMNS_NAME ,batch_size:int = 4):
    df = pd.read_csv(df_dir)

    split_data(df ,train_split=train_split ,val_split=val_split,
               seed = seed ,col_name=col_name)
    train = InstructionFineTuning(train_df_dir ,Max_lenght=Max_length)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_function)
    validation = InstructionFineTuning(val_df_dir , Max_lenght=Max_length)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_function)
    test = InstructionFineTuning(test_df_dir ,Max_lenght=Max_length)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_function)
    return  train_loader, val_loader , test_loader