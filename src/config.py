import os
import tiktoken

COLUMNS_NAME = ["instruction" , "input" , "response"]
PARENT_DIR= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PARENT_DIR ,'data')
train_df_dir = os.path.join(DATA_PATH, "train.csv")
val_df_dir = os.path.join(DATA_PATH, "val.csv")
test_df_dir = os.path.join(DATA_PATH, "test.csv")
TOKENIZER = tiktoken.encoding_for_model('gpt-2')
PADDINGTEXT = "<|endoftext|>"
PADDINGTOKEN = TOKENIZER.encode(PADDINGTEXT,allowed_special = 'all')[0]
BASE_MODELS_DIR = os.path.join(PARENT_DIR, 'BASEModel')
SEED = 1234