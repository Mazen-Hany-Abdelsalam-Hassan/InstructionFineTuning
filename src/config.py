import os
import tiktoken
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATHS = {
              "S" : "gpt2-small-124M.pth",
              "M" : "gpt2-medium-355M.pth",
              "L" : "gpt2-large-774M.pth",
              "XL": "gpt2-xl-1558M.pth"              }

BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

MODEL_CONFIGS = {
    "S": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "L": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "XL": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


COLUMNS_NAME = ["instruction" , "input" , "response"]
PARENT_DIR= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PARENT_DIR ,'data')
train_df_dir = os.path.join(DATA_PATH, "train.csv")
val_df_dir = os.path.join(DATA_PATH, "val.csv")
test_df_dir = os.path.join(DATA_PATH, "test.csv")
TOKENIZER = tiktoken.encoding_for_model('gpt-2')
MAX_LENGTH = 550
PADDINGTEXT = "<|endoftext|>"
PADDINGTOKEN = TOKENIZER.encode(PADDINGTEXT,allowed_special = 'all')[0]



INSTRUCTION_DIR = os.path.join(PARENT_DIR ,"INSTRUCTION_FINE_TUNED_MODEL" )
BASE_MODELS_DIR = os.path.join(PARENT_DIR, 'BASEModel')



SYS_prompt = """Below are sql tables schemas paired with instruction that describes a task. Using valid SQLite,
write a response that appropriately completes the request for the provided tables
### Instruction:
{Instruction}
### Input:
{Input}
### Response:
{Response}"""
SEED = 1234