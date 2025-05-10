from config import TOKENIZER
from model_modefied import GPT_INSTRUCTION_FINE_TUNED
import tiktoken
import  torch
from config import BASE_CONFIG ,PADDINGTOKEN,test_df_dir ,SYS_prompt , SEED
import pandas as pd

from src.config import DEVICE


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        # Append sampled index to the running sequence
        if idx_next  == PADDINGTOKEN:
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
            break
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate_text_full(text:str,
                  model:GPT_INSTRUCTION_FINE_TUNED,
                  tokenizer= TOKENIZER , max_new_token:int = 100,
                  context_size = BASE_CONFIG["context_length"] ):
    encoded = tokenizer.encode(text)
    start_from = len(encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    encoded_tensor = encoded_tensor.to(DEVICE)
    model.to(DEVICE)
    generated_text_encoded = generate_text_simple(model= model,
                                                  idx = encoded_tensor ,
                                                  max_new_tokens=max_new_token ,
                                                  context_size=context_size)
    #return generated_text_encoded
    decoded_text = tokenizer.decode(generated_text_encoded[0].to('cpu').tolist()[start_from:])
    return decoded_text

def take_sample(template:str=SYS_prompt , num_sample:int = 10):
    df = pd.read_csv(test_df_dir)
    df = df.sample(n=num_sample,ignore_index=True)
    x = []
    y = []
    for i in df.index:
        instruction = df.loc[i,"instruction"]
        Input = df.loc[i , "input"]
        response = df.loc[i , "response"]
        x.append(template.format(Instruction = instruction , Input = Input , Response = ''))
        y.append(response)
    return x , y