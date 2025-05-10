import  torch
from torch.nn.functional import  cross_entropy
from config import DEVICE,TOKENIZER
from model_modefied import GPT_INSTRUCTION_FINE_TUNED
from torch.utils.data import DataLoader
from text_generation import generate_text_full , take_sample

def loss_on_batch(model:GPT_INSTRUCTION_FINE_TUNED,
                  x:torch.tensor , y:torch.tensor , device=DEVICE):
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    loss = cross_entropy(logits.flatten(0,1) , y.flatten())
    return  loss

def evaluate(loader:DataLoader ,
             model:GPT_INSTRUCTION_FINE_TUNED , device = DEVICE , num_batch:int = 50):
    model.to(device)
    model.eval()
    total_loss= 0.0

    if num_batch is not None :
        num_batch = min(len(loader), num_batch)

    else:
        num_batch = len(loader)

    for i,(x , y) in enumerate(loader):
        if i >= num_batch:
            break

        with torch.no_grad():
            loss= loss_on_batch(model= model ,x=x,y=y,device=device )
        total_loss += loss.item()


    model.train()
    return total_loss / num_batch

def visual_inspection(model:GPT_INSTRUCTION_FINE_TUNED ,tokenizer = TOKENIZER
                      ,num_sample:int = 5,max_new_tokens:int= 100):
    model.eval()
    samples = take_sample(num_sample=num_sample)
    for x,y  in zip(samples[0],samples[1]):
        text_generated= generate_text_full(text=x,
                                     model=model,
                                     tokenizer=tokenizer,
                                     max_new_token=max_new_tokens)

        print(f"generated_response  : {text_generated} \n  ground_truth_reponse :  {y} \n\n\n ")


    model.train()


def train(model:GPT_INSTRUCTION_FINE_TUNED ,
          train_loader:DataLoader , val_loader:DataLoader ,
         optimizer:torch.optim,num_epochs:int=2 ,
          log_freq:int=50 ,
          num_batch:int = 10 ,
          device = DEVICE):
    #epoch_loss = dict()
    #step_loss = dict()
    model.train()
    model.to(device)
    for epoch in range( num_epochs):
        batch_loss = 0
        train_loss = 0
        step = 0
        for batch_idx ,(x,y) in enumerate (train_loader):
            optimizer.zero_grad()
            loss = loss_on_batch(model=model , x=x , y = y , device=device)
            loss.backward()
            optimizer.step()
            step+=1
            batch_loss+=loss.item()
            train_loss+=loss.item()
            if batch_idx % log_freq ==0:
                sample_val_loss = evaluate(loader=val_loader,model=model ,
                                           device=device , num_batch=num_batch )
                print(f" epoch num {epoch + 1}  step num {batch_idx} / {len(train_loader)} loss training ={batch_loss / step} , val_loss = {sample_val_loss}")
                visual_inspection(model=model)

                batch_loss = 0
                step = 0


        val_loss = evaluate(loader=val_loader , model=model , device=device , num_batch=None)
        print(f"epoch num {epoch+1} val_loss = {val_loss} , train_loss = {train_loss /len(val_loader)}")
        visual_inspection(model=model , num_sample=10)





