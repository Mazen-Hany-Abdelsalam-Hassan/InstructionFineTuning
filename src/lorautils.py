import torch
from torch.nn import  Module
import torch.nn as nn
from config import SEED
class LoRA_layer(Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 16,
                 alpha: float = 1.5):
        super().__init__()
        self.input_transformation = nn.Linear(in_features=input_dim,
                                              out_features=rank, bias=False)
        self.output_transformation = nn.Linear(in_features=rank
                                               , out_features=output_dim, bias=False)
        nn.init.zeros_(self.output_transformation.weight)
        nn.init.kaiming_uniform_(self.input_transformation.weight, a=5 ** .5)
        self.alpha = alpha

    def forward(self, x):
        state = self.input_transformation(x)
        return self.output_transformation(state) * self.alpha


class Linear_with_LoRA(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int = 16, alpha: float = 1.5):
        super().__init__()
        self.linear_layer = linear_layer
        input_dim = self.linear_layer.in_features
        output_dim = self.linear_layer.out_features
        self.lora_layer = LoRA_layer(input_dim=input_dim
                                     , output_dim=output_dim, rank=rank, alpha=alpha)

    def forward(self, x):
        return self.linaer_layer(x) + self.lora_layer(x)



def Replace_Linear(model:Module , rank:int = 16 , alpha:float= 1.5):
    for child_name , child_layer in model.named_children():
        if isinstance(child_layer , nn.Linear):
            replace_with = Linear_with_LoRA(child_layer ,rank = rank , alpha = alpha)
            setattr(model , child_name , replace_with )
        else:
            Replace_Linear( model = child_layer , rank = rank , alpha = alpha)