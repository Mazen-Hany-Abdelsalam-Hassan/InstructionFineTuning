import  torch
from torch.nn.functional import  cross_entropy

def calc_loss(prediction: torch.tensor, target: torch.tensor):
    prediction = prediction.flatten(0, 1)
    target = target.flatten(0)
    loss = cross_entropy(input=prediction, target=target, reduction='mean')
    return loss, target.shape[0]

