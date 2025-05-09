import torch

from lorautils import Replace_Linear
from config import MODEL_CONFIGS , BASE_CONFIG
from torch.nn import   Module
from GPT2 import  GPTModel
from  model_download import download

class GPT_INSTRUCTION_FINE_TUNED(Module):
    def __init__(self, model_variant: str = "L",
                 rank: int = 16,
                 alpha: float = 1.5,
                 dropout: float = .0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        save_dir = download(model_variant)
        weights = torch.load(save_dir, weights_only=True)
        model_config = MODEL_CONFIGS[model_variant]
        model_config.update(BASE_CONFIG)
        model_config["drop_rate"] = dropout
        self.model = GPTModel(model_config)
        self.model.load_state_dict(weights)
        self._prepare_model()



    def _prepare_model(self):
        for _, layer in self.model.named_parameters():
            layer.requirs_grad = False
        Replace_Linear(model=self.model, rank=self.rank, alpha=self.alpha)

    def forward(self, x):
        return self.model(x)
