from models import VisualTransformer
import flax.linen as nn
from functools import partial
from typing import Dict

def trainable_model(config: Dict) -> nn.Module:
    config["training"] = True
    return partial(VisualTransformer, **config)

def inference_model(config: Dict) -> nn.Module:
    config["training"] = False
    return partial(VisualTransformer, **config)
    