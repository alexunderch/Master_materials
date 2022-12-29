
import sys
sys.path.append(".")
from functools import partial
import flax.linen as nn
from typing import Dict, List, Optional, Tuple
import jax.numpy as jnp
try:
    from .modules import TransformerEncoderBlock
    from .pos_embeddings import TransformerEmbeddings
except:
    from modules import TransformerEncoderBlock
    from pos_embeddings import TransformerEmbeddings


class TransformerHead(nn.Module):
    output_dim: int
    training: bool
    dropout_rate: Optional[float] = .1
    use_bias: Optional[bool] = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = nn.Dropout(rate = self.dropout_rate, deterministic = not self.training)(x)
        out = nn.Dense(self.output_dim, use_bias = self.use_bias)(out)
        return out

class VisualTransformer(nn.Module):
    training: bool 
    n_blocks: int
    block_config: Dict
    output_dim: int
    dropout_embedding: float
    img_params: Tuple[int, int]
    cls_index: int = 0
    
    
    def setup(self) -> None:
        
        self.apply_embedding = TransformerEmbeddings(
            dropout_rate = self.dropout_embedding,
            latent_dim = self.block_config["latent_dim"],
            image_size = self.img_params[0],
            patch_size = self.img_params[1],
            training = self.training
        )
        
        self.encoder_layers = [
            TransformerEncoderBlock(**self.block_config.pop("latent_dim")[0], 
                                    training = self.training) \
            for _ in range(self.n_blocks)
        ]

        self.head = TransformerHead(training = self.training, output_dim = self.output_dim)
                                   

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        
        out = self.apply_embedding(x)
        out = self.encoder_layers[0](out, mask)
        for layer in self.encoder_layers[1:]:
            out = layer(out, mask)

        out = self.head(out[:, self.cls_index, :])
        return out

    def get_attention_maps(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> List[jnp.ndarray]:
        out = self.apply_embedding(x)
        attention_maps_= []
        for layer in self.encoder_layers:
            print(layer)
            attention_maps_.append(layer.get_attention_map(out, mask))
            out = layer(out, mask)

        return attention_maps_ 
