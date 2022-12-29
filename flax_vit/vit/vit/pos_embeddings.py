from typing import Callable, Any, Tuple
import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class PatchEmbeddings(nn.Module):

    image_size: int
    patch_size: int
    latent_dim: int
    dtype: jnp.dtype = jnp.float32  

    def setup(self):
        self.num_patches =  (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv(
            self.latent_dim,
            kernel_size = (self.patch_size, self.patch_size),
            strides = (self.patch_size, self.patch_size),
            padding = "VALID",
            dtype = self.dtype,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.projection(x)
        batch_size, _, _, channels = x.shape
        return jnp.reshape(x, (batch_size, -1, channels))


def sinusoidal_init(max_len: int = 2048,
                    min_scale: float = 1.0,
                    max_scale: float = 1e+4)-> Callable:

  def init(key: Any, shape: Tuple, dtype: Any = np.float32) -> jnp.ndarray:
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=dtype)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init

class TransformerEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    dropout_rate: float
    latent_dim: int
    image_size: int
    patch_size: int
    training: bool
    type: Literal["learnable", "sinusoid"] = "learnable"
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.cls_token = self.param("cls_token", 
                                    nn.initializers.zeros, 
                                    (1, 1, self.latent_dim))
        self.patch_embeddings = PatchEmbeddings(latent_dim = self.latent_dim,
                                                image_size = self.image_size,
                                                patch_size= self.patch_size,
                                                dtype = self.dtype)
        num_patches = self.patch_embeddings.num_patches

        pos_emb_shape = (1, num_patches + 1, self.latent_dim)
        if self.type == "learnable":
            self.position_embeddings = self.param(
                "position_embeddings", 
                nn.initializers.zeros, 
                pos_emb_shape
            )
        elif self.type == "sinusoid":
            self.position_embeddings = sinusoidal_init(max_len = num_patches + 1)(
                None,
                pos_emb_shape,
            )
            
        self.dropout = nn.Dropout(rate=self.dropout_rate, deterministic = self.training)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size = x.shape[0]

        embeddings = self.patch_embeddings(x)
        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.latent_dim))
        embeddings = jax.lax.concatenate([cls_tokens, embeddings], dimension = 1) 
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    