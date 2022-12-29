from typing import Callable, Optional, Any, Tuple 
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
import jax
from functools import partial


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -jnp.inf, attn_logits)
    attention = nn.softmax(attn_logits, axis = -1)
    values = jnp.matmul(attention, v)
    return values, attention


class ResidualWrapper(nn.Module):
    fn: nn.Module

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        result = self.fn(x, **kwargs) 
        result =  result if isinstance(result, jnp.ndarray) else result[0] 
        return x + result    

class PreNormLayer(nn.Module):
    fn: nn.Module
    norm: Optional[nn.Module] = nn.LayerNorm

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        result = self.fn(self.norm(x), **kwargs)
        return result if isinstance(result, jnp.ndarray) else result[0]    

class FeedForwardLayer(nn.Module):
    training: bool
    latent_dim: int
    dropout_rate: float
    activation: nn.activation
    use_bias: Optional[bool] = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        feature_dim = x.shape[-1]
        out = nn.Dense(self.latent_dim, use_bias = self.use_bias)(x)
        out = nn.Dropout(rate = self.dropout_rate, deterministic = not self.training)(out)
        out = self.activation(out)
        out = nn.Dense(feature_dim, use_bias = self.use_bias)(out)
        return out



class MultiHeadSelfAttentionLayer(nn.Module):
    training: bool
    dropout_rate: float
    n_heads: int 
    use_bias: Optional[bool] = False
    attention_function: Optional[Callable] = scaled_dot_product

    
    def prepare_qkv(self, x: jnp.ndarray):
        batch_size, seq_length, hidden_dim = x.shape
        assert hidden_dim // self.n_heads
        qkv = nn.Dense(3 * hidden_dim, 
                       use_bias = self.use_bias,
                       kernel_init = nn.initializers.xavier_uniform(),  
                       bias_init = nn.initializers.zeros)(x)
        qkv = jnp.swapaxes(qkv.reshape(batch_size, seq_length, self.n_heads, -1), 1, 2)
        q, k, v = jnp.array_split(qkv, 3, axis = -1)
        return q, k, v

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        batch_size, seq_length, hidden_dim = x.shape
        o_dropout = nn.Dropout(self.dropout_rate, deterministic = not self.training)

        q, k, v = self.prepare_qkv(x)
        values, attention = jax.vmap(self.attention_function)(q, k, v, mask = mask)
        values = jnp.swapaxes(values, 1, 2)
        values = values.reshape(batch_size, seq_length, hidden_dim)

        values = nn.Dense(hidden_dim, 
                          use_bias = self.use_bias,
                          kernel_init = nn.initializers.xavier_uniform(),  
                          bias_init = nn.initializers.zeros)(values)
        values =  o_dropout(values)
                           
        return values, attention
        

class TransformerEncoderBlock(nn.Module):
    training: bool
    n_heads: int
    latent_ffd_dim: int
    dropout_rate_ffd: float
    dropout_rate_att: float
    use_bias_att: Optional[bool] = False
    use_bias_ffd: Optional[bool] = True
    attention_function: Optional[Callable] = scaled_dot_product
    
    def setup(self) -> None:
        self.self_attn = MultiHeadSelfAttentionLayer(
                            **dict(training = self.training,
                                    dropout_rate = self.dropout_rate_att,
                                    n_heads = self.n_heads,
                                    use_bias = self.use_bias_att,
                                    attention_function = self.attention_function)
                            )

        self.norm_attention = PreNormLayer(fn = self.self_attn,
                                           norm  = nn.LayerNorm())

        self.residual_norm_attention = ResidualWrapper(fn = self.norm_attention)
        
        ffd = FeedForwardLayer(
                                training = self.training,     
                                latent_dim= self.latent_ffd_dim,
                                dropout_rate = self.dropout_rate_ffd,
                                activation = nn.gelu,
                                use_bias = self.use_bias_ffd
                            )

        norm_ffd = PreNormLayer(fn = ffd,
                                norm  = nn.LayerNorm())

        self.residual_norm_ffd = ResidualWrapper(fn = norm_ffd)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        return self.residual_norm_ffd(
            self.residual_norm_attention(x, mask=mask)
        )
    
    def get_attention_map(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        return self.self_attn(
            #NOTE: getting a map of prenormalized input without residual
            self.norm_attention(x, mask=mask)
        )[1]


        