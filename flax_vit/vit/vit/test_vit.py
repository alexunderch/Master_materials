
from os import stat
from statistics import mode
from models import VisualTransformer
import jax
import jax.numpy as jnp

def test_transformer():
    rng = jax.random.PRNGKey(0)
    _, dropout_rng = jax.random.split(rng)

    x = jnp.ones(shape = (2, 32, 32, 3))
    model = VisualTransformer(  n_blocks = 3,
                                block_config = {
                                    "latent_dim": 512 ,
                                    "latent_ffd_dim": 1203,
                                    "n_heads": 8 ,
                                    "dropout_rate_ffd": .1 ,
                                    "dropout_rate_att": .5
                                },
                                output_dim = 2,
                                dropout_embedding = .5,
                                training = True,
                                img_params = (32, 16))
    params = model.init({"params":rng, "dropout": dropout_rng}, x)
    print(jax.tree_map(lambda x: x.shape, params))
    _ = model.apply(params, x, rngs={'dropout': dropout_rng})
    binded_mod = model.bind(params, rngs={'dropout': dropout_rng})
    att_maps = binded_mod.get_attention_maps(x = x)
    print(att_maps[0].shape)


if __name__ == "__main__":
    test_transformer()
