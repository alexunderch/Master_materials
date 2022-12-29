import sys


sys.path.append("..")
sys.path.append(".")

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from flax.training.train_state import TrainState
import jax

import jax.numpy as jnp
import optax
from vit.models import VisualTransformer
from train_utils import (create_learning_rate_fn, 
                         make_infer_fn, 
                         make_update_fn,
                         init_train_state)
from train_main import full_trainining


import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=3'

def test_step():
    #TODO
    print(jax.devices("cpu"))
    rng = jax.random.PRNGKey(0)
    _, dropout_rng = jax.random.split(rng)
    model = VisualTransformer(n_blocks = 3,
                               block_config = {
                                    "latent_dim": 512 ,
                                    "latent_ffd_dim": 1203,
                                    "n_heads": 8 ,
                                    "dropout_rate_ffd": .1 ,
                                    "dropout_rate_att": .5
                                },
                                output_dim = 2,
                                dropout_embedding = .5,
                                img_params = (32, 16))

    
    test_config = dict(
                        weight_decay = .001,
                        lr = 1e-5,
                        warmup_epochs = 3,
                        num_epochs=  1,
                        num_classes = 2)

    lr_schedule = create_learning_rate_fn(config = test_config,
                                          base_learning_rate = 1e-5,
                                          steps_per_epoch = 2)

    test_batch = {
        "image": jnp.zeros((2, 32, 32, 3)),
        "label": jnp.zeros((2, 1))
    }
    params = model.init(rng, test_batch['image'])["params"]


    test_state = TrainState.create(apply_fn=model.apply,
                                   params=params,
                                   tx= optax.adam(learning_rate = 1e-5))
    

    eval_step = make_infer_fn(
                                num_classes = 2,
                                config =test_config  
                             )
    train_step = make_update_fn(
                                create_learning_rate_fn(test_config, 2), 
                                num_classes = 2, 
                                config = test_config
                               )
    
    print(
        train_step(state = test_state, 
               batch = test_batch, 
               dropout_rng = dropout_rng)[1]
    )
    

    print(
        eval_step(state = test_state, 
               batch = test_batch, 
               dropout_rng = dropout_rng)
    )

def test_full_training():
    
    image_size = 224
    patch_size = 16
    print(jax.device_count())
    
    model_config = dict(
                        n_blocks = 3,
                        block_config = {
                            "latent_dim": 512 ,
                            "latent_ffd_dim": 768,
                            "n_heads": 8 ,
                            "dropout_rate_ffd": .1 ,
                            "dropout_rate_att": .5
                        },
                        dropout_embedding = .5,
                        img_params = (image_size, patch_size)
                        ) 
    seed = 42

    config = dict(model_config = model_config)

    config.update(
                    dict(weight_decay = .001,
                        learning_rate = 1e-5,
                        warmup_epochs = 1,
                        num_epochs=  1,
                        clip_parameter = 1.,
                        batch_size = 10
                        )
                )

    logger_kwargs = dict(
        project_name = "training ViT with flax",
        wandb_config = dict()
    )

    dataset_kwargs = dict(
        dataset_name = "caltech101",
        validation_split = .3,
        force_download = False,

    )

    full_trainining(
        config = config,
        seed = seed,
        logger_kwargs = logger_kwargs,
        dataset_kwargs = dataset_kwargs
    )

if __name__ == "__main__":
    # test_step()
    test_full_training()