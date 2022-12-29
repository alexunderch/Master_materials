from multiprocessing.connection import wait
import sys
sys.path.append("..")
sys.path.append(".")

from typing import Dict, List, Optional, Tuple
try:
    from .train_utils import (make_update_fn, 
                            make_infer_fn,
                            accumulate_metrics, 
                            create_learning_rate_fn, 
                            init_train_state,
                            copy_train_state
                            )

    from .common_utils import (convert_hidden_state_to_image, 
                            save_checkpoint_wandb, 
                            restore_checkpoint_wandb,
                            )
    from .wandb_logger import WandbLogger, make_cli_logger

except:
    from train_utils import (make_update_fn, 
                            make_infer_fn,
                            accumulate_metrics, 
                            create_learning_rate_fn, 
                            init_train_state,
                            copy_train_state
                            )

    from common_utils import (convert_hidden_state_to_image, 
                            save_checkpoint_wandb, 
                            restore_checkpoint_wandb,
                            )
    from wandb_logger import WandbLogger, make_cli_logger

from tf_data_processing.input_pipeline import (prepare_data, 
                                              get_classes, 
                                              prefetch
                                              )
from vit.models import VisualTransformer
import jax, flax
from tqdm import tqdm
import jax.numpy as jnp
from flax.training.train_state import TrainState
import tensorflow_datasets as tfds
from wandb.errors import CommError

def full_trainining(
                    config: Dict,
                    seed: int,
                    dataset_kwargs: Dict,
                    logger_kwargs: Dict,
                    state: Optional[TrainState] = None,
                    logger: WandbLogger = WandbLogger,
                    ) -> Tuple[TrainState, TrainState]:

    rng = jax.random.PRNGKey(seed = seed)
    n_prefetch = jax.device_count()
    train_dataset, eval_dataset, test_dataset, ds_info = prepare_data(
                                                                      batch_size = config["batch_size"],
                                                                      image_size = config["model_config"]["img_params"][0],
                                                                      **dataset_kwargs,
                                                                     )
    num_classes = len(get_classes(ds_info))
    config["model_config"].update({"output_dim": num_classes})

    if state is None:
        state = init_train_state(
                                model = VisualTransformer(training = True, **config["model_config"]), 
                                random_key = rng, 
                                shape = next(iter(tfds.as_numpy(train_dataset)))[0].shape,
                                config = config,
                                steps_per_epoch = len(train_dataset)
                                )

    eval_state = init_train_state(model = VisualTransformer(training = False, **config["model_config"]), 
                                random_key = rng, 
                                shape = next(iter(tfds.as_numpy(eval_dataset)))[0].shape,
                                config = config,
                                steps_per_epoch = 1
                                ) 
    level, cli_logger = make_cli_logger()
    wandb_logger = logger(
                          training_config = config,
                          **logger_kwargs
                          )
    eval_step = make_infer_fn(
                                num_classes = num_classes,
                                config =config  
                             )
    train_step = make_update_fn(
                                create_learning_rate_fn(config, len(train_dataset)), 
                                num_classes = num_classes, 
                                config = config
                               )
    
    _, dropout_rng =  jax.random.split(rng)

    batch_metrics = dict(
                    train = [],
                    eval = [],
                    test = []
                    )
    for epoch in tqdm(range(1, config["num_epochs"] + 1)):

        best_epoch_eval_loss = jnp.inf
        # #### training phase ####
        cli_logger.log(level, f"training epoch {epoch}")
        state = flax.jax_utils.replicate(state)
        for batch in tqdm(iter(tfds.as_numpy(train_dataset)), total = len(train_dataset)):
            batch = dict(
                        image = batch[0],
                        label = batch[1]
                        )
            state, metrics = train_step(
                                        state = state,
                                        batch = flax.jax_utils.replicate(batch),
                                        rng = flax.jax_utils.replicate(dropout_rng),        
                                        )
            batch_metrics["train"].append(metrics)

        batch_metrics["train"] = accumulate_metrics(batch_metrics["train"])

        cli_logger.log(20, "train logs:\n" + "\n".join([f"{k}: {v}" for k, v in batch_metrics["train"].items()]))
        wandb_logger.log_metrics(batch_metrics["train"], step = epoch, prefix = "train")

        state = flax.jax_utils.unreplicate(state)
        #### evaluation phase ####
        cli_logger.log(level, f"evaluating epoch {epoch}")
        eval_state = copy_train_state(apply_fn = eval_state.apply_fn,
                                      params = state.params)
        for batch in tqdm(iter(tfds.as_numpy(eval_dataset)), total = len(eval_dataset)):
            batch = dict(
                        image = batch[0],
                        label = batch[1]
                        )

            metrics = eval_step( 
                                state = flax.jax_utils.replicate(eval_state), 
                                batch = flax.jax_utils.replicate(batch),
                                rng = flax.jax_utils.replicate(dropout_rng),  
                                )
            batch_metrics["eval"].append(metrics)

        batch_metrics["eval"] = accumulate_metrics(batch_metrics["eval"])
        cli_logger.log(20, "eval logs:\n" + "\n".join([f"{k}: {v}" for k, v in batch_metrics["eval"].items()]))

        if batch_metrics["eval"]["loss"] < best_epoch_eval_loss:
            cli_logger.log(level, f"saving checkpoint")

            best_epoch_eval_loss = batch_metrics["eval"]["loss"]
            save_checkpoint_wandb(ckpt_path = "ckpt_file.pth", 
                                  state = eval_state, 
                                  step = epoch)
        
        wandb_logger.log_metrics(batch_metrics["eval"], step = epoch, prefix = "eval")
        batch_metrics["train"], batch_metrics["eval"] = [], []
        

    #### testing phase ####
    cli_logger.log(level, f"testing")
    restored_best_state = eval_state
    try:
        restored_best_state = restore_checkpoint_wandb("ckpt_file.pth", restored_best_state)
    except:
        restored_best_state = TrainState(step = eval_state.step,
                                        apply_fn = eval_state.apply_fn,
                                        params = eval_state.params,
                                        tx = state.tx,
                                        opt_state = state.opt_state)
    for batch in tqdm(iter(tfds.as_numpy(test_dataset)), total = len(test_dataset)):
        batch = dict(
                    image = batch[0],
                    label = batch[1]
                    )
        metrics =  eval_step( 
                            state = flax.jax_utils.replicate(restored_best_state), 
                            batch = flax.jax_utils.replicate(batch),
                            rng = flax.jax_utils.replicate(dropout_rng), 
                            )
        batch_metrics["test"].append(metrics)

    batch_metrics["test"] = accumulate_metrics(batch_metrics["test"])
    cli_logger.log(20, "test logs:\n" + "\n".join([f"{k}: {v}" for k, v in batch_metrics["test"].items()]))
    wandb_logger.log_metrics(batch_metrics["test"], step = epoch, prefix = "test")

    return state, eval_state, restored_best_state

        
                            
