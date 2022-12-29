import sys
from typing import Dict
sys.path.append("..")
sys.path.append(".")

import jax
import jax.numpy as jnp
from nn.train_main import full_trainining
import wandb
import yaml
from pathlib import Path

def main():

    print("devices for training:", jax.device_count())
    config = yaml.safe_load(Path('config.yaml').read_text())
    wandb.init(
        project = config["logger_kwargs"]["project_name"],
        job_type = config["logger_kwargs"]["wandb_config"]["job_type"])

    seed = config["seed"]
    training_config = config["config"]

    logger_kwargs = config["logger_kwargs"]
    dataset_kwargs = config["dataset_kwargs"]

    ##############################################
    # SWEEP KWARGS
    ##############################################
    #TODO
    ##############################################
    # SWEEP KWARGS
    ##############################################
    
    full_trainining(
        config = training_config,
        seed = seed,
        logger_kwargs = logger_kwargs,
        dataset_kwargs = dataset_kwargs
    )

def hyperparameter_sweep(
                        project_name: str,
                        parameters_dict: Dict
                        ):
    sweep_configuration = {
        'method': 'bayes',
        'name': 'SWEEP',
        'metric': {'goal': 'maximize', 'name': 'test_f1'},
        'parameters': parameters_dict
    }

    sweep_id = wandb.sweep(sweep = sweep_configuration, project = project_name)
    return sweep_id

if __name__ == "__main__":
    main()
    #todo sweep config
    # parameters_dict =  {
    #     'config':
    #     {
    #         'batch_size': {'values': [40]},
    #         'num_epochs': {'values': [20]},
    #         'learning_rate': {'max': 0.1, 'min': 1e-4},
    #         'weight_decay': {'max': 0.1, 'min': 1e-4},
    #         'clip_parameter': {'max': 10., 'min': .1},
    #         'model_config': {'n_blocks': {'values': [4, 8, 12]}}
    #     }

    # }
    # sweep_id = hyperparameter_sweep("training-ViT-with-flax", 
    #                                 parameters_dict)
    # wandb.agent(sweep_id, function = main, count = 4)