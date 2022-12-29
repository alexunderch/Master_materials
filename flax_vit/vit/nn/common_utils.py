from pathlib import Path
from flax.training import checkpoints
import os
from flax.training.train_state import TrainState
import jax.numpy as jnp
import optax
from dataclasses import dataclass, field
from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)
import matplotlib.pyplot as plt
import jax
import wandb


def save_checkpoint(state: TrainState,
                    checkpoint_dir: Path) -> None:
    os.makedirs(checkpoint_dir, exist_ok = True)
    checkpoints.save_checkpoint(
        ckpt_dir = checkpoint_dir,
        prefix = "chkpnt_",
        target = state,
        step = state.step,
        overwrite = False, 
        keep = 1
    )

def restore_checkpoint(state: TrainState, 
                      checkpoint_dir: Path) -> TrainState:
    
    assert os.path.isdir(checkpoint_dir)
    return checkpoints.restore_checkpoint(
        ckpt_dir = checkpoint_dir,
        target = state,
        step = state.step,
        prefix = "chkpnt_"
    )


def checkpoint_exists(ckpt_file) -> bool:
    return os.path.isfile(ckpt_file)

####wandb utils
def save_checkpoint_wandb(ckpt_path, state: TrainState, step: int):
    
    artifact = wandb.Artifact(
        f'{wandb.run.name}-checkpoint', type='model'
    )
    with open(ckpt_path, "wb") as outfile:
        outfile.write(msgpack_serialize(to_state_dict(state)))
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact, aliases=["latest", f"step_{step}"])


def restore_checkpoint_wandb(ckpt_file, state: TrainState):
    assert checkpoint_exists(ckpt_file)
    artifact = wandb.use_artifact(
                                    f'{wandb.run.name}-checkpoint:latest',
                                 )
    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, ckpt_file)
    with open(ckpt_path, "rb") as data_file:
        byte_data = data_file.read()
    return from_bytes(state, byte_data)

# def hyperparameter_sweep(sweep_config: Dict):
#     wandb.sweep(sweep_config, project = project_name)
####wandb utils

def plot_attention_maps(attn_maps: list, idx: int=0, batch_idx: int = 0) -> None:

    input_data = jnp.arange(attn_maps[batch_idx][idx].shape[-1])
    attn_maps = [jax.device_get(m[idx]) for m in attn_maps]

    num_heads = attn_maps[batch_idx].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
    fig.subplots_adjust(hspace=0.5)
    plt.show()

