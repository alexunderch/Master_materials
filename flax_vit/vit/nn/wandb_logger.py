from typing import Dict, Optional, Tuple
import wandb
import jax.numpy as jnp
from dataclasses import dataclass
import logging
import jax
import sys

def make_cli_logger() -> Tuple[int, logging.Logger]:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logging.INFO, logger
@dataclass
class WandbLogger():
    project_name: str
    wandb_config: Dict
    training_config: Dict
    
    def __post_init__(self) -> None:
        wandb.init(
            config = self.training_config,
            project = self.project_name,
            reinit = True,
            **self.wandb_config
        )
    
    def log_metrics(self, metrics: Dict, 
                          step: int,
                          prefix: Optional[str] = None
                    ) -> None:
        if prefix is  None:
            prefix = "x"
        for k in list(metrics.keys()):
            prefix_key = f"{prefix}_{k}"
            wandb.log({prefix_key: metrics.pop(k)}, step = step)
        
    def log_attentionheatmaps(self, data: jnp.ndarray,
                                    metadata: Dict,
                                    idx: int = 0
                            ) -> None:
        
        attn_maps = [m[idx] for m in jax.device_get(data[idx])]
        # wandb.log({'heatmap_with_text': wandb.plots.HeatMap(x_labels, y_labels, matrix_values, show_text=False)})
        # wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
        #                 y_true=ground_truth, preds=predictions,
        #                 class_names=class_names)})

    
        # #TODO: more appropriate logging
        # images = wandb.Image(images, caption = metadata["caption"])
        # wandb.log({metadata["log_as"]: images})