import argparse
from src.core.linear_eval import linear_eval

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--ssl-config",
        default="/Users/riccardomusmeci/Developer/experiments/lightning-ssl/dino-vit-tiny-sagemaker-stl10-96/dino.yml",
        # required=True,
        help="path to SSL model YAML configuration file",
    )
    
    parser.add_argument(
        "--linear-config",
        default="config/linear/config.yml",
        help="path to YAML configuration file for the linear classifier.",
    )
    
    parser.add_argument(
        "--ssl-ckpt",
        default="/Users/riccardomusmeci/Developer/experiments/lightning-ssl/dino-vit-tiny-sagemaker-stl10-96/checkpoints/epoch=221-step=346764-val_loss=3.415.ckpt",
        # required=True,
        help="path to ssl model checkpoints",
    )
    
    parser.add_argument(
        "--data-dir",
        default="/Users/riccardomusmeci/Developer/data/stl10",
        help="dataset path"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        default="/Users/riccardomusmeci/Developer/experiments/lightning-ssl/dino-vit-tiny-sagemaker-stl10-96/linear/checkpoints",
        help="where to save checkpoints during training"
    )
    
    parser.add_argument(
        "--resume-from",
        default=None,
        help="path to checkpoint (ckpt file) to resume training from"
    )
    
    parser.add_argument(
        "--seed",
        default=42
    )

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    linear_eval(args)    