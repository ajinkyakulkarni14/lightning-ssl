import argparse
from src.core.train_ssl import train_ssl

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        choices=["dino", "byol"],
        default="dino",
        help="model name (dino/byol).",
    )
    
    parser.add_argument(
        "--config",
        default="config/ssl/dino.yml",
        help="path to YAML configuration file.",
    )
    
    parser.add_argument(
        "--data-dir",
        default="/Users/riccardomusmeci/Developer/data/stl10",
        help="dataset path"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        default="/Users/riccardomusmeci/Developer/experiments/lightning-ssl",
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
    train_ssl(args)    