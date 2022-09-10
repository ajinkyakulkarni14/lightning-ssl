import argparse
from src.core.train import train

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        default="dino",
        help="dino/byol",
        choices=["dino", "byol"]
    )
    
    parser.add_argument(
        "--config",
        help="path to YAML configuration file.",
        required=True
    )
    
    parser.add_argument(
        "--data-dir",
        default="/opt/ml/input/data/dataset",
        help="dataset path"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        default="/opt/ml/checkpoints"
    )

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    train(args)    