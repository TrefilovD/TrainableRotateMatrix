import argparse
import os

from engine import train

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--frequency_vis", type=int, default=10, help="частота визуализации во время обучения (в эпохах)")
    parser.add_argument("--frequency_log", type=int, default=10, help="частота логирования во время обучения (в эпохах)")
    parser.add_argument("--save_dir", type=str, default="./src/results", help="путь до папки, куда будут сохраняться визуализированные результаты")
    parser.add_argument("--checkpoint_path", type=str, default="./src/checkpoint", help="путь до лучшего чекпоинта")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)