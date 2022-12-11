import os
import argparse
from tqdm import tqdm
from os.path import join
from utils import run, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/pmsData', help='data root where all datasets are provided as directories, no naming rules required')
    parser.add_argument('--device', default='cuda', help='if you have a GPU, use cuda for fast reconstruction, if device is \'cpu\', will use multi-core torch')
    args = parser.parse_args()

    data_paths = [d for d in sorted(os.listdir(args.data_root))]
    data_paths = [join(args.data_root, d) for d in data_paths]
    data_paths = [d for d in data_paths if os.path.isdir(d)]

    for data_path in data_paths:
        run(f'python main.py --data_root {data_path} --device {args.device} --restart --output_dir lambertian')  # lambertian model
        run(f'cp {join(data_path, "lambertian/lambertian.pth")} {join(data_path, "difference/lambertian.pth")}')
        run(f'python main.py --data_root {data_path} --device {args.device} --restart --output_dir pixel_vals --use_pix')  # use global pixel value removal
        run(f'python main.py --data_root {data_path} --device {args.device} --restart --output_dir difference --use_opt')  # use difference in rendered values


if __name__ == '__main__':
    main()
