import os
import argparse
from tqdm import tqdm
from os.path import join
from utils import run, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/pmsData', help='data root where all datasets are provided as directories, no naming rules required')
    parser.add_argument('--num_correction', default=50, help='number of self coorection iterations to perform')
    parser.add_argument('--device', default='cuda', help='if you have a GPU, use cuda for fast reconstruction, if device is \'cpu\', will use multi-core torch')
    parser.add_argument('--skip', default=[], nargs='*')
    args = parser.parse_args()

    data_paths = [d for d in sorted(os.listdir(args.data_root))]
    data_paths = [d for d in data_paths if d not in args.skip]
    data_paths = [join(args.data_root, d) for d in data_paths]
    data_paths = [d for d in data_paths if os.path.isdir(d)]

    for data_path in data_paths:
        run(f'python main.py --data_root {data_path} --device {args.device} --output_dir lambertian')  # lambertian model
        run(f'python main.py --data_root {data_path} --device {args.device} --output_dir pixel_vals --use_pix')  # use global pixel value removal

        # perform self-correction photometric stereo
        run(f'cp {join(data_path, "pixel_vals/lambertian.pth")} {join(data_path, "difference/lambertian.pth")}')  # for second stage optimization
        run(f'python main.py --data_root {data_path} --device {args.device} --output_dir difference --use_opt --repeat {args.num_correction} --rtol_hi_opt 0.100 --rtol_lo_opt 0.100')  # use difference in rendered values


if __name__ == '__main__':
    main()
