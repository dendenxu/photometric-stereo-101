import os
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from os.path import join
from torch.optim import Adam
from utils import load_mask, load_unchanged, load_image, save_image, parallel_execution, make_params, make_buffer, normalize, mse, save_unchanged, log, run, dotdict


def load_image_list(path: str, data_root: str = ''):
    with open(path, 'r') as f:
        image_list = [join(data_root, line.strip()) for line in f]
        return image_list


def load_vec3(path: str):
    with open(path, 'r') as f:
        value = [line.strip().split() for line in f]
        value = np.array(value, dtype=np.float32)
        return value


def save_model(chkpt_path: str, net: nn.Module, optim: torch.optim.Optimizer, iter: int):
    os.makedirs(os.path.dirname(chkpt_path), exist_ok=True)
    state = dotdict()
    state.net = net.state_dict()
    state.optim = optim.state_dict()
    state.iter = iter
    torch.save(state, chkpt_path)


def load_model(chkpt_path: str, net: nn.Module, optim: torch.optim.Optimizer):
    state = torch.load(chkpt_path)
    state = dotdict(state)
    net.load_state_dict(state.net)
    optim.load_state_dict(state.optim)
    return state.iter


def rgb_to_gray(rgb: torch.Tensor):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


class Lambertian(nn.Module):
    def __init__(self, P: int) -> None:
        # P: number of pixels to recover
        super().__init__()
        self.P = P

        self._albedo = make_params(torch.rand(P, 3))  # P,
        self._normal = make_params(torch.rand(P, 3))  # P,

        self._albedo_actvn = nn.Softplus()
        self._normal_actvn = normalize

    @property
    def albedo(self) -> torch.Tensor:
        return self._albedo_actvn(self._albedo)

    @property
    def normal(self) -> torch.Tensor:
        return self._normal_actvn(self._normal)

    def forward(self, dirs: torch.Tensor, ints: torch.Tensor, valid: torch.Tensor = slice(None)) -> torch.Tensor:
        # dirs: N, 3
        # ints: N, 3
        N, C = dirs.shape
        P = self.P

        albedo = self.albedo[None].expand(N, P, 3)[valid]  # 1, P, 3
        normal = self.normal[None].expand(N, P, 3)[valid]  # 1, P, 3
        dirs = dirs[:, None].expand(N, P, 3)[valid]  # N, 1, 3
        ints = ints[:, None].expand(N, P, 3)[valid]  # N, 1, 3
        # lambertian model: I = albedo * normal @ dirs * ints
        render = albedo * ints * (normal * dirs).sum(dim=-1, keepdim=True)  # N, P, 3
        return render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/pmsData/buddhaPNG')
    parser.add_argument('--mask_file', default='mask.png')
    parser.add_argument('--light_dir_file', default='light_directions.txt')
    parser.add_argument('--light_int_file', default='light_intensities.txt')
    parser.add_argument('--image_list', default='filenames.txt')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--iter', default=0, type=int)
    parser.add_argument('--high_low_rtol', default=0.05, type=float, help='difference ratio in the rendered value with gt value to discard a pixel')
    parser.add_argument('--high_low_iter', default=10000, type=int, help='number of iterations to perform after discarding highlights and shadows')
    parser.add_argument('--lr', default=5e-2, type=float)
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--use_opt', action='store_true', help='use difference in rendered and gt value')
    args = parser.parse_args()

    # reconfigure paths based on data_root setting
    args.mask_file = join(args.data_root, args.mask_file)
    args.light_dir_file = join(args.data_root, args.light_dir_file)
    args.light_int_file = join(args.data_root, args.light_int_file)
    args.image_list = join(args.data_root, args.image_list)
    args.output_dir = join(args.data_root, args.output_dir)

    # load images & mask & light direction and intensity from disk
    img_list = load_image_list(args.image_list, args.data_root)
    # just a normalization for stable optimization (the input should alreay have beed linearized)
    imgs = np.array(parallel_execution(img_list, action=load_unchanged)).astype(np.float32) / 65536
    mask = load_mask(args.mask_file)
    dirs = load_vec3(args.light_dir_file)
    ints = load_vec3(args.light_int_file)

    # move loaded data onto gpu
    imgs = torch.from_numpy(imgs).to(args.device, non_blocking=True)  # N, H, W, 3 (linearized)
    mask = torch.from_numpy(mask).to(args.device, non_blocking=True)  # H, W, 1
    dirs = torch.from_numpy(dirs).to(args.device, non_blocking=True)  # N, 3
    ints = torch.from_numpy(ints).to(args.device, non_blocking=True)  # N, 3

    # load shapes and pixels to optimize
    N, H, W, C = imgs.shape
    rgbs = imgs[torch.arange(N)[..., None], mask[..., 0]]  # get valid pixels
    N, P, C = rgbs.shape

    # construct the simple lambertian model and corresponding optimizer
    lambertian = Lambertian(P).to(args.device, non_blocking=True)
    optim = Adam(lambertian.parameters(), lr=args.lr)

    # maybe reload previous model from disk, since the training is merely a few seconds, we won't implement a resume option
    chkpt_path = join(args.output_dir, 'lambertian.pth')
    if os.path.exists(chkpt_path) and not args.restart:
        iter = load_model(chkpt_path, lambertian, optim)
    else:
        iter = 0

    if iter < args.iter:
        # perform optimization on all pixels
        pbar = tqdm(total=args.iter - iter)
        for i in range(args.iter - iter):
            optim.zero_grad()
            render = lambertian(dirs, ints)
            loss = mse(rgbs, render)
            psnr = 10 * torch.log10(1 / loss)
            loss.backward()
            optim.step()
            pbar.update(1)
            pbar.set_description(f'loss: {loss.item():.8f}, psnr: {psnr.item():.6f}')

        if not args.no_save:
            # save the optimized model to disk (simple lambertian model for now)
            save_model(chkpt_path, lambertian, optim, args.iter)

    if iter < args.iter:
        # find extra pixels, mark them as highlights or shadows
        if args.use_opt:
            render = lambertian(dirs, ints)
            diff = (render - rgbs).norm(dim=-1)  # N, P
        else:
            diff = rgb_to_gray(rgbs)  # N, P
        atol = diff.ravel().topk(int(args.high_low_rtol * (N * P)))[0].min()
        valid = diff < atol
        valid = valid.nonzero(as_tuple=True)

        if args.scratch:
            # reset network to random initialization
            lambertian = Lambertian(P).to(args.device, non_blocking=True)
            optim = Adam(lambertian.parameters(), lr=args.lr)

        # perform second stage training without highlights or shadows
        pbar = tqdm(total=args.iter - iter)
        for i in range(args.iter - iter):
            optim.zero_grad()
            render = lambertian(dirs, ints, valid)
            loss = mse(rgbs[valid], render)
            psnr = 10 * torch.log10(1 / loss)
            loss.backward()
            optim.step()
            pbar.update(1)
            pbar.set_description(f'loss: {loss.item():.8f}, psnr: {psnr.item():.6f}')

        if not args.no_save:
            # save the optimized model to disk (simple lambertian model for now)
            save_model(chkpt_path, lambertian, optim, args.iter)

    # save normal image, regular images 0-255
    normal = lambertian.normal.detach()
    normal_image = normal.new_zeros(H, W, 3)
    normal_image[mask[..., 0]] = (normal + 1) / 2
    normal_image = normal_image.detach().cpu().numpy()
    save_unchanged(join(args.output_dir, 'normal.png'), (normal_image.clip(0, 1) * 255).astype(np.uint8))

    # save albedo image as 16-bit png
    albedo = lambertian.albedo.detach()
    albedo_image = albedo.new_zeros(H, W, 3)
    albedo_image[mask[..., 0]] = albedo
    albedo_image = albedo_image.detach().cpu().numpy()
    save_unchanged(join(args.output_dir, 'albedo.png'), (albedo_image.clip(0) * 65536).astype(np.uint16))

    # save re-rendered image as 16-bit png
    viewdir = normalize(torch.tensor([0., 0., 1.], dtype=torch.float, device=args.device))  # 3,
    intensity = torch.tensor([2., 2., 2.], dtype=torch.float, device=args.device)  # 3,
    render = lambertian(viewdir[None], intensity[None])[0]  # use light intensity same as viewing direction
    render_image = render.new_zeros(H, W, 3)
    render_image[mask[..., 0]] = render
    render_image = render_image.detach().cpu().numpy()
    save_unchanged(join(args.output_dir, 'render.png'), (render_image.clip(0) * 65536).astype(np.uint16))


if __name__ == "__main__":
    main()
