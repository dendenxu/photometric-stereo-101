import os
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from os.path import join
from torch.optim import Adam
from termcolor import colored
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


def get_atol(diff: torch.Tensor, rtol_hi: float, rtol_lo: float):
    atol_hi = diff.ravel().topk(int(rtol_hi * diff.numel()), largest=True)[0].min()
    atol_lo = diff.ravel().topk(int(rtol_lo * diff.numel()), largest=False)[0].max()
    log(f'atol_hi: {colored(atol_hi.item(), "magenta")}, atol_lo: {colored(atol_lo.item(), "magenta")}')
    return atol_hi, atol_lo


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
        render = albedo * ints * (normal * dirs).sum(dim=-1, keepdim=True)  # N, P,
        return render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/pmsData/buddhaPNG')
    parser.add_argument('--mask_file', default='mask.png')
    parser.add_argument('--light_dir_file', default='light_directions.txt')
    parser.add_argument('--light_int_file', default='light_intensities.txt')
    parser.add_argument('--image_list', default='filenames.txt')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--device', default='cuda', help='if you have a GPU, use cuda for fast reconstruction, if device is \'cpu\', will use multi-core torch')
    parser.add_argument('--iter', default=20000, type=int, help='number of iterations to perform (maybe after discarding highlights and shadows)')
    parser.add_argument('--rtol_hi', default=0.100, type=float, help='difference ratio in the rendered value with gt value to discard a pixel')
    parser.add_argument('--rtol_lo', default=0.005, type=float, help='difference ratio in the rendered value with gt value to discard a pixel')
    parser.add_argument('--rtol_hi_opt', default=0.050, type=float, help='difference ratio in the rendered value with gt value to discard a pixel')
    parser.add_argument('--rtol_lo_opt', default=0.050, type=float, help='difference ratio in the rendered value with gt value to discard a pixel')
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--restart', action='store_true', help='ignore pretrained weights')
    parser.add_argument('--no_save', action='store_true', help='do not save trained weights (pixels) to disk')
    parser.add_argument('--use_pix', action='store_true', help='use sorted global pixel values')
    parser.add_argument('--use_opt', action='store_true', help='use sorted difference in rendered and gt value (this required training the model with a vanilla model first)')
    args = parser.parse_args()

    # reconfigure paths based on data_root setting
    args.mask_file = join(args.data_root, args.mask_file)
    args.light_dir_file = join(args.data_root, args.light_dir_file)
    args.light_int_file = join(args.data_root, args.light_int_file)
    args.image_list = join(args.data_root, args.image_list)
    args.output_dir = join(args.data_root, args.output_dir)
    log(f'output will be saved to: {colored(args.output_dir, "yellow")}')

    # load images & mask & light direction and intensity from disk
    img_list = load_image_list(args.image_list, args.data_root)
    # just a normalization for stable optimization (the input should alreay have beed linearized)
    imgs = np.array(parallel_execution(img_list, action=load_unchanged)).astype(np.float32) / 65535.0
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
    iter = 0
    chkpt_path = join(args.output_dir, 'lambertian.pth')
    if os.path.exists(chkpt_path) and (not args.restart or args.use_opt):
        iter = load_model(chkpt_path, lambertian, optim)
        if args.use_opt:
            iter = 0
    elif args.use_opt:
        log(f'not reloading from disk, --use_opt will be ignored', 'yellow')
        args.use_opt = False

    if args.repeat != 1 and not args.use_opt:
        log(f'not using self-correction, will set repeat to 1', 'yellow')
        args.repeat = 1

    for i in range(args.repeat):
        if iter < args.iter:
            if args.use_opt or args.use_pix:
                # find extra pixels, mark them as highlights or shadows
                diff = rgb_to_gray(rgbs)  # N, P
                atol_hi, atol_lo = get_atol(diff, args.rtol_hi, args.rtol_lo)
                valid = (diff < atol_hi) & (diff > atol_lo)

                if args.use_opt:
                    # use relative error to determine invalid pixels
                    render = lambertian(dirs, ints)
                    diff = rgb_to_gray(rgbs) - rgb_to_gray(render)  # N, P
                    atol_hi, atol_lo = get_atol(diff, args.rtol_hi_opt, args.rtol_lo_opt)
                    valid = valid & (diff < atol_hi) & (diff > atol_lo)

                # perform this async operation only once
                valid = valid.nonzero(as_tuple=True)

            # perform second stage training without highlights or shadows
            prev = 0  # previous psnr value
            pbar = tqdm(total=args.iter - iter)
            for i in range(args.iter - iter):
                optim.zero_grad()
                if args.use_opt or args.use_pix:
                    render = lambertian(dirs, ints, valid)
                    loss = mse(rgbs[valid], render)
                else:
                    render = lambertian(dirs, ints)
                    loss = mse(rgbs, render)

                # early termination
                psnr = 10 * torch.log10(1 / loss)
                if (psnr - prev) < torch.finfo(loss.dtype).eps:
                    log(f'early termination, reason: {colored("[convergence]", "green")}, diff in psnr: {colored(f"{psnr - prev:.12f}", "magenta")}')
                    break
                prev = psnr

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
    save_unchanged(join(args.output_dir, 'normal.png'), (normal_image.clip(0, 1) * 255.0).astype(np.uint8))

    # save albedo image as 16-bit png
    albedo = lambertian.albedo.detach()
    albedo_image = albedo.new_zeros(H, W, 3)
    albedo_image[mask[..., 0]] = albedo
    albedo_image = albedo_image.detach().cpu().numpy()
    save_unchanged(join(args.output_dir, 'albedo.png'), (albedo_image.clip(0) * 65535.0).astype(np.uint16))

    # save re-rendered image as 16-bit png
    viewdir = normalize(torch.tensor([0., 0., 1.], dtype=torch.float, device=args.device))  # 3,
    intensity = torch.tensor([2., 2., 2.], dtype=torch.float, device=args.device)  # 3,
    render = lambertian(viewdir[None], intensity[None])[0]  # use light intensity same as viewing direction
    render_image = render.new_zeros(H, W, 3)
    render_image[mask[..., 0]] = render
    render_image = render_image.detach().cpu().numpy()
    save_unchanged(join(args.output_dir, 'render.png'), (render_image.clip(0) * 65535.0).astype(np.uint16))


if __name__ == "__main__":
    main()
