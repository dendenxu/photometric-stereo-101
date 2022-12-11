# Computer Vision Homework #1 Report

## Homework Settings

> Given linearized render of an object from one view with multiple light direction and light intensity, recover the normal and albedo image of the object using a) a simple linear Lambertian model and b) a simple linear Lambertian model with highlight and shadow removal.

### Data specification

- 96 images of an object with a resolution of 612 * 512, stored as 16-bit `png`
- A mask file for valid object pixels
- Light directions and light intensities used for all rendered objects in `txt`

### Code structure

This code is implemented in `python`, powered by `pytorch`, other required packages are listed in `requirements.txt`

Assume you've got a valid `python` installation, run:

```shell
pip install -r requirements.txt # to install dependencies
```

We actually recommend installing `pytorch` with `cuda` support through `conda` (possibly with the help of `mamba`), but that hustle is beyond the scope of this small project... (just like the hustle of installing `matlab`, luckily we don't need `matlab`)

- `main.py`: for the main logic of the algorithm implemented in this project
- `utils.py`: for utility functions used by the main program
- `run_exps.py`: a script for running all experiment settings on all provided datasets (given a `--data_root` directory)

Both `main.py` and `run_exps.py` provides command line interfaces, run them with `--help` to see help to the command line arguments. Optionally, see `run_exps.py` for example use cases of `main.py`.

## Experiment Results

### Experiment Settings

We implemented our algorithms in `pytorch` by considering the noisy data and abundant input as a perfect candidate for optimization based algorithms. Another benefit is to easily implement parallel algorithms running on GPU to make these largely parallel optimization finish in time.

We tested three versions of the simple photometric algorithm:

- A vanilla version that uses linear Lambertian model to reconstruct the images

  - We use mask to compute indices for valid pixels to optimize, we set this number to `P`, note that this is shared across lights.
  - We set `normal` and `albedo` as optimizable to perform gradient descent on them (both 3D tensors)
  - We activate raw` normal` values with an `normalize` operation to make it represent a 3D direction (2 DoF). We chose this 3D Cartesian representation for easier implementation. 
  - We activate raw `albedo` values with a `softplus` activation to make them non-negative (no limit on upper bound). This achieves better results than a simple `relu` since `relu` cuts gradients to negative values
  - We perform forward pass on all pixels to be optimized all at once (~1G VRAM)
  - We use an `Adam` optimizer with a learning rate of `1e-1` and optimize for 5k iterations (takes 15 seconds on a 3090)
  - We do not clip rendered `rgb` values during training, since the initialized normal maybe incorrect, producing incorrect unshaded area. Although this violates the Lambert shading model, it's better for optimization. (And simpler to implement)

- A simple global pixel value sorting version to remove effects of shadows and highlights

  - We sort all pixel values (masked) and get indices of valid pixels within masked pixels, note that this is **not** shared across lights. 

  - The reason for computing indices instead of using the original `mask` is that `cpu` code and `gpu` code are executed asynchronously as long as no explicit sync signals or syncing required operation is performed (a rule of thumb: as long as we won't need `gpu`' s output for launching the next `cuda` kernel, we are good. `cuda` kernel launches require sizes to be predetermined.). And taking indices from a mask is such **syncing required operation** since future `cuda` kernel launches will need the size of the indices (computed from the `mask` tensor).

  - We use an `--rtol_hi` and `--rtol_lo` parameter to control ratio to discard the pixel values. We set this value to 0.05 (5%) and 0.01 (1%) respectively in all experiments.

  - We convert the `rgb` values to grayscale using:

    ```python
    0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ```

     before sorting them.
